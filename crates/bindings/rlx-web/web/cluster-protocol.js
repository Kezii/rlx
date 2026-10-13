// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Elastic data-parallel coordination for a cluster of browser tabs.
//
// This is the *reduce* half of the contract that `src/cluster.rs` leaves open:
// `ClusterWorker` produces a flat gradient bucket and consumes an averaged one,
// exactly like the `reduce: FnMut(&[f32]) -> Vec<f32>` closure in
// `rlx_runtime::dist::training::run_train`. Here the reduce is a message bus
// across same-origin tabs rather than `rlx-collectives` over TCP.
//
// Kept free of wasm, DOM and BroadcastChannel so the protocol can be tested in
// Node against a stub rank (`tests/cluster_protocol.test.mjs`) — the failure
// modes here (split brain, a stale joiner, a stalled cohort) all produce
// plausible-looking training rather than an error.
//
// Protocol, leader-driven and elastic:
//   hello  rank → all    I exist; here is my batch / ready / running state
//   beat   rank → all    liveness, every `beat` ms
//   round  leader → all  the parameters for step N **and** each rank's shard
//   grad   rank → leader my bucket for step N (+ loss, sample count)
//
// **One message per round, carrying both the parameters and the assignment.**
// This was two — a `commit` with the previous result, then an `open` with the
// next assignment — and over a WebRTC mesh that cost a factor of ~2000 in
// throughput: a rank only contributed when its parameter version matched the
// step being opened, so a `commit` that was reordered, chunked late, or lost
// left it skipping every round while the leader waited out the full collect
// window. Fusing them removes the dependency rather than tuning the timeout:
// the parameters a round is computed against travel *with* the assignment, so
// a rank that missed everything before can still contribute to this one, and
// there is no version to mismatch.
//
// **Gradients go up, parameters come down.** Broadcasting the averaged
// *gradient* instead would be the same number of bytes, but it makes every rank
// accumulate its own parameter history: a rank that misses one broadcast —
// because it was joining, mid-compute, or briefly outside the cohort — applies
// a different subset of updates and diverges from the cohort permanently, with
// nothing in the loss curve to show it. Assigning parameters from one source of
// truth makes that class of bug unrepresentable, and it collapses the elastic
// join into the normal path: a joining rank is synced by the next commit.
//
// The leader is the lowest live id that is synced and running; if it vanishes,
// the next one takes over.

export const DEFAULT_TIMING = {
  beat: 400,      // liveness broadcast period at small cohort sizes
  drop: 1600,     // silence after which a rank is presumed gone
  collect: 2000,  // how long a leader waits for a straggler's bucket
  takeover: 3200, // silence from the leader before re-election
};

/// Cohort-size-dependent policy. Two things that are right for three ranks are
/// wrong for thirty:
///
/// * **Unanimity.** Waiting for every rank's bucket is fine when they all
///   arrive; with enough ranks, the slowest one owns the step rate, and the
///   leader burns the whole `collect` window every single step. Above a few
///   ranks the step resolves on a quorum instead — which costs nothing in
///   correctness, because the reduce is already sample-weighted over whoever
///   arrived, and every included bucket is checked against the step's
///   parameter version.
/// * **Heartbeat fan-out.** Each rank beats to every other, so mesh-wide beat
///   traffic grows as n²/period. Stretching the period with `n` holds it
///   roughly constant; `drop` and `takeover` are derived from the stretched
///   beat, or a larger cohort would evict members that are merely quiet.
export function policyFor(n, T = DEFAULT_TIMING) {
  const beat = Math.round(T.beat * Math.max(1, n / 4));
  return {
    n,
    // Unanimous while the cohort is small enough that every bucket arrives.
    quorum: n <= 3 ? n : Math.max(2, Math.ceil(n * 0.6)),
    beat,
    drop: Math.max(T.drop, beat * 4),
    takeover: Math.max(T.takeover, beat * 8),
    // A bigger cohort gets more slack, bounded — the quorum is what actually
    // keeps the step rate up, this is only the floor for a tiny cohort.
    collect: Math.min(T.collect * 3, Math.round(T.collect * (1 + n / 16))),
  };
}

/// Wire one rank into a bus. `rank` is the wasm `ClusterWorker` (or anything
/// with the same shape: `batch`, `bucket_len`, `grads`, `apply`, `params`,
/// `set_params`). `post` receives state snapshots for the UI.
/// Yield to the message queue without paying the timer clamp.
///
/// `setTimeout(…, 0)` is clamped to ~4ms once nesting passes a few levels, so a
/// loop that yields that way caps at ~250 rounds/s *no matter how fast the rank
/// computes* — which is what a 1,200 samples/s CPU rank was actually hitting,
/// not a compute limit. A `MessageChannel` task is not clamped and still lets
/// inbound messages through.
///
/// The waiter list is a FIFO, and it is not optional: a single pending slot
/// loses a resolver whenever two yields overlap, and that loop never resumes.
/// With one rank per Worker that cannot happen in the browser — but it happens
/// immediately with several ranks in one process, which is how the tests run,
/// and a wedged leader there looks exactly like divergence.
function makeYield() {
  if (typeof MessageChannel === "undefined") {
    return () => new Promise((r) => setTimeout(r, 0));
  }
  const channel = new MessageChannel();
  const waiting = [];
  channel.port1.onmessage = () => waiting.shift()?.();
  // Node keeps the loop alive for an open port; the browser has no such notion.
  channel.port1.unref?.();
  channel.port2.unref?.();
  return () =>
    new Promise((resolve) => {
      waiting.push(resolve);
      channel.port2.postMessage(0);
    });
}

export function createRank({
  id, bus, rank, post = () => {}, model = "?", backend = "cpu", precision = "f32",
  timing = {},
}) {
  const T = { ...DEFAULT_TIMING, ...timing };
  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
  const send = (msg) => bus.postMessage(msg);
  const bucketLen = rank.bucket_len;

  let running = false;  // participating in steps right now?
  let ready = false;    // parameters consistent with the cohort?
  let step = 0;
  let lr = 0.05;
  let looping = false;
  let pending = null;   // leader: in-flight bucket collection for one step
  let lastOpen = 0;
  let joining = null;
  let disposed = false;
  let contributing = false; // a bucket is being computed on this rank
  let lastSamples = 0;      // samples in the round just completed, for the UI
  let roundsFrom = null;    // id whose rounds we are currently following
  const yieldToQueue = makeYield();
  let fault = null;         // last backend error, surfaced in the UI

  const members = new Map(); // id → { seen, batch, ready, running, step }
  const stat = { steps: 0, samples: 0, bytesAt: 0, loss: NaN, since: now() };

  // Recomputed once per heartbeat from the live peer count. Cached rather than
  // derived on demand because `live()` prunes using `policy.drop`, so deriving
  // it from `live()` would be circular.
  let policy = policyFor(1, T);

  // Traffic comes from the bus, which is the only layer that knows the encoded
  // size. Estimating it here as 4 bytes an element reported f32 volumes however
  // the wire was configured — and made a halved payload look like *more*
  // traffic, because the saved bandwidth bought extra rounds.
  const wireBytes = () => bus.bytes?.() ?? 0;

  function live() {
    const t = now();
    for (const [k, m] of members) if (t - m.seen > policy.drop) members.delete(k);
    return members;
  }

  // Ranks that can contribute a *valid* bucket: alive, synced, and running. A
  // tab that stopped must leave the cohort, or the leader blocks for `collect`
  // every step waiting for a bucket that will never arrive.
  function cohortIds() {
    const ids = running && ready ? [id] : [];
    for (const [k, m] of live()) if (m.ready && m.running) ids.push(k);
    return ids.sort((a, b) => a - b);
  }

  // Electing on liveness alone would hand the lead to a stopped tab, which
  // never opens a step and stalls everyone.
  function leaderId() {
    let best = running && ready ? id : null;
    for (const [k, m] of live()) {
      if (!(m.ready && m.running)) continue;
      if (best === null || k < best) best = k;
    }
    return best;
  }

  // Set when this rank stood down for a lower id: it must not drive again
  // until the window lapses, or the two would trade the lead back and forth.
  let deferUntil = 0;

  const isLeader = () =>
    running && ready && leaderId() === id && now() >= deferUntil;

  function announce() {
    send({ t: "hello", from: id, batch: rank.batch, ready, running, step, model, backend });
  }

  function beat() {
    send({ t: "beat", from: id, batch: rank.batch, ready, running, step, backend });
  }

  // ── leader: drive one round ─────────────────────────────────────────────
  async function driveRound() {
    const ids = cohortIds();
    const batches = new Map([[id, rank.batch]]);
    for (const k of ids) if (k !== id) batches.set(k, members.get(k)?.batch ?? rank.batch);
    const total = [...batches.values()].reduce((a, b) => a + b, 0);

    // Consecutive, non-overlapping windows of the shared dataset, so N ranks
    // see N× the samples per round rather than all grinding the same ones.
    const assign = {};
    let offset = 0;
    for (const k of ids) {
      assign[k] = (step * total + offset) >>> 0;
      offset += batches.get(k);
    }

    // Install the collection state *before* announcing the round. A rank can
    // reply with its bucket as soon as it sees one, and if `pending` does not
    // exist yet that bucket is dropped on the floor — the round then waits out
    // the full `collect` timeout and the cohort runs at 1/collect rounds per
    // second while still looking perfectly healthy.
    const got = new Map();
    let settle = null;
    const collected = ids.length > 1
      ? new Promise((resolve) => {
          settle = resolve;
          pending = {
            step,
            want: new Set(ids),
            // Never ask for more buckets than the cohort can supply.
            quorum: Math.min(ids.length, policy.quorum),
            got,
            resolve,
          };
        })
      : null;

    send({
      t: "round",
      from: id,
      step,
      lr,
      assign,
      params: rank.params(),
      loss: stat.loss,
      samples: lastSamples,
      cohort: ids.length,
    });
    lastOpen = now();

    // WebGPU readback resolves on a later task, so this is awaited on every
    // backend: a CPU rank resolves immediately, a GPU rank when its buffer maps.
    const mine = await rank.grads(assign[id]);
    got.set(id, { loss: mine[0], g: mine.subarray(1), samples: rank.batch });

    if (collected) {
      // Our own bucket may have been the last one needed to reach quorum.
      if (pending && got.size >= pending.quorum) settle();
      else if (pending) pending.timer = setTimeout(settle, policy.collect);
      await collected;
      if (pending?.timer) clearTimeout(pending.timer);
      pending = null;
    }

    // Sample-weighted mean — the weighting the native fabric's `mean_reduce`
    // uses too, so a cohort of unequal batch sizes (or one that lost a
    // straggler mid-round) still averages to an unbiased gradient.
    const avg = new Float32Array(bucketLen);
    let samples = 0, loss = 0;
    for (const e of got.values()) samples += e.samples;
    for (const e of got.values()) {
      const w = e.samples / samples;
      loss += w * e.loss;
      for (let i = 0; i < bucketLen; i++) avg[i] += w * e.g[i];
    }

    // The optimizer stays on the rlx side of the boundary: the leader applies
    // the averaged bucket through `ClusterWorker::apply`, and the next round's
    // message publishes the result, so every rank runs the same update rule.
    rank.apply(avg, lr);
    // Adopt exactly what the peers will receive. At reduced precision the
    // published parameters are quantized, so a leader keeping its own f32 copy
    // would hold values no other rank has — and "every rank holds identical
    // parameters", which elastic join depends on, would quietly become false.
    const published = bus.quantize?.(rank.params());
    if (published) rank.set_params(published);
    step += 1;
    stat.steps += 1;
    stat.samples += samples;
    stat.loss = loss;
    lastSamples = samples;
    report(got.size);
  }

  /// Adopt the parameters a round is to be computed against.
  ///
  /// Idempotent and order-independent: a rank that missed every previous round
  /// is fully caught up by this one, which is what makes elastic join free of a
  /// separate snapshot exchange.
  function adopt(m) {
    rank.set_params(m.params);
    step = m.step;
    ready = true;
    stat.steps += 1;
    stat.samples += m.samples ?? 0;
    if (Number.isFinite(m.loss)) stat.loss = m.loss;
    if (joining) { clearInterval(joining); joining = null; }
    report(m.cohort);
  }

  async function loop() {
    if (looping || disposed) return;
    looping = true;
    try {
      while (running && ready && isLeader() && !disposed) {
        // Wait for the transport to drain before opening another round. A round
        // carries the whole parameter vector, and a mesh rejects a send once too
        // many transfers are pending — so pace on reported pressure instead of
        // handing it work it cannot take.
        for (let waited = 0; waited < 20 && (bus.pressure?.() ?? 0) > 1; waited++) {
          await new Promise((r) => setTimeout(r, policy.beat));
        }
        try {
          await driveRound();
          fault = null;
        } catch (e) {
          // A GPU fault would otherwise end the loop with the UI still showing
          // "training": surface it and let the heartbeat retry or hand over.
          fault = String(e?.message ?? e);
          if (pending) { clearTimeout(pending.timer); pending.resolve(); pending = null; }
          report();
          await new Promise((r) => setTimeout(r, policy.beat));
        }
        await yieldToQueue(); // let inbound messages through, unclamped
      }
    } finally {
      looping = false;
      // Publish the *last applied* parameters, which is why this is here and
      // not in `stop()`: stopping only ends the loop, so a round already in
      // flight still applies an update afterwards. Publishing before that
      // would leave the cohort one round behind the leader forever.
      //
      // Only when the run ended (`!running`). Exiting the loop because
      // leadership was lost must not publish — that would fight the new leader.
      if (!running && ready && !disposed) publish();
    }
  }

  function report(cohortSize) {
    const dt = (now() - stat.since) / 1000;
    post({
      t: "state",
      id: id.toString(16).padStart(8, "0"),
      leader: isLeader(),
      ready,
      running,
      step,
      cohort: cohortSize ?? Math.max(1, cohortIds().length),
      loss: stat.loss,
      stepsPerSec: dt > 0 ? stat.steps / dt : 0,
      samplesPerSec: dt > 0 ? stat.samples / dt : 0,
      bytesPerSec: dt > 0 ? (wireBytes() - stat.bytesAt) / dt : 0,
      peers: [...live().entries()].map(([k, m]) => ({
        id: k.toString(16).padStart(8, "0"),
        backend: m.backend ?? "?",
      })),
      bucket: bucketLen,
      batch: rank.batch,
      model,
      backend,
      precision,
      fault,
      // Surfaced so the adaptation is visible rather than a hidden heuristic.
      policy: { n: policy.n, quorum: policy.quorum, beat: policy.beat },
    });
    if (dt > 2) {
      stat.steps = 0;
      stat.samples = 0;
      stat.bytesAt = wireBytes();
      stat.since = now();
    }
  }

  // ── message handling ─────────────────────────────────────────────────────
  function onMessage(m) {
    if (disposed || !m || m.from === id) return;
    if (m.to !== undefined && m.to !== id) return;

    const peer = members.get(m.from) ?? {};
    members.set(m.from, {
      ...peer,
      seen: now(),
      batch: m.batch ?? peer.batch,
      ready: m.ready ?? peer.ready,
      running: m.running ?? peer.running,
      step: m.step ?? peer.step,
      backend: m.backend ?? peer.backend,
    });

    switch (m.t) {
      case "hello":
        // Answer so the newcomer learns the roster. It needs no parameter
        // snapshot: the next round carries them.
        beat();
        break;

      case "round": {
        lastOpen = now();
        // Two ranks can briefly both believe they lead (disagreeing membership
        // views during a join or a takeover). Both would keep publishing their
        // own parameters and the cohort would oscillate between two models, so
        // the higher id stands down: under every view the lower id is the
        // rightful leader.
        if (looping || pending) {
          if (m.from < id) standDown();
          break;
        }
        // Reject a *pretender*, not a timeline. Step numbers are only
        // comparable within one leader's run: two islands that each elected a
        // leader count independently, so after they merge the rank with the
        // higher step would reject the rightful leader's rounds forever as
        // "stale" and never converge. So: follow the lowest live id, and only
        // treat a step as stale when it comes from the leader we are already
        // following (an out-of-order duplicate).
        const lead = leaderId();
        if (lead !== null && m.from > lead) break; // higher id is not the leader
        if (m.from === roundsFrom && m.step < step) break; // duplicate, out of order
        roundsFrom = m.from;
        adopt(m);
        if (!running) break;
        if (m.assign?.[id] === undefined) break; // leader has not counted us yet
        contribute(m);
        break;
      }

      case "grad":
        if (pending && m.step === pending.step && pending.want.has(m.from)) {
          pending.got.set(m.from, { loss: m.loss, g: m.g, samples: m.samples });
          if (pending.got.size >= pending.quorum) pending.resolve();
        }
        break;
    }
  }

  /// Compute and send this rank's bucket. Not awaited by the message handler:
  /// a GPU rank's gradients resolve on a later task, and blocking the handler
  /// would stall the heartbeat that keeps it in the cohort.
  async function contribute(m) {
    if (contributing) return; // a bucket is already in flight here
    contributing = true;
    try {
      const out = await rank.grads(m.assign[id]);
      // A newer round arrived while we computed: this bucket is against the
      // older parameters, so drop it rather than bias the average. Note this
      // is a *supersession* check, not a version guard — the round we answered
      // carried the parameters we used, so there was never a mismatch to skip.
      if (m.step !== step) return;
      send({
        t: "grad", from: id, to: m.from, step: m.step,
        loss: out[0], samples: rank.batch, g: out.subarray(1),
      });
      fault = null;
    } catch (e) {
      fault = String(e?.message ?? e);
      report();
    } finally {
      contributing = false;
    }
  }

  /// Publish the current parameters without asking anyone to compute.
  ///
  /// With parameters riding on the *next* round, the leader's last update would
  /// otherwise never reach anyone: it applies the average, advances, and then
  /// stops. Every rank should hold the final model when a run ends, so stopping
  /// sends one assignment-free round.
  function publish() {
    send({
      t: "round",
      from: id,
      step,
      lr,
      assign: {},
      params: rank.params(),
      loss: stat.loss,
      samples: lastSamples,
      cohort: cohortIds().length,
    });
  }

  function standDown() {
    if (pending) { clearTimeout(pending.timer); pending.resolve(); pending = null; }
    // Stay `ready`: parameters arrive with the rightful leader's next commit,
    // and until then this rank's own values are a valid (if stale) model. What
    // it must stop doing is *publishing* them.
    deferUntil = now() + T.takeover;
    announce();
    report();
  }

  // Self-promoting to `ready` is only safe when nobody can hand us parameters:
  // a tab joining a run in progress must not train from fresh init params, or
  // it diverges from the cohort for the rest of the run. So retry the announce,
  // and bootstrap only when no peer holds parameters — then the lowest id
  // decides, deterministically.
  function beginJoining() {
    if (joining) clearInterval(joining);
    let tries = 0;
    joining = setInterval(() => {
      if (disposed || ready) { clearInterval(joining); joining = null; return; }
      const peers = [...live().entries()];
      if (!peers.some(([, m]) => m.ready) && ++tries >= 2) {
        const lowest = peers.reduce((lo, [k]) => Math.min(lo, k), id);
        if (lowest === id) {
          ready = true;
          clearInterval(joining); joining = null;
          report();
          if (running) loop();
          return;
        }
      }
      announce();
    }, T.beat);
  }

  // Heartbeat + takeover. A rank that stops hearing `open` while it is now the
  // rightful leader starts driving; identical parameters make that safe.
  //
  // Self-rescheduling rather than `setInterval`, because the period adapts to
  // the cohort size and an interval's period is fixed when it is created.
  let heart = null;
  const tick = () => {
    if (disposed) return;
    policy = policyFor(live().size + 1, T);
    beat();
    if (running && ready && isLeader() && !looping && !pending
        && now() - lastOpen > policy.takeover) {
      lastOpen = now();
      loop();
    }
    report();
    heart = setTimeout(tick, policy.beat);
  };
  heart = setTimeout(tick, T.beat);

  bus.onmessage = (ev) => onMessage(ev.data ?? ev);

  return {
    start(nextLr) {
      if (nextLr !== undefined) lr = nextLr;
      running = true;
      announce();
      if (!ready) beginJoining();
      else loop();
      report();
    },
    stop() {
      // Capture leadership before clearing `running`, which is part of the
      // election predicate.
      const wasLeader = isLeader();
      running = false;
      // A leader with a round in flight publishes when its loop drains; one
      // that is idle has to do it here.
      if (wasLeader && ready && !looping) publish();
      beat();
      report();
    },
    setLr(nextLr) { lr = nextLr; },
    dispose() {
      disposed = true;
      running = false;
      clearTimeout(heart);
      if (joining) clearInterval(joining);
      if (pending) { clearTimeout(pending.timer); pending.resolve(); pending = null; }
    },
    // Test/debug surface.
    get state() { return { id, ready, running, step, leader: isLeader(), cohort: cohortIds() }; },
  };
}
