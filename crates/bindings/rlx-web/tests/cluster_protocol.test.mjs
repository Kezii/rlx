// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Protocol tests for the tab cluster (`web/cluster-protocol.js`), run in Node
// against a stub rank and an in-process bus:
//
//   node --test crates/bindings/rlx-web/tests/
//
// Every failure mode here produces *plausible* training rather than an error —
// a split-brain cohort, a joiner starting from stale parameters, or a stopped
// tab throttling everyone still shows a falling loss curve. So each is asserted
// directly: ranks must stay bit-identical, and the step rate must not collapse.

import { test } from "node:test";
import assert from "node:assert/strict";
import { createRank, policyFor, DEFAULT_TIMING } from "../web/cluster-protocol.js";

const FAST = { beat: 20, drop: 90, collect: 70, takeover: 140 };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/// A rank shaped like the wasm `ClusterWorker`. Gradients depend on the current
/// parameters, so any divergence between ranks compounds instead of healing —
/// which is what makes "stay bit-identical" a real assertion.
/// `latency` makes `grads` resolve on a later task, the way a WebGPU rank does
/// (its buffer map cannot block the event loop). A cohort must tolerate ranks
/// that answer at different speeds, because that is what a mixed
/// WebGPU/CPU cluster is.
function stubRank({ batch = 4, n = 16, seed = 1, latency = 0 } = {}) {
  const params = Float32Array.from({ length: n }, (_, i) => Math.sin(seed * 0.7 + i));
  return {
    batch,
    bucket_len: n,
    grads(start) {
      const out = this.bucket(start);
      return latency > 0
        ? new Promise((r) => setTimeout(() => r(out), latency))
        : out;
    },
    bucket(start) {
      const out = new Float32Array(n + 1);
      let loss = 0;
      for (let i = 0; i < n; i++) {
        const g = params[i] * 0.5 + 0.01 * ((start + i) % 7);
        out[i + 1] = g;
        loss += params[i] * params[i];
      }
      out[0] = loss / n;
      return out;
    },
    apply(avg, lr) {
      for (let i = 0; i < n; i++) params[i] -= lr * avg[i];
    },
    params: () => params.slice(),
    set_params: (p) => params.set(p),
  };
}

/// In-process stand-in for a BroadcastChannel: one hub, many endpoints, with
/// structured-clone semantics so a bucket cannot be shared by reference (which
/// would hide divergence bugs the browser would expose).
function makeHub() {
  const subs = new Set();
  return {
    channel() {
      const ch = {
        onmessage() {},
        postMessage(msg) {
          // Delivered on a later task, like a real BroadcastChannel: a
          // synchronous fake would let a reply re-enter the sender mid-call and
          // test a condition the browser never produces.
          const copy = structuredClone(msg);
          setTimeout(() => {
            for (const s of subs) if (s !== ch) s.onmessage({ data: structuredClone(copy) });
          }, 0);
        },
      };
      subs.add(ch);
      return ch;
    },
    drop(ch) { subs.delete(ch); },
    bridge(other) {
      // Join two hubs so each sees the other's traffic — used to force the
      // split-brain case by letting two established leaders meet.
      const a = this.channel(), b = other.channel();
      a.onmessage = (ev) => b.postMessage(ev.data);
      b.onmessage = (ev) => a.postMessage(ev.data);
    },
  };
}

function spawn(hub, id, opts = {}) {
  const rank = stubRank(opts);
  const bus = hub.channel();
  const node = createRank({
    id, bus, rank, model: "stub",
    backend: opts.latency ? "webgpu" : "cpu",
    timing: FAST,
  });
  return { id, rank, bus, node };
}

/// Run a scenario with guaranteed teardown: a rank that outlives its test keeps
/// a heartbeat interval alive, which hangs the whole run.
async function scenario(fn) {
  const made = [];
  const make = (hub, id, opts) => {
    const r = spawn(hub, id, opts);
    made.push(r);
    return r;
  };
  try {
    await fn(make);
  } finally {
    for (const r of made) r.node.dispose();
  }
}

const paramsOf = (r) => Array.from(r.rank.params());

/// Stop every rank and let the bus drain. Parameters are compared at a
/// quiescent point because a running cohort always has a commit in flight: the
/// leader is a fraction of a step ahead of the members by construction, which
/// is lag, not drift. What must hold is that they converge once traffic stops.
async function quiesce(ranks, ms = 150) {
  for (const r of ranks) r.node.stop();
  await sleep(ms);
}

function assertIdentical(ranks, label) {
  const first = paramsOf(ranks[0]);
  for (const r of ranks.slice(1)) {
    assert.deepEqual(paramsOf(r), first, `${label}: rank ${r.id} diverged from ${ranks[0].id}`);
  }
}

test("a single rank trains alone", async () => scenario(async (make) => {
  const hub = makeHub();
  const a = make(hub, 1);
  const before = paramsOf(a);
  a.node.start(0.05);
  await sleep(250);
  assert.ok(a.node.state.step > 2, `expected steps, got ${a.node.state.step}`);
  assert.ok(a.node.state.leader, "a lone rank must lead itself");
  assert.notDeepEqual(paramsOf(a), before, "parameters never changed");
}));

test("three ranks stay bit-identical and elect one leader", async () => scenario(async (make) => {
  const hub = makeHub();
  const ranks = [make(hub, 10), make(hub, 20), make(hub, 30)];
  for (const r of ranks) r.node.start(0.05);
  await sleep(400);

  const leaders = ranks.filter((r) => r.node.state.leader);
  assert.equal(leaders.length, 1, `expected one leader, got ${leaders.length}`);
  assert.equal(leaders[0].id, 10, "lowest id must lead");
  for (const r of ranks) assert.ok(r.node.state.step > 1, `rank ${r.id} did not step`);
  await quiesce(ranks);
  assertIdentical(ranks, "steady state");
}));

test("a rank joining mid-run adopts the cohort's parameters", async () => scenario(async (make) => {
  const hub = makeHub();
  const ranks = [make(hub, 10), make(hub, 20)];
  for (const r of ranks) r.node.start(0.05);
  await sleep(300);

  // Different seed: if the joiner trained from its own init it would diverge
  // permanently, and the loss curve would not reveal it.
  const joiner = make(hub, 40, { seed: 99 });
  assert.notDeepEqual(paramsOf(joiner), paramsOf(ranks[0]), "joiner should start different");
  joiner.node.start(0.05);
  await sleep(300);

  assert.ok(joiner.node.state.ready, "joiner never synced");
  await quiesce([...ranks, joiner]);
  assertIdentical([...ranks, joiner], "after join");
}));

test("a stopped rank leaves the cohort instead of stalling it", async () => scenario(async (make) => {
  const hub = makeHub();
  const ranks = [make(hub, 10), make(hub, 20), make(hub, 30)];
  for (const r of ranks) r.node.start(0.05);
  await sleep(300);

  ranks[2].node.stop();
  const at = ranks[0].node.state.step;
  await sleep(300);
  const progressed = ranks[0].node.state.step - at;

  // Waiting on the stopped rank would cap progress at one step per `collect`
  // (70ms) ≈ 4 steps; a cohort that dropped it does far better.
  assert.ok(progressed > 8, `cohort stalled on a stopped rank: ${progressed} steps in 300ms`);
  await quiesce([ranks[0], ranks[1]]);
  assertIdentical([ranks[0], ranks[1]], "after a rank stopped");
}));

test("losing the leader re-elects without divergence", async () => scenario(async (make) => {
  const hub = makeHub();
  const ranks = [make(hub, 10), make(hub, 20), make(hub, 30)];
  for (const r of ranks) r.node.start(0.05);
  await sleep(300);
  assert.ok(ranks[0].node.state.leader, "rank 10 should lead first");

  // The leader's tab closes: dispose it and remove it from the bus entirely.
  ranks[0].node.dispose();
  hub.drop(ranks[0].bus);

  const survivors = [ranks[1], ranks[2]];
  await sleep(400);
  const leaders = survivors.filter((r) => r.node.state.leader);
  assert.equal(leaders.length, 1, `expected one new leader, got ${leaders.length}`);
  assert.equal(leaders[0].id, 20, "next-lowest id must take over");

  const at = survivors[0].node.state.step;
  await sleep(200);
  assert.ok(survivors[0].node.state.step > at, "cohort did not resume after takeover");
  await quiesce(survivors);
  assertIdentical(survivors, "after takeover");
}));

test("two established leaders meeting converge instead of splitting", async () => scenario(async (make) => {
  // Each hub is its own island, so both ranks legitimately lead. Bridging the
  // hubs is the split-brain moment: without a stand-down rule both would keep
  // applying their own average and drift apart while still looking healthy.
  const left = makeHub(), right = makeHub();
  const a = make(left, 10);
  const b = make(right, 20, { seed: 5 });
  a.node.start(0.05);
  b.node.start(0.05);
  await sleep(250);
  assert.ok(a.node.state.leader && b.node.state.leader, "both islands should lead");
  assert.notDeepEqual(paramsOf(a), paramsOf(b), "islands should have drifted apart");

  left.bridge(right);
  // Merging is the slowest path in the protocol: the higher id has to notice a
  // foreign round, stand down, and then adopt the next one from the rightful
  // leader. Give it several beats.
  await sleep(900);

  const leaders = [a, b].filter((r) => r.node.state.leader);
  assert.equal(leaders.length, 1, `split brain persisted: ${leaders.length} leaders`);
  assert.equal(leaders[0].id, 10, "lower id must keep the lead");
  await quiesce([a, b]);
  assertIdentical([a, b], "after the islands merged");
}));

test("a mixed cohort tolerates a rank that answers asynchronously", async () => scenario(async (make) => {
  // One rank resolves its bucket immediately (CPU) and one on a later task
  // (WebGPU readback). Both must land inside the collection window and the
  // cohort must stay identical — a GPU rank whose bucket always arrived late
  // would be silently excluded from every average while the UI still counted
  // it as a member.
  const hub = makeHub();
  const fast = make(hub, 10);
  const slow = make(hub, 20, { latency: 25 });
  let cohortSeen = 0;
  const watch = setInterval(() => {
    cohortSeen = Math.max(cohortSeen, fast.node.state.cohort.length);
  }, 20);

  fast.node.start(0.05);
  slow.node.start(0.05);
  await sleep(500);
  clearInterval(watch);

  assert.equal(cohortSeen, 2, "the async rank never joined the cohort");
  assert.ok(slow.node.state.step > 1, `async rank did not advance: ${slow.node.state.step}`);
  await quiesce([fast, slow]);
  assertIdentical([fast, slow], "mixed-latency cohort");
}));

test("the policy adapts to the number of peers", () => {
  const T = DEFAULT_TIMING;
  // Small cohorts wait for everyone: every bucket matters and they all arrive.
  for (const n of [1, 2, 3]) assert.equal(policyFor(n, T).quorum, n, `n=${n}`);
  // Larger ones must not let the slowest rank own the step rate.
  assert.ok(policyFor(10, T).quorum < 10, "10 ranks should not need unanimity");
  assert.ok(policyFor(100, T).quorum < 100, "100 ranks should not need unanimity");
  assert.ok(policyFor(100, T).quorum >= 2, "quorum must never fall below two");

  // Beat traffic grows as n²/period across a mesh, so the period stretches —
  // and `drop`/`takeover` must stretch with it, or a bigger cohort would evict
  // members that are merely quiet.
  let prev = policyFor(1, T);
  for (const n of [4, 10, 40, 100]) {
    const p = policyFor(n, T);
    assert.ok(p.beat >= prev.beat, `beat regressed at n=${n}`);
    assert.ok(p.drop >= p.beat * 3, `drop too tight for beat at n=${n}`);
    assert.ok(p.takeover > p.drop, `takeover must outlast drop at n=${n}`);
    assert.ok(p.collect <= T.collect * 3, `collect unbounded at n=${n}`);
    prev = p;
  }
});

test("one slow rank does not throttle a larger cohort", async () => scenario(async (make) => {
  // Five ranks, one answering far slower than the collect window. Requiring
  // every bucket would cap the cohort at one step per `collect` — visible as a
  // healthy-looking loss curve advancing at a crawl. The quorum lets the fast
  // ranks proceed, and the slow one stays in sync off the commits.
  const hub = makeHub();
  const fast = [make(hub, 10), make(hub, 20), make(hub, 30), make(hub, 40)];
  const slow = make(hub, 50, { latency: 300 });
  for (const r of [...fast, slow]) r.node.start(0.05);
  await sleep(200);

  const at = fast[0].node.state.step;
  await sleep(600);
  const progressed = fast[0].node.state.step - at;

  // Unanimity would allow ~6 steps in 600ms (collect ≈ 92ms at n=5).
  assert.ok(progressed > 15, `cohort throttled by one slow rank: ${progressed} steps in 600ms`);

  await quiesce([...fast, slow]);
  assertIdentical([...fast, slow], "slow rank left behind");
}));

test("a peer joining mid-run does not collapse throughput", async () => scenario(async (make) => {
  // Observed in a browser over the WebRTC mesh: a second peer joined and the
  // cohort fell from ~1000 samples/s to 0.5 — exactly one round per `collect`
  // timeout, i.e. the newcomer never contributed and the leader waited it out
  // every round. The cause was two messages per round: a rank only answered
  // when its parameter version already matched the round being opened, so a
  // parameters message that was late, reordered or lost left it skipping
  // forever. One fused round removes the dependency, and this is the guard.
  const hub = makeHub();
  const leader = make(hub, 10);
  leader.node.start(0.05);
  await sleep(300);

  const solo = leader.node.state.step;
  assert.ok(solo > 5, `leader should be running freely, got ${solo} steps`);

  const joiner = make(hub, 20);
  joiner.node.start(0.05);
  await sleep(600);

  const after = leader.node.state.step - solo;
  // A cohort is paced by message round-trips, so it is legitimately slower than
  // a rank spinning alone — comparing against solo pace would measure the
  // transport, not the bug. What the bug looked like is unmistakable: one round
  // per `collect` timeout, i.e. ~8 rounds in 600ms at collect=70ms. Require an
  // order of magnitude above that.
  const stalled = 600 / FAST.collect;
  assert.ok(
    after > stalled * 10,
    `throughput collapsed after a peer joined: ${after} rounds in 600ms ` +
      `(a stalled cohort would manage ~${Math.round(stalled)})`,
  );
  // And the newcomer must actually be contributing, not merely present.
  assert.ok(joiner.node.state.step > 1, "joiner never advanced");
  assert.equal(leader.node.state.cohort.length, 2, "joiner not in the cohort");

  await quiesce([leader, joiner]);
  assertIdentical([leader, joiner], "after a mid-run join");
}));

test("a rank that missed every prior round still contributes to the next", async () => scenario(async (make) => {
  // The property that makes the above true: parameters travel *with* the
  // assignment, so there is no version for a starved rank to mismatch.
  const hub = makeHub();
  const leader = make(hub, 10);
  leader.node.start(0.05);
  await sleep(400);
  const at = leader.node.state.step;
  assert.ok(at > 3, "leader should have advanced alone first");

  const late = make(hub, 90, { seed: 123 }); // different init, far behind
  late.node.start(0.05);
  await sleep(250);

  assert.ok(late.node.state.ready, "late rank never became ready");
  assert.ok(
    late.node.state.step >= at,
    `late rank stuck at ${late.node.state.step} while cohort is at ${leader.node.state.step}`,
  );
  await quiesce([leader, late]);
  assertIdentical([leader, late], "late rank");
}));
