// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The transport path, end to end *except* WebRTC itself:
//
//   protocol → encode → transfer to the main thread → mesh → transfer back
//   → decode → protocol
//
//   node --test crates/bindings/rlx-web/tests/cluster_bus.test.mjs
//
// WebRTC negotiation cannot run outside a browser, so the mesh here is an
// in-process hub — but every byte still goes through the real wire format and
// real transfer semantics (`structuredClone` with a transfer list detaches the
// sender's buffer, exactly as `postMessage` does). That covers the two failure
// modes this layer can produce silently: a message whose fields do not survive
// JSON, and a buffer read after it was transferred away.

import { test } from "node:test";
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { createRank } from "../web/cluster-protocol.js";
import {
  encode, decode, proxyBus, proxyHost, f32ToF16, f16ToF32, quantizeF16,
  dequantizeF16, roundTrip, quantize, dequantize, decodeMini, WIRE_FORMATS,
} from "../web/cluster-bus.js";

const FAST = { beat: 20, drop: 90, collect: 70, takeover: 140 };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function stubRank({ batch = 4, n = 64, seed = 1 } = {}) {
  const params = Float32Array.from({ length: n }, (_, i) => Math.sin(seed * 0.7 + i));
  return {
    batch,
    bucket_len: n,
    grads(start) {
      const out = new Float32Array(n + 1);
      let loss = 0;
      for (let i = 0; i < n; i++) {
        out[i + 1] = params[i] * 0.5 + 0.01 * ((start + i) % 7);
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

/// A `postMessage` pair with real transfer semantics: the transferred buffer is
/// detached from the sender, so a use-after-transfer bug fails here instead of
/// only in a browser.
function portPair() {
  const a = { onmessage: null, postMessage: (m, t) => hop(b, m, t) };
  const b = { onmessage: null, postMessage: (m, t) => hop(a, m, t) };
  const hop = (dest, msg, transfer) => {
    const copy = structuredClone(msg, { transfer: transfer ?? [] });
    setTimeout(() => dest.onmessage?.({ data: copy }), 0);
  };
  return [a, b];
}

/// Stands in for the WebRTC mesh: delivers each buffer to every other peer.
function makeMesh() {
  const peers = new Set();
  return {
    add: (p) => peers.add(p),
    drop: (p) => peers.delete(p),
    broadcast(sender, buf) {
      for (const p of peers) {
        if (p !== sender) p.deliver(structuredClone(buf));
      }
    },
  };
}

function meshRank(mesh, id, opts = {}) {
  const [workerPort, mainPort] = portPair();
  const rank = stubRank(opts);
  // Worker side: the protocol talks to a proxy bus.
  const node = createRank({
    id, rank, bus: proxyBus(workerPort),
    model: "stub", backend: opts.backend ?? "cpu", timing: FAST,
  });
  // Main-thread side: ferry buffers between the worker and the mesh.
  const peer = { deliver: null };
  const host = proxyHost(mainPort, (buf) => mesh.broadcast(peer, buf));
  peer.deliver = (buf) => host.deliver(buf);
  mesh.add(peer);
  return { id, rank, node, peer };
}

test("the wire format round-trips a bucket at every header alignment", () => {
  // `new Float32Array(buffer, offset)` throws on a misaligned offset, and the
  // header's length is whatever JSON produces — so every residue mod 4 must
  // work, or only *some* message types break.
  for (let padding = 0; padding < 8; padding++) {
    const g = Float32Array.from({ length: 300 }, (_, i) => i * 0.25 - 7);
    const msg = { t: "grad", from: 7, to: 9, step: 3, loss: 1.5, samples: 4, g, pad: "x".repeat(padding) };
    const back = decode(encode(msg));
    assert.equal(back.step, 3);
    assert.deepEqual(Array.from(back.g), Array.from(g), `padding ${padding}`);
    assert.ok(!("_f32" in back), "wire marker leaked into the decoded message");
  }
});

test("a message with no float payload round-trips unchanged", () => {
  const msg = { t: "open", from: 1, step: 5, lr: 0.05, assign: { 1: 0, 2: 4 } };
  assert.deepEqual(decode(encode(msg)), msg);
});

test("a non-finite loss survives the wire as a non-finite value", () => {
  // JSON has no NaN: `stat.loss` starts as NaN and rides along on `snapshot`,
  // so it arrives as null. The protocol must treat that as "no loss yet"
  // rather than adopting it as a number.
  const back = decode(encode({ t: "snapshot", from: 1, step: 0, loss: NaN, samples: 0 }));
  assert.equal(back.loss, null);
  assert.ok(!Number.isFinite(back.loss));
});

test("a cohort trains through the encoded proxy path", async () => {
  const mesh = makeMesh();
  const ranks = [meshRank(mesh, 10), meshRank(mesh, 20), meshRank(mesh, 30)];
  try {
    for (const r of ranks) r.node.start(0.05);
    await sleep(600);

    for (const r of ranks) assert.ok(r.node.state.step > 1, `rank ${r.id} did not step`);
    assert.equal(ranks.filter((r) => r.node.state.leader).length, 1, "expected one leader");

    for (const r of ranks) r.node.stop();
    await sleep(200);
    const first = Array.from(ranks[0].rank.params());
    for (const r of ranks.slice(1)) {
      assert.deepEqual(Array.from(r.rank.params()), first, `rank ${r.id} diverged over the wire`);
    }
  } finally {
    for (const r of ranks) { r.node.dispose(); mesh.drop(r.peer); }
  }
});

test("a rank joining over the proxy path adopts the cohort's parameters", async () => {
  const mesh = makeMesh();
  const ranks = [meshRank(mesh, 10), meshRank(mesh, 20)];
  let joiner;
  try {
    for (const r of ranks) r.node.start(0.05);
    await sleep(400);

    // A different seed: without the parameter snapshot crossing the wire this
    // rank would train from its own init and diverge for the rest of the run.
    joiner = meshRank(mesh, 40, { seed: 99 });
    assert.notDeepEqual(
      Array.from(joiner.rank.params()),
      Array.from(ranks[0].rank.params()),
    );
    joiner.node.start(0.05);
    await sleep(500);

    assert.ok(joiner.node.state.ready, "joining rank never synced");
    for (const r of [...ranks, joiner]) r.node.stop();
    await sleep(200);
    assert.deepEqual(
      Array.from(joiner.rank.params()),
      Array.from(ranks[0].rank.params()),
      "the joining rank did not converge",
    );
  } finally {
    for (const r of [...ranks, joiner]) if (r) { r.node.dispose(); mesh.drop(r.peer); }
  }
});

// ── outbound backpressure ──────────────────────────────────────────────────
// A WebRTC data channel chunks a few hundred KB per round, and Trystero refuses
// a send once too many transfers are pending — observed in a browser as
// "mesh: send failed: Trystero: too many pending transfers". The host queue
// must never hand the transport more than it can drain.

/// A `worker`-shaped stub for the host side of the proxy.
function fakeWorker() {
  const w = { onmessage: null, posted: [] };
  w.postMessage = (msg) => w.posted.push(msg);
  return w;
}

/// A `send` that settles only when the test says so.
function gatedSend() {
  const calls = [];
  const send = (buf) => {
    let release;
    const promise = new Promise((r) => {
      release = r;
    });
    calls.push({ buf, release });
    return promise;
  };
  return { send, calls };
}

test("only one transfer is in flight at a time", async () => {
  const worker = fakeWorker();
  const { send, calls } = gatedSend();
  proxyHost(worker, send, { maxInflight: 1 });

  worker.onmessage({ data: { t: "bus", kind: "grad", buf: encode({ t: "grad", from: 1 }) } });
  worker.onmessage({ data: { t: "bus", kind: "grad", buf: encode({ t: "grad", from: 2 }) } });
  worker.onmessage({ data: { t: "bus", kind: "grad", buf: encode({ t: "grad", from: 3 }) } });
  assert.equal(calls.length, 1, "handed the transport more than one transfer");

  calls[0].release();
  await sleep(5);
  assert.equal(calls.length, 2, "next transfer should start when the first settles");
  calls[1].release();
  await sleep(5);
  assert.equal(calls.length, 3);
});

test("a superseded round is dropped rather than queued behind the newest", async () => {
  const worker = fakeWorker();
  const { send, calls } = gatedSend();
  const drops = [];
  proxyHost(worker, send, { maxInflight: 1, onDrop: (t) => drops.push(t) });

  const round = (step) => ({ t: "bus", kind: "round", buf: encode({ t: "round", from: 1, step }) });
  worker.onmessage({ data: round(1) }); // starts sending immediately
  worker.onmessage({ data: round(2) });
  worker.onmessage({ data: round(3) });
  worker.onmessage({ data: round(4) });

  assert.equal(calls.length, 1, "only one in flight");
  assert.equal(decode(calls[0].buf).step, 1);
  assert.ok(drops.some((d) => /superseded/.test(d)), "drops should be reported");

  calls[0].release();
  await sleep(5);
  // Rounds 2 and 3 were superseded by 4; delivering them would only push stale
  // parameters, since every round carries its own.
  assert.equal(calls.length, 2);
  assert.equal(decode(calls[1].buf).step, 4, "the newest round must win");
});

test("small messages keep their order and the queue stays bounded", async () => {
  const worker = fakeWorker();
  const { send, calls } = gatedSend();
  const drops = [];
  proxyHost(worker, send, { maxInflight: 1, maxQueue: 3, onDrop: (t) => drops.push(t) });

  for (let i = 0; i < 6; i++) {
    worker.onmessage({
      data: { t: "bus", kind: "beat", buf: encode({ t: "beat", from: i, step: i }) },
    });
  }
  assert.ok(drops.some((d) => /queue full/.test(d)), "overflow should be reported");

  const seen = [decode(calls[0].buf).from];
  for (let i = 0; i < 4 && i < calls.length + 3; i++) {
    calls.at(-1).release();
    await sleep(5);
    if (calls.length > seen.length) seen.push(decode(calls.at(-1).buf).from);
  }
  // FIFO among the survivors: ascending, never reordered.
  const ascending = seen.every((v, i) => i === 0 || v > seen[i - 1]);
  assert.ok(ascending, `small messages were reordered: ${seen}`);
});

test("queue depth is reported back so a leader can pace itself", async () => {
  // The two halves live on opposite ends of a port in the real system; sharing
  // one object here would have them fight over the same tag.
  const worker = fakeWorker();
  const { send, calls } = gatedSend();
  proxyHost(worker, send, { maxInflight: 1 });

  worker.onmessage({ data: { t: "bus", kind: "round", buf: encode({ t: "round", from: 1, step: 1 }) } });
  worker.onmessage({ data: { t: "bus", kind: "round", buf: encode({ t: "round", from: 1, step: 2 }) } });

  const pressure = worker.posted.filter((m) => m.t === "bus:pressure").at(-1);
  assert.ok(pressure, "no pressure update was posted");
  assert.ok(pressure.depth >= 1, `depth should count work in flight, got ${pressure.depth}`);

  calls[0].release();
  await sleep(5);

  // The worker-side bus turns those updates into the number a leader paces on.
  const port = { onmessage: null, postMessage: () => {} };
  const bus = proxyBus(port);
  assert.equal(bus.pressure(), 0);
  port.onmessage({ data: { t: "bus:pressure", depth: 7 } });
  assert.equal(bus.pressure(), 7, "pressure did not reach the protocol");
});

// ── half-precision wire ────────────────────────────────────────────────────
// A round ships the whole parameter vector, so the wire format is the
// bandwidth. f16 halves it; the catch is that f16's smallest normal is ~6e-5,
// so an unscaled gradient bucket around 1e-8 would quantize to all zeros and
// training would stop with nothing reporting an error.

test("f16 conversion is exact for representable values and keeps NaN/Inf", () => {
  for (const v of [0, 1, -1, 0.5, -2, 2048, 65504, 6.103515625e-5]) {
    assert.equal(f16ToF32(f32ToF16(v)), v, `${v} should survive exactly`);
  }
  assert.equal(f16ToF32(f32ToF16(Infinity)), Infinity);
  assert.equal(f16ToF32(f32ToF16(-Infinity)), -Infinity);
  assert.ok(Number.isNaN(f16ToF32(f32ToF16(NaN))), "NaN must not decode as Inf");
  // Above f16's range it saturates rather than wrapping to zero or NaN.
  assert.equal(f16ToF32(f32ToF16(1e30)), Infinity);
});

test("the dynamic scale is what keeps tiny gradients alive", () => {
  // Magnitudes a gradient bucket reaches late in training. f16's smallest
  // *subnormal* is 2^-24 ≈ 6e-8, so everything here is under it.
  const tiny = Float32Array.from({ length: 256 }, (_, i) => (i - 128) * 1e-10);

  // Unscaled, essentially the whole bucket quantizes to zero — the update
  // vanishes and nothing reports an error.
  const unscaled = [...tiny].map((v) => f16ToF32(f32ToF16(v)));
  const deadUnscaled = unscaled.filter((v) => v === 0).length;
  assert.ok(
    deadUnscaled > 250,
    `expected the bucket to vanish without a scale, only ${deadUnscaled}/256 did`,
  );

  // Scaled, the signal survives with ~1e-3 relative error.
  const { scale, payload } = quantizeF16(tiny);
  const back = dequantizeF16(payload, scale);
  const dead = [...back].filter((v, i) => v === 0 && tiny[i] !== 0).length;
  assert.equal(dead, 0, "no non-zero value should quantize away");
  let worst = 0;
  for (let i = 0; i < tiny.length; i++) {
    if (tiny[i] !== 0) worst = Math.max(worst, Math.abs(back[i] - tiny[i]) / Math.abs(tiny[i]));
  }
  assert.ok(worst < 2e-3, `relative error ${worst.toExponential(2)} too large`);
});

test("an all-zero bucket does not produce a degenerate scale", () => {
  const zeros = new Float32Array(32);
  const { scale, payload } = quantizeF16(zeros);
  assert.ok(Number.isFinite(scale) && scale !== 0, `scale ${scale} is unusable`);
  assert.deepEqual([...dequantizeF16(payload, scale)], [...zeros]);
});

test("f16 halves the wire and still decodes the right message", () => {
  const params = Float32Array.from({ length: 1024 }, (_, i) => Math.sin(i) * 0.01);
  const msg = { t: "round", from: 3, step: 9, lr: 0.05, assign: { 3: 0 }, params };

  const wideBytes = encode(msg).byteLength;
  const halfBytes = encode(msg, { precision: "f16" }).byteLength;
  assert.ok(halfBytes < wideBytes * 0.55, `f16 saved too little: ${wideBytes} → ${halfBytes}`);

  const back = decode(encode(msg, { precision: "f16" }));
  assert.equal(back.t, "round");
  assert.equal(back.step, 9);
  assert.deepEqual(back.assign, { 3: 0 });
  // The wire markers must not leak into the decoded message.
  for (const k of ["_f32", "_bits", "_scale"]) assert.ok(!(k in back), `${k} leaked`);
  let worst = 0;
  for (let i = 0; i < params.length; i++) {
    worst = Math.max(worst, Math.abs(back.params[i] - params[i]));
  }
  assert.ok(worst < 1e-4, `absolute error ${worst.toExponential(2)} too large`);
});

test("quantization is idempotent, so ranks converge rather than drift", () => {
  // The leader adopts its own published values; if a second trip through the
  // wire changed them again, every round would nudge the cohort apart.
  const values = Float32Array.from({ length: 512 }, (_, i) => Math.cos(i) * 0.3);
  const once = roundTrip(values, "f16");
  const twice = roundTrip(once, "f16");
  assert.deepEqual([...twice], [...once], "a second round trip changed the values");
  // And f32 reports "nothing changes", so the caller can skip the copy.
  assert.equal(roundTrip(values, "f32"), null);
});

test("a bus exposes the round trip so the leader can match its peers", () => {
  const worker = fakeWorker();
  const half = proxyBus(worker, { precision: "f16" });
  const wide = proxyBus(worker, { precision: "f32" });
  const values = Float32Array.from({ length: 64 }, (_, i) => i * 0.001);
  assert.ok(half.quantize(values) !== null, "f16 bus must report a change");
  assert.equal(wide.quantize(values), null, "f32 bus must report none");
});

test("the 8-bit formats match the OCP FP8 definitions", () => {
  // Enumerating all 256 codes is the whole point of building the tables
  // exhaustively: the format is checkable, not asserted.
  const finite = (f) => {
    const out = [];
    for (let c = 0; c < 1 << f.bits; c++) {
      const v = decodeMini(c, f);
      if (Number.isFinite(v)) out.push(v);
    }
    return out;
  };

  const e4m3 = finite(WIRE_FORMATS.f8e4m3);
  assert.equal(Math.max(...e4m3), 448, "E4M3 max finite is 448");
  assert.equal(e4m3.length, 254, "E4M3 reserves only ±NaN — no infinities");

  const e5m2 = finite(WIRE_FORMATS.f8e5m2);
  assert.equal(Math.max(...e5m2), 57344, "E5M2 max finite is 57344");
  // E5M2 does have infinities, so more codes are reserved than E4M3's.
  assert.ok(e5m2.length < e4m3.length, "E5M2 reserves Inf as well as NaN");

  // E4M3 has no Inf: an out-of-range value saturates instead.
  const { scale, payload } = quantize(Float32Array.of(1e30), WIRE_FORMATS.f8e4m3);
  assert.ok(Number.isFinite(dequantize(payload, scale, WIRE_FORMATS.f8e4m3)[0]));
});

test("precision trades bytes for error in the expected order", () => {
  const values = Float32Array.from({ length: 2048 }, (_, i) => Math.sin(i) * 1e-6);
  const msg = { t: "round", from: 1, step: 1, params: values };

  // Error is measured against the bucket's largest magnitude, not per element.
  // A scaled quantizer has roughly uniform *absolute* resolution across the
  // bucket, so per-element relative error is unbounded wherever a value passes
  // near zero — which says nothing about the format.
  let maxAbs = 0;
  for (const v of values) maxAbs = Math.max(maxAbs, Math.abs(v));
  const measure = (precision) => {
    const buf = encode(msg, { precision });
    const back = decode(buf).params;
    let worst = 0;
    for (let i = 0; i < values.length; i++) {
      worst = Math.max(worst, Math.abs(back[i] - values[i]) / maxAbs);
    }
    return { bytes: buf.byteLength, worst };
  };

  const f32 = measure("f32");
  const f16 = measure("f16");
  const e4m3 = measure("f8e4m3");
  const e5m2 = measure("f8e5m2");

  assert.equal(f32.worst, 0, "f32 must be exact");
  assert.ok(f16.bytes < f32.bytes * 0.55, "f16 should roughly halve the wire");
  assert.ok(e4m3.bytes < f32.bytes * 0.3, "f8 should roughly quarter it");
  assert.equal(e4m3.bytes, e5m2.bytes, "both f8 formats are one byte per element");

  // Error grows as bytes shrink, and e4m3 beats e5m2 — the dynamic scale
  // already covers range, so e5m2's extra exponent bit only costs a mantissa
  // bit. This is why e4m3 is the f8 worth offering first.
  assert.ok(f16.worst < 1e-3, `f16 error ${f16.worst}`);
  assert.ok(e4m3.worst > 1e-3, "f8 should be visibly coarser than f16");
  assert.ok(e4m3.worst > f16.worst, "f8 must be worse than f16");
  assert.ok(e4m3.worst < 0.1, `e4m3 error ${e4m3.worst} unexpectedly large`);
  assert.ok(e5m2.worst > e4m3.worst, "e5m2 should be the less accurate f8");
});

test("ranks set to different precisions still interoperate", () => {
  // The format id and scale ride in each message, so there is no handshake to
  // get wrong and no cohort-wide setting to keep in sync.
  const values = Float32Array.from({ length: 128 }, (_, i) => (i - 64) * 0.01);
  for (const sender of ["f32", "f16", "f8e4m3", "f8e5m2"]) {
    const back = decode(encode({ t: "round", params: values }, { precision: sender }));
    assert.equal(back.params.length, values.length, `${sender} length changed`);
    assert.ok(!("_fmt" in back) && !("_scale" in back), `${sender} leaked wire keys`);
  }
});

test("every precision the UI offers is a format the codec knows", () => {
  const html = readFileSync(
    new URL("../web/cluster.html", import.meta.url),
    "utf8",
  );
  const block = html.slice(html.indexOf('id="precision"'));
  const offered = [...block.slice(0, block.indexOf("</select>")).matchAll(/value="([^"]+)"/g)]
    .map((m) => m[1]);
  assert.ok(offered.length >= 3, `only found ${offered.length} options`);
  for (const id of offered) {
    assert.ok(id === "f32" || WIRE_FORMATS[id], `UI offers ${id} with no grammar`);
  }
});
