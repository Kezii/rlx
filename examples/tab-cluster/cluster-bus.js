// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The bus the cluster protocol talks over, and the wire format it uses.
//
// The page is a **static site** — no server of ours is involved at runtime — so
// there are two transports and no relay we host:
//
//   BroadcastChannel  tabs of this origin, in this browser.
//   WebRTC mesh       ranks on other devices. Signalling (a few KB, once per
//                     peer) goes over public infrastructure; gradient buckets
//                     (tens to hundreds of KB, every step) go peer-to-peer and
//                     never touch a third party. See `cluster-p2p.js`.
//
// `RTCPeerConnection` does not exist in a Web Worker, so the transport lives on
// the main thread while the rank and the protocol stay in the Worker. The
// Worker therefore talks to a `proxyBus`, which encodes each message to one
// ArrayBuffer and **transfers** it — so the UI thread moves the bucket without
// copying or parsing it.

// ── wire format ───────────────────────────────────────────────────────────
// [u32 LE header length][JSON header, space-padded to a multiple of 4][f32 payload]
//
// One message carries at most one float array (`grad.g`, `commit.params`,
// `snapshot.params`), named by `_f32` in the header.
//
// The padding is load-bearing: `new Float32Array(buffer, offset)` throws unless
// `offset` is a multiple of 4, and a JSON header is whatever length it is. The
// alternative — copying the payload into a fresh aligned buffer on every
// message — costs a few hundred KB of copy per step.
const F32_FIELDS = ["g", "params", "avg"];

// ── half-precision payloads ───────────────────────────────────────────────
//
// A round ships the whole parameter vector, so the wire format *is* the
// bandwidth. f16 halves it for ~5e-4 relative error — far below the noise in a
// minibatch gradient, and it is the same trade mixed-precision training makes
// everywhere else.
//
// Raw f16 would not survive gradients, though: its smallest normal is ~6e-5, so
// a bucket whose values live around 1e-8 would quantize to all zeros — training
// would simply stop, with no error anywhere. Hence a **dynamic scale** per
// message (exactly what loss scaling does in AMP): the largest magnitude is
// mapped to ~1024, every value rides along in the normal range, and the scale
// travels in the header.

const scratch = new ArrayBuffer(4);
const scratchF32 = new Float32Array(scratch);
const scratchU32 = new Uint32Array(scratch);

/// f32 → f16 bits, round-to-nearest-even, saturating to ±Inf on overflow.
export function f32ToF16(value) {
  scratchF32[0] = value;
  const bits = scratchU32[0];
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  let mantissa = bits & 0x7fffff;

  if (exponent === 0xff) {
    // Inf stays Inf; NaN must stay NaN (a zero payload would read as Inf).
    return sign | 0x7c00 | (mantissa ? 0x200 : 0);
  }
  let e = exponent - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign; // underflows past subnormal → signed zero
    mantissa |= 0x800000;
    const shift = 14 - e;
    const rounded = (mantissa + (1 << (shift - 1))) >>> shift;
    return sign | rounded;
  }
  // Round-to-nearest-even before truncating 13 bits.
  const rounded = mantissa + 0x0fff + ((mantissa >>> 13) & 1);
  if (rounded & 0x800000) {
    e += 1;
    if (e >= 0x1f) return sign | 0x7c00;
    return sign | (e << 10);
  }
  return sign | (e << 10) | ((rounded >>> 13) & 0x3ff);
}

/// f16 bits → f32.
export function f16ToF32(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa ? NaN : sign * Infinity;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

/// The wire formats the UI offers.
///
/// `target` is where the largest magnitude in a message is placed by the
/// dynamic scale — comfortably inside the format's normal range, with decades
/// of headroom underneath so small values do not fall into subnormals.
///
/// `exp`/`mant`/`bias`/`inf` describe the minifloat exactly, which is what lets
/// one generic decoder serve all of them. The 8-bit pair follows OCP FP8:
/// **E4M3 has no infinities** (the all-ones exponent holds finite values up to
/// 448, with only all-ones mantissa reserved for NaN), while **E5M2 does**.
export const WIRE_FORMATS = {
  f32: null,
  f16: { id: "f16", bits: 16, exp: 5, mant: 10, bias: 15, inf: true, target: 1024 },
  f8e4m3: { id: "f8e4m3", bits: 8, exp: 4, mant: 3, bias: 7, inf: false, target: 192 },
  f8e5m2: { id: "f8e5m2", bits: 8, exp: 5, mant: 2, bias: 15, inf: true, target: 1024 },
};

/// Decode one minifloat code. Generic over the format, so E4M3's missing
/// infinities and E5M2's presence of them are data, not branches in three
/// separate implementations.
export function decodeMini(code, f) {
  const maxExp = (1 << f.exp) - 1;
  const sign = code >>> (f.exp + f.mant) ? -1 : 1;
  const exp = (code >>> f.mant) & maxExp;
  const mant = code & ((1 << f.mant) - 1);
  if (exp === maxExp) {
    if (f.inf) return mant ? NaN : sign * Infinity;
    // E4M3: only the all-ones mantissa is NaN; the rest are finite.
    if (mant === (1 << f.mant) - 1) return NaN;
    return sign * (1 + mant / (1 << f.mant)) * 2 ** (maxExp - f.bias);
  }
  if (exp === 0) return (sign * mant) / (1 << f.mant) * 2 ** (1 - f.bias);
  return sign * (1 + mant / (1 << f.mant)) * 2 ** (exp - f.bias);
}

/// Every finite code of an 8-bit format, sorted by value.
///
/// An 8-bit format has 256 codes, so the exact set is cheap to enumerate — and
/// nearest-value lookup over the real table is correct by construction, which
/// hand-rolled rounding for two more formats would not be.
const tables = new Map();

function table(f) {
  if (!tables.has(f.id)) {
    const entries = [];
    for (let code = 0; code < 1 << f.bits; code++) {
      const value = decodeMini(code, f);
      if (Number.isFinite(value)) entries.push({ code, value });
    }
    entries.sort((a, b) => a.value - b.value);
    tables.set(f.id, {
      values: Float64Array.from(entries.map((e) => e.value)),
      codes: Uint16Array.from(entries.map((e) => e.code)),
    });
  }
  return tables.get(f.id);
}

/// Nearest finite code for `value`, saturating at the ends.
function encodeMini(value, f) {
  const { values, codes } = table(f);
  if (Number.isNaN(value)) return (1 << (f.exp + f.mant)) - 1 | (((1 << f.exp) - 1) << f.mant);
  if (value <= values[0]) return codes[0];
  if (value >= values[values.length - 1]) return codes[values.length - 1];
  let lo = 0;
  let hi = values.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (values[mid] <= value) lo = mid;
    else hi = mid;
  }
  // Ties go to the even code, matching round-to-nearest-even.
  const dLo = value - values[lo];
  const dHi = values[hi] - value;
  if (dLo < dHi) return codes[lo];
  if (dHi < dLo) return codes[hi];
  return codes[lo] % 2 === 0 ? codes[lo] : codes[hi];
}

const ARRAYS = { 8: Uint8Array, 16: Uint16Array };

/// f16 code → nearest code in an 8-bit format, for all 65,536 inputs.
///
/// The binary search over the value table is correct but was *slower than the
/// bandwidth it saved*: at ~100k elements a round, f8 ran fewer rounds than f16
/// in the browser. Going through f16 first — cheap bit math — and then a single
/// 64 KB lookup makes f8 cost f16 plus one array read.
const bridges = new Map();

function bridge(f) {
  if (!bridges.has(f.id)) {
    const map = new Uint8Array(0x10000);
    for (let code = 0; code < 0x10000; code++) {
      map[code] = encodeMini(f16ToF32(code), f);
    }
    bridges.set(f.id, map);
  }
  return bridges.get(f.id);
}

/// Quantize to a scaled minifloat. Returns the scale and the payload.
///
/// The scale is per message and travels in the header — the same trick AMP's
/// loss scaling uses, and the reason a gradient bucket living around 1e-9 does
/// not quantize to all zeros.
export function quantize(values, f) {
  let maxAbs = 0;
  for (let i = 0; i < values.length; i++) {
    const a = Math.abs(values[i]);
    if (a > maxAbs && Number.isFinite(a)) maxAbs = a;
  }
  const scale = maxAbs > 0 ? f.target / maxAbs : 1;
  const out = new ARRAYS[f.bits](values.length);
  if (f.id === "f16") {
    for (let i = 0; i < values.length; i++) out[i] = f32ToF16(values[i] * scale);
  } else {
    const via = bridge(f);
    for (let i = 0; i < values.length; i++) out[i] = via[f32ToF16(values[i] * scale)];
  }
  return { scale, payload: out };
}

export function dequantize(payload, scale, f) {
  const out = new Float32Array(payload.length);
  const inv = 1 / scale;
  if (f.id === "f16") {
    for (let i = 0; i < payload.length; i++) out[i] = f16ToF32(payload[i]) * inv;
  } else {
    for (let i = 0; i < payload.length; i++) out[i] = decodeMini(payload[i], f) * inv;
  }
  return out;
}

/// Back-compat shims for the f16-specific helpers.
export const quantizeF16 = (values) => quantize(values, WIRE_FORMATS.f16);
export const dequantizeF16 = (payload, scale) =>
  dequantize(payload, scale, WIRE_FORMATS.f16);


/// `msg` → one ArrayBuffer. `precision` picks the payload format; the format id
/// and its scale travel in the header, so the receiver needs no configuration
/// and ranks set to different precisions still interoperate.
export function encode(msg, { precision = "f32" } = {}) {
  const field = F32_FIELDS.find((k) => msg[k] instanceof Float32Array);
  let payload = field ? msg[field] : null;

  const header = { ...msg };
  let format = null;
  if (field) {
    delete header[field];
    header._f32 = field;
    const f = WIRE_FORMATS[precision];
    if (f && payload.length) {
      const q = quantize(payload, f);
      header._fmt = f.id;
      header._scale = q.scale;
      payload = q.payload;
      format = f;
    }
  }

  let json = new TextEncoder().encode(JSON.stringify(header));
  const pad = (4 - ((4 + json.length) % 4)) % 4;
  if (pad) {
    const padded = new Uint8Array(json.length + pad);
    padded.set(json);
    padded.fill(0x20, json.length); // spaces: still valid JSON
    json = padded;
  }

  const buf = new ArrayBuffer(4 + json.length + (payload ? payload.byteLength : 0));
  new DataView(buf).setUint32(0, json.length, true);
  new Uint8Array(buf, 4, json.length).set(json);
  if (payload) {
    // 8-bit needs no alignment, 16-bit needs 2, f32 needs 4 — the header is
    // padded to 4, which satisfies all three.
    const Ctor = format ? ARRAYS[format.bits] : Float32Array;
    new Ctor(buf, 4 + json.length, payload.length).set(payload);
  }
  return buf;
}

export function decode(buf) {
  const view = new DataView(buf);
  const headerLen = view.getUint32(0, true);
  const json = new TextDecoder().decode(new Uint8Array(buf, 4, headerLen));
  const msg = JSON.parse(json);
  if (msg._f32) {
    const offset = 4 + headerLen;
    const field = msg._f32;
    const f = msg._fmt ? WIRE_FORMATS[msg._fmt] : null;
    if (f) {
      const Ctor = ARRAYS[f.bits];
      const raw = new Ctor(buf, offset, (buf.byteLength - offset) / (f.bits / 8));
      msg[field] = dequantize(raw, msg._scale, f);
      delete msg._fmt;
      delete msg._scale;
    } else {
      msg[field] = new Float32Array(buf, offset, (buf.byteLength - offset) / 4);
    }
    delete msg._f32;
  }
  return msg;
}

// ── transports ────────────────────────────────────────────────────────────

/// What a vector becomes after a trip through the wire at this precision.
///
/// The leader needs this: it applies an f32 update but publishes a quantized
/// one, so without adopting its own published values it would hold something no
/// other rank has, and "every rank holds identical parameters" — the property
/// the whole elastic-join design rests on — would quietly stop being true.
export function roundTrip(values, precision) {
  // `null` means "the wire changes nothing", so the caller can skip the copy.
  const f = WIRE_FORMATS[precision];
  if (!f || !values.length) return null;
  const { scale, payload } = quantize(values, f);
  return dequantize(payload, scale, f);
}

/// Same-browser bus. Encoded like the P2P path so both share one wire format
/// (and one set of tests) rather than relying on structured clone for one and
/// a codec for the other.
export function channelBus(name, { onError = () => {}, precision = "f32" } = {}) {
  const ch = new BroadcastChannel(name);
  // Counted here because only the bus knows the encoded size — the protocol
  // used to assume 4 bytes an element, which reported f32 volumes no matter
  // what was actually on the wire and hid the whole point of f16.
  let bytes = 0;
  const bus = {
    onmessage() {},
    postMessage: (msg) => {
      const buf = encode(msg, { precision });
      bytes += buf.byteLength;
      ch.postMessage(buf);
    },
    close: () => ch.close(),
    quantize: (values) => roundTrip(values, precision),
    bytes: () => bytes,
  };
  ch.onmessage = (ev) => {
    try {
      bytes += ev.data.byteLength ?? 0;
      bus.onmessage({ data: decode(ev.data) });
    } catch (e) {
      // A frame that cannot be decoded is indistinguishable from one that was
      // never sent, and a silently dropped round of parameters stalls the
      // cohort at one round per collect timeout. Say so instead.
      onError(`undecodable frame: ${e?.message ?? e}`);
    }
  };
  return bus;
}

/// Worker-side bus that forwards through a `postMessage` port (the main thread,
/// which owns the real transport). Buffers are transferred, never copied.
///
/// `port` needs `postMessage(msg, transfer)` and an assignable `onmessage` —
/// `self` inside a dedicated Worker, or a `MessagePort`.
///
/// Each outbound message is tagged with its `kind` so the host can apply
/// backpressure without parsing the payload, and `pressure()` reports how many
/// messages are waiting there so a leader can pace itself.
export function proxyBus(port, { tag = "bus", onError = () => {}, precision = "f32" } = {}) {
  let depth = 0;
  let bytes = 0;
  const bus = {
    onmessage() {},
    postMessage(msg) {
      const buf = encode(msg, { precision });
      bytes += buf.byteLength;
      port.postMessage({ t: tag, kind: msg.t, buf }, [buf]);
    },
    pressure: () => depth,
    quantize: (values) => roundTrip(values, precision),
    bytes: () => bytes,
    close() {},
  };
  const prior = port.onmessage?.bind(port);
  port.onmessage = (ev) => {
    if (ev.data?.t === `${tag}:pressure`) {
      depth = ev.data.depth;
      return;
    }
    if (ev.data?.t !== tag) {
      prior?.(ev);
      return;
    }
    try {
      bytes += ev.data.buf?.byteLength ?? 0;
      bus.onmessage({ data: decode(ev.data.buf) });
    } catch (e) {
      onError(`undecodable frame: ${e?.message ?? e}`);
    }
  };
  return bus;
}

/// Main-thread end of a `proxyBus`: hands encoded buffers to `send`, and
/// `deliver` pushes inbound ones back down to the Worker.
///
/// ## Backpressure
///
/// `send` is allowed to be slow — a WebRTC data channel chunks a few hundred KB
/// and Trystero rejects a send outright once too many transfers are pending
/// ("too many pending transfers"). A leader that fires a round per iteration
/// will outrun the channel, so this queues:
///
/// * **At most `maxInflight` transfers at a time.** No new send starts until
///   the previous one's promise settles.
/// * **Rounds coalesce; the newest wins.** Dropping a superseded round is
///   *safe* precisely because each round carries the parameters it is to be
///   computed against — a newer round is strictly better than the one it
///   replaces, so queueing both would only deliver stale parameters late.
/// * **Small messages keep their order**, bounded by `maxQueue`; the oldest is
///   dropped with a report rather than growing without limit.
export function proxyHost(
  worker,
  send,
  { tag = "bus", maxInflight = 1, maxQueue = 32, onDrop = () => {} } = {},
) {
  let inflight = 0;
  let round = null; // the newest round waiting to go out
  const small = [];

  const depth = () => small.length + (round ? 1 : 0) + inflight;
  const reportPressure = () => worker.postMessage({ t: `${tag}:pressure`, depth: depth() });

  function take() {
    if (small.length) return small.shift();
    if (round) {
      const r = round;
      round = null;
      return r;
    }
    return null;
  }

  function pump() {
    while (inflight < maxInflight) {
      const next = take();
      if (!next) break;
      inflight += 1;
      let settled;
      try {
        settled = send(next);
      } catch (e) {
        inflight -= 1;
        onDrop(`send threw: ${e?.message ?? e}`);
        continue;
      }
      Promise.resolve(settled)
        .catch((e) => onDrop(`send failed: ${e?.message ?? e}`))
        .finally(() => {
          inflight -= 1;
          reportPressure();
          pump();
        });
    }
    reportPressure();
  }

  const prior = worker.onmessage?.bind(worker);
  worker.onmessage = (ev) => {
    if (ev.data?.t !== tag) {
      prior?.(ev);
      return;
    }
    if (ev.data.kind === "round") {
      if (round) onDrop("superseded a queued round");
      round = ev.data.buf;
    } else {
      if (small.length >= maxQueue) {
        small.shift();
        onDrop("outbound queue full, dropped the oldest message");
      }
      small.push(ev.data.buf);
    }
    pump();
  };

  return {
    deliver(buf) {
      worker.postMessage({ t: tag, buf }, [buf]);
    },
    pressure: depth,
  };
}
