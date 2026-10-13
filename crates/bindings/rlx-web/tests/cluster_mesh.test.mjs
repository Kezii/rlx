// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The mesh adapter's contract with Trystero.
//
//   node --test crates/bindings/rlx-web/tests/cluster_mesh.test.mjs
//
// WebRTC cannot run outside a browser, so the transport itself is untestable
// here — but the *API shape* is exactly what broke in a browser: 0.26's
// `makeAction` returns an object, and array-destructuring it threw
// "{} is not iterable", which surfaced only as "mesh unavailable". These tests
// pin the shape so the next version bump fails here instead of there.

import { test } from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import path from "node:path";
import {
  joinMesh, roomCode, turnServers, iceProbe, iceVerdict, browserName,
  DEFAULT_ICE_SERVERS,
} from "../web/cluster-p2p.js";
import { encode, decode } from "../web/cluster-bus.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const modern = path.join(HERE, "fixtures", "fake-trystero.mjs");
const legacy = path.join(HERE, "fixtures", "fake-trystero-legacy.mjs");
const url = (p) => `file://${p}`;

/// `onMessage` is async (a `Blob` payload's bytes are only reachable
/// asynchronously), so delivery completes a microtask after `_deliver`.
const settle = () => new Promise((r) => setTimeout(r, 0));

test("the 0.26 object-shaped action is driven correctly", async () => {
  const seen = [];
  const peerCounts = [];
  const mesh = await joinMesh({
    room: "ROOM1",
    moduleUrl: url(modern),
    onData: (buf) => seen.push(decode(buf)),
    onPeers: (n) => peerCounts.push(n),
  });
  const { rooms } = await import(url(modern));
  const room = rooms.at(-1);

  // `onMessage` must have been *assigned*, not called: it starts as null, so
  // calling it would have been a silent no-op and no message would arrive.
  assert.equal(typeof room.onPeerJoin, "function", "onPeerJoin must be assigned");

  // Inbound: a real encoded message round-trips to the protocol.
  const msg = { t: "commit", from: 7, step: 3, loss: 0.5, samples: 8, params: Float32Array.from([1, 2, 3]) };
  room._deliver("rlxbus", new Uint8Array(encode(msg)));
  await settle();
  assert.equal(seen.length, 1, "no message reached onData");
  assert.equal(seen[0].t, "commit");
  assert.deepEqual(Array.from(seen[0].params), [1, 2, 3]);

  // Outbound: sent as bytes the wire format can decode back.
  mesh.send(encode({ t: "beat", from: 7, step: 3 }));
  assert.equal(room.sent.length, 1);
  const sent = room.sent[0];
  assert.ok(sent instanceof Uint8Array, "payload should be binary");
  assert.equal(decode(sent.buffer).t, "beat");

  room._addPeer("p1");
  room._addPeer("p2");
  assert.deepEqual(peerCounts.slice(-2), [1, 2], "peer count should track joins");
  room._dropPeer("p1");
  assert.equal(peerCounts.at(-1), 1);
  assert.equal(mesh.peers(), 1);

  mesh.close();
  assert.ok(room.left, "close should leave the room");
});

test("a pre-0.26 tuple-shaped action still works", async () => {
  const seen = [];
  const mesh = await joinMesh({
    room: "ROOM2",
    moduleUrl: url(legacy),
    onData: (buf) => seen.push(decode(buf)),
  });
  const { rooms } = await import(url(legacy));
  const room = rooms.at(-1);

  room._deliver(new Uint8Array(encode({ t: "beat", from: 1, step: 0 })));
  await settle();
  assert.equal(seen.at(-1)?.t, "beat", "tuple-shaped receive was not wired");
  mesh.send(encode({ t: "hello", from: 1 }));
  assert.equal(decode(room.sent.at(-1).buffer).t, "hello");
});

/// Trystero hands binary back as a `Uint8Array` that may be a *view* into a
/// larger buffer. Decoding from offset 0 would then read the wrong bytes —
/// the same alignment/offset trap the wire format already guards.
test("a payload delivered as an offset view decodes correctly", async () => {
  const seen = [];
  await joinMesh({
    room: "ROOM3",
    moduleUrl: url(modern),
    onData: (buf) => seen.push(decode(buf)),
  });
  const { rooms } = await import(url(modern));
  const room = rooms.at(-1);

  const payload = new Uint8Array(encode({ t: "beat", from: 9, step: 1 }));
  const backing = new Uint8Array(payload.byteLength + 7);
  backing.set(payload, 3);
  const view = new Uint8Array(backing.buffer, 3, payload.byteLength);
  assert.notEqual(view.byteOffset, 0, "test needs a non-zero offset");

  room._deliver("rlxbus", view);
  await settle();
  assert.equal(seen.at(-1)?.t, "beat");
  assert.equal(seen.at(-1)?.from, 9);
});

test("the room is configured with headroom for a parameter vector", async () => {
  await joinMesh({ room: "ROOM4", moduleUrl: url(modern), relayUrls: ["wss://a", "wss://b"] });
  const { rooms } = await import(url(modern));
  const room = rooms.at(-1);
  // A bucket for mnist-mlp is ~398 KB; the default cap must not reject it.
  assert.ok(room.config.maxReceiveBytes >= 1_000_000, "receive cap too small for a bucket");
  assert.equal(room.config.appId, "rlx-tab-cluster");
  assert.deepEqual(room.config.relayConfig.urls, ["wss://a", "wss://b"]);
});

/// `onJoinError` is **per peer** — `JoinError` carries a `peerId`, and the room
/// stays usable while one peer is unreachable. Reporting it as "join failed"
/// pointed at NAT configuration when the usual cause is a peer that is gone:
/// a tab closed without a clean leave stays advertised on a public relay, and a
/// backgrounded tab is throttled too hard to finish ICE.
test("an unreachable peer is reported per peer, not as a room failure", async () => {
  const peerErrors = [];
  const roomErrors = [];
  await joinMesh({
    room: "ROOM5",
    moduleUrl: url(modern),
    onError: (e) => roomErrors.push(e),
    onPeerError: (peerId, why) => peerErrors.push([peerId, why]),
  });
  const { rooms } = await import(url(modern));
  rooms.at(-1)._failJoin("could not connect after exchanging SDP", "XmvyuAs4iB96l2rzxekh");

  assert.equal(peerErrors.length, 1, "peer failure did not reach onPeerError");
  assert.equal(peerErrors[0][0], "XmvyuAs4iB96l2rzxekh", "the peer id must be carried through");
  assert.match(peerErrors[0][1], /could not connect/);
  assert.deepEqual(roomErrors, [], "a single peer failing is not a room-level error");
});

test("a dead peer is abandoned rather than retried forever", async () => {
  await joinMesh({ room: "ROOM7", moduleUrl: url(modern) });
  const { rooms } = await import(url(modern));
  const timeout = rooms.at(-1).callbacks?.handshakeTimeoutMs;
  assert.ok(timeout > 0, "no handshake timeout — a gone peer is retried indefinitely");
  assert.ok(timeout <= 30000, `handshake timeout ${timeout}ms is too patient`);
});

test("a module without joinRoom fails with a message naming it", async () => {
  await assert.rejects(
    () => joinMesh({ room: "R", moduleUrl: "data:text/javascript,export const nope=1" }),
    /does not export joinRoom/,
  );
});

test("room codes avoid look-alike glyphs", () => {
  for (let i = 0; i < 50; i++) {
    const code = roomCode();
    assert.equal(code.length, 6);
    assert.doesNotMatch(code, /[AEIOU01BY]/, `ambiguous glyph in ${code}`);
  }
});

/// `DataPayload` permits a `Blob`, whose bytes are only reachable
/// asynchronously. Reading one as a view yields garbage or throws, and either
/// way the frame vanishes — which looks exactly like a peer that never answers.
test("a payload delivered as a Blob decodes correctly", async () => {
  const seen = [];
  await joinMesh({
    room: "ROOM6",
    moduleUrl: url(modern),
    onData: (buf) => seen.push(decode(buf)),
  });
  const { rooms } = await import(url(modern));
  const room = rooms.at(-1);

  const bytes = new Uint8Array(encode({ t: "beat", from: 11, step: 2 }));
  room._deliver("rlxbus", new Blob([bytes]));
  await settle();
  assert.equal(seen.at(-1)?.t, "beat");
  assert.equal(seen.at(-1)?.from, 11);
});

// ── ICE configuration ──────────────────────────────────────────────────────
// Without STUN a peer gathers only `host` candidates — its own LAN addresses.
// Two tabs on one machine and two devices on one network both work that way,
// so this only shows up across networks: the SDP exchange succeeds and ICE
// then fails with "could not connect to peer … configure TURN servers".

test("STUN servers are configured by default", async () => {
  await joinMesh({ room: "ICE1", moduleUrl: url(modern) });
  const { rooms } = await import(url(modern));
  const ice = rooms.at(-1).config.rtcConfig?.iceServers;
  assert.ok(ice?.length, "no iceServers were passed — ICE will only see host candidates");
  const urls = ice.flatMap((s) => (Array.isArray(s.urls) ? s.urls : [s.urls]));
  assert.ok(urls.some((u) => u.startsWith("stun:")), `no STUN in ${urls.join(", ")}`);
  // STUN only reveals a peer's own address; nothing is relayed through it, so
  // the "payloads never touch a third party" property survives the default.
  assert.ok(!urls.some((u) => u.startsWith("turn")), "TURN must never be a default");
});

test("explicit iceServers replace the defaults", async () => {
  await joinMesh({
    room: "ICE2",
    moduleUrl: url(modern),
    iceServers: [{ urls: ["stun:example.test:3478"] }],
  });
  const { rooms } = await import(url(modern));
  const urls = rooms.at(-1).config.rtcConfig.iceServers.flatMap((s) => s.urls);
  assert.deepEqual(urls, ["stun:example.test:3478"]);
});

test("TURN entries parse and are appended after STUN", async () => {
  const parsed = turnServers("turn:a.test:3478|alice|secret;turns:b.test:5349|bob|hunter2");
  assert.deepEqual(parsed, [
    { urls: "turn:a.test:3478", username: "alice", credential: "secret" },
    { urls: "turns:b.test:5349", username: "bob", credential: "hunter2" },
  ]);
  assert.deepEqual(turnServers(""), []);
  assert.deepEqual(turnServers(undefined), []);
  // A bare URL with no credentials is still usable.
  assert.deepEqual(turnServers("turn:open.test:3478"), [{ urls: "turn:open.test:3478" }]);

  await joinMesh({ room: "ICE3", moduleUrl: url(modern), turn: parsed });
  const { rooms } = await import(url(modern));
  const ice = rooms.at(-1).config.rtcConfig.iceServers;
  assert.equal(ice.length, DEFAULT_ICE_SERVERS.length + 2);
  // Order matters: a direct path should be tried before paying for a relay.
  assert.ok(String(ice.at(-1).urls).startsWith("turns:"));
  assert.ok(String(ice[0].urls ?? ice[0]).includes("stun:"));
});

// ── ICE preflight ──────────────────────────────────────────────────────────
// "Could not connect to peer after exchanging SDP" has three causes needing
// three different answers. A throwaway RTCPeerConnection distinguishes them
// before any peer exists; `PeerConnection` is injectable so the tally and the
// verdict are testable without a browser.

/// A fake `RTCPeerConnection` that emits the candidate types it is told to.
function fakePeerConnection(types, { fail = null } = {}) {
  return class {
    constructor(config) {
      this.config = config;
      this.onicecandidate = null;
    }
    createDataChannel() {}
    async createOffer() {
      if (fail) throw new Error(fail);
      return { type: "offer", sdp: "" };
    }
    async setLocalDescription() {
      // Candidates trickle in after the local description is set.
      for (const [type, address] of types) {
        this.onicecandidate?.({ candidate: { type, address } });
      }
      this.onicecandidate?.({ candidate: null }); // gathering complete
    }
    close() {
      this.closed = true;
    }
  };
}

test("the preflight tallies candidate types and the public address", async () => {
  const probe = await iceProbe({
    PeerConnection: fakePeerConnection([
      ["host", "192.168.1.9"],
      ["host", "10.0.0.5"],
      ["srflx", "203.0.113.7"],
      ["srflx", "203.0.113.7"],
    ]),
  });
  assert.equal(probe.host, 2);
  assert.equal(probe.srflx, 2);
  assert.equal(probe.relay, 0);
  assert.deepEqual(probe.publicIps, ["203.0.113.7"], "duplicate addresses should collapse");
  assert.equal(probe.error, null);
});

test("the verdict names the cause, not just the symptom", async () => {
  const verdict = async (types) => iceVerdict(await iceProbe({ PeerConnection: fakePeerConnection(types) }));

  // Nothing at all: WebRTC is blocked outright.
  let v = await verdict([]);
  assert.equal(v.ok, false);
  assert.match(v.text, /blocked/);

  // Host only. Two causes — blocked UDP, or the browser restricting ICE — and
  // the verdict must name both plus the one-step way to tell them apart,
  // because "STUN unreachable" alone sends people to reconfigure a firewall
  // that was never the problem.
  v = await verdict([["host", "192.168.1.9"]]);
  assert.equal(v.ok, false);
  assert.match(v.text, /outbound UDP is blocked/);
  assert.match(v.text, /another browser/);
  assert.match(v.text, /one non-isolated Wi-Fi/, "must say host candidates still work");

  // Host + srflx: healthy, but no relay — the TURN case must be named as
  // conditional rather than demanded of everyone.
  v = await verdict([["host", "192.168.1.9"], ["srflx", "203.0.113.7"]]);
  assert.equal(v.ok, true);
  assert.match(v.text, /direct paths only/);
  assert.match(v.text, /symmetric NAT/);

  // All three: configuration is not the problem.
  v = await verdict([
    ["host", "192.168.1.9"],
    ["srflx", "203.0.113.7"],
    ["relay", "198.51.100.2"],
  ]);
  assert.equal(v.ok, true);
  assert.match(v.text, /both available/);
});

test("the preflight reports a failure instead of hanging", async () => {
  const probe = await iceProbe({
    PeerConnection: fakePeerConnection([], { fail: "offer rejected" }),
    timeoutMs: 50,
  });
  assert.match(probe.error ?? "", /offer rejected/);
  assert.equal(iceVerdict(probe).ok, false);
});

test("no WebRTC at all is reported as such", async () => {
  const probe = await iceProbe({ PeerConnection: undefined });
  assert.match(probe.error ?? "", /WebRTC unavailable/);
});

test("the verdict names the browser, and warns Safari specifically", () => {
  const hostOnly = { host: 1, srflx: 0, relay: 0, prflx: 0, publicIps: [], error: null };

  const safari = iceVerdict(hostOnly, "Safari");
  assert.match(safari.text, /^Safari: /, "the line should start with the browser");
  assert.match(safari.text, /camera\/mic permission/, "Safari's own restriction should be named");

  const chrome = iceVerdict(hostOnly, "Chrome");
  assert.match(chrome.text, /^Chrome: /);
  assert.match(chrome.text, /VPN or privacy extension/);
  assert.doesNotMatch(chrome.text, /camera\/mic/, "do not blame Safari's rule in Chrome");

  // Healthy verdicts carry the browser too, so a pasted line is self-describing.
  const healthy = { host: 2, srflx: 1, relay: 0, prflx: 0, publicIps: ["203.0.113.7"], error: null };
  assert.match(iceVerdict(healthy, "Firefox").text, /^Firefox: host ✓ · srflx ✓ \(203\.0\.113\.7\)/);
});

test("browser detection distinguishes the engines that matter", () => {
  const ua = {
    Safari: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
    Chrome: "Mozilla/5.0 (Macintosh) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36",
    Firefox: "Mozilla/5.0 (Macintosh; rv:131.0) Gecko/20100101 Firefox/131.0",
    Edge: "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36 Edg/155.0.0.0",
  };
  for (const [expected, agent] of Object.entries(ua)) {
    assert.equal(browserName(agent), expected, agent);
  }
  // Chrome's UA contains "Safari/" and Edge's contains "Chrome/": order matters.
  assert.equal(browserName(""), "this browser");
});
