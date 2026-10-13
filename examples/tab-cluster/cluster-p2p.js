// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Cross-device transport for the tab cluster, with no server of ours in it.
//
// Ranks connect as a **WebRTC mesh**: signalling (an offer, an answer and a few
// ICE candidates — kilobytes, once per peer) goes over public infrastructure,
// and from then on every gradient bucket travels directly peer-to-peer. That
// split is the whole reason to use WebRTC here rather than piping the cluster
// through a public broker or an iroh relay: buckets are tens to hundreds of KB
// *every step*, which is abusive to relay and slow to route, while signalling
// is tiny and one-shot.
//
// Mesh and signalling come from Trystero (MIT), loaded as a pinned ES module.
// Hand-rolling them was the alternative, but WebRTC negotiation is the one part
// of this demo that cannot be tested outside a browser — so it belongs in
// maintained code, not in ours. `?p2p=<url>` overrides the module, so pointing
// at a vendored or self-hosted copy is a URL change, not a code change.

/// Pinned so a static deploy cannot change behaviour under us. 0.26's default
/// export carries the nostr strategy, i.e. signalling over public relays.
export const DEFAULT_P2P_MODULE = "https://esm.sh/trystero@0.26.0";

/// Public STUN servers, used unless `iceServers` overrides them.
///
/// Without STUN a peer only gathers `host` candidates — its own LAN addresses.
/// That is enough for two tabs on one machine and for two devices on one
/// network, which is why it took a cross-network run to notice: ICE fails after
/// the SDP exchange succeeds, so signalling looks healthy and the mesh reports
/// *"could not connect to peer … configure TURN servers"*.
///
/// STUN only tells a peer its own public address; **no data flows through it**,
/// so the property that gradients never touch a third party holds. That is not
/// true of TURN — see `turnServers`.
export const DEFAULT_ICE_SERVERS = [
  { urls: ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"] },
  { urls: ["stun:stun.cloudflare.com:3478"] },
];

/// Parse `turn:host:3478|user|credential;turns:host:5349|user|credential`.
///
/// TURN is a **relay**: when both peers sit behind symmetric NAT there is no
/// direct path, and every byte goes through that server. For this cluster that
/// means the whole parameter vector each round — tens of MB/s — which is why
/// there is no default and no public TURN baked in. Supply your own if you need
/// it, and know what it carries.
export function turnServers(spec) {
  if (!spec) return [];
  return spec
    .split(";")
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => {
      const [urls, username, credential] = entry.split("|").map((f) => f?.trim());
      const server = { urls };
      if (username) server.username = username;
      if (credential) server.credential = credential;
      return server;
    });
}

/// Normalise whatever the mesh hands back into an ArrayBuffer `decode` can read
/// from offset 0.
///
/// Two shapes bite here. A `Uint8Array` may be a *view* into a larger buffer,
/// so decoding from offset 0 would read the wrong bytes. And `DataPayload`
/// permits a `Blob`, whose bytes are only reachable asynchronously — reading it
/// as a view yields garbage or throws, and either way the frame is lost, which
/// looks exactly like a cohort stalling on a peer that never answers.
async function ownedBuffer(data) {
  if (data instanceof ArrayBuffer) return data;
  if (typeof Blob !== "undefined" && data instanceof Blob) return data.arrayBuffer();
  const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
  return u8.byteOffset === 0 && u8.byteLength === u8.buffer.byteLength
    ? u8.buffer
    : u8.slice().buffer;
}

/// Join `room` and return `{ send, close, peers }`. `onData` receives an
/// ArrayBuffer per message; `onPeers` the current peer count.
///
/// Written against Trystero 0.26's API, which is not the tuple-returning one
/// most examples show: `makeAction` returns an **object** (`{ send, onMessage }`)
/// and `onPeerJoin` / `onPeerLeave` are properties to **assign**, not functions
/// to call. Getting either wrong fails in a way worth naming — array-destructuring
/// the action throws `{} is not iterable`, and `handle.onPeerJoin?.(cb)` silently
/// does nothing because the property starts as `null`. Both shapes are accepted
/// below so `?p2p=` can point at another version.
export async function joinMesh({
  room,
  appId = "rlx-tab-cluster",
  moduleUrl = DEFAULT_P2P_MODULE,
  relayUrls,
  maxReceiveBytes = 64 * 1024 * 1024,
  iceServers = DEFAULT_ICE_SERVERS,
  turn = [],
  handshakeTimeoutMs = 15000,
  onData = () => {},
  onPeers = () => {},
  onError = () => {},
  onPeerError = () => {},
}) {
  const mod = await import(/* @vite-ignore */ moduleUrl);
  const joinRoom = mod.joinRoom ?? mod.default?.joinRoom;
  if (typeof joinRoom !== "function") {
    throw new Error(`${moduleUrl} does not export joinRoom`);
  }

  // `maxReceiveBytes` is a real cap, not a hint: a parameter vector for
  // mnist-mlp is ~398 KB and the bigger models are larger, so leave headroom
  // rather than have buckets rejected on arrival.
  const config = { appId, maxReceiveBytes };
  // ICE needs more than host candidates the moment the peers are not on one
  // network. TURN entries go in the same list; Trystero also accepts them via
  // `turnConfig`, but one `rtcConfig` keeps the ordering explicit.
  const ice = [...iceServers, ...turn];
  if (ice.length) config.rtcConfig = { iceServers: ice };
  if (relayUrls?.length) config.relayConfig = { urls: relayUrls };
  const handle = joinRoom(config, room, {
    // Per-*peer*, not per-room: `JoinError` carries a `peerId`, and the room
    // stays perfectly usable while one peer is unreachable. Reporting this as
    // "join failed" sent the reader looking for a NAT problem when the usual
    // cause is a peer that is gone — a tab closed without a clean leave stays
    // advertised on a public relay for a while, and a backgrounded tab has its
    // timers throttled hard enough to never finish ICE.
    onJoinError: (e) => onPeerError(e?.peerId ?? "unknown", e?.error ?? String(e)),
    // Abandon a peer that never completes its handshake instead of retrying it
    // indefinitely.
    handshakeTimeoutMs,
  });

  // One action namespace carries the whole protocol: the payload is already a
  // single encoded buffer, and Trystero chunks anything over a data channel's
  // message limit — which the 398 KB bucket of `mnist-mlp` exceeds.
  const made = handle.makeAction("rlxbus");
  const action = Array.isArray(made)
    ? { send: made[0], setOnMessage: (cb) => made[1](cb) } // pre-0.26 tuple
    : { send: made?.send, setOnMessage: (cb) => { made.onMessage = cb; } };
  if (typeof action.send !== "function") {
    throw new Error("makeAction returned neither {send} nor [send, receive]");
  }
  action.setOnMessage(async (data) => {
    try {
      onData(await ownedBuffer(data));
    } catch (e) {
      onError(`undecodable payload: ${e?.message ?? e}`);
    }
  });

  const count = () => Object.keys(handle.getPeers?.() ?? {}).length;
  const announce = () => onPeers(count());
  if (typeof handle.onPeerJoin === "function") {
    // Older API: a registration function.
    handle.onPeerJoin(announce);
    handle.onPeerLeave?.(announce);
  } else {
    handle.onPeerJoin = announce;
    handle.onPeerLeave = announce;
  }

  return {
    selfId: mod.selfId,
    peers: count,
    /// The live `RTCPeerConnection`s, so a caller can read `getStats()` and see
    /// what the transport actually did — bytes, round-trip time, and whether
    /// the selected candidate pair is direct or relayed.
    connections: () => handle.getPeers?.() ?? {},
    // Returns the promise rather than swallowing it: the caller's queue uses
    // it for backpressure, and a mesh that is handed more than it can drain
    // refuses outright ("too many pending transfers").
    send: (buf) => action.send(new Uint8Array(buf)),
    close: () => handle.leave?.(),
  };
}

/// Short, unambiguous room code. Avoids vowels and look-alike glyphs because
/// the usual way this reaches a second device is someone typing it.
export function roomCode(length = 6) {
  const alphabet = "23456789CDFGHJKLMNPQRSTVWXZ";
  const bytes = crypto.getRandomValues(new Uint8Array(length));
  return Array.from(bytes, (b) => alphabet[b % alphabet.length]).join("");
}

// ── ICE preflight ──────────────────────────────────────────────────────────

/// What this machine can obtain *before* any peer exists.
///
/// "Could not connect to peer after exchanging SDP" has three causes that need
/// three different answers, and the message cannot tell them apart:
///
///   * no `srflx` → STUN is unreachable (UDP blocked), so the peer only knows
///     its own LAN addresses and can never be found from another network;
///   * `srflx` but no `relay` → a direct path is the only option, which fails
///     when both ends are behind symmetric NAT or the Wi-Fi has client
///     isolation. That needs TURN, and nothing in this page can substitute;
///   * both present → the configuration is fine, so an unreachable peer is
///     almost certainly gone or suspended rather than unroutable.
///
/// A throwaway `RTCPeerConnection` answers all three in a couple of seconds,
/// with no peer and no room. `PeerConnection` is injectable so the tally and
/// the classification can be tested without a browser.
export async function iceProbe({
  iceServers = DEFAULT_ICE_SERVERS,
  turn = [],
  timeoutMs = 4000,
  PeerConnection = globalThis.RTCPeerConnection,
} = {}) {
  const result = { host: 0, srflx: 0, relay: 0, prflx: 0, publicIps: [], error: null };
  if (typeof PeerConnection !== "function") {
    result.error = "WebRTC unavailable (not a secure context?)";
    return result;
  }

  let pc;
  try {
    pc = new PeerConnection({ iceServers: [...iceServers, ...turn] });
    // A data channel is what makes the browser gather candidates at all.
    pc.createDataChannel("rlx-ice-probe");
    await new Promise((resolve) => {
      const done = () => resolve();
      const timer = setTimeout(done, timeoutMs);
      pc.onicecandidate = (ev) => {
        if (!ev.candidate) {
          clearTimeout(timer);
          done();
          return;
        }
        const type = ev.candidate.type ?? "";
        if (type in result) result[type] += 1;
        // The reflexive address is this machine as the internet sees it —
        // worth showing, because two peers reporting the same one are behind
        // the same NAT and should be using host candidates.
        if (type === "srflx" && ev.candidate.address) {
          if (!result.publicIps.includes(ev.candidate.address)) {
            result.publicIps.push(ev.candidate.address);
          }
        }
      };
      pc.createOffer()
        .then((offer) => pc.setLocalDescription(offer))
        .catch((e) => {
          result.error = String(e?.message ?? e);
          clearTimeout(timer);
          done();
        });
    });
  } catch (e) {
    result.error = String(e?.message ?? e);
  } finally {
    try {
      pc?.close();
    } catch {
      /* already closed */
    }
  }
  return result;
}

/// Coarse engine detection, for hints that only apply to one of them.
///
/// Not for feature decisions — those are probed. This is purely so the ICE
/// line names the browser, because the same page on the same machine can
/// gather different candidates in different browsers and the first question is
/// always "which one is this?".
export function browserName(ua = globalThis.navigator?.userAgent ?? "") {
  if (/Firefox\//.test(ua)) return "Firefox";
  if (/Edg\//.test(ua)) return "Edge";
  if (/OPR\//.test(ua)) return "Opera";
  if (/Chrome\//.test(ua)) return "Chrome";
  if (/Safari\//.test(ua)) return "Safari";
  return "this browser";
}

/// Turn a probe into the one sentence a reader can act on.
export function iceVerdict(probe, browser = browserName()) {
  if (probe.error) return { ok: false, text: `ICE unavailable: ${probe.error}` };
  const flags = ["host", "srflx", "relay"]
    .map(
      (k) =>
        `${k} ${probe[k] ? "✓" : "✗"}` +
        (k === "srflx" && probe.publicIps.length ? ` (${probe.publicIps.join(", ")})` : ""),
    )
    .join(" · ");
  const seen = `${browser}: ${flags}`;
  if (!probe.host) {
    return { ok: false, text: `${seen} — no candidates at all; WebRTC is blocked here` };
  }
  if (!probe.srflx) {
    // Two very different causes, and the page cannot tell them apart alone —
    // but the reader can, in one step, because it is cheap to try another
    // browser on the same machine. Safari and privacy extensions both restrict
    // ICE candidate gathering; a firewall blocking outbound UDP does not care
    // which browser is asking.
    const safari = browser === "Safari";
    return {
      ok: false,
      text:
        `${seen} — ${browser} got no reflexive candidate. Either outbound UDP is` +
        " blocked here, or this browser is restricting ICE" +
        (safari
          ? " (Safari limits candidates on pages with no camera/mic permission)"
          : " (a VPN or privacy extension will do this)") +
        ". Try the same room in another browser on this machine: if that one" +
        " shows srflx ✓ it is the browser, not the network. Either way host" +
        " candidates still work, so two devices on one non-isolated Wi-Fi will" +
        " pair — only cross-network needs the reflexive address.",
    };
  }
  if (!probe.relay) {
    return {
      ok: true,
      text: `${seen} — direct paths only. Fine on one network; if the other device` +
        " is behind symmetric NAT or client isolation, add ?turn=…",
    };
  }
  return { ok: true, text: `${seen} — direct and relayed paths both available` };
}
