// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// UI thread for the tab cluster. It owns no training state — the Worker holds
// the wasm rank and the protocol — but it does own the *transport*, because
// `RTCPeerConnection` is unavailable in a Worker.
//
//   no ?room=   same-browser tabs, over a BroadcastChannel in the Worker.
//   ?room=CODE  a WebRTC mesh: this thread ferries one transferred buffer per
//               message between the Worker and the peers. Nothing is parsed
//               here, and the page stays a static site — no server of ours.

import { proxyHost } from "./cluster-bus.js";
import qrcode from "./vendor/qrcode-generator.mjs";
import { highlightAll } from "./highlight.js";
import {
  joinMesh, roomCode, turnServers, iceProbe, iceVerdict, DEFAULT_P2P_MODULE,
} from "./cluster-p2p.js";

const $ = (id) => document.getElementById(id);
const fmt = (v, n = 2) => (Number.isFinite(v) ? v.toFixed(n) : "—");
const params = new URLSearchParams(location.search);

const worker = new Worker("./cluster-worker.js", { type: "module" });

const history = [];
const MAX_POINTS = 240;

const note = (text, cls = "warn") => {
  $("fault").textContent = text;
  $("fault").className = "v " + cls;
};

worker.onmessage = (ev) => {
  const s = ev.data;
  if (s.t === "log") return note(s.text);
  if (s.t !== "state") return;

  $("rank-id").textContent = s.id;
  $("role").textContent = s.leader ? "leader (drives the step)" : "member";
  $("role").className = "v " + (s.leader ? "ok" : "");
  $("state").textContent = !s.ready ? "syncing parameters…" : s.running ? "training" : "idle";
  $("state").className = "v " + (s.ready ? (s.running ? "ok" : "warn") : "warn");
  $("backend").textContent = s.backend ?? "—";

  $("model").textContent = `${s.model} · batch ${s.batch} · ${s.bucket.toLocaleString()} params`;
  $("cohort").textContent = `${s.cohort} rank${s.cohort === 1 ? "" : "s"}`;
  // Each peer with its backend, so a cohort mixing WebGPU and CPU ranks is
  // visible rather than assumed.
  $("peers").textContent = s.peers.length
    ? s.peers.map((p) => `${p.id} (${p.backend})`).join(", ")
    : "(alone)";
  $("step").textContent = s.step.toLocaleString();
  $("loss").textContent = fmt(s.loss, 4);
  $("steps-sec").textContent = `${fmt(s.stepsPerSec, 1)} steps/s`;
  $("samples-sec").textContent = `${fmt(s.samplesPerSec, 0)} samples/s (cluster)`;
  $("comm").textContent =
    `${fmt(s.bytesPerSec / 1e6, 2)} MB/s` +
    (s.precision && s.precision !== "f32" ? ` (${s.precision} wire)` : "");
  if (s.policy) {
    const unanimous = s.policy.quorum >= s.policy.n;
    $("policy").textContent =
      `quorum ${s.policy.quorum}/${s.policy.n}` +
      `${unanimous ? " (unanimous)" : ""} · beat ${s.policy.beat}ms`;
  }
  if (s.fault) note(s.fault, "err");

  if (Number.isFinite(s.loss)) {
    history.push(s.loss);
    if (history.length > MAX_POINTS) history.shift();
    drawLoss();
  }
};

// ── loss curve ─────────────────────────────────────────────────────────────
function drawLoss() {
  const c = $("curve");
  const ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  if (history.length < 2) return;

  const lo = Math.min(...history), hi = Math.max(...history);
  const span = hi - lo || 1;
  const x = (i) => (i / (history.length - 1)) * (w - 2) + 1;
  const y = (v) => h - 1 - ((v - lo) / span) * (h - 2);

  ctx.strokeStyle = "#0a7";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  history.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
  ctx.stroke();

  ctx.fillStyle = "#666";
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillText(hi.toFixed(3), 4, 12);
  ctx.fillText(lo.toFixed(3), 4, h - 4);
}

/// Draw `text` as a QR into the canvas, sized to whatever the data needs.
///
/// Type 0 lets the encoder pick the smallest version that fits, and level `M`
/// survives a phone camera at an angle better than `L`. Drawn at an integer
/// module size with a 4-module quiet zone — a QR without its margin, or with
/// fractional module edges, is the usual reason a scan fails.
function renderQr(text, canvasId = "qr", box = 180) {
  const qr = qrcode(0, "M");
  qr.addData(text);
  qr.make();

  const modules = qr.getModuleCount();
  const quiet = 4;
  const canvas = $(canvasId);
  const scale = Math.max(2, Math.floor(box / (modules + quiet * 2)));
  const size = (modules + quiet * 2) * scale;
  canvas.width = size;
  canvas.height = size;

  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = "#000";
  for (let row = 0; row < modules; row++) {
    for (let col = 0; col < modules; col++) {
      if (qr.isDark(row, col)) {
        ctx.fillRect((col + quiet) * scale, (row + quiet) * scale, scale, scale);
      }
    }
  }
  if (canvasId === "qr") $("qr-row").hidden = false;
}

/// Report what ICE actually negotiated, per peer.
///
/// "Could not connect to peer" has several causes that need different answers —
/// no STUN (only `host` candidates), a peer that is gone, or symmetric NAT on
/// both sides (needs TURN) — and they are indistinguishable from the message
/// alone. The candidate types say which it is.
async function renderIce(mesh) {
  const conns = Object.entries(mesh.connections());
  if (!conns.length) {
    $("ice").textContent = "no peer connections yet";
    return;
  }
  const rows = [];
  for (const [peerId, pc] of conns) {
    const gathered = new Map();
    let remoteCandidates = 0;
    let path = "?";
    try {
      const stats = await pc.getStats();
      const byId = new Map();
      stats.forEach((s) => byId.set(s.id, s));
      stats.forEach((s) => {
        if (s.type === "local-candidate") {
          gathered.set(s.candidateType, (gathered.get(s.candidateType) ?? 0) + 1);
        }
        if (s.type === "remote-candidate") remoteCandidates += 1;
        if (s.type === "candidate-pair" && (s.selected || s.nominated)) {
          const local = byId.get(s.localCandidateId)?.candidateType ?? "?";
          const remote = byId.get(s.remoteCandidateId)?.candidateType ?? "?";
          path = `${local}→${remote}`;
        }
      });
    } catch {
      /* stats unavailable */
    }
    const kinds = [...gathered].map(([k, n]) => `${k}×${n}`).join(" ") || "none";
    rows.push(
      `${peerId.slice(0, 8)} ${pc.connectionState} · ${path} · gathered ${kinds}` +
        ` · ${remoteCandidates} remote`,
    );
  }
  $("ice").textContent = rows.join(" | ");
}

// ── transport ──────────────────────────────────────────────────────────────
// `seed` is shared by every rank so a joining one starts from the same
// parameters the cohort did; the leader still overrides it with a snapshot.
const SEED = 42;
const room = params.get("room");

/// `?model=` / `?backend=` / `?batch=` / `?lr=` preselect the controls, so a
/// harness (or a shared link) can pin a configuration instead of depending on
/// whatever the selects happen to show.
function applyUrlOverrides() {
  const set = (id, key) => {
    const v = params.get(key);
    if (v === null) return;
    const el = $(id);
    el.value = v;
    // A `<select>` silently drops a value that matches no option, leaving it
    // empty — which then reads as 0 and the rank clamps it to 1. Measuring a
    // "batch 256" run that was really batch 1 is worse than refusing.
    if (el.value !== v) {
      note(`${key}=${v} is not a valid option; using ${el.value || "the default"}`);
    }
  };
  set("model-select", "model");
  set("backend-select", "backend");
  set("batch", "batch");
  // Mirror a URL-provided batch onto the slider without rounding the value
  // itself: an explicit `?batch=100` stays 100, the slider just shows the
  // nearest notch.
  {
    const n = Number($("batch").value);
    if (Number.isFinite(n) && n > 0) {
      $("batch-exp").value = String(Math.round(Math.log2(n)));
      $("batch-shown").textContent = n.toLocaleString();
    }
  }
  set("precision", "precision");
  set("lr", "lr");
  $("lr-shown").textContent = $("lr").value;
}

let started = false;
let meshPeers = 0;
let icePreflight = null;
// Peers already reported as unreachable, so a retried handshake does not
// overwrite the Notes line every few seconds.
const peerFailures = new Set();

async function begin() {
  started = true;
  let transport = "local";

  if (room) {
    // Preflight before any peer exists: this is what tells "STUN is blocked"
    // apart from "that peer is gone" apart from "you need TURN".
    const turn = turnServers(params.get("turn"));
    iceProbe({
      turn,
      ...(params.get("ice")
        ? { iceServers: [{ urls: params.get("ice").split(",").filter(Boolean) }] }
        : {}),
    })
      .then((probe) => {
        icePreflight = iceVerdict(probe); // names the browser itself
        // Its own row: the per-peer line refreshes every few seconds and would
        // otherwise overwrite the one datum that does not change.
        $("ice-preflight").textContent = icePreflight.text;
        $("ice-preflight").className = "v " + (icePreflight.ok ? "ok" : "warn");
      })
      .catch(() => {});

    $("room").textContent = room;
    $("join-url").textContent = location.href;
    try {
      renderQr(location.href);
    } catch (e) {
      note(`QR unavailable (${e?.message ?? e}) — use the join URL above`);
    }
    // The host is wired up *before* joining: `joinMesh` registers its receive
    // handler before it resolves, so a peer's first message can arrive while we
    // are still awaiting. Outbound traffic sent before the mesh is up is
    // dropped, which the protocol's heartbeat and rejoin already tolerate.
    let meshSend = null;
    const host = proxyHost(worker, (buf) => meshSend?.(buf), {
      // A dropped message is tolerated by the protocol but should still be
      // visible — a cohort quietly running on stale rounds looks like a slow
      // link rather than a full queue.
      onDrop: (text) => note(`mesh queue: ${text}`),
    });
    try {
      const mesh = await joinMesh({
        room,
        moduleUrl: params.get("p2p") ?? DEFAULT_P2P_MODULE,
        relayUrls: params.get("relays")?.split(",").filter(Boolean),
        // `?ice=stun:host:3478,stun:other:3478` replaces the default STUN set;
        // `?turn=turn:host:3478|user|cred;…` adds relays for the case where no
        // direct path exists. TURN carries every byte, so it is never implicit.
        ...(params.get("ice")
          ? { iceServers: [{ urls: params.get("ice").split(",").filter(Boolean) }] }
          : {}),
        turn,
        onData: (buf) => host.deliver(buf),
        onPeers: (n) => {
          meshPeers = n;
          $("transport").textContent = `WebRTC mesh · room ${room} · ${n} peer${n === 1 ? "" : "s"}`;
        },
        onError: (text) => note(`mesh: ${text}`, "err"),
        onPeerError: (peerId, why) => {
          // One unreachable peer is routine; say so without implying the room
          // failed, and do not repeat it for the same peer.
          const short = String(peerId).slice(0, 8);
          if (peerFailures.has(short)) return;
          peerFailures.add(short);
          const live = meshPeers;
          // Say which of the three cases this is, using the preflight rather
          // than suggesting TURN at everyone unconditionally.
          let tail;
          if (live > 0) {
            tail = `continuing with ${live} connected peer${live === 1 ? "" : "s"}`;
          } else if (icePreflight && !icePreflight.ok) {
            tail = `and ICE preflight already failed here — ${icePreflight.text}`;
          } else {
            tail =
              "no peer has connected yet. Either that peer is gone or suspended" +
              " (a closed tab stays advertised on the relay for a while), or both" +
              " ends are behind symmetric NAT / client isolation — the ICE line" +
              " below says which.";
          }
          note(`peer ${short} unreachable (${why}) — ${tail}`);
        },
      });
      meshSend = mesh.send;
      transport = "mesh";
      // Keep the ICE line current: it is the only way to tell "no STUN" from
      // "peer is gone" from "needs TURN".
      renderIce(mesh);
      setInterval(() => renderIce(mesh), 3000);
      // Debug handle: lets a harness (tools/browser_cluster_smoke.mjs) read
      // WebRTC stats off the live connections instead of trusting the label.
      window.__rlxCluster = { room, mesh };
      $("transport").textContent = `WebRTC mesh · room ${room} · 0 peers`;
      $("transport").className = "v ok";
    } catch (e) {
      // A blocked CDN or a signalling failure must not take the demo down: the
      // same-browser cluster still works, and the reason is visible.
      const why = String(e?.message ?? e);
      const hint = /TURN|could not connect to peer/i.test(why)
        ? " — signalling worked but no direct path exists between these networks;" +
          " pass ?turn=turn:host:3478|user|cred (it relays every byte) or put both" +
          " devices on one network"
        : " — this browser's tabs only";
      note(`mesh unavailable (${why})${hint}`, "err");
      $("transport").textContent = "BroadcastChannel (this browser only)";
      $("transport").className = "v warn";
    }
  } else {
    $("transport").textContent = "BroadcastChannel (this browser only)";
    $("transport").className = "v warn";
  }

  worker.postMessage({
    t: "start",
    model: $("model-select").value,
    backend: $("backend-select").value,
    batch: Number($("batch").value),
    lr: Number($("lr").value),
    seed: SEED,
    transport,
    // `?precision=f16` halves every round's payload for ~5e-4 relative error.
    precision: $("precision").value,
  });
  $("start").disabled = true;
  $("stop").disabled = false;
}

// ── controls ───────────────────────────────────────────────────────────────
/// Start, or re-start with whatever the controls now say.
///
/// The page autostarts, so by the time anyone touches the selects a rank
/// already exists — and it is built once, holding compiled-graph and wasm
/// state. Picking `webgpu` after load therefore did nothing at all, with the
/// backend line still reading `cpu` and no hint why. Reloading into the chosen
/// configuration is honest and keeps one code path for "how a rank is built".
$("start").onclick = () => {
  if (!started) {
    begin();
    return;
  }
  const next = new URL(location.href);
  next.searchParams.set("model", $("model-select").value);
  next.searchParams.set("backend", $("backend-select").value);
  next.searchParams.set("batch", $("batch").value);
  next.searchParams.set("precision", $("precision").value);
  next.searchParams.set("lr", $("lr").value);
  if (room) next.searchParams.set("room", room);
  location.href = next.href;
};

$("stop").onclick = () => {
  worker.postMessage({ t: "stop" });
  $("start").disabled = false;
  $("stop").disabled = true;
};

// Batch doubles per notch: 2, 4, 8, … 65536. WebGPU happily trains at the top
// of that range; CPU gets very slow past a few thousand.
$("batch-exp").oninput = () => {
  const n = 2 ** Number($("batch-exp").value);
  $("batch").value = String(n);
  $("batch-shown").textContent = n.toLocaleString();
};

$("lr").oninput = () => {
  $("lr-shown").textContent = $("lr").value;
  worker.postMessage({ t: "lr", lr: Number($("lr").value) });
};

$("spawn").onclick = () => window.open(location.href, "_blank");

// Switching to a mesh changes the transport the Worker was built with, so it
// reloads into a room rather than hot-swapping a bus mid-run.
$("open-room").onclick = () => {
  const next = new URL(location.href);
  next.searchParams.set("room", room ?? roomCode());
  location.href = next.href;
};

// The colophon is a clone of the masthead rather than a second copy of the
// markup: two hand-maintained copies of the same links drift, and this one is
// built from the element that is already on the page.
{
  const masthead = document.querySelector(".masthead");
  const clone = masthead.cloneNode(true);
  // Canvas contents are not cloned, and ids must stay unique — both sets are
  // drawn below.
  clone.querySelectorAll("canvas").forEach((c) => {
    c.id += "-foot";
  });
  clone.classList.add("colophon");
  $("colophon").replaceWith(clone);
}

// QRs are generated locally so they work with no network at all — unlike the
// avatar, which is a remote image by nature.
const LINKS = [
  ["qr-repo", "https://github.com/MIT-RLX/rlx"],
  ["qr-models", "https://github.com/MIT-RLX/rlx-models"],
  ["qr-paper", "https://arxiv.org/abs/2609.37916"],
];
for (const [id, url] of LINKS) {
  for (const target of [id, `${id}-foot`]) {
    try {
      renderQr(url, target, 104);
    } catch (e) {
      note(`QR for ${target} unavailable (${e?.message ?? e})`);
    }
  }
}

// Colour the examples. Done from `textContent`, so the source of truth stays
// the markup and highlighting cannot change what the code says.
highlightAll();

// Autostart: a cluster demo where every new tab needs a click is a worse demo.
applyUrlOverrides();
begin();
