// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Drive several *independent* browser instances at the tab cluster at once and
// report what each rank actually did.
//
//   node crates/bindings/rlx-web/tools/browser_cluster_smoke.mjs [--seconds 20]
//
// Separate `--user-data-dir` profiles are separate browsers: they share no
// `BroadcastChannel`, so the only way they can form one cohort is the WebRTC
// mesh. That makes this a real test of the cross-device path without needing a
// second device.
//
// Uses the DevTools Protocol over Node's built-in WebSocket — no Playwright, no
// driver binaries. Served over http://localhost, which is a secure context, so
// `RTCPeerConnection` is available without a certificate.

import { spawn } from "node:child_process";
import { mkdtempSync, openSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const args = process.argv.slice(2);
const seconds = Number(args[args.indexOf("--seconds") + 1]) || 20;
const origin = args.includes("--origin")
  ? args[args.indexOf("--origin") + 1]
  : "http://localhost:8000";
const instances = Number(args[args.indexOf("--instances") + 1]) || 2;
// Pin the rank config so the run tests the *mesh*, not whatever backend the
// selects happen to show. `--backend auto` to let each browser choose.
const backend = args.includes("--backend") ? args[args.indexOf("--backend") + 1] : "cpu";
// Other browser *engines* to invite into the room part-way through. They are
// only launched, not driven — Safari needs Remote Automation enabled and
// Firefox needs geckodriver, neither of which is required to observe them:
// the Chrome ranks report the cohort they can see.
const invite = args.includes("--invite")
  ? args[args.indexOf("--invite") + 1].split(",").filter(Boolean)
  : [];
const model = args.includes("--model") ? args[args.indexOf("--model") + 1] : "mnist-cnn";
const batch = args.includes("--batch") ? args[args.indexOf("--batch") + 1] : "8";
// `--turn 'turn:host:3478|user|cred'` reaches the page as `?turn=`, so the ICE
// preflight reports whether that relay actually yields `relay` candidates.
const turn = args.includes("--turn") ? args[args.indexOf("--turn") + 1] : "";
const precision = args.includes("--precision") ? args[args.indexOf("--precision") + 1] : "f32";
// The standalone site serves the demo as `index.html`; in-tree it is
// `cluster.html`. Same page either way.
const page = args.includes("--page") ? args[args.indexOf("--page") + 1] : "cluster.html";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const room = Array.from({ length: 6 }, () =>
  "23456789CDFGHJKLMNPQRSTVWXZ"[Math.floor(Math.random() * 27)],
).join("");

/// Everything spawned, registered before it can fail so nothing leaks.
///
/// This matters more than it looks: an instance that failed *during* startup
/// used to be dropped on the floor, and the orphan kept holding its debugging
/// port — so the next run found a live endpoint with an empty target list and
/// failed for a reason that had nothing to do with the code under test.
const spawned = [];

function cleanup() {
  for (const b of spawned) {
    try {
      b.ws?.close();
      b.proc?.kill("SIGTERM");
      rmSync(b.profile, { recursive: true, force: true });
    } catch {
      /* best effort */
    }
  }
  spawned.length = 0;
}

process.on("SIGINT", () => {
  cleanup();
  process.exit(130);
});

/// One Chrome instance with its own profile and debugging port.
async function launch(index) {
  const port = 9222 + index;
  const profile = mkdtempSync(path.join(tmpdir(), `rlx-chrome-${index}-`));
  const url =
    `${origin}/${page}?room=${room}&backend=${backend}&model=${model}&batch=${batch}` +
    `&precision=${precision}` +
    (turn ? `&turn=${encodeURIComponent(turn)}` : "");
  // Chrome's startup diagnostics go to a file: discarding them turned a
  // launch failure into a bare "no page target".
  const log = path.join(profile, "chrome.log");
  const logFd = openSync(log, "w");
  const proc = spawn(
    CHROME,
    [
      `--user-data-dir=${profile}`,
      `--remote-debugging-port=${port}`,
      "--headless=new",
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-background-timer-throttling",
      ...(backend === "cpu" ? [] : ["--enable-unsafe-webgpu"]),
      url,
    ],
    { stdio: ["ignore", logFd, logFd] },
  );
  const handle = { index, port, profile, proc, ws: null, url };
  spawned.push(handle);

  // Wait for the debugging endpoint, then find the page target.
  let target = null;
  let lastErr = "";
  for (let i = 0; i < 150 && !target; i++) {
    await sleep(200);
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json/list`);
      const list = await res.json();
      target =
        list.find((t) => t.type === "page" && t.url.includes(page)) ??
        list.find((t) => t.type === "page");
      if (!target) lastErr = `targets: ${list.map((t) => t.type).join(",")}`;
    } catch (e) {
      lastErr = String(e?.cause?.code ?? e?.message ?? e);
    }
  }
  if (!target) console.error(`  instance ${index}: last probe → ${lastErr}`);
  if (!target) {
    let tail = "";
    try {
      tail = readFileSync(log, "utf8").split("\n").slice(-12).join("\n");
    } catch {
      /* no log */
    }
    throw new Error(`instance ${index}: no page target on ${port}\n${tail}`);
  }

  const ws = new WebSocket(target.webSocketDebuggerUrl);
  handle.ws = ws;
  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = () => reject(new Error(`instance ${index}: devtools socket failed`));
  });

  let nextId = 1;
  const pending = new Map();
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (pending.has(msg.id)) {
      pending.get(msg.id)(msg.result);
      pending.delete(msg.id);
    }
  };
  const evaluate = (expression) =>
    new Promise((resolve) => {
      const id = nextId++;
      pending.set(id, (r) => resolve(r?.result?.value));
      ws.send(JSON.stringify({ id, method: "Runtime.evaluate", params: { expression, returnByValue: true } }));
    });

  handle.evaluate = evaluate;
  // `Runtime.evaluate` returns the promise object unless told to await it.
  handle.evaluateAsync = (expression) =>
    new Promise((resolve) => {
      const rid = nextId++;
      pending.set(rid, (r) => resolve(r?.result?.value));
      ws.send(
        JSON.stringify({
          id: rid,
          method: "Runtime.evaluate",
          params: { expression, returnByValue: true, awaitPromise: true },
        }),
      );
    });
  return handle;
}

/// Read the rank's state straight off the page.
const READ = `JSON.stringify({
  rank: document.getElementById("rank-id")?.textContent,
  role: document.getElementById("role")?.textContent,
  state: document.getElementById("state")?.textContent,
  backend: document.getElementById("backend")?.textContent,
  transport: document.getElementById("transport")?.textContent,
  cohort: document.getElementById("cohort")?.textContent,
  peers: document.getElementById("peers")?.textContent,
  step: document.getElementById("step")?.textContent,
  loss: document.getElementById("loss")?.textContent,
  rate: document.getElementById("samples-sec")?.textContent,
  comm: document.getElementById("comm")?.textContent,
  policy: document.getElementById("policy")?.textContent,
  notes: document.getElementById("fault")?.textContent,
  ice: document.getElementById("ice")?.textContent,
  icePreflight: document.getElementById("ice-preflight")?.textContent,
  masthead: (() => {
    const shot = (id) => {
      const c = document.getElementById(id);
      if (!c) return id + ":missing";
      const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
      let dark = 0;
      for (let i = 0; i < d.length; i += 4) if (d[i] < 128) dark++;
      return id + ":" + c.width + "px/" + dark + "dark";
    };
    const links = [...document.querySelectorAll(".masthead a")].map((a) => a.href);
    const logo = document.querySelector(".masthead img");
    return shot("qr-repo") + " " + shot("qr-paper") +
      " | logo " + (logo?.complete && logo.naturalWidth > 0 ? "loaded" : "not loaded") +
      " | links " + links.length;
  })(),
  // Did the highlighter run, and does the code still read correctly?
  highlighted: (() => {
    const pres = [...document.querySelectorAll("pre[data-lang]")];
    if (!pres.length) return "no code blocks";
    const spans = pres.reduce((n, p) => n + p.querySelectorAll("span[class^=tok-]").length, 0);
    const raw = pres.reduce((n, p) => n + p.textContent.length, 0);
    return pres.length + " blocks, " + spans + " tokens, " + raw + " chars";
  })(),
  // Is the QR actually drawn, or an empty canvas?
  qr: (() => {
    const c = document.getElementById("qr");
    if (!c || document.getElementById("qr-row")?.hidden) return "hidden";
    const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
    let dark = 0;
    for (let i = 0; i < d.length; i += 4) if (d[i] < 128) dark++;
    return c.width + "x" + c.height + ", " + dark + " dark px";
  })(),
})`;

/// Read what the WebRTC transport actually did, straight off the live
/// connections — bytes, round-trip time, and whether the selected candidate
/// pair is direct (`host`/`srflx`) or relayed (`relay`). The transport label on
/// the page says "WebRTC mesh"; this checks it.
const STATS = `(async () => {
  const mesh = window.__rlxCluster?.mesh;
  if (!mesh) return JSON.stringify({ error: "no mesh handle (not in a room?)" });
  const out = [];
  for (const [peerId, pc] of Object.entries(mesh.connections())) {
    const row = { peer: peerId.slice(0, 8), state: pc.connectionState, ice: pc.iceConnectionState };
    try {
      const stats = await pc.getStats();
      const byId = new Map();
      stats.forEach((s) => byId.set(s.id, s));
      stats.forEach((s) => {
        if (s.type === "candidate-pair" && (s.selected || s.nominated)) {
          row.rttMs = s.currentRoundTripTime === undefined ? null : s.currentRoundTripTime * 1000;
          row.sentKB = Math.round((s.bytesSent ?? 0) / 1024);
          row.recvKB = Math.round((s.bytesReceived ?? 0) / 1024);
          const local = byId.get(s.localCandidateId);
          const remote = byId.get(s.remoteCandidateId);
          row.path = \`\${local?.candidateType ?? "?"}→\${remote?.candidateType ?? "?"}\`;
          row.protocol = local?.protocol ?? "?";
        }
        if (s.type === "local-candidate") {
          row.gathered = row.gathered ?? {};
          row.gathered[s.candidateType] = (row.gathered[s.candidateType] ?? 0) + 1;
        }
        if (s.type === "data-channel") {
          row.channel = s.state;
          row.msgsSent = s.messagesSent;
          row.msgsRecv = s.messagesReceived;
          row.dcSentKB = Math.round((s.bytesSent ?? 0) / 1024);
          row.dcRecvKB = Math.round((s.bytesReceived ?? 0) / 1024);
        }
      });
    } catch (e) {
      row.error = String(e);
    }
    out.push(row);
  }
  return JSON.stringify(out);
})()`;

const browsers = [];
try {
  console.log(
    `room ${room} · ${instances} independent Chrome profiles · ${origin}` +
      ` · model ${model} · backend ${backend} · batch ${batch} · ${precision}`,
  );
  for (let i = 0; i < instances; i++) {
    browsers.push(await launch(i));
    console.log(`  instance ${i} up on :${9222 + i}`);
    await sleep(1500); // stagger so one leads and the others join a live run
  }

  if (invite.length) {
    const url = `${origin}/${page}?room=${room}&backend=${backend}&model=${model}`;
    for (const app of invite) {
      console.log(`  inviting ${app}`);
      spawn("open", ["-a", app, url], { stdio: "ignore" });
      await sleep(2000);
    }
  }

  for (let t = 5; t <= seconds; t += 5) {
    await sleep(5000);
    console.log(`\n── t+${t}s ──`);
    for (const b of browsers) {
      const raw = await b.evaluate(READ);
      const s = JSON.parse(raw ?? "{}");
      console.log(
        `  [${b.index}] ${s.rank ?? "?"} ${s.role ?? ""} | ${s.backend ?? "?"} | ` +
          `cohort ${s.cohort ?? "?"} | step ${s.step ?? "?"} | loss ${s.loss ?? "?"} | ` +
          `${s.rate ?? "?"} | ${s.comm ?? "?"}`,
      );
      console.log(`       transport: ${s.transport ?? "?"}`);
      console.log(`       peers: ${s.peers ?? "?"} | ${s.policy ?? ""}`);
      if (s.masthead) console.log(`       masthead: ${s.masthead}`);
      if (s.highlighted) console.log(`       code: ${s.highlighted}`);
      if (s.icePreflight) console.log(`       ice preflight: ${s.icePreflight}`);
      if (s.ice) console.log(`       ice per peer: ${s.ice}`);
      if (s.qr) console.log(`       qr: ${s.qr}`);
      if (s.notes && s.notes !== "—") console.log(`       notes: ${s.notes}`);
    }
  }
  // What the mesh actually carried, per peer connection.
  console.log("\n── WebRTC transport ──");
  for (const b of browsers) {
    const raw = await b.evaluateAsync(STATS);
    let rows;
    try {
      rows = JSON.parse(raw ?? "null");
    } catch {
      rows = null;
    }
    if (!rows || rows.error) {
      console.log(`  [${b.index}] ${rows?.error ?? "no stats"}`);
      continue;
    }
    for (const r of rows) {
      console.log(
        `  [${b.index}] peer ${r.peer} | ${r.state}/${r.ice} | path ${r.path ?? "?"}` +
          ` over ${r.protocol ?? "?"} | rtt ${r.rttMs?.toFixed(1) ?? "?"}ms`,
      );
      console.log(
        `       gathered: ${
          r.gathered ? Object.entries(r.gathered).map(([k, v]) => `${k}×${v}`).join(", ") : "?"
        }`,
      );
      console.log(
        `       ice: sent ${r.sentKB ?? "?"} KB, recv ${r.recvKB ?? "?"} KB` +
          ` | datachannel ${r.channel ?? "?"}: ${r.msgsSent ?? "?"} msgs out / ` +
          `${r.msgsRecv ?? "?"} in, ${r.dcSentKB ?? "?"} KB out / ${r.dcRecvKB ?? "?"} KB in`,
      );
    }
  }
} finally {
  cleanup();
}
