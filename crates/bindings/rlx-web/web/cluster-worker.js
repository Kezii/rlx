// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// One tab's rank, in a dedicated module Worker: bring up a device, load wasm,
// build the rank, and hand it to the coordination protocol.
//
// Transport depends on reach. Same-browser tabs use a `BroadcastChannel` from
// right here. Cross-device ranks need WebRTC, and `RTCPeerConnection` does not
// exist in a Worker — so the main thread owns that transport and this Worker
// talks to it through a `proxyBus`, which transfers one encoded buffer per
// message (no copy, no parse on the UI thread). Compute never leaves here.
//
// All protocol logic lives in `cluster-protocol.js` so it can be tested in Node
// without a browser; this file is only the browser wiring.

import init, * as rlx from "./pkg/rlx_web.js";
import { createRank } from "./cluster-protocol.js";
import { channelBus, proxyBus } from "./cluster-bus.js";

// Random 32-bit id doubles as the election key (lowest wins), so electing a
// leader needs no extra round trip.
const id = crypto.getRandomValues(new Uint32Array(1))[0] >>> 0;

let node = null;

/// A frame the bus could not decode is a lost round, which shows up as a
/// stalled cohort rather than an error — so it goes to the UI.
const busError = (text) => self.postMessage({ t: "log", text: `bus: ${text}` });

/// Build the rank on the requested backend, falling back to CPU rather than
/// leaving the tab dead. WebGPU needs its adapter+device *before* the rank is
/// constructed: `ClusterWorker::new` probes availability rather than assuming
/// it, so without this it would correctly refuse to compile for WebGPU.
async function buildRank(want, model, batch, seed) {
  const tryGpu = want === "webgpu" || want === "auto";
  if (tryGpu && typeof rlx.init_webgpu === "function") {
    try {
      await rlx.init_webgpu();
    } catch (e) {
      self.postMessage({ t: "log", text: `init_webgpu failed: ${e?.message ?? e}` });
    }
  }
  try {
    return new rlx.ClusterWorker(model, batch, seed, want);
  } catch (e) {
    if (want === "cpu") throw e;
    self.postMessage({
      t: "log",
      text: `${want} unavailable (${e?.message ?? e}) — falling back to cpu`,
    });
    return new rlx.ClusterWorker(model, batch, seed, "cpu");
  }
}

/// Make a backend prove itself before a rank trusts it.
///
/// A backend can compile a graph, report success, and return **zeros**: in
/// Chrome the WebGPU path builds an invalid compute pipeline
/// (`[Invalid ComputePipeline "rlx-wgpu reduce"]`), every dispatch after it
/// fails, and `run_async` still resolves — so the rank trained happily on a
/// loss of exactly 0.0000 and poisoned the cohort's average with it. Nothing
/// errors; the page just reports `webgpu` and learns nothing.
///
/// A freshly initialised classifier must produce a finite loss near
/// `ln(num_classes)` — never 0 and never NaN. One step is enough to tell.
async function validate(rank) {
  const probe = await rank.grads(0);
  const loss = probe[0];
  if (!Number.isFinite(loss)) return `${rank.backend} returned a non-finite loss (${loss})`;
  if (Math.abs(loss) < 1e-6) {
    return `${rank.backend} returned a loss of exactly ${loss} — the backend ran but produced zeros`;
  }
  return null;
}

self.onmessage = async (ev) => {
  const m = ev.data;

  if (m.t === "start") {
    if (!node) {
      await init();
      let rank = await buildRank(m.backend ?? "auto", m.model, m.batch, m.seed);
      const bad = await validate(rank);
      if (bad) {
        self.postMessage({ t: "log", text: `${bad} — falling back to cpu` });
        rank = new rlx.ClusterWorker(m.model, m.batch, m.seed, "cpu");
      }
      // "Why is it only on CPU?" should be answerable from the page.
      if (rank.fallback) {
        self.postMessage({ t: "log", text: `${rank.backend}: ${rank.fallback}` });
      }
      node = createRank({
        id,
        // `proxyBus` chains onto this handler, so inbound bus traffic is routed
        // to the protocol and control messages still land here.
        bus:
          m.transport === "mesh"
            ? proxyBus(self, { onError: busError, precision: m.precision })
            : channelBus("rlx-cluster", { onError: busError, precision: m.precision }),
        rank,
        model: m.model,
        backend: rank.backend,
        precision: m.precision ?? "f32",
        post: (state) => self.postMessage(state),
      });
    }
    node.start(m.lr);
  } else if (m.t === "stop") {
    node?.stop();
  } else if (m.t === "lr") {
    node?.setLr(m.lr);
  }
};
