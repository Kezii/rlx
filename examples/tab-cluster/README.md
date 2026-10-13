---
title: RLX Tab Cluster
emoji: 🦀
colorFrom: indigo
colorTo: green
sdk: static
app_file: index.html
header: mini
pinned: false
license: apache-2.0
short_description: Data-parallel training across browser tabs, in Rust via wasm
tags:
  - rust
  - webassembly
  - webgpu
  - distributed-training
  - compiler
  - webrtc
---

# RLX tab cluster

Data-parallel training across browser tabs and devices, with no backend. Every
tab is a **rank**: it compiles an [RLX](https://github.com/MIT-RLX/rlx) backward
graph in a Web Worker, computes gradients over its own shard of the data, and
the cohort averages them each round.

- **Open another tab** — it joins over a `BroadcastChannel` and samples/s rises.
- **Open a room** — other devices join a **WebRTC mesh**; scan the QR code on a
  phone. Signalling goes to public relays, every gradient travels peer-to-peer.
- **Pick a backend** per rank: `webgpu`, `webgl` or `cpu`. A mixed cohort shows
  up in the roster.
- **Pick a wire format**: `f32`, `f16` or `f8` payloads, trading bandwidth for
  precision — f8 moves 3.1× less data.

The page ends with the same model written three ways — Rust, JavaScript and
Python — each building a graph, letting autodiff rewrite it into a backward
graph, and training it.

## Deploy

The directory is self-contained: copy it to any static host.

    # Hugging Face Space — `sdk: static` serves index.html
    git push  # with this folder at the Space root

    # GitHub Pages, Netlify, Vercel, S3 — upload as-is
    python3 -m http.server 8000     # or serve locally

`.nojekyll` is for GitHub Pages, which otherwise skips directories it treats as
private and would drop the wasm bundle. It is harmless elsewhere.

## Two things worth knowing

**Cross-device rooms need HTTPS.** `RTCPeerConnection` is only available in a
secure context, with `localhost` the sole exception — so a LAN IP over plain
http pairs tabs within one browser but never reaches a second device. Hugging
Face Spaces, Pages and Netlify all serve HTTPS already.

**Do not add COOP/COEP headers.** They would cross-origin isolate the page,
which it does not need (the wasm is single-threaded), and would break the two
cross-origin resources it does use: the GitHub avatar and the mesh module.

Built from `crates/bindings/rlx-web` in <https://github.com/MIT-RLX/rlx>, which
also carries the measurements and the failure modes behind these choices.
Licensed MIT OR Apache-2.0.
