// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Which vision models can a **browser** WebGPU rank actually train?
//!
//! Browser WebGPU refuses any schedule containing host-executed steps — it
//! cannot block to run them — but compile succeeds anyway, so the refusal only
//! appears once per `run_async`. That is why a cluster rank can join on
//! `webgpu` and silently contribute nothing.
//!
//! The lowered schedule is the same natively, so this answers the question
//! here, where a GPU exists and the test can run:
//! `CompiledGraph::host_execution_summary()` names the host steps.
//!
//! Skips when no wgpu adapter is present rather than failing — the gate runs on
//! machines with no GPU.

#![cfg(feature = "webgpu")]

use rlx_runtime::{Device, Session, device_ext};

// A source include, not a link against the lib: this test target compiles the
// module afresh and runs its own dead-code pass over all of it, while the test
// below uses only `build_train`/`all`/`slug`. Without the allow, every item the
// test happens not to call is a hard error under `-D warnings`.
#[path = "../src/vision/mod.rs"]
#[allow(dead_code)]
mod vision;

#[test]
fn report_host_steps_per_model_on_webgpu() {
    if !device_ext::is_available(Device::WebGpu) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }

    let mut trainable = Vec::new();
    let mut hosted = Vec::new();
    for &id in vision::VisionModelId::all() {
        let graph = vision::build_train(id, 1);
        let compiled = Session::new(Device::WebGpu).compile(graph);
        match compiled.host_execution_summary() {
            None => {
                eprintln!("{:>10}: entirely on device", id.slug());
                trainable.push(id.slug());
            }
            Some(summary) => {
                eprintln!("{:>10}: {summary}", id.slug());
                hosted.push(id.slug());
            }
        }
    }

    eprintln!("\nbrowser-WebGPU trainable: {trainable:?}\nfalls back to CPU: {hosted:?}");

    // The MLP is the one a GPU rank can train; anything with a convolution
    // lowers it to a host step. If this ever flips, the cluster's backend
    // selection and its UI copy are both out of date.
    assert!(
        trainable.contains(&"mnist-mlp"),
        "mnist-mlp should run entirely on device"
    );
}
