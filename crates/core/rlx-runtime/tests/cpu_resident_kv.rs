// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CPU backend's resident-KV contract, driven the way a bucketed decode
//! loop drives it (`rlx-qwen3`'s `decode_step_gpu_resident`):
//!
//! 1. bind the padded `past_k` once (`bind_gpu_handle`) and register which
//!    output carries the grown cache (`register_kv_row_feed`);
//! 2. per step, run WITHOUT passing `past_k`, reading back only the "logits"
//!    (`run_read_outputs(.., Some(&[0]))`);
//! 3. read the new row (`read_output_row`) and fold it into the handle at the
//!    next free slot (`feed_kv_row`).
//!
//! Each step is checked against a host-side model of the cache, so a feed
//! that writes the wrong row, a handle the run ignores, or a stale arena all
//! show up as wrong numbers.

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

const UPPER: usize = 4; // padded cache rows (bucket upper bound)
const D: usize = 3; // row width (kv_dim)

/// `out0 = sum(past_k) + sum(k_new)` stands in for the logits;
/// `out1 = concat(past_k, k_new)` is the grown cache, new row at `UPPER`.
fn decode_graph() -> Graph {
    let mut g = Graph::new("resident_kv");
    let past = g.input("past_k", Shape::new(&[UPPER, D], DType::F32));
    let new = g.input("k_new", Shape::new(&[1, D], DType::F32));
    let grown = g.concat(vec![past, new], 0, Shape::new(&[UPPER + 1, D], DType::F32));
    let total = g.sum(grown, vec![0, 1], false);
    g.set_outputs(vec![total, grown]);
    g
}

#[test]
fn cpu_resident_kv_feeds_one_row_per_step() {
    let mut c = Session::new(Device::Cpu).compile(decode_graph());
    let caps = c.capabilities();
    assert!(caps.gpu_handles && caps.kv_resident);

    // Host model of the padded cache; starts with one real row.
    let mut model = vec![0f32; UPPER * D];
    model[..D].copy_from_slice(&[1.0, 2.0, 3.0]);
    assert!(c.bind_gpu_handle("past_k", &model));
    assert!(c.register_kv_row_feed("past_k", 1));
    assert!(c.has_gpu_handle("past_k"));

    for (step, past_len) in (1..=UPPER).enumerate() {
        let k_new: Vec<f32> = (0..D).map(|j| (10 * (step + 1) + j) as f32).collect();
        // `past_k` is NOT passed: the run must use the resident handle.
        let outs = c.run_read_outputs(&[("k_new", &k_new)], Some(&[0]));
        assert_eq!(outs.len(), 1, "only the requested output comes back");
        let want: f32 = model.iter().sum::<f32>() + k_new.iter().sum::<f32>();
        assert!(
            (outs[0][0] - want).abs() < 1e-4,
            "step {step}: logits {} != {want} — the run did not see the resident cache",
            outs[0][0]
        );

        let row = c
            .read_output_row(1, UPPER, D)
            .expect("new row of the grown cache");
        assert_eq!(
            row, k_new,
            "step {step}: read_output_row returned the wrong row"
        );
        assert!(c.feed_kv_row(UPPER, past_len, D));
        if past_len < UPPER {
            model[past_len * D..(past_len + 1) * D].copy_from_slice(&k_new);
        }
        // past_len == UPPER fills the bucket: the feed has no slot and must
        // leave the handle alone (the caller rebinds on the bucket change).
        assert_eq!(
            c.read_gpu_handle("past_k").as_deref(),
            Some(model.as_slice()),
            "step {step}: handle diverged from the host model"
        );
    }

    // Explicit inputs still override a handle of the same name.
    let zeros = vec![0f32; UPPER * D];
    let outs = c.run(&[("past_k", &zeros), ("k_new", &[1.0, 1.0, 1.0])]);
    assert!((outs[0][0] - 3.0).abs() < 1e-6);
    assert_eq!(outs.len(), 2, "plain run still returns every output");
}

#[test]
fn cpu_row_reads_stay_in_bounds() {
    let mut c = Session::new(Device::Cpu).compile(decode_graph());
    c.run(&[("past_k", &[0.0; UPPER * D]), ("k_new", &[0.0; D])]);
    assert!(
        c.read_output_row(1, UPPER + 1, D).is_none(),
        "row past the tensor"
    );
    assert!(c.read_output_row(9, 0, D).is_none(), "no such output");
    assert!(
        c.read_gpu_handle_row("past_k", 0, D).is_none(),
        "nothing bound yet"
    );
}
