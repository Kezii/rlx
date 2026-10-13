// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`rlx_compile::shard`] claims a cut is **independent**: each shard's piece
//! of the output depends only on its own slice of one operand plus the operands
//! it reads whole. Everything downstream — running host steps concurrently,
//! fanning an op out across nodes — is unsafe if that claim is wrong, and wrong
//! in the worst way: a shard reading the wrong slice still returns numbers.
//!
//! So each option is executed the way a worker would be fed it — given only its
//! slice — and the recombined result is compared against the unsharded op.
//!
//! How *equal* the result is depends on the axis, and the difference is not in
//! the decomposition:
//!
//! * **Batch and row splits are bit-identical.** Every output element is
//!   computed by the same reduction over the same inputs, and the kernel keeps
//!   the same shape class, so the arithmetic order is untouched.
//! * **Out-channel and column splits agree to ~5e-7.** The maths is equally
//!   exact, but the backend picks a different blocking for the narrower shape
//!   (5 output channels vs 2, 9 columns vs 2) and reassociates the accumulation.
//!
//! The tolerance below is therefore deliberately tight — f32 epsilon scale, not
//! "close enough". A real decomposition bug (a shard fed the wrong slice)
//! produces an O(1) difference and still fails loudly. The practical
//! consequence is recorded in [`rlx_compile::shard`]: a placement that must
//! stay bit-identical has to shard along batch/rows, not channels/columns.

use rlx_compile::shard::{Residency, ShardAxis, shardings};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

/// Element-wise comparison with a stated tolerance, reporting the worst
/// offender — an index bug shows up as an O(1) difference, not an epsilon one.
fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        worst <= tol,
        "{what}: worst difference {worst:e} at {at} (got {}, want {}) exceeds {tol:e}",
        got[at],
        want[at]
    );
}

fn ramp(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * 0.37 + seed).sin() * 0.5 + (i % 7) as f32 * 0.1)
        .collect()
}

fn conv_graph(n: usize, c_in: usize, c_out: usize, hw: usize) -> (Graph, Shape) {
    let f = DType::F32;
    let mut g = Graph::new("conv");
    let x = g.input("x", Shape::new(&[n, c_in, hw, hw], f));
    let w = g.param("w", Shape::new(&[c_out, c_in, 3, 3], f));
    let out = g.conv2d(x, w, [3, 3], [1, 1], [0, 0], [1, 1], 1);
    let shape = g.node(out).shape.clone();
    g.set_outputs(vec![out]);
    (g, shape)
}

fn run_conv(n: usize, c_in: usize, c_out: usize, hw: usize, x: &[f32], w: &[f32]) -> Vec<f32> {
    let (g, _) = conv_graph(n, c_in, c_out, hw);
    let mut compiled = Session::new(Device::Cpu).compile(g);
    compiled.set_param("w", w);
    compiled.run(&[("x", x)]).remove(0)
}

fn matmul_graph(m: usize, k: usize, n: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("mm");
    let a = g.input("a", Shape::new(&[m, k], f));
    let b = g.param("b", Shape::new(&[k, n], f));
    let out = g.matmul(a, b, Shape::new(&[m, n], f));
    g.set_outputs(vec![out]);
    g
}

fn run_matmul(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut compiled = Session::new(Device::Cpu).compile(matmul_graph(m, k, n));
    compiled.set_param("b", b);
    compiled.run(&[("a", a)]).remove(0)
}

/// Batch split: every shard gets its own samples and the weights whole. This is
/// the data-parallel cut, and the one a conv host step would use locally.
#[test]
fn conv_batch_shards_recombine_exactly() {
    let (n, c_in, c_out, hw) = (6usize, 3usize, 4usize, 7usize);
    let x = ramp(n * c_in * hw * hw, 0.1);
    let w = ramp(c_out * c_in * 3 * 3, 0.9);
    let reference = run_conv(n, c_in, c_out, hw, &x, &w);

    let (g, _) = conv_graph(n, c_in, c_out, hw);
    let out_id = *g.outputs.first().unwrap();
    // 4 shards of 6 samples: deliberately uneven (2,2,1,1).
    let option = shardings(&g, out_id, 4)
        .into_iter()
        .find(|o| o.axis == ShardAxis::Batch && o.count() == 4)
        .expect("batch sharding");
    assert!(option.shards.iter().any(|s| s.len != option.shards[0].len));

    let sample = c_in * hw * hw;
    let mut recombined = Vec::with_capacity(reference.len());
    for shard in &option.shards {
        let slice = &x[shard.start * sample..(shard.start + shard.len) * sample];
        recombined.extend(run_conv(shard.len, c_in, c_out, hw, slice, &w));
    }
    assert_eq!(recombined.len(), reference.len());
    assert_eq!(recombined, reference, "batch shards changed the result");
}

/// Out-channel split: every shard gets its own filters and the activations
/// whole. The output is interleaved per sample, so recombining is a gather —
/// the cost that does not show up in the byte count.
#[test]
fn conv_out_channel_shards_recombine_exactly() {
    let (n, c_in, c_out, hw) = (3usize, 2usize, 5usize, 6usize);
    let x = ramp(n * c_in * hw * hw, 0.2);
    let w = ramp(c_out * c_in * 3 * 3, 0.4);
    let reference = run_conv(n, c_in, c_out, hw, &x, &w);

    let (g, out_shape) = conv_graph(n, c_in, c_out, hw);
    let out_id = *g.outputs.first().unwrap();
    let option = shardings(&g, out_id, 3)
        .into_iter()
        .find(|o| o.axis == ShardAxis::OutChannel && o.count() == 3)
        .expect("out-channel sharding");

    let hw_out = match (out_shape.dims()[2], out_shape.dims()[3]) {
        (rlx_ir::Dim::Static(h), rlx_ir::Dim::Static(w)) => h * w,
        _ => unreachable!(),
    };
    let filter = c_in * 3 * 3;
    let mut recombined = vec![0.0f32; reference.len()];
    for shard in &option.shards {
        let w_slice = &w[shard.start * filter..(shard.start + shard.len) * filter];
        let part = run_conv(n, c_in, shard.len, hw, &x, w_slice);
        // part is [n, shard.len, h_out, w_out] → scatter into [n, c_out, …].
        for b in 0..n {
            for ch in 0..shard.len {
                let src = (b * shard.len + ch) * hw_out;
                let dst = (b * c_out + shard.start + ch) * hw_out;
                recombined[dst..dst + hw_out].copy_from_slice(&part[src..src + hw_out]);
            }
        }
    }
    // Not bit-identical: a 5-channel conv and a 2-channel conv block
    // differently inside the kernel. Epsilon-scale, so a wrong slice still fails.
    assert_close(&recombined, &reference, 5e-6, "out-channel shards");
}

/// Row split: `A`'s rows are contiguous, so a worker's slice is one copy.
#[test]
fn matmul_row_shards_recombine_exactly() {
    let (m, k, n) = (10usize, 6usize, 4usize);
    let a = ramp(m * k, 0.3);
    let b = ramp(k * n, 0.7);
    let reference = run_matmul(m, k, n, &a, &b);

    let g = matmul_graph(m, k, n);
    let out_id = *g.outputs.first().unwrap();
    let option = shardings(&g, out_id, 3)
        .into_iter()
        .find(|o| o.axis == ShardAxis::Rows && o.count() == 3)
        .expect("row sharding");

    let mut recombined = Vec::with_capacity(reference.len());
    for shard in &option.shards {
        let rows = &a[shard.start * k..(shard.start + shard.len) * k];
        recombined.extend(run_matmul(shard.len, k, n, rows, &b));
    }
    assert_eq!(recombined, reference, "row shards changed the result");
}

/// Column split: `B`'s columns are **strided**, so the worker's slice has to be
/// gathered before it is sent. Same numbers, different IO shape — worth knowing
/// before choosing this axis over a row split of equal byte count.
#[test]
fn matmul_column_shards_recombine_exactly() {
    let (m, k, n) = (4usize, 5usize, 9usize);
    let a = ramp(m * k, 0.5);
    let b = ramp(k * n, 0.15);
    let reference = run_matmul(m, k, n, &a, &b);

    let g = matmul_graph(m, k, n);
    let out_id = *g.outputs.first().unwrap();
    let option = shardings(&g, out_id, 4)
        .into_iter()
        .find(|o| o.axis == ShardAxis::Cols && o.count() == 4)
        .expect("column sharding");

    let mut recombined = vec![0.0f32; reference.len()];
    for shard in &option.shards {
        // Gather the strided column block.
        let mut b_cols = Vec::with_capacity(k * shard.len);
        for row in 0..k {
            let base = row * n + shard.start;
            b_cols.extend_from_slice(&b[base..base + shard.len]);
        }
        let part = run_matmul(m, k, shard.len, a.as_slice(), &b_cols);
        for row in 0..m {
            let dst = row * n + shard.start;
            recombined[dst..dst + shard.len]
                .copy_from_slice(&part[row * shard.len..(row + 1) * shard.len]);
        }
    }
    // Same story as the out-channel split: sgemm blocks a 9-column and a
    // 2-column product differently.
    assert_close(&recombined, &reference, 5e-6, "column shards");
}

/// The IO numbers the chooser ranks on must match what the shards actually
/// need: sliced operands cross once in total, shared ones once per worker.
#[test]
fn moved_bytes_matches_what_the_shards_are_fed() {
    let (n, c_in, c_out, hw) = (8usize, 3usize, 4usize, 5usize);
    let (g, _) = conv_graph(n, c_in, c_out, hw);
    let out_id = *g.outputs.first().unwrap();
    let batch = shardings(&g, out_id, 4)
        .into_iter()
        .find(|o| o.axis == ShardAxis::Batch && o.count() == 4)
        .unwrap();

    let x_bytes = (n * c_in * hw * hw * 4) as u64;
    let w_bytes = (c_out * c_in * 3 * 3 * 4) as u64;
    let out_bytes = batch.output_elems * 4;

    // Cold: activations once, weights to each of the four workers, output back.
    assert_eq!(
        batch.moved_bytes(Residency::default()),
        x_bytes + 4 * w_bytes + out_bytes
    );
    // Warm (data-parallel: weights already on every node): activations + output.
    assert_eq!(
        batch.moved_bytes(Residency {
            shared_resident: true
        }),
        x_bytes + out_bytes
    );
}
