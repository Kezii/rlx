// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! BitNet b1.58 QAT (`rlx_compile::inject_bitnet_qat`) executed on CPU vs the
//! host definition in `rlx_ir::ternary`:
//!
//! * forward loss = loss of `int8_tok(x) @ ternary(w)` computed on the host;
//! * `dW` = the plain-matmul gradient at the quantized operands — the STE
//!   is exactly identity, independent of `γ`;
//! * both weight-scale scopes (per-tensor = BitNet, per-channel).
//!
//!   cargo test -p rlx-runtime --features cpu --test bitnet_qat

#![cfg(feature = "cpu")]

use rlx_autodiff::grad_with_loss;
use rlx_compile::{BitNetQatConfig, inject_bitnet_qat};
use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::ternary::{
    BITNET_EPS, TernaryConfig, TernaryScale, fake_quantize_act_per_token, fake_ternarize,
};
use rlx_ir::{DType, Graph, Philox4x32, Shape};
use rlx_runtime::{Device, Session};

const B: usize = 6;
const K: usize = 16;
const N: usize = 5;

fn f32s(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// `loss = mean((x @ w − target)²)`; returns `(graph, w)`.
fn linear_mse() -> (Graph, rlx_ir::NodeId) {
    let mut g = Graph::new("bitnet_linear");
    let x = g.input("x", Shape::new(&[B, K], DType::F32));
    let w = g.input("w", Shape::new(&[K, N], DType::F32));
    let t = g.input("target", Shape::new(&[B, N], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[B, N], DType::F32));
    let d = g.binary(BinaryOp::Sub, y, t, Shape::new(&[B, N], DType::F32));
    let sq = g.binary(BinaryOp::Mul, d, d, Shape::new(&[B, N], DType::F32));
    let loss = g.reduce(
        sq,
        ReduceOp::Mean,
        vec![0, 1],
        false,
        Shape::scalar(DType::F32),
    );
    g.set_outputs(vec![loss]);
    (g, w)
}

/// Host fake-ternary of a `[K, N]` RHS: output channel = column.
fn host_wq(w: &[f32], cfg: &TernaryConfig) -> Vec<f32> {
    let mut nk = vec![0f32; K * N];
    for k in 0..K {
        for n in 0..N {
            nk[n * K + k] = w[k * N + n];
        }
    }
    let q = fake_ternarize(&nk, N, K, cfg);
    let mut kn = vec![0f32; K * N];
    for k in 0..K {
        for n in 0..N {
            kn[k * N + n] = q[n * K + k];
        }
    }
    kn
}

fn check(scale: TernaryScale) {
    let mut rng = Philox4x32::new(11);
    let mut x = vec![0f32; B * K];
    let mut w = vec![0f32; K * N];
    let mut t = vec![0f32; B * N];
    rng.fill_normal(&mut x);
    rng.fill_normal(&mut w);
    rng.fill_normal(&mut t);

    let cfg = BitNetQatConfig {
        ternary: TernaryConfig {
            scale,
            ..TernaryConfig::default()
        },
        ..BitNetQatConfig::default()
    };
    let (fwd, w_id) = linear_mse();
    let (qat, rep) = inject_bitnet_qat(&fwd, &cfg);
    assert_eq!(rep.ternary_weights, vec!["w"]);
    assert_eq!(rep.quantized_activations, 1);
    let w_qat = qat.input_id("w").expect("w survives the rewrite");
    assert_eq!(
        w_qat, w_id,
        "leaf ids are stable when the leaf precedes the rewrite"
    );

    let bwd = grad_with_loss(&qat, &[w_qat]);
    let mut exe = Session::new(Device::Cpu).compile(bwd);
    let outs = exe.run_typed(&[
        ("x", &f32s(&x), DType::F32),
        ("w", &f32s(&w), DType::F32),
        ("target", &f32s(&t), DType::F32),
        ("d_output", &f32s(&[1.0]), DType::F32),
    ]);
    let loss = to_f32(&outs[0].0)[0];
    let dw = to_f32(&outs[1].0);

    // Host reference.
    let xq = fake_quantize_act_per_token(&x, B, K, BITNET_EPS);
    let wq = host_wq(&w, &cfg.ternary);
    let mut r = [0f32; B * N];
    for b in 0..B {
        for n in 0..N {
            let y: f32 = (0..K).map(|k| xq[b * K + k] * wq[k * N + n]).sum();
            r[b * N + n] = y - t[b * N + n];
        }
    }
    let ref_loss = r.iter().map(|v| v * v).sum::<f32>() / (B * N) as f32;
    let mut ref_dw = vec![0f32; K * N];
    for k in 0..K {
        for n in 0..N {
            ref_dw[k * N + n] = (0..B)
                .map(|b| xq[b * K + k] * 2.0 * r[b * N + n] / (B * N) as f32)
                .sum();
        }
    }

    let rel = |a: f32, b: f32| (a - b).abs() / b.abs().max(1e-3);
    assert!(
        rel(loss, ref_loss) < 1e-5,
        "{scale:?}: loss {loss} vs host {ref_loss}"
    );
    let max_err = dw
        .iter()
        .zip(&ref_dw)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-5, "{scale:?}: dW max err {max_err}");
}

#[test]
fn bitnet_qat_per_tensor_matches_host() {
    check(TernaryScale::PerTensor);
}

#[test]
fn bitnet_qat_per_channel_matches_host() {
    check(TernaryScale::PerChannel);
}
