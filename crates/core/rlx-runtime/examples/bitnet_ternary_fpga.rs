// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! BitNet b1.58 end to end: train f32 → ternary weights (PTQ vs QAT) →
//! FPGA SystemVerilog.
//!
//! 1. Train an f32 MLP classifier on CPU.
//! 2. **PTQ:** ternarize those weights with the BitNet absmean rule
//!    (`rlx_ir::ternary`) and measure the accuracy drop.
//! 3. **QAT:** fine-tune from the f32 weights with
//!    `rlx_compile::inject_bitnet_qat` (ternary weights + int8 per-token
//!    activations in the forward, identity STE in the backward).
//! 4. Run the QAT model through the integer FPGA reference
//!    (`rlx_fpga::reference`, bit-exact with the emitted RTL).
//! 5. Export SystemVerilog with `ExportQuantMode::Ternary` — 2-bit packed
//!    weights on the multiplier-free add/sub/skip MAC.
//!
//! ```sh
//! cargo run -p rlx-runtime --release --example bitnet_ternary_fpga --features cpu,fpga -- [out_dir] [hw]
//! ```
//!
//! `hw` is an `HwTarget` string; `xc7:arty-a7-35` (or `xc7:basys3`, …) adds
//! the openXC7 bitstream flow and UART host link: `./synth.sh`,
//! `./program.sh`, then `python3 run_uart.py tb_image.mem`.

use rlx_autodiff::grad_with_loss;
use rlx_compile::{BitNetQatConfig, inject_bitnet_qat};
use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::ternary::{BITNET_EPS, TernaryConfig, fake_quantize_act_per_token, fake_ternarize};
use rlx_ir::{DType, Graph, NodeId, Philox4x32, Shape};
use rlx_runtime::export::ExportSession;
use rlx_runtime::{Device, Session};

const IN: usize = 16;
const HID: usize = 64;
const CLASSES: usize = 4;
const TRAIN: usize = 512;
const TEST: usize = 512;
const F32_STEPS: usize = 300;
const QAT_STEPS: usize = 300;

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

struct Data {
    x: Vec<f32>,
    labels: Vec<usize>,
    onehot: Vec<f32>,
}

/// Gaussian blobs around random class centres.
fn make_data(rng: &mut Philox4x32, centres: &[f32], n: usize) -> Data {
    let mut noise = vec![0f32; n * IN];
    rng.fill_normal(&mut noise);
    let mut x = vec![0f32; n * IN];
    let mut labels = Vec::with_capacity(n);
    let mut onehot = vec![0f32; n * CLASSES];
    for i in 0..n {
        let c = i % CLASSES;
        labels.push(c);
        onehot[i * CLASSES + c] = 1.0;
        for d in 0..IN {
            x[i * IN + d] = centres[c * IN + d] + 1.6 * noise[i * IN + d];
        }
    }
    Data { x, labels, onehot }
}

/// `loss = mean((relu(x @ w1) @ w2 − onehot)²)`.
fn train_graph() -> (Graph, NodeId, NodeId) {
    let mut g = Graph::new("bitnet_mlp_train");
    let x = g.input("x", Shape::new(&[TRAIN, IN], DType::F32));
    let w1 = g.input("w1", Shape::new(&[IN, HID], DType::F32));
    let w2 = g.input("w2", Shape::new(&[HID, CLASSES], DType::F32));
    let t = g.input("target", Shape::new(&[TRAIN, CLASSES], DType::F32));
    let h = g.matmul(x, w1, Shape::new(&[TRAIN, HID], DType::F32));
    let h = g.activation(Activation::Relu, h, Shape::new(&[TRAIN, HID], DType::F32));
    let y = g.matmul(h, w2, Shape::new(&[TRAIN, CLASSES], DType::F32));
    let d = g.binary(
        BinaryOp::Sub,
        y,
        t,
        Shape::new(&[TRAIN, CLASSES], DType::F32),
    );
    let sq = g.binary(
        BinaryOp::Mul,
        d,
        d,
        Shape::new(&[TRAIN, CLASSES], DType::F32),
    );
    let loss = g.reduce(
        sq,
        ReduceOp::Mean,
        vec![0, 1],
        false,
        Shape::scalar(DType::F32),
    );
    g.set_outputs(vec![loss]);
    (g, w1, w2)
}

/// Inference graph for export: weights are named Params the legalizer binds.
fn inference_graph() -> Graph {
    let mut g = Graph::new("bitnet_mlp");
    let x = g.input("x", Shape::new(&[1, IN], DType::F32));
    let w1 = g.param("fc1_w", Shape::new(&[IN, HID], DType::F32));
    let w2 = g.param("fc2_w", Shape::new(&[HID, CLASSES], DType::F32));
    let h = g.matmul(x, w1, Shape::new(&[1, HID], DType::F32));
    let h = g.activation(Activation::Relu, h, Shape::new(&[1, HID], DType::F32));
    let y = g.matmul(h, w2, Shape::new(&[1, CLASSES], DType::F32));
    g.set_outputs(vec![y]);
    g
}

/// Adam on the host over `(w1, w2)`.
fn train(
    graph: Graph,
    w1_id: NodeId,
    w2_id: NodeId,
    w: &mut [Vec<f32>; 2],
    data: &Data,
    steps: usize,
    lr: f32,
) -> f32 {
    let bwd = grad_with_loss(&graph, &[w1_id, w2_id]);
    let mut exe = Session::new(Device::Cpu).compile(bwd);
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let mut m = [vec![0f32; w[0].len()], vec![0f32; w[1].len()]];
    let mut v = m.clone();
    let x = f32s(&data.x);
    let t = f32s(&data.onehot);
    let seed = f32s(&[1.0]);
    let mut loss = 0.0;
    for step in 1..=steps {
        let outs = exe.run_typed(&[
            ("x", &x, DType::F32),
            ("w1", &f32s(&w[0]), DType::F32),
            ("w2", &f32s(&w[1]), DType::F32),
            ("target", &t, DType::F32),
            ("d_output", &seed, DType::F32),
        ]);
        loss = to_f32(&outs[0].0)[0];
        for k in 0..2 {
            let g = to_f32(&outs[1 + k].0);
            let (c1, c2) = (1.0 - b1.powi(step as i32), 1.0 - b2.powi(step as i32));
            for i in 0..g.len() {
                m[k][i] = b1 * m[k][i] + (1.0 - b1) * g[i];
                v[k][i] = b2 * v[k][i] + (1.0 - b2) * g[i] * g[i];
                w[k][i] -= lr * (m[k][i] / c1) / ((v[k][i] / c2).sqrt() + eps);
            }
        }
    }
    loss
}

/// Host forward; `ternary = true` applies BitNet W1.58A8 numerics.
fn accuracy(w: &[Vec<f32>; 2], data: &Data, ternary: bool) -> f32 {
    let n = data.labels.len();
    let cfg = TernaryConfig::default();
    let (w1, w2) = if ternary {
        (
            fake_ternarize(&w[0], IN, HID, &cfg),
            fake_ternarize(&w[1], HID, CLASSES, &cfg),
        )
    } else {
        (w[0].clone(), w[1].clone())
    };
    let act = |x: &[f32], rows: usize, cols: usize| {
        if ternary {
            fake_quantize_act_per_token(x, rows, cols, BITNET_EPS)
        } else {
            x.to_vec()
        }
    };
    let x = act(&data.x, n, IN);
    let mut h = vec![0f32; n * HID];
    for i in 0..n {
        for o in 0..HID {
            h[i * HID + o] = (0..IN)
                .map(|k| x[i * IN + k] * w1[k * HID + o])
                .sum::<f32>()
                .max(0.0);
        }
    }
    let h = act(&h, n, HID);
    let mut correct = 0;
    for i in 0..n {
        let y: Vec<f32> = (0..CLASSES)
            .map(|o| (0..HID).map(|k| h[i * HID + k] * w2[k * CLASSES + o]).sum())
            .collect();
        let pred = (0..CLASSES).max_by(|&a, &b| y[a].total_cmp(&y[b])).unwrap();
        correct += usize::from(pred == data.labels[i]);
    }
    correct as f32 / n as f32
}

/// Static per-tensor activation scales from the training set (the FPGA
/// datapath requantizes with fixed scales).
fn calibrate(w: &[Vec<f32>; 2], data: &Data) -> (f32, f32, f32) {
    let cfg = TernaryConfig::default();
    let w1 = fake_ternarize(&w[0], IN, HID, &cfg);
    let w2 = fake_ternarize(&w[1], HID, CLASSES, &cfg);
    let amax = |v: f32, m: f32| m.max(v.abs());
    let (mut ax, mut ah, mut ay) = (0f32, 0f32, 0f32);
    for i in 0..data.labels.len() {
        let x = &data.x[i * IN..(i + 1) * IN];
        ax = x.iter().copied().fold(ax, |m, v| amax(v, m));
        let h: Vec<f32> = (0..HID)
            .map(|o| {
                (0..IN)
                    .map(|k| x[k] * w1[k * HID + o])
                    .sum::<f32>()
                    .max(0.0)
            })
            .collect();
        ah = h.iter().copied().fold(ah, |m, v| amax(v, m));
        for o in 0..CLASSES {
            let y: f32 = (0..HID).map(|k| h[k] * w2[k * CLASSES + o]).sum();
            ay = amax(y, ay);
        }
    }
    (ax / 127.0, ah / 127.0, ay / 127.0)
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("rlx_bitnet_fpga"));
    let hw = std::env::args()
        .nth(2)
        .map(|s| rlx_fpga::HwTarget::parse(&s).expect("hw target"))
        .unwrap_or_default();

    let mut rng = Philox4x32::new(2026);
    let mut centres = vec![0f32; CLASSES * IN];
    rng.fill_normal(&mut centres);
    let train_set = make_data(&mut rng, &centres, TRAIN);
    let test_set = make_data(&mut rng, &centres, TEST);

    let mut w = [vec![0f32; IN * HID], vec![0f32; HID * CLASSES]];
    rng.fill_normal(&mut w[0]);
    rng.fill_normal(&mut w[1]);
    w[0].iter_mut().for_each(|v| *v *= (1.0 / IN as f32).sqrt());
    w[1].iter_mut()
        .for_each(|v| *v *= (1.0 / HID as f32).sqrt());

    // 1. f32 baseline.
    let (g, w1, w2) = train_graph();
    let loss = train(g, w1, w2, &mut w, &train_set, F32_STEPS, 1e-2);
    let f32_acc = accuracy(&w, &test_set, false);
    // 2. PTQ.
    let ptq_acc = accuracy(&w, &test_set, true);
    println!(
        "f32           loss {loss:.4}  test acc {:.1}%",
        100.0 * f32_acc
    );
    println!("ternary PTQ               test acc {:.1}%", 100.0 * ptq_acc);

    // 3. QAT fine-tune from the f32 weights.
    let (g, w1, w2) = train_graph();
    let (qat, report) = inject_bitnet_qat(&g, &BitNetQatConfig::default());
    println!(
        "QAT rewrite: ternary weights {:?}, {} int8 activation taps",
        report.ternary_weights, report.quantized_activations
    );
    let loss = train(qat, w1, w2, &mut w, &train_set, QAT_STEPS, 2e-3);
    let qat_acc = accuracy(&w, &test_set, true);
    println!(
        "ternary QAT   loss {loss:.4}  test acc {:.1}%",
        100.0 * qat_acc
    );

    // 4. Integer FPGA reference (bit-exact with the RTL).
    let (x_scale, h_scale, y_scale) = calibrate(&w, &train_set);
    let mut opts = rlx_fpga::LegalizeOptions::new();
    opts.weights_f32.insert("fc1_w".into(), w[0].clone());
    opts.weights_f32.insert("fc2_w".into(), w[1].clone());
    opts.act_scales.insert("input".into(), x_scale);
    opts.act_scales.insert("fc1_w".into(), h_scale);
    opts.act_scales.insert("fc2_w".into(), y_scale);
    let fg = rlx_fpga::FpgaExportConfig::default()
        .with_quant_mode(rlx_fpga::ExportQuantMode::Ternary)
        .from_graph;
    let model = rlx_fpga::prepare_model(
        &inference_graph(),
        rlx_fpga::ExportQuantMode::Ternary,
        &opts,
        &fg,
    )
    .expect("ternary legalize");
    let mut correct = 0;
    for i in 0..TEST {
        let xq: Vec<i8> = test_set.x[i * IN..(i + 1) * IN]
            .iter()
            .map(|&v| (v / x_scale).round().clamp(-127.0, 127.0) as i8)
            .collect();
        let (pred, _) = rlx_fpga::reference::run(&model, &xq);
        correct += usize::from(pred == test_set.labels[i]);
    }
    println!(
        "FPGA int ref              test acc {:.1}%  (static int8 activations)",
        100.0 * correct as f32 / TEST as f32
    );

    // 5. SystemVerilog.
    let arts = ExportSession::fpga(&out_dir)
        .quant_mode(rlx_fpga::ExportQuantMode::Ternary)
        .hw_target(hw)
        .legalize(opts)
        .export(&inference_graph())
        .expect("fpga export");
    println!(
        "\nwrote {} files to {}",
        arts.files.len(),
        arts.out_dir.display()
    );
    if let Some(s) = arts.estimate_summary {
        println!("{s}");
    }

    // Stimulus for the emitted `tb.sv`: test sample 0 + the prediction the
    // RTL must reproduce.
    let xq: Vec<i8> = test_set.x[..IN]
        .iter()
        .map(|&v| (v / x_scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    let mem: String = xq.iter().map(|&b| format!("{:02x}\n", b as u8)).collect();
    std::fs::write(out_dir.join("tb_image.mem"), mem).expect("write tb_image.mem");
    let (pred, _) = rlx_fpga::reference::run(&model, &xq);
    println!("tb_image.mem = test sample 0 → reference pred = {pred}");
}
