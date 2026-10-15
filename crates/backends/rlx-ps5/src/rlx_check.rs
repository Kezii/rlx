// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compiles and runs an RLX graph on the PS5's CPU through `rlx-runtime`
//! (`Device::Cpu`): `softmax(x @ w + b)`, checked against an f64 reference,
//! then times a square matmul. Results also go to `/data/rlx-ps5/rlx.txt`.

use rlx_ir::*;
use rlx_runtime::{Device, Session};
use std::fmt::Write as _;
use std::time::Instant;

/// Deterministic values in [-1, 1).
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn reference(x: &[f32], w: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0f64; m * n];
    for i in 0..m {
        let row = &mut out[i * n..(i + 1) * n];
        for (j, r) in row.iter_mut().enumerate() {
            *r = b[j] as f64
                + (0..k)
                    .map(|p| x[i * k + p] as f64 * w[p * n + j] as f64)
                    .sum::<f64>();
        }
        let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = row.iter().map(|v| (v - max).exp()).sum();
        row.iter_mut().for_each(|v| *v = (*v - max).exp() / sum);
    }
    out
}

fn main() {
    let workers = rlx_ps5::init();
    let mut log = String::new();
    let mut say = |s: String| {
        println!("{s}");
        let _ = writeln!(log, "{s}");
    };

    let (m, k, n) = (64, 256, 128);
    let mut g = Graph::new("ps5_check");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[k, n], DType::F32));
    let b = g.param("b", Shape::new(&[n], DType::F32));
    let mm = g.matmul(x, w, Shape::new(&[m, n], DType::F32));
    let z = g.binary(op::BinaryOp::Add, mm, b, Shape::new(&[m, n], DType::F32));
    let y = g.softmax(z, 1, Shape::new(&[m, n], DType::F32));
    g.set_outputs(vec![y]);

    let (xd, wd, bd) = (fill(m * k, 1), fill(k * n, 2), fill(n, 3));
    let t = Instant::now();
    let mut c = Session::new(Device::Cpu).compile(g);
    say(format!(
        "compile: {:.2} ms",
        t.elapsed().as_secs_f64() * 1e3
    ));
    c.set_param("w", &wd);
    c.set_param("b", &bd);
    let out = c.run(&[("x", &xd)]);
    let want = reference(&xd, &wd, &bd, m, k, n);
    let err = out[0]
        .iter()
        .zip(&want)
        .map(|(&a, &r)| (a as f64 - r).abs())
        .fold(0f64, f64::max);
    say(format!(
        "{} softmax(x@w+b) [{m}x{k}]x[{k}x{n}]: max |err| vs f64 = {err:.3e}",
        if err < 1e-5 { "ok  " } else { "FAIL" }
    ));

    for &s in &[256usize, 512, 1024] {
        let mut g = Graph::new("ps5_gemm");
        let a = g.input("a", Shape::new(&[s, s], DType::F32));
        let bb = g.param("b", Shape::new(&[s, s], DType::F32));
        let o = g.matmul(a, bb, Shape::new(&[s, s], DType::F32));
        g.set_outputs(vec![o]);
        let mut c = Session::new(Device::Cpu).compile(g);
        let (ad, bdat) = (fill(s * s, 4), fill(s * s, 5));
        c.set_param("b", &bdat);
        let _ = c.run(&[("a", &ad)]);
        let reps = 5;
        let t = Instant::now();
        for _ in 0..reps {
            let _ = c.run(&[("a", &ad)]);
        }
        let dt = t.elapsed().as_secs_f64() / reps as f64;
        say(format!(
            "gemm {s}^3: {:.2} ms, {:.1} GFLOP/s",
            dt * 1e3,
            2.0 * (s as f64).powi(3) / dt / 1e9
        ));
    }
    say(format!(
        "workers: {workers} (affinity {:#x}, available_parallelism {})",
        rlx_ps5::affinity_mask().unwrap_or(0),
        std::thread::available_parallelism().map_or(0, |n| n.get())
    ));
    let _ = std::fs::create_dir_all("/data/rlx-ps5");
    let _ = std::fs::write("/data/rlx-ps5/rlx.txt", &log);
    rlx_ps5::exit(i32::from(log.contains("FAIL")));
}
