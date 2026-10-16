// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! F64 graphs on every backend, against CPU's native f64. Ill-conditioned
//! data (large cancelling terms) makes an f32 computation visibly wrong, so
//! the error separates real f64 / double-single from f32 promotion.

use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const N: usize = 4096;
const M: usize = 16;
const K: usize = 512;
const P: usize = 8;

fn bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f64s(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

/// `(a + b, a * b, A @ B, sum(a))` in F64.
fn graph() -> Graph {
    let f = DType::F64;
    let mut g = Graph::new("f64_probe");
    let a = g.input("a", Shape::new(&[N], f));
    let b = g.input("b", Shape::new(&[N], f));
    let ma = g.input("ma", Shape::new(&[M, K], f));
    let mb = g.input("mb", Shape::new(&[K, P], f));
    let add = g.binary(BinaryOp::Add, a, b, Shape::new(&[N], f));
    let mul = g.binary(BinaryOp::Mul, a, b, Shape::new(&[N], f));
    let mm = g.matmul(ma, mb, Shape::new(&[M, P], f));
    let sum = g.reduce(a, ReduceOp::Sum, vec![0], false, Shape::new(&[1], f));
    g.set_outputs(vec![add, mul, mm, sum]);
    g
}

struct Data {
    a: Vec<f64>,
    b: Vec<f64>,
    ma: Vec<f64>,
    mb: Vec<f64>,
}

fn data() -> Data {
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    // Cancelling large terms + small detail: the sum is dominated by the
    // detail, which f32 (24 bits) loses entirely next to 1e8.
    let a: Vec<f64> = (0..N)
        .map(|i| if i % 2 == 0 { 1e8 } else { -1e8 } + r() * 1e-3)
        .collect();
    let b: Vec<f64> = (0..N).map(|_| 1.0 + r() * 1e-9).collect();
    let ma: Vec<f64> = (0..M * K)
        .map(|i| if i % 2 == 0 { 1e6 } else { -1e6 } + r())
        .collect();
    let mb: Vec<f64> = (0..K * P).map(|_| 1.0 + r() * 1e-7).collect();
    Data { a, b, ma, mb }
}

fn run(device: Device, d: &Data) -> Vec<Vec<f64>> {
    let mut cg = Session::new(device).compile(graph());
    let f = DType::F64;
    let (a, b, ma, mb) = (bytes(&d.a), bytes(&d.b), bytes(&d.ma), bytes(&d.mb));
    cg.run_typed(&[("a", &a, f), ("b", &b, f), ("ma", &ma, f), ("mb", &mb, f)])
        .into_iter()
        .map(|(bytes, dt)| {
            assert_eq!(dt, DType::F64, "{device:?} returned {dt:?}");
            f64s(&bytes)
        })
        .collect()
}

/// Max error relative to the largest reference magnitude.
fn rel_err(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0f64, |m, x| m.max(x.abs())).max(f64::MIN_POSITIVE);
    got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0f64, f64::max) / scale
}

/// The same ops in f32, for scale.
fn f32_reference(d: &Data) -> Vec<Vec<f64>> {
    let a: Vec<f32> = d.a.iter().map(|&x| x as f32).collect();
    let b: Vec<f32> = d.b.iter().map(|&x| x as f32).collect();
    let add = a.iter().zip(&b).map(|(x, y)| (x + y) as f64).collect();
    let mul = a.iter().zip(&b).map(|(x, y)| (x * y) as f64).collect();
    let mut mm = vec![0f64; M * P];
    for i in 0..M {
        for j in 0..P {
            let mut acc = 0f32;
            for k in 0..K {
                acc += d.ma[i * K + k] as f32 * d.mb[k * P + j] as f32;
            }
            mm[i * P + j] = acc as f64;
        }
    }
    let sum = vec![a.iter().sum::<f32>() as f64];
    vec![add, mul, mm, sum]
}

#[test]
fn f64_ops_on_every_backend() {
    let d = data();
    let want = run(Device::Cpu, &d);
    let f32_err: Vec<f64> = f32_reference(&d)
        .iter()
        .zip(&want)
        .map(|(g, w)| rel_err(g, w))
        .collect();
    eprintln!("f32 baseline   add {:.1e}  mul {:.1e}  matmul {:.1e}  sum {:.1e}", f32_err[0], f32_err[1], f32_err[2], f32_err[3]);
    for device in [Device::Metal, Device::Mlx, Device::Gpu, Device::Vulkan] {
        if !rlx_runtime::is_available(device) {
            continue;
        }
        let got = std::panic::catch_unwind(|| run(device, &d));
        match got {
            Ok(got) => {
                let e: Vec<f64> = got.iter().zip(&want).map(|(g, w)| rel_err(g, w)).collect();
                eprintln!("{device:<12?}   add {:.1e}  mul {:.1e}  matmul {:.1e}  sum {:.1e}", e[0], e[1], e[2], e[3]);
            }
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                eprintln!("{device:<12?}   FAILED: {}", msg.lines().next().unwrap_or(""));
            }
        }
    }
}
