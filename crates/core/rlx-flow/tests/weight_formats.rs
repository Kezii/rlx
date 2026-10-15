// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The file-backed `WeightSource`s (`--features formats`): safetensors
//! dtypes and shards, and a GGUF tensor served as F32 and as packed blocks —
//! through a real flow, where both must give the same projection.
#![cfg(feature = "formats")]

use rlx_flow::formats::{GgufSource, SafetensorsSource};
use rlx_flow::{ModelFlow, WeightSource, WeightSourceExt};
use rlx_gguf::{GgmlType, GgufWriter};
use rlx_ir::{DType, Shape};
use rlx_runtime::{Device, Session};

fn scratch(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("rlx-flow-formats-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A safetensors file from `(name, dtype, shape, bytes)`.
fn write_st(path: &std::path::Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    header.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
    let h = serde_json::to_vec(&header).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend_from_slice(&data);
    std::fs::write(path, out).unwrap();
}

fn le<T: Copy, const N: usize>(v: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    v.iter().flat_map(|&x| f(x)).collect()
}

#[test]
fn safetensors_dtypes_shards_and_errors() {
    let d = scratch("st");
    let w = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    write_st(
        &d.join("model-00001-of-00002.safetensors"),
        &[
            ("model.w", "F32", vec![2, 3], le(&w, f32::to_le_bytes)),
            (
                "model.h",
                "F16",
                vec![2],
                le(
                    &[half::f16::from_f32(0.5), half::f16::from_f32(-2.0)],
                    half::f16::to_le_bytes,
                ),
            ),
        ],
    );
    write_st(
        &d.join("model-00002-of-00002.safetensors"),
        &[(
            "model.b",
            "BF16",
            vec![1],
            le(&[half::bf16::from_f32(3.0)], half::bf16::to_le_bytes),
        )],
    );
    let mut src = SafetensorsSource::open(&d).unwrap();
    assert_eq!(src.keys(), ["model.b", "model.h", "model.w"]);
    assert_eq!(src.info("model.h"), Some(("F16", &[2usize][..])));
    // Non-consuming, transposed on request.
    for _ in 0..2 {
        assert_eq!(
            src.take("model.w", true).unwrap(),
            (vec![1., 4., 2., 5., 3., 6.], vec![3, 2])
        );
    }
    assert_eq!(src.take("model.h", false).unwrap().0, vec![0.5, -2.0]);
    assert_eq!(src.take("model.b", false).unwrap().0, vec![3.0]);
    // A miss names the key the checkpoint does have.
    let err = src.take("w", false).unwrap_err().to_string();
    assert!(err.contains("closest: model.w"), "{err}");
    let err = src.take("layers.model.w", false).unwrap_err().to_string();
    assert!(err.contains("closest: model.w"), "{err}");
    // Through a prefix, keys come back stripped.
    let p = SafetensorsSource::open(&d).unwrap().with_prefix("model.");
    assert_eq!(p.keys(), ["b", "h", "w"]);

    // The same name in two shards is refused.
    write_st(
        &d.join("model-00003.safetensors"),
        &[("model.b", "F32", vec![1], le(&[1f32], f32::to_le_bytes))],
    );
    assert!(SafetensorsSource::open(&d).is_err());
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn gguf_packed_and_f32_give_the_same_projection() {
    let (k, n, b) = (64usize, 4usize, 2usize);
    let wf: Vec<f32> = (0..n * k)
        .map(|i| ((i * 37 % 29) as f32 - 14.0) * 0.05)
        .collect();
    let d = scratch("gguf");
    let path = d.join("m.gguf");
    let mut gw = GgufWriter::new();
    // GGML order: innermost (in = k) first.
    gw.add_tensor_bytes(
        "blk.0.w",
        vec![k, n],
        GgmlType::Q8_0,
        rlx_gguf::quantize(&wf, GgmlType::Q8_0).unwrap(),
    )
    .unwrap();
    gw.write_to_path(&path).unwrap();

    let mut f32_src = GgufSource::open(&path).unwrap();
    let (data, shape) = f32_src.take("blk.0.w", false).unwrap();
    assert_eq!(shape, vec![n, k], "outermost-first like safetensors");
    assert!(
        f32_src.take_packed("blk.0.w").unwrap().is_none(),
        "packing is opt-in"
    );
    let mut packed_src = GgufSource::open(&path).unwrap().packed(true);
    let p = packed_src
        .take_packed("blk.0.w")
        .unwrap()
        .expect("Q8_0 packs");
    assert_eq!((p.in_dim, p.out_dim), (k, n));
    assert_eq!(p.scheme, rlx_ir::quant::QuantScheme::GgufQ8_0);

    let x: Vec<f32> = (0..b * k)
        .map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.1)
        .collect();
    let run = |src: &mut dyn WeightSource| -> (Vec<f32>, bool) {
        // The builder asks for HF-style names; the file has GGUF ones.
        let mut src = src.renamed(|key| key.replace("layers.0.", "blk.0."));
        let built = ModelFlow::new("proj")
            .input("x", Shape::new(&[b, k], DType::F32))
            .linear("layers.0.w", true)
            .build(&mut src)
            .unwrap();
        let typed = built.typed_params.clone();
        let (g, params) = built.into_graph_parts().unwrap();
        let dequant = g
            .nodes()
            .iter()
            .any(|nd| matches!(nd.op, rlx_ir::Op::DequantMatMul { .. }));
        let mut c = Session::new(Device::Cpu).compile(g);
        for (name, v) in &params {
            c.set_param(name, v);
        }
        for (name, bytes, dt) in &typed {
            c.set_param_typed(name, bytes, *dt);
        }
        (c.run(&[("x", &x)]).remove(0), dequant)
    };
    let (y_f32, f32_dequant) = run(&mut f32_src);
    let (y_packed, packed_dequant) = run(&mut packed_src);
    assert!(
        !f32_dequant && packed_dequant,
        "packed weights must reach the graph"
    );
    // Reference: x · Wᵀ over the dequantized weights.
    for (j, (a, p)) in y_f32.iter().zip(&y_packed).enumerate() {
        let (r, c) = (j / n, j % n);
        let want: f32 = (0..k).map(|i| x[r * k + i] * data[c * k + i]).sum();
        assert!((a - want).abs() < 1e-4, "f32 [{j}] {a} vs {want}");
        assert!((p - want).abs() < 1e-3, "packed [{j}] {p} vs {want}");
    }
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn open_picks_the_format_by_path() {
    let d = scratch("open");
    write_st(
        &d.join("a.safetensors"),
        &[("x", "F32", vec![1], le(&[7f32], f32::to_le_bytes))],
    );
    let mut s = rlx_flow::formats::open(d.join("a.safetensors")).unwrap();
    assert_eq!(s.take("x", false).unwrap().0, vec![7.0]);
    assert!(rlx_flow::formats::open(d.join("a.bin")).is_err());
    let _ = std::fs::remove_dir_all(&d);
}
