// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decode safetensors / NumPy dtypes to host buffers.

use anyhow::{Result, bail};
use half::{bf16, f16};

pub fn decode_f32(raw: &[u8], dtype: safetensors::Dtype) -> Result<Vec<f32>> {
    use safetensors::Dtype as D;
    Ok(match dtype {
        D::F32 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        D::F64 => raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
            .collect(),
        D::F16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        D::BF16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        D::I64 => raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c) as f32)
            .collect(),
        D::I32 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c) as f32)
            .collect(),
        D::I16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32)
            .collect(),
        D::U8 => raw.iter().map(|&b| b as f32).collect(),
        D::I8 => raw.iter().map(|&b| b as i8 as f32).collect(),
        D::BOOL => raw.iter().map(|&b| (b != 0) as u8 as f32).collect(),
        other => bail!("cannot decode safetensors dtype {other:?} to f32"),
    })
}

#[allow(dead_code)]
pub fn decode_u8_raw(raw: &[u8]) -> Vec<u8> {
    raw.to_vec()
}
