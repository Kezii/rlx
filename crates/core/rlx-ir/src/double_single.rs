// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Double-single tensors** — f64 values for backends without native f64.
//!
//! A double-single (df) value is an unevaluated pair of f32, `hi + lo` with
//! `|lo| ≤ ulp(hi)/2`: ~48 significand bits against f64's 53, and f32's
//! exponent range. A df tensor of logical shape `[d0, …, dn]` is an F32 tensor
//! of shape `[d0, …, dn, 2]` (`[…, 0]` = hi, `[…, 1]` = lo), 8 bytes per
//! element like f64. `rlx_compile::lower_f64` rewrites F64 graphs into this
//! form, so data movement (reshape, transpose, narrow, concat, gather) runs on
//! existing F32 kernels with the trailing axis carried along, and only the
//! arithmetic in [`DfOp`] needs dedicated kernels.
//!
//! The arithmetic is the error-free-transform library of `rlxsl::dw`, which
//! must be compiled without fast math (reassociation zeroes the error terms).

/// Arithmetic on df tensors (`Op::DoubleSingle`). All shapes below are
/// logical; the F32 tensors carry a trailing axis of 2.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DfOp {
    /// Elementwise, numpy broadcasting: `[a, b] -> out`.
    Add,
    Sub,
    Mul,
    Div,
    /// `[.., m, k] @ [.., k, n] -> [.., m, n]` with df products and sums
    /// (leading batch axes broadcast).
    MatMul,
    /// Sum over `axes` of the single input.
    Sum { axes: Vec<usize>, keep_dim: bool },
}

impl DfOp {
    pub fn num_inputs(&self) -> usize {
        match self {
            DfOp::Sum { .. } => 1,
            _ => 2,
        }
    }
}

impl std::fmt::Display for DfOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DfOp::Add => write!(f, "df_add"),
            DfOp::Sub => write!(f, "df_sub"),
            DfOp::Mul => write!(f, "df_mul"),
            DfOp::Div => write!(f, "df_div"),
            DfOp::MatMul => write!(f, "df_matmul"),
            DfOp::Sum { axes, keep_dim } => write!(f, "df_sum(axes={axes:?},keep={keep_dim})"),
        }
    }
}

/// Split each f64 into its df pair, interleaved `(hi, lo)`. Exact: `hi` is
/// the f64 rounded to f32 and `lo` the (representable) remainder, also rounded.
pub fn split_f64(values: &[f64]) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        let hi = v as f32;
        out.push(hi);
        out.push((v - hi as f64) as f32);
    }
    out
}

/// Inverse of [`split_f64`]: `hi + lo` evaluated in f64.
pub fn join_f64(pairs: &[f32]) -> Vec<f64> {
    pairs
        .chunks_exact(2)
        .map(|p| p[0] as f64 + p[1] as f64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_join_keeps_48_bits() {
        let v = [1e8 + 1e-3, -std::f64::consts::PI, 1.0 / 3.0, 0.0, -0.0, 6.02e23];
        for (a, b) in v.iter().zip(join_f64(&split_f64(&v))) {
            assert!((a - b).abs() <= a.abs() * 2f64.powi(-47), "{a} -> {b}");
        }
    }
}
