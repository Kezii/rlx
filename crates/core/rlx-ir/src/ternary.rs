// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Ternary (BitNet b1.58) weight quantization** — the one numeric
//! definition shared by QAT (`rlx_compile::bitnet_qat`), post-training
//! conversion, and export (`rlx-fpga` `ExportQuantMode::Ternary`).
//!
//! BitNet b1.58 (Ma et al. 2024, `arXiv:2402.17764`) "W1.58A8":
//!
//! ```text
//!   weights      γ  = max(mean|W|, ε)                    (absmean, per tensor)
//!                Wq = clamp(round(W / γ), -1, 1)         ∈ {-1, 0, 1}
//!                W ≈ Wq · γ
//!   activations  s  = 127 / max(max|x|, ε)               (absmax, per token)
//!                xq = clamp(round(x · s), -128, 127)     ∈ i8
//!                x ≈ xq / s
//! ```
//!
//! `round` is half-to-even, matching `Op::Activation(Round)` and
//! `torch.round`, so the graph-side QAT subgraph and this host reference
//! produce the same codes. Keeping both sides on one definition is the
//! point: a model trained with per-tensor scales must be exported with
//! per-tensor scales.

/// Scope of the weight scale `γ`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TernaryScale {
    /// One `γ` for the whole tensor — BitNet b1.58.
    #[default]
    PerTensor,
    /// One `γ` per output channel (row of the `[out, in]` view). Free on
    /// per-channel requant hardware; must match what training used.
    PerChannel,
}

impl TernaryScale {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "tensor" | "per-tensor" | "per_tensor" | "bitnet" => Ok(Self::PerTensor),
            "channel" | "per-channel" | "per_channel" | "row" => Ok(Self::PerChannel),
            other => Err(format!(
                "unknown TernaryScale {other:?}; expected per-tensor | per-channel"
            )),
        }
    }
}

/// Ternarization options. `Default` is BitNet b1.58.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TernaryConfig {
    pub scale: TernaryScale,
    /// Lower clamp on `γ` (and on activation `max|x|`).
    pub eps: f32,
}

impl Default for TernaryConfig {
    fn default() -> Self {
        Self {
            scale: TernaryScale::PerTensor,
            eps: BITNET_EPS,
        }
    }
}

/// `ε` of the BitNet reference implementation.
pub const BITNET_EPS: f32 = 1e-5;

/// Activation code range for BitNet A8 (`[-128, 127]`, scale on 127).
pub const BITNET_ACT_QMAX: f32 = 127.0;

/// Ternary codes for a `[rows, cols]` matrix (rows = output channels).
#[derive(Debug, Clone, PartialEq)]
pub struct TernaryWeights {
    /// Row-major `{-1, 0, 1}` codes.
    pub codes: Vec<i8>,
    /// `γ` per row: length 1 for `PerTensor`, `rows` for `PerChannel`.
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl TernaryWeights {
    /// `γ` that applies to `row`.
    pub fn row_scale(&self, row: usize) -> f32 {
        if self.scales.len() == 1 {
            self.scales[0]
        } else {
            self.scales[row]
        }
    }

    /// `γ` broadcast to one entry per row — the shape per-channel requant wants.
    pub fn row_scales(&self) -> Vec<f32> {
        (0..self.rows).map(|r| self.row_scale(r)).collect()
    }

    /// `Wq · γ` back in f32, row-major `[rows, cols]`.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.codes.len());
        for r in 0..self.rows {
            let s = self.row_scale(r);
            let row = &self.codes[r * self.cols..(r + 1) * self.cols];
            out.extend(row.iter().map(|&c| f32::from(c) * s));
        }
        out
    }

    /// Fraction of zero codes — the share of MACs ternary hardware skips.
    pub fn sparsity(&self) -> f32 {
        if self.codes.is_empty() {
            return 0.0;
        }
        self.codes.iter().filter(|&&c| c == 0).count() as f32 / self.codes.len() as f32
    }
}

/// `max(mean|w|, eps)`.
pub fn absmean_scale(w: &[f32], eps: f32) -> f32 {
    if w.is_empty() {
        return eps;
    }
    let sum: f64 = w.iter().map(|&v| f64::from(v.abs())).sum();
    ((sum / w.len() as f64) as f32).max(eps)
}

/// `clamp(round(w / γ), -1, 1)`. Divides (rather than multiplying by `1/γ`)
/// so it agrees with the reference to the last ulp at the ±0.5 boundaries.
#[inline]
pub fn ternary_code(w: f32, gamma: f32) -> i8 {
    (w / gamma).round_ties_even().clamp(-1.0, 1.0) as i8
}

/// BitNet absmean ternarization of a row-major `[rows, cols]` matrix.
pub fn ternarize(w: &[f32], rows: usize, cols: usize, cfg: &TernaryConfig) -> TernaryWeights {
    assert_eq!(
        w.len(),
        rows * cols,
        "ternarize: len {} != rows*cols {}",
        w.len(),
        rows * cols
    );
    let scales = match cfg.scale {
        TernaryScale::PerTensor => vec![absmean_scale(w, cfg.eps)],
        TernaryScale::PerChannel => (0..rows)
            .map(|r| absmean_scale(&w[r * cols..(r + 1) * cols], cfg.eps))
            .collect(),
    };
    let mut codes = Vec::with_capacity(w.len());
    for r in 0..rows {
        let g = if scales.len() == 1 {
            scales[0]
        } else {
            scales[r]
        };
        codes.extend(
            w[r * cols..(r + 1) * cols]
                .iter()
                .map(|&v| ternary_code(v, g)),
        );
    }
    TernaryWeights {
        codes,
        scales,
        rows,
        cols,
    }
}

/// Fake-ternarize in place of the original values: `Wq · γ`. Host
/// counterpart of the QAT forward.
pub fn fake_ternarize(w: &[f32], rows: usize, cols: usize, cfg: &TernaryConfig) -> Vec<f32> {
    ternarize(w, rows, cols, cfg).dequantize()
}

/// BitNet A8 per-token activation quantization of a row-major
/// `[tokens, features]` matrix. Returns `(codes, scale)` with
/// `x ≈ codes / scale[token]`.
pub fn quantize_act_per_token(
    x: &[f32],
    tokens: usize,
    features: usize,
    eps: f32,
) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(x.len(), tokens * features);
    let mut codes = Vec::with_capacity(x.len());
    let mut scales = Vec::with_capacity(tokens);
    for t in 0..tokens {
        let row = &x[t * features..(t + 1) * features];
        let amax = row.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(eps);
        let s = BITNET_ACT_QMAX / amax;
        scales.push(s);
        codes.extend(
            row.iter()
                .map(|&v| (v * s).round_ties_even().clamp(-128.0, 127.0) as i8),
        );
    }
    (codes, scales)
}

/// Host counterpart of the QAT activation forward: `xq / s`.
pub fn fake_quantize_act_per_token(
    x: &[f32],
    tokens: usize,
    features: usize,
    eps: f32,
) -> Vec<f32> {
    let (codes, scales) = quantize_act_per_token(x, tokens, features, eps);
    codes
        .chunks(features.max(1))
        .zip(&scales)
        .flat_map(|(row, &s)| row.iter().map(move |&c| f32::from(c) / s))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitnet_absmean_codes() {
        // mean|w| = (0.9+0.1+0.4+1.2)/4 = 0.65
        let w = [0.9, -0.1, 0.4, -1.2];
        let t = ternarize(&w, 1, 4, &TernaryConfig::default());
        assert!((t.scales[0] - 0.65).abs() < 1e-7);
        // 0.9/0.65=1.38→1, -0.15→0, 0.62→1, -1.85→-1
        assert_eq!(t.codes, vec![1, 0, 1, -1]);
        assert_eq!(t.sparsity(), 0.25);
    }

    #[test]
    fn per_channel_has_row_scales() {
        let w = [1.0, -1.0, 0.0, 0.0, 10.0, 0.0];
        let cfg = TernaryConfig {
            scale: TernaryScale::PerChannel,
            ..TernaryConfig::default()
        };
        let t = ternarize(&w, 2, 3, &cfg);
        assert_eq!(t.scales.len(), 2);
        assert_eq!(t.codes, vec![1, -1, 0, 0, 1, 0]);
        // Per-tensor would zero the small row out entirely.
        let pt = ternarize(&w, 2, 3, &TernaryConfig::default());
        assert_eq!(&pt.codes[..3], &[0, 0, 0]);
    }

    #[test]
    fn all_zero_weights_use_eps() {
        let t = ternarize(&[0.0; 4], 2, 2, &TernaryConfig::default());
        assert_eq!(t.scales, vec![BITNET_EPS]);
        assert!(t.codes.iter().all(|&c| c == 0));
    }

    #[test]
    fn ties_round_to_even() {
        // w/γ = 0.5 exactly → 0 under half-to-even, matching Op::Round.
        assert_eq!(ternary_code(0.5, 1.0), 0);
        assert_eq!(ternary_code(-0.5, 1.0), 0);
        assert_eq!(ternary_code(1.5, 1.0), 1);
    }

    #[test]
    fn act_per_token_roundtrip() {
        let x = [0.5, -1.0, 0.25, 2.0, 0.0, -0.5];
        let (codes, scales) = quantize_act_per_token(&x, 2, 3, BITNET_EPS);
        assert_eq!(&codes[..3], &[64, -127, 32]);
        assert_eq!(&codes[3..], &[127, 0, -32]);
        assert!((scales[0] - 127.0).abs() < 1e-6);
        let back = fake_quantize_act_per_token(&x, 2, 3, BITNET_EPS);
        for (a, b) in x.iter().zip(&back) {
            assert!((a - b).abs() <= 2.0 / 127.0 * 0.5 + 1e-6, "{a} vs {b}");
        }
    }
}
