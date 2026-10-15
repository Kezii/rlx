// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! How numbers are represented when a graph runs: `f32`, `f64`, `f64` as two
//! `f32` (double-single), or any binary float format from `f16` down to 4-bit
//! minifloats, emulated on top of `f32`.
//!
//! [`FloatFormat::round`] is the canonical scalar definition of "round an `f32`
//! to this format": round to nearest, ties to even, gradual underflow, ±0 and
//! NaN pass through, everything at or beyond the largest finite value
//! (infinities included) saturates. Every backend's `Op::RoundToFormat` kernel
//! must be bit-identical to it, and `rlx_tensor::lowp` delegates to it.
//!
//! Inputs and results below `f32`'s smallest normal (`~1.2e-38`) are flushed
//! to a signed zero. Apple GPUs flush `f32` subnormals in hardware even with
//! precise math, so without this the same graph would round differently per
//! backend. It only affects `bf16`'s own subnormals, which most `bf16`
//! hardware also flushes; `f16` and every narrower format keep theirs.
//!
//! Formats with 8 or fewer bits are normally used with a shared scale
//! ([`FormatScaling`]). Scales are powers of two chosen by the OCP MX rule
//! ([`FloatFormat::pow2_scale`]), so applying them is exact on every backend,
//! where dividing by an arbitrary `f32` scale is not (GPU division is not
//! always correctly rounded).

use std::fmt;
use std::str::FromStr;

use crate::quant::ScaledFormat;

/// A binary floating-point format emulated on `f32`: a sign bit, `exp_bits`
/// exponent bits with `bias`, and `man_bits` explicit mantissa bits.
///
/// `max_normal` is the largest finite magnitude. IEEE-style formats (`f16`,
/// `bf16`, `f8e5m2`, the `fXmYeZ` family) reserve the top exponent for inf/NaN;
/// the OCP MX element formats (`f4e2m1`, `f6e2m3`, `f6e3m2`, the `fNeXmY` family)
/// are all-finite. Both are expressed by `max_normal` alone.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloatFormat {
    pub exp_bits: u8,
    pub man_bits: u8,
    pub bias: i32,
    pub max_normal: f32,
}

// `max_normal` is never NaN, so equality is reflexive.
impl Eq for FloatFormat {}

impl std::hash::Hash for FloatFormat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (
            self.exp_bits,
            self.man_bits,
            self.bias,
            self.max_normal.to_bits(),
        )
            .hash(state);
    }
}

impl FloatFormat {
    /// IEEE binary16.
    pub const F16: Self = Self::ieee(5, 10, 65504.0);
    /// bfloat16.
    pub const BF16: Self = Self::ieee(8, 7, 3.389_531_4e38);

    /// An IEEE-style format (bias `2^(exp_bits-1) - 1`) with an explicit maximum.
    pub const fn ieee(exp_bits: u8, man_bits: u8, max_normal: f32) -> Self {
        Self {
            exp_bits,
            man_bits,
            bias: (1i32 << (exp_bits - 1)) - 1,
            max_normal,
        }
    }

    /// The element grid of a [`ScaledFormat`] (FP8 / FP6 / FP4 / custom minifloat).
    pub fn from_scaled(fmt: ScaledFormat) -> Self {
        let (e, m, bias) = fmt.fields();
        Self {
            exp_bits: e as u8,
            man_bits: m as u8,
            bias,
            max_normal: fmt.max_finite(),
        }
    }

    /// Total bits: sign + exponent + mantissa.
    pub const fn bits(self) -> u32 {
        1 + self.exp_bits as u32 + self.man_bits as u32
    }

    /// Smallest normal exponent.
    pub const fn emin(self) -> i32 {
        1 - self.bias
    }

    /// Exponent of `max_normal` (`floor(log2(max_normal))`).
    pub fn emax(self) -> i32 {
        f32_exponent(self.max_normal)
    }

    /// Round `x` to this format. See the module docs for the exact semantics.
    pub fn round(self, x: f32) -> f32 {
        let bits = x.to_bits();
        let abs_bits = bits & 0x7fff_ffff;
        let sign = bits & 0x8000_0000;
        // ±0 and NaN pass through; f32 subnormals flush.
        if abs_bits == 0 || abs_bits > 0x7f80_0000 {
            return x;
        }
        if abs_bits < 0x0080_0000 {
            return f32::from_bits(sign);
        }
        let ax = f32::from_bits(abs_bits);
        if ax >= self.max_normal {
            return f32::from_bits(sign | self.max_normal.to_bits());
        }
        // Below the smallest normal the step stays at the subnormal step.
        let ee = f32_exponent(ax).max(self.emin());
        let shift = self.man_bits as i32 - ee;
        // `ax * 2^shift` has at most `man_bits + 1` integer bits, so every step
        // here is exact in f64 and the result is representable in f32.
        let v = (ax as f64) * pow2_f64(shift);
        let q = (v.round_ties_even() * pow2_f64(-shift)) as f32;
        f32::from_bits(sign | flush_subnormal(q.min(self.max_normal)).to_bits())
    }

    /// Round every element of `xs` in place.
    pub fn round_slice(self, xs: &mut [f32]) {
        for x in xs {
            *x = self.round(*x);
        }
    }

    /// Shared power-of-two scale for a block whose largest magnitude is `amax`
    /// (the OCP MX rule): `2^(floor(log2(amax)) - emax)`, so the block's largest
    /// element lands in the format's top binade. A zero or non-finite `amax`
    /// gives `1.0`. The exponent is clamped to the normal `f32` range.
    pub fn pow2_scale(self, amax: f32) -> f32 {
        if amax == 0.0 || !amax.is_finite() {
            return 1.0;
        }
        let e = (f32_exponent(amax) - self.emax()).clamp(-126, 127);
        f32::from_bits(((e + 127) as u32) << 23)
    }

    /// Round with a power-of-two scale: `round(x / s) * s`.
    #[inline]
    pub fn round_scaled(self, x: f32, scale: f32) -> f32 {
        let y = self.round(x / scale) * scale;
        if y.is_nan() {
            y
        } else {
            y.signum() * flush_subnormal(y.abs())
        }
    }

    /// Round `xs` with `scaling`, treating it as rows of length `last_dim`
    /// (blocks never cross a row). This is the reference for every backend.
    pub fn round_slice_scaled(self, xs: &mut [f32], scaling: FormatScaling, last_dim: usize) {
        match scaling {
            FormatScaling::None => self.round_slice(xs),
            FormatScaling::PerTensor => {
                let amax = xs.iter().fold(0f32, |a, &x| a.max(x.abs()));
                let s = self.pow2_scale(amax);
                for x in xs {
                    *x = self.round_scaled(*x, s);
                }
            }
            FormatScaling::Block(n) => {
                let n = n as usize;
                let last_dim = last_dim.max(1);
                for row in xs.chunks_mut(last_dim) {
                    for block in row.chunks_mut(n) {
                        let amax = block.iter().fold(0f32, |a, &x| a.max(x.abs()));
                        let s = self.pow2_scale(amax);
                        for x in block {
                            *x = self.round_scaled(*x, s);
                        }
                    }
                }
            }
        }
    }

    /// Parse a format name. Accepts:
    ///
    /// - `f16` / `half`, `bf16`
    /// - `f8` / `fp8` / `e4m3` (OCP E4M3), `bf8` / `e5m2`, `f6` (OCP E2M3),
    ///   `f4` / `fp4` / `e2m1` / `nvf4`
    /// - `fNeXmY` (e.g. `f4e3m0`): an all-finite minifloat, as [`ScaledFormat`]
    ///   parses it (≤ 8 bits)
    /// - `fXmYeZ` (e.g. `f12m7e4`): an IEEE-style format with the top exponent
    ///   reserved, up to `e8m23`
    pub fn parse(spec: &str) -> Option<Self> {
        let s = spec.trim().to_ascii_lowercase();
        let named = match s.as_str() {
            "f16" | "half" | "fp16" => Some(Self::F16),
            "bf16" => Some(Self::BF16),
            "f8" | "fp8" | "e4m3" | "f8e4m3" => Some(Self::from_scaled(ScaledFormat::F8E4M3)),
            "bf8" | "e5m2" | "f8e5m2" => Some(Self::ieee(5, 2, 57344.0)),
            "f6" | "fp6" | "e2m3" => Some(Self::from_scaled(ScaledFormat::F6E2M3)),
            "e3m2" => Some(Self::from_scaled(ScaledFormat::F6E3M2)),
            "f4" | "fp4" | "e2m1" | "nvf4" => Some(Self::from_scaled(ScaledFormat::F4E2M1)),
            _ => None,
        };
        if named.is_some() {
            return named;
        }
        if let Ok(fmt) = s.parse::<ScaledFormat>() {
            return Some(Self::from_scaled(fmt));
        }
        // Generic IEEE-style `f{total}m{man}e{exp}`.
        let rest = s.strip_prefix('f')?;
        let (total, rest) = rest.split_once('m')?;
        let (man, exp) = rest.split_once('e')?;
        let (total, man, exp): (u32, u32, u32) =
            (total.parse().ok()?, man.parse().ok()?, exp.parse().ok()?);
        if exp == 0 || exp > 8 || man > 23 || 1 + man + exp != total {
            return None;
        }
        let bias = (1i32 << (exp - 1)) - 1;
        let max = (2.0 - pow2_f64(-(man as i32))) * pow2_f64(bias);
        Some(Self::ieee(
            exp as u8,
            man as u8,
            max.min(f32::MAX as f64) as f32,
        ))
    }
}

impl fmt::Display for FloatFormat {
    /// The first spelling that [`FloatFormat::parse`]s back to `self`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (n, e, m) = (self.bits(), self.exp_bits, self.man_bits);
        let candidates = [
            "f16".to_string(),
            "bf16".to_string(),
            format!("f{n}e{e}m{m}"),
            format!("f{n}e{e}m{m}fnuz"),
            format!("f{n}m{m}e{e}"),
        ];
        match candidates.iter().find(|c| Self::parse(c) == Some(*self)) {
            Some(name) => f.write_str(name),
            // Not reachable through `parse`; spell out every field.
            None => write!(
                f,
                "f{n}e{e}m{m}(bias {}, max {})",
                self.bias, self.max_normal
            ),
        }
    }
}

/// How values share a scale before rounding to a [`FloatFormat`].
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatScaling {
    /// Round the raw values (the norm for `f16` / `bf16`).
    None,
    /// One power-of-two scale for the whole tensor.
    PerTensor,
    /// One power-of-two scale per run of `n` consecutive elements along the
    /// last axis (OCP MX uses 32). A shorter final block is allowed.
    Block(u32),
}

impl FormatScaling {
    /// The default for a format: none above 8 bits, MX 32-element blocks at or below.
    pub fn default_for(fmt: FloatFormat) -> Self {
        if fmt.bits() > 8 {
            Self::None
        } else {
            Self::Block(32)
        }
    }
}

/// How numbers are represented when a graph runs.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NumberFormat {
    /// IEEE binary32 — the default; nothing is rewritten.
    #[default]
    F32,
    /// Binary64: native where the backend has it, double-single elsewhere.
    F64,
    /// Double-single: each value is an unevaluated sum of two `f32`
    /// (~48 significant bits), on every backend including CPU.
    F64x2,
    /// Values rounded to `format` (weights once, every intermediate after the op
    /// that produces it); arithmetic inside an op accumulates in `f32`.
    Float {
        format: FloatFormat,
        scaling: FormatScaling,
    },
}

impl NumberFormat {
    /// `format` with its default scaling.
    pub fn float(format: FloatFormat) -> Self {
        Self::Float {
            format,
            scaling: FormatScaling::default_for(format),
        }
    }
}

impl FromStr for NumberFormat {
    type Err = String;

    /// Parse `f32`, `f64`, `f64x2` (alias `df64`), or a [`FloatFormat`] name
    /// with an optional scaling suffix: `@none`, `@tensor`, `@b<N>` (e.g.
    /// `f4@b16`). Without a suffix the format's default scaling applies.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim().to_ascii_lowercase();
        match s.as_str() {
            "f32" | "fp32" | "float" => return Ok(Self::F32),
            "f64" | "fp64" | "double" => return Ok(Self::F64),
            "f64x2" | "df64" | "double-single" => return Ok(Self::F64x2),
            _ => {}
        }
        let (name, suffix) = match s.split_once('@') {
            Some((n, sfx)) => (n, Some(sfx)),
            None => (s.as_str(), None),
        };
        let format = FloatFormat::parse(name).ok_or_else(|| {
            format!(
                "unknown number format {s:?}: expected f32, f64, f64x2, f16, bf16, f8, f6, f4, \
                 fNeXmY (e.g. f4e3m0) or fXmYeZ (e.g. f12m7e4), optionally with @none, @tensor or @b<N>"
            )
        })?;
        let scaling = match suffix {
            None => FormatScaling::default_for(format),
            Some("none") => FormatScaling::None,
            Some("tensor") => FormatScaling::PerTensor,
            Some(b) => {
                let n: u32 = b
                    .strip_prefix('b')
                    .and_then(|n| n.parse().ok())
                    .filter(|&n| n > 0)
                    .ok_or_else(|| {
                        format!("bad scaling suffix @{b}: expected none, tensor or b<N>")
                    })?;
                FormatScaling::Block(n)
            }
        };
        Ok(Self::Float { format, scaling })
    }
}

impl fmt::Display for NumberFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F64 => write!(f, "f64"),
            Self::F64x2 => write!(f, "f64x2"),
            Self::Float { format, scaling } => {
                write!(f, "{format}")?;
                if *scaling != FormatScaling::default_for(*format) {
                    match scaling {
                        FormatScaling::None => write!(f, "@none")?,
                        FormatScaling::PerTensor => write!(f, "@tensor")?,
                        FormatScaling::Block(n) => write!(f, "@b{n}")?,
                    }
                }
                Ok(())
            }
        }
    }
}

/// `0` for a non-negative `x` below `f32`'s smallest normal, else `x`.
#[inline]
fn flush_subnormal(x: f32) -> f32 {
    if x.to_bits() < 0x0080_0000 { 0.0 } else { x }
}

/// Unbiased exponent of a non-negative finite `f32` (`-127` for subnormals).
#[inline]
fn f32_exponent(x: f32) -> i32 {
    ((x.to_bits() >> 23) & 0xff) as i32 - 127
}

#[inline]
fn pow2_f64(n: i32) -> f64 {
    f64::from_bits(((n + 1023) as u64) << 52)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_formats_parse() {
        assert_eq!(FloatFormat::parse("f16"), Some(FloatFormat::F16));
        assert_eq!(FloatFormat::parse("f4").unwrap().max_normal, 6.0);
        assert_eq!(FloatFormat::parse("f6").unwrap().max_normal, 7.5);
        assert_eq!(FloatFormat::parse("f8").unwrap().max_normal, 448.0);
        // All-finite fNeXmY vs IEEE-style fXmYeZ keep their existing meanings.
        assert_eq!(FloatFormat::parse("f4e3m0").unwrap().max_normal, 16.0);
        assert_eq!(FloatFormat::parse("f4m0e3").unwrap().max_normal, 8.0);
        assert_eq!(FloatFormat::parse("f12m7e4").unwrap().man_bits, 7);
        assert!(FloatFormat::parse("f5m1e2").is_none(), "bits must add up");
    }

    #[test]
    fn number_format_round_trips_through_text() {
        for s in [
            "f32",
            "f64",
            "f64x2",
            "f16",
            "bf16",
            "f4e2m1",
            "f4e2m1@b16",
            "f16@tensor",
            "f8e4m3@none",
        ] {
            let f: NumberFormat = s.parse().unwrap();
            assert_eq!(f.to_string(), s, "{s}");
        }
        assert_eq!("f4".parse::<NumberFormat>().unwrap().to_string(), "f4e2m1");
        assert!("f4@b0".parse::<NumberFormat>().is_err());
    }

    #[test]
    fn f16_rounding_matches_hardware_half() {
        // Exhaustive over a stride of f32 bit patterns, vs the `half` semantics
        // reproduced by hand: f32 -> nearest-even f16 -> f32.
        let f16 = FloatFormat::F16;
        let mut checked = 0;
        for bits in (0u32..=0x7f7f_ffff).step_by(997) {
            let x = f32::from_bits(bits);
            let want = f16_round_reference(x);
            assert_eq!(f16.round(x).to_bits(), want.to_bits(), "x = {x:e}");
            checked += 1;
        }
        assert!(checked > 2_000_000);
    }

    /// f32 → f16 → f32 via integer arithmetic, saturating instead of producing inf.
    fn f16_round_reference(x: f32) -> f32 {
        let ax = x.abs();
        if ax >= 65504.0 {
            return 65504f32.copysign(x);
        }
        // Scale into the f16 integer grid of the right binade, round half-even.
        let e = (ax.log2().floor() as i32).max(-14);
        let ulp = (2f64).powi(e - 10);
        let q = ((ax as f64) / ulp).round_ties_even() * ulp;
        // log2 can land on the next binade just below a power of two; the
        // f64 step above is then twice too coarse, but such inputs round to
        // the power of two either way at 10 mantissa bits.
        (q as f32).min(65504.0).copysign(x)
    }

    #[test]
    fn ties_go_to_even_and_subnormals_survive() {
        let f4 = FloatFormat::parse("f4").unwrap(); // e2m1: 0, .5, 1, 1.5, 2, 3, 4, 6
        assert_eq!(f4.round(2.5), 2.0);
        assert_eq!(f4.round(3.5), 4.0);
        assert_eq!(f4.round(0.25), 0.0);
        assert_eq!(f4.round(0.26), 0.5);
        assert_eq!(f4.round(-5.0), -4.0);
        assert_eq!(f4.round(f32::NEG_INFINITY), -6.0);
        assert!(f4.round(f32::NAN).is_nan());
        assert_eq!(f4.round(-0.0).to_bits(), (-0.0f32).to_bits());
        let bf16 = FloatFormat::BF16;
        let tiny = f32::from_bits(1); // smallest f32 subnormal
        assert_eq!(bf16.round(tiny), 0.0, "f32 subnormals flush");
        assert_eq!(bf16.round(-tiny).to_bits(), (-0.0f32).to_bits());
        // f16 subnormals are f32 normals, so they survive.
        assert_eq!(FloatFormat::F16.round(6.0e-8), 5.9604645e-8);
    }

    #[test]
    fn matches_scaled_format_quantize() {
        for fmt in ScaledFormat::NAMED
            .into_iter()
            .chain([ScaledFormat::custom(3, 0)])
        {
            let ff = FloatFormat::from_scaled(fmt);
            for i in -4000..4000 {
                let x = i as f32 * 0.0137;
                assert_eq!(ff.round(x), fmt.quantize(x), "{fmt} x={x}");
            }
        }
    }

    #[test]
    fn mx_block_scaling() {
        let f4 = FloatFormat::parse("f4").unwrap();
        // amax 0.3: floor(log2 0.3) = -2, emax(e2m1) = 2 → scale 2^-4.
        assert_eq!(f4.pow2_scale(0.3), 0.0625);
        let mut xs = [0.3f32, 0.01, -0.2, 0.0];
        f4.round_slice_scaled(&mut xs, FormatScaling::Block(32), 4);
        assert_eq!(xs, [0.25, 0.0, -0.1875, 0.0]);
    }
}
