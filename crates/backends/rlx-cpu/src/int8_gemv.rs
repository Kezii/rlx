// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Int8 decode GEMVs (`m == 1`) for the GGUF formats that otherwise dequantize
//! every block to f32 before the dot — Q5_K, Q6_K, Q8_0, Q4_0, Q4_1 — and for
//! Q4_K (feature `int8-gemv`, x86-64 AVX2).
//!
//! Same scheme as the Q4_K arm in `gguf_matmul`: quantize the activation row
//! once (Q8_K for the 256-element K-quants, a 32-element i8 block for Q8_0 /
//! Q4_0), then every weight row is an integer dot over the packed bytes. The
//! f32 path spends most of its time unpacking; this one reads each byte once
//! and multiplies it in place.
//!
//! The kernels follow llama.cpp's AVX2 ones: per-block scales and the
//! bsums/min corrections stay in vector registers. Scalar per-block scale
//! work is what limits a GEMV here — LLVM loads an f16 scale with `vpinsrw`
//! into whatever register is free, often the accumulator, which chains every
//! block on the previous block's FMA. The 32-element formats therefore convert
//! eight scales per `vcvtph2ps` and broadcast them from memory.
//!
//! Not bit-identical to the f32 path: the activation is rounded to int8, as in
//! llama.cpp. That is why it sits behind a default-off feature.

use rayon::prelude::*;
use rlx_gguf::{Q4K_BLOCK_BYTES, Q8K_BLOCK_BYTES, QK_K};
use rlx_ir::quant::QuantScheme;
use std::arch::x86_64::*;

const QK32: usize = 32;
const Q5K_BLOCK_BYTES: usize = 176;
const Q6K_BLOCK_BYTES: usize = 210;
const Q8_0_BLOCK_BYTES: usize = 34;
const Q4_0_BLOCK_BYTES: usize = 18;
const Q4_1_BLOCK_BYTES: usize = 20;

fn available() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| {
        std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("f16c")
    })
}

/// `out[n] = W[n,k] · x[k]` for `scheme` when this module covers it. `false`
/// means "not handled" (other scheme, ragged `k`, no AVX2) and the caller keeps
/// its own path.
pub(crate) fn gemv(scheme: QuantScheme, x: &[f32], w: &[u8], out: &mut [f32], k: usize) -> bool {
    if !available() || x.len() < k {
        return false;
    }
    match scheme {
        QuantScheme::GgufQ6K if k.is_multiple_of(QK_K) => {
            let y = q8k_row(&x[..k]);
            rows(w, out, k / QK_K * Q6K_BLOCK_BYTES, |row| {
                // SAFETY: `available()` checked AVX2+FMA; `row` and `y` hold
                // whole blocks (`k / QK_K` of each).
                unsafe { dot_q6k(row, &y) }
            })
        }
        QuantScheme::GgufQ4K if k.is_multiple_of(QK_K) => {
            let y = q8k_row(&x[..k]);
            rows(w, out, k / QK_K * Q4K_BLOCK_BYTES, |row| {
                // SAFETY: as above.
                unsafe { dot_q4k(row, &y) }
            })
        }
        QuantScheme::GgufQ5K if k.is_multiple_of(QK_K) => {
            let y = q8k_row(&x[..k]);
            rows(w, out, k / QK_K * Q5K_BLOCK_BYTES, |row| {
                // SAFETY: as above.
                unsafe { dot_q5k(row, &y) }
            })
        }
        QuantScheme::GgufQ8_0 if k.is_multiple_of(QK32) => {
            let y = q8_32_row(&x[..k], false);
            rows(w, out, k / QK32 * Q8_0_BLOCK_BYTES, |row| {
                // SAFETY: as above.
                unsafe { dot_q8_0(row, &y) }
            })
        }
        QuantScheme::GgufQ4_0 if k.is_multiple_of(QK32) => {
            let y = q8_32_row(&x[..k], true);
            rows(w, out, k / QK32 * Q4_0_BLOCK_BYTES, |row| {
                // SAFETY: as above.
                unsafe { dot_q4_0(row, &y) }
            })
        }
        QuantScheme::GgufQ4_1 if k.is_multiple_of(QK32) => {
            let y = q8_32_row(&x[..k], true);
            rows(w, out, k / QK32 * Q4_1_BLOCK_BYTES, |row| {
                // SAFETY: as above.
                unsafe { dot_q4_1(row, &y) }
            })
        }
        _ => false,
    }
}

/// Row-parallel driver: `dot(row_bytes)` per output. `false` when `w` is short.
fn rows(w: &[u8], out: &mut [f32], row_bytes: usize, dot: impl Fn(&[u8]) -> f32 + Sync) -> bool {
    if w.len() < out.len() * row_bytes {
        return false;
    }
    out.par_iter_mut()
        .enumerate()
        .for_each(|(j, o)| *o = dot(&w[j * row_bytes..(j + 1) * row_bytes]));
    true
}

/// The activation for the K-quants: `rlx_gguf::quantize_q8_k_row`, the same
/// bytes, vectorized — it runs on one thread ahead of every GEMV.
fn q8k_row(x: &[f32]) -> Vec<u8> {
    let mut y = vec![0u8; x.len() / QK_K * Q8K_BLOCK_BYTES];
    for (yb, xs) in y
        .as_chunks_mut::<Q8K_BLOCK_BYTES>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<QK_K>().0)
    {
        // SAFETY: `gemv` checked AVX2 (`available`).
        let amax = unsafe { amax32(xs) };
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        yb[..4].copy_from_slice(&d.to_le_bytes());
        for c in 0..QK_K / QK32 {
            // SAFETY: as above.
            let (q, s0, s1) = unsafe { quant32(&xs[QK32 * c..QK32 * (c + 1)], id, -128) };
            // SAFETY: 32 bytes inside the block's `qs`.
            unsafe { _mm256_storeu_si256(yb[4 + QK32 * c..].as_mut_ptr().cast(), q) };
            let o = 4 + QK_K + 4 * c;
            yb[o..o + 2].copy_from_slice(&(s0 as i16).to_le_bytes());
            yb[o + 2..o + 4].copy_from_slice(&(s1 as i16).to_le_bytes());
        }
    }
    y
}

/// `max |x|` over a multiple of 8 floats.
#[target_feature(enable = "avx2")]
fn amax32(x: &[f32]) -> f32 {
    let abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let mut m = _mm256_setzero_ps();
    for c in x.as_chunks::<8>().0 {
        // SAFETY: `c` is 8 floats.
        m = _mm256_max_ps(
            m,
            _mm256_and_ps(unsafe { _mm256_loadu_ps(c.as_ptr()) }, abs),
        );
    }
    let m = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
    let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    _mm_cvtss_f32(_mm_max_ss(m, _mm_movehdup_ps(m)))
}

/// 32 floats times `id`, rounded half away from zero as `rlx_gguf` does (add
/// ±0.5, truncate) and clamped to `lo..=127`: the i8 bytes in order and the
/// sums of each 16.
#[target_feature(enable = "avx2")]
fn quant32(x: &[f32], id: f32, lo: i32) -> (__m256i, i32, i32) {
    let vid = _mm256_set1_ps(id);
    let sign = _mm256_set1_ps(-0.0);
    let half = _mm256_set1_ps(0.5);
    let (vlo, vhi) = (_mm256_set1_epi32(lo), _mm256_set1_epi32(127));
    let mut v = [_mm256_setzero_si256(); 4];
    for (i, vi) in v.iter_mut().enumerate() {
        // SAFETY: `x` holds 32 floats.
        let p = _mm256_mul_ps(unsafe { _mm256_loadu_ps(x[8 * i..].as_ptr()) }, vid);
        let r = _mm256_add_ps(p, _mm256_or_ps(half, _mm256_and_ps(p, sign)));
        *vi = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvttps_epi32(r), vlo), vhi);
    }
    // `packs` interleaves the 128-bit halves; the permute puts the bytes back
    // in order.
    let p = _mm256_packs_epi16(
        _mm256_packs_epi32(v[0], v[1]),
        _mm256_packs_epi32(v[2], v[3]),
    );
    let q = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    (
        q,
        hsum_i32(_mm256_add_epi32(v[0], v[1])),
        hsum_i32(_mm256_add_epi32(v[2], v[3])),
    )
}

#[target_feature(enable = "avx2")]
fn hsum_i32(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    _mm_cvtsi128_si32(_mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01)))
}

/// The activation for the 32-element formats: symmetric i8 per 32 values
/// with an f32 scale (kept f32 rather than Q8_0's f16 so the activation loses
/// no more than the int8 rounding), clamped to ±127 so `_mm256_sign_epi8`
/// never negates -128. `d` and `sum` (`d · Σq` per block, for the Q4_0 / Q4_1
/// offset terms) are zero-padded to a multiple of 8 blocks.
///
/// `paired` lays `q` out for [`blocks32_u4`]: each pair of blocks `2p, 2p+1`
/// as `[2p][..16], [2p+1][..16], [2p][16..], [2p+1][16..]`, the order of the
/// two blocks' low and high nibbles side by side; an unpaired last block stays
/// as it is.
struct Q8x32 {
    d: Vec<f32>,
    sum: Vec<f32>,
    q: Vec<i8>,
}

fn q8_32_row(x: &[f32], paired: bool) -> Q8x32 {
    let nb = x.len() / QK32;
    let mut d = vec![0f32; nb.next_multiple_of(8)];
    let mut sum = vec![0f32; nb.next_multiple_of(8)];
    let mut q = vec![0i8; x.len()];
    for b in 0..nb {
        let xs = &x[b * QK32..(b + 1) * QK32];
        // SAFETY: `gemv` checked AVX2 (`available`).
        let amax = unsafe { amax32(xs) };
        if amax == 0.0 {
            continue;
        }
        d[b] = amax / 127.0;
        // SAFETY: as above.
        let (v, s0, s1) = unsafe { quant32(xs, 127.0 / amax, -127) };
        sum[b] = d[b] * (s0 + s1) as f32;
        let mut qb = [0i8; QK32];
        // SAFETY: `qb` is 32 bytes.
        unsafe { _mm256_storeu_si256(qb.as_mut_ptr().cast(), v) };
        if paired && (b ^ 1) < nb {
            let at = QK32 * (b & !1) + 16 * (b & 1);
            q[at..at + 16].copy_from_slice(&qb[..16]);
            q[at + 32..at + 48].copy_from_slice(&qb[16..]);
        } else {
            q[QK32 * b..QK32 * (b + 1)].copy_from_slice(&qb);
        }
    }
    Q8x32 { d, sum, q }
}

#[inline]
fn f16_at(b: &[u8], i: usize) -> f32 {
    half::f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

#[target_feature(enable = "avx2")]
fn hsum_f32x4(v: __m128) -> f32 {
    let s = _mm_add_ps(v, _mm_movehl_ps(v, v));
    _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
}

#[target_feature(enable = "avx2")]
fn hsum_f32(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
}

/// `((qh >> s) & mask) << 4` per byte: the high bits of a 5/6-bit value moved
/// above its nibble. The 16-bit shifts bleed bits across byte boundaries, but
/// the mask keeps only bits that came from the same byte. (A helper rather
/// than a closure: closures do not inherit `#[target_feature]`.)
#[target_feature(enable = "avx2")]
#[inline]
fn high_bits(qh: __m256i, s: i32, mask: __m256i) -> __m256i {
    let v = _mm256_srl_epi16(qh, _mm_cvtsi32_si128(s));
    _mm256_slli_epi16(_mm256_and_si256(v, mask), 4)
}

/// `pshufb` masks: lane pair `2i, 2i+1` of an i16 vector broadcast across
/// all 16 lanes (llama.cpp `get_scale_shuffle_k4`).
const SHUF_K4: [[u8; 32]; 8] = {
    let mut m = [[0u8; 32]; 8];
    let mut i = 0;
    while i < 8 {
        let mut l = 0;
        while l < 32 {
            m[i][l] = (2 * i + l % 2) as u8;
            l += 1;
        }
        i += 1;
    }
    m
};

/// `pshufb` masks: byte `2i` into the low 8 lanes, `2i+1` into the high 8
/// (llama.cpp `get_scale_shuffle`), for 16-element scale groups.
const SHUF_Q6: [[u8; 16]; 8] = {
    let mut m = [[0u8; 16]; 8];
    let mut i = 0;
    while i < 8 {
        let mut l = 0;
        while l < 16 {
            m[i][l] = (2 * i + l / 8) as u8;
            l += 1;
        }
        i += 1;
    }
    m
};

#[inline]
fn q8k_d(yb: &[u8; Q8K_BLOCK_BYTES]) -> f32 {
    f32::from_le_bytes([yb[0], yb[1], yb[2], yb[3]])
}

/// Q6_K × Q8_K, one row. Q6_K block: `ql[128]` low nibbles, `qh[64]` high
/// 2-bit pairs, `scales[16]` i8, `d` f16. Values are `q - 32` with `q` in
/// 0..64, so the dot is `Σ sc·q·y − 32·Σ sc·bsum` and `q` stays unsigned for
/// `maddubs` (63·127·2 fits i16).
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_q6k(row: &[u8], y: &[u8]) -> f32 {
    let m3 = _mm256_set1_epi8(0x03);
    let m15 = _mm256_set1_epi8(0x0F);
    let mut acc = _mm256_setzero_ps();
    for (w, yb) in row
        .as_chunks::<Q6K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(y.as_chunks::<Q8K_BLOCK_BYTES>().0.iter())
    {
        let q8 = &yb[4..4 + QK_K];
        // SAFETY: every load reads 16/32 bytes inside the 210-byte block, the
        // 292-byte activation block, or a 16/32-byte mask.
        unsafe {
            let bsums = _mm256_loadu_si256(yb[4 + QK_K..].as_ptr().cast());
            let scales = _mm_loadu_si128(w[192..].as_ptr().cast());
            let corr = _mm256_slli_epi32(_mm256_madd_epi16(bsums, _mm256_cvtepi8_epi16(scales)), 5);
            let mut sumi = _mm256_setzero_si256();
            for h in 0..2 {
                let qla = _mm256_loadu_si256(w[64 * h..].as_ptr().cast());
                let qlb = _mm256_loadu_si256(w[64 * h + 32..].as_ptr().cast());
                let qh = _mm256_loadu_si256(w[128 + 32 * h..].as_ptr().cast());
                let q = [
                    _mm256_or_si256(
                        _mm256_and_si256(qla, m15),
                        _mm256_slli_epi16(_mm256_and_si256(qh, m3), 4),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(qlb, m15),
                        _mm256_slli_epi16(_mm256_and_si256(qh, _mm256_set1_epi8(0x0C)), 2),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(qla, 4), m15),
                        _mm256_and_si256(qh, _mm256_set1_epi8(0x30)),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(qlb, 4), m15),
                        _mm256_srli_epi16(_mm256_and_si256(qh, _mm256_set1_epi8(0xC0u8 as i8)), 2),
                    ),
                ];
                for (c, q) in q.into_iter().enumerate() {
                    let yv = _mm256_loadu_si256(q8[128 * h + 32 * c..].as_ptr().cast());
                    let sc = _mm_shuffle_epi8(
                        scales,
                        _mm_loadu_si128(SHUF_Q6[4 * h + c].as_ptr().cast()),
                    );
                    let p =
                        _mm256_madd_epi16(_mm256_cvtepi8_epi16(sc), _mm256_maddubs_epi16(q, yv));
                    sumi = _mm256_add_epi32(sumi, p);
                }
            }
            let d = _mm256_set1_ps(f16_at(w, 208) * q8k_d(yb));
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(_mm256_sub_epi32(sumi, corr)), acc);
        }
    }
    hsum_f32(acc)
}

/// Q4_K/Q5_K 6-bit scales and mins, unpacked to `[sc0..sc8, m0..m8]` bytes
/// (llama.cpp's `utmp` shuffle).
#[inline]
fn scales_mins_k4(q: &[u8]) -> [u32; 4] {
    const M1: u32 = 0x3f3f_3f3f;
    const M2: u32 = 0x0f0f_0f0f;
    const M3: u32 = 0x0303_0303;
    let u = |i: usize| u32::from_le_bytes([q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]]);
    let (u0, u1, u2) = (u(0), u(1), u(2));
    [
        u0 & M1,
        (u2 & M2) | (((u0 >> 6) & M3) << 4),
        u1 & M1,
        ((u2 >> 4) & M2) | (((u1 >> 6) & M3) << 4),
    ]
}

/// Q4_K × Q8_K, one row: [`dot_k4`] without the 5th bit.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_q4k(row: &[u8], y: &[u8]) -> f32 {
    dot_k4::<Q4K_BLOCK_BYTES, false>(row, y)
}

/// Q5_K × Q8_K, one row: [`dot_k4`] with the 5th bit.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_q5k(row: &[u8], y: &[u8]) -> f32 {
    dot_k4::<Q5K_BLOCK_BYTES, true>(row, y)
}

/// Q4_K / Q5_K × Q8_K, one row. Block: `d`, `dmin` f16, `scales[12]`, then
/// for Q5_K `qh[32]` (bit `2p`/`2p+1` = 5th bit of pair `p`'s low/high
/// group), then `qs[128]`. Values are `d·sc·q − dmin·m`, so the min term is
/// `dmin · Σ m·bsum` over the eight 32-groups.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
fn dot_k4<const BB: usize, const Q5: bool>(row: &[u8], y: &[u8]) -> f32 {
    let m4 = _mm256_set1_epi8(0x0F);
    let one = _mm256_set1_epi8(0x01);
    let mut acc = _mm256_setzero_ps();
    let mut acc_m = _mm_setzero_ps();
    for (w, yb) in row
        .as_chunks::<BB>()
        .0
        .iter()
        .zip(y.as_chunks::<Q8K_BLOCK_BYTES>().0.iter())
    {
        let q8 = &yb[4..4 + QK_K];
        let qs0 = if Q5 { 48 } else { 16 };
        let u = scales_mins_k4(&w[4..16]);
        let yd = q8k_d(yb);
        // SAFETY: every load reads 32 bytes inside the block, the 292-byte
        // activation block, or a 32-byte mask.
        unsafe {
            let sm = _mm256_cvtepu8_epi16(_mm_set_epi32(
                u[3] as i32,
                u[2] as i32,
                u[1] as i32,
                u[0] as i32,
            ));
            let bsums = _mm256_loadu_si256(yb[4 + QK_K..].as_ptr().cast());
            let q8s = _mm_hadd_epi16(
                _mm256_castsi256_si128(bsums),
                _mm256_extracti128_si256(bsums, 1),
            );
            let mins = _mm_madd_epi16(_mm256_extracti128_si256(sm, 1), q8s);
            acc_m = _mm_fmadd_ps(
                _mm_set1_ps(-f16_at(w, 2) * yd),
                _mm_cvtepi32_ps(mins),
                acc_m,
            );
            let sc128 = _mm256_castsi256_si128(sm);
            let scales = _mm256_set_m128i(sc128, sc128);
            let qh = if Q5 {
                _mm256_loadu_si256(w[16..].as_ptr().cast())
            } else {
                _mm256_setzero_si256()
            };
            let mut sumi = _mm256_setzero_si256();
            for p in 0..4 {
                let qs = _mm256_loadu_si256(w[qs0 + 32 * p..].as_ptr().cast());
                let mut lo = _mm256_and_si256(qs, m4);
                let mut hi = _mm256_and_si256(_mm256_srli_epi16(qs, 4), m4);
                if Q5 {
                    lo = _mm256_or_si256(lo, high_bits(qh, 2 * p as i32, one));
                    hi = _mm256_or_si256(hi, high_bits(qh, 2 * p as i32 + 1, one));
                }
                let s0 =
                    _mm256_shuffle_epi8(scales, _mm256_loadu_si256(SHUF_K4[2 * p].as_ptr().cast()));
                let s1 = _mm256_shuffle_epi8(
                    scales,
                    _mm256_loadu_si256(SHUF_K4[2 * p + 1].as_ptr().cast()),
                );
                let ylo = _mm256_loadu_si256(q8[64 * p..].as_ptr().cast());
                let yhi = _mm256_loadu_si256(q8[64 * p + 32..].as_ptr().cast());
                let a = _mm256_madd_epi16(s0, _mm256_maddubs_epi16(lo, ylo));
                let b = _mm256_madd_epi16(s1, _mm256_maddubs_epi16(hi, yhi));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(a, b));
            }
            let d = _mm256_set1_ps(f16_at(w, 0) * yd);
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi), acc);
        }
    }
    hsum_f32(acc) + hsum_f32x4(acc_m)
}

/// Signed i8 × i8 over 32 lanes → 8 × i32, via `|w| · sign(y, w)`.
#[target_feature(enable = "avx2")]
#[inline]
fn dot_i8x32(w: __m256i, y: __m256i) -> __m256i {
    let p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(y, w));
    _mm256_madd_epi16(p, _mm256_set1_epi16(1))
}

/// Eight f16 values at byte `off` of eight consecutive blocks (fewer at a
/// row's end; the rest are zero), as f32.
#[target_feature(enable = "avx2,f16c")]
#[inline]
fn f16x8<const BB: usize>(blocks: &[[u8; BB]], off: usize) -> __m256 {
    let mut h = [0u16; 8];
    for (o, b) in h.iter_mut().zip(blocks) {
        *o = u16::from_le_bytes([b[off], b[off + 1]]);
    }
    // SAFETY: `h` is 16 bytes.
    _mm256_cvtph_ps(unsafe { _mm_loadu_si128(h.as_ptr().cast()) })
}

/// Q8_0's loop: `Σ_b d_b·dy_b·dot(b)` with the per-block scales `d_b` at
/// byte 0, converted eight blocks at a time, and two accumulators so
/// consecutive blocks do not wait on one FMA chain.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
fn blocks32<const BB: usize>(
    row: &[u8],
    y: &Q8x32,
    dot: impl Fn(&[u8; BB], &[i8]) -> __m256i,
) -> f32 {
    let mut acc = [_mm256_setzero_ps(); 2];
    let mut sc = [0f32; 8];
    for (g, grp) in row.as_chunks::<BB>().0.chunks(8).enumerate() {
        // SAFETY: `y.d` is padded to whole groups of 8; the store fills `sc`,
        // and broadcasts read inside it.
        unsafe {
            let yd = _mm256_loadu_ps(y.d[8 * g..].as_ptr());
            _mm256_storeu_ps(sc.as_mut_ptr(), _mm256_mul_ps(f16x8(grp, 0), yd));
            for (i, w) in grp.iter().enumerate() {
                let b = 8 * g + i;
                let p = _mm256_cvtepi32_ps(dot(w, &y.q[QK32 * b..QK32 * (b + 1)]));
                acc[i % 2] = _mm256_fmadd_ps(p, _mm256_broadcast_ss(&sc[i]), acc[i % 2]);
            }
        }
    }
    hsum_f32(_mm256_add_ps(acc[0], acc[1]))
}

/// Q8_0 × i8/32, one row. Q8_0 block: `d` f16, `qs[32]` i8.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_q8_0(row: &[u8], y: &Q8x32) -> f32 {
    blocks32::<Q8_0_BLOCK_BYTES>(row, y, |w, yq| {
        // SAFETY: 32-byte loads inside the block / the block's activations.
        unsafe {
            dot_i8x32(
                _mm256_loadu_si256(w[2..].as_ptr().cast()),
                _mm256_loadu_si256(yq.as_ptr().cast()),
            )
        }
    })
}

/// Q4_0 × i8/32, one row. Q4_0 block: `d` f16, `qs[16]`, values `q - 8`:
/// the dot runs on the unsigned `q` and subtracts `8·d·Σy` per block.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_q4_0(row: &[u8], y: &Q8x32) -> f32 {
    let (dot, off) = blocks32_u4::<Q4_0_BLOCK_BYTES>(row, 2, y, 0);
    dot - 8.0 * off
}

/// Q4_1 × i8/32, one row. Q4_1 block: `d`, `m` f16, `qs[16]`, values
/// `d·q + m`: `Σ d·dy·(q·y) + Σ m·dy·Σy`.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_q4_1(row: &[u8], y: &Q8x32) -> f32 {
    let (dot, min) = blocks32_u4::<Q4_1_BLOCK_BYTES>(row, 4, y, 2);
    dot + min
}

/// The 4-bit formats' loop: [`blocks32`] for blocks whose `qs[16]` (at byte
/// `qs` of the block) hold unsigned nibbles, two blocks per step. Both blocks'
/// nibble bytes share one register, so one `maddubs` pair covers 64 values
/// against the `paired` activation layout, and one scale vector (block `2p`
/// in the low four lanes, `2p+1` in the high) applies both scales. The second
/// result is `Σ_b e_b·sum_b` for the f16 `e_b` at byte `extra`.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
fn blocks32_u4<const BB: usize>(row: &[u8], qs: usize, y: &Q8x32, extra: usize) -> (f32, f32) {
    let m4 = _mm256_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);
    let two = _mm256_set1_epi32(2);
    // Two accumulators, swapped every pair, kept in registers (indexing an
    // array of them by the loop counter spills both to the stack).
    let (mut acc, mut acc_odd) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let mut accx = _mm256_setzero_ps();
    for (g, grp) in row.as_chunks::<BB>().0.chunks(8).enumerate() {
        // SAFETY: `y.d` / `y.sum` are padded to whole groups of 8; the nibble
        // loads read 16 bytes inside a block and the activation loads 32 bytes
        // inside `y.q`.
        unsafe {
            let d = f16x8(grp, 0);
            let sc = _mm256_mul_ps(d, _mm256_loadu_ps(y.d[8 * g..].as_ptr()));
            let e = if extra == 0 { d } else { f16x8(grp, extra) };
            accx = _mm256_fmadd_ps(e, _mm256_loadu_ps(y.sum[8 * g..].as_ptr()), accx);
            // Lanes of `sc` for the current pair: block 2p low, 2p+1 high.
            let mut lane = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
            let (pairs, rest) = grp.as_chunks::<2>();
            for (p, [w0, w1]) in pairs.iter().enumerate() {
                let yq = &y.q[QK32 * (8 * g + 2 * p)..];
                let v = _mm256_set_m128i(
                    _mm_loadu_si128(w1[qs..].as_ptr().cast()),
                    _mm_loadu_si128(w0[qs..].as_ptr().cast()),
                );
                let lo = _mm256_and_si256(v, m4);
                let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
                let v = _mm256_add_epi16(
                    _mm256_maddubs_epi16(lo, _mm256_loadu_si256(yq.as_ptr().cast())),
                    _mm256_maddubs_epi16(hi, _mm256_loadu_si256(yq[QK32..].as_ptr().cast())),
                );
                let dot = _mm256_cvtepi32_ps(_mm256_madd_epi16(v, ones));
                acc = _mm256_fmadd_ps(dot, _mm256_permutevar8x32_ps(sc, lane), acc);
                std::mem::swap(&mut acc, &mut acc_odd);
                lane = _mm256_add_epi32(lane, two);
            }
            // An unpaired last block: its low and high nibbles, against its
            // natural activation layout.
            if let [w] = rest {
                let i = 2 * pairs.len();
                let q = _mm_loadu_si128(w[qs..].as_ptr().cast());
                let v = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(q, 4), q), m4);
                let yq = &y.q[QK32 * (8 * g + i)..];
                let v = _mm256_maddubs_epi16(v, _mm256_loadu_si256(yq.as_ptr().cast()));
                let dot = _mm256_cvtepi32_ps(_mm256_madd_epi16(v, ones));
                let s = _mm256_permutevar8x32_ps(sc, _mm256_set1_epi32(i as i32));
                acc = _mm256_fmadd_ps(dot, s, acc);
            }
        }
    }
    (hsum_f32(_mm256_add_ps(acc, acc_odd)), hsum_f32(accx))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    /// Random but well-formed blocks: finite f16 scales, arbitrary payload.
    fn blocks(scheme: QuantScheme, n_blocks: usize, seed: &mut u64) -> Vec<u8> {
        let bb = scheme.gguf_block_bytes() as usize;
        let mut w: Vec<u8> = (0..n_blocks * bb).map(|_| lcg(seed) as u8).collect();
        for b in w.chunks_exact_mut(bb) {
            let d = half::f16::from_f32(0.001 + (lcg(seed) % 1000) as f32 * 1e-5).to_le_bytes();
            let at = match scheme {
                QuantScheme::GgufQ6K => vec![208],
                QuantScheme::GgufQ5K | QuantScheme::GgufQ4K | QuantScheme::GgufQ4_1 => vec![0, 2],
                _ => vec![0],
            };
            for o in at {
                b[o..o + 2].copy_from_slice(&d);
            }
        }
        w
    }

    /// f64 dot of the dequantized weights against the DEQUANTIZED activation,
    /// so the only difference from the kernel is f32 vs f64 accumulation: any
    /// wrong lane, nibble, or scale index shows up as a large error.
    fn reference(scheme: QuantScheme, w: &[u8], x: &[f32], n: usize, k: usize) -> Vec<f64> {
        let xq: Vec<f32> = match scheme {
            QuantScheme::GgufQ6K | QuantScheme::GgufQ5K | QuantScheme::GgufQ4K => {
                let y = q8k_row(x);
                y.as_chunks::<Q8K_BLOCK_BYTES>()
                    .0
                    .iter()
                    .flat_map(|b| {
                        let mut o = [0f32; QK_K];
                        rlx_gguf::dequant_q8_k_block(b, &mut o);
                        o
                    })
                    .collect()
            }
            _ => {
                let y = q8_32_row(x, false);
                y.q.iter()
                    .enumerate()
                    .map(|(i, &v)| y.d[i / QK32] * v as f32)
                    .collect()
            }
        };
        let (be, bb) = (
            scheme.gguf_block_size() as usize,
            scheme.gguf_block_bytes() as usize,
        );
        let mut wf = Vec::with_capacity(n * k);
        let mut blk = [0f32; QK_K];
        for b in w.chunks_exact(bb) {
            crate::gguf_matmul::dequant_block(scheme, b, &mut blk);
            wf.extend_from_slice(&blk[..be]);
        }
        (0..n)
            .map(|j| (0..k).map(|p| wf[j * k + p] as f64 * xq[p] as f64).sum())
            .collect()
    }

    #[test]
    fn int8_gemv_matches_dequantized_reference() {
        if !available() {
            eprintln!("skip: no AVX2+FMA");
            return;
        }
        let mut seed = 7u64;
        let n = 37;
        // 544 = 17 blocks of 32: a partial group of eight scales.
        for (scheme, k) in [
            (QuantScheme::GgufQ4K, 512),
            (QuantScheme::GgufQ6K, 512),
            (QuantScheme::GgufQ5K, 512),
            (QuantScheme::GgufQ8_0, 512),
            (QuantScheme::GgufQ8_0, 544),
            (QuantScheme::GgufQ4_0, 512),
            (QuantScheme::GgufQ4_0, 544),
            (QuantScheme::GgufQ4_1, 544),
        ] {
            let be = scheme.gguf_block_size() as usize;
            let w = blocks(scheme, n * k / be, &mut seed);
            let x: Vec<f32> = (0..k)
                .map(|_| (lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0)
                .collect();
            let mut got = vec![f32::NAN; n];
            assert!(gemv(scheme, &x, &w, &mut got, k), "{scheme:?} not handled");
            let want = reference(scheme, &w, &x, n, k);
            let scale = want.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-6);
            for (j, (g, r)) in got.iter().zip(&want).enumerate() {
                assert!(
                    ((*g as f64) - r).abs() <= 1e-4 * scale,
                    "{scheme:?} k={k} row {j}: {g} vs {r}"
                );
            }
        }
    }

    /// The vectorized Q8_K activation must be byte-for-byte the scalar one —
    /// rounding of exact halves, -0.0, and an all-zero block included.
    #[test]
    fn q8k_row_matches_rlx_gguf() {
        if !available() {
            eprintln!("skip: no AVX2+FMA");
            return;
        }
        let mut seed = 11u64;
        let mut x: Vec<f32> = (0..3 * QK_K)
            .map(|_| (lcg(&mut seed) % 20001) as f32 / 1000.0 - 10.0)
            .collect();
        // Block 1: amax 127 so `id` is 1 and x·id lands on exact halves.
        for (i, v) in x[QK_K..2 * QK_K].iter_mut().enumerate() {
            *v = (i as f32 - 128.0) + if i % 2 == 0 { 0.5 } else { -0.5 };
        }
        x[QK_K] = -127.0;
        x[QK_K + 5] = -0.0;
        // Block 2: all zero.
        x[2 * QK_K..].fill(0.0);
        let mut want = vec![0u8; 3 * Q8K_BLOCK_BYTES];
        rlx_gguf::quantize_q8_k_row(&x, &mut want);
        assert_eq!(q8k_row(&x), want);
    }

    #[test]
    fn unhandled_shapes_fall_through() {
        let mut out = [0f32; 1];
        assert!(!gemv(
            QuantScheme::GgufQ6K,
            &[0.0; 100],
            &[0; 210],
            &mut out,
            100
        ));
        assert!(!gemv(
            QuantScheme::GgufQ2K,
            &[0.0; 256],
            &[0; 84],
            &mut out,
            256
        ));
    }
}
