// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Op::RoundToFormat: round to a binary float format emulated on f32, with an
// optional power-of-two scale. Bit-identical to
// `rlx_ir::FloatFormat::round_slice_scaled`. Exponent, NaN/inf and subnormal
// decisions read the f32 bits (independent of -ftz / fast-math); ldexpf and
// rintf (ties to even) are exact here. Shared by CUDA (NVRTC) and ROCm (HIP).
// One thread per element, or per block when mode == 2.

__device__ __forceinline__ int rtf_exp(float a) {
    return (int)((__float_as_uint(a) >> 23) & 0xffu) - 127;
}

__device__ __forceinline__ float rtf_flush(float y) {
    unsigned int b = __float_as_uint(y);
    return (b & 0x7fffffffu) < 0x00800000u ? __uint_as_float(b & 0x80000000u) : y;
}

__device__ __forceinline__ float rtf_round(float x, int man_bits, int emin, unsigned int max_bits) {
    unsigned int bits = __float_as_uint(x);
    unsigned int ab = bits & 0x7fffffffu;
    unsigned int sgn = bits & 0x80000000u;
    if (ab == 0u || ab > 0x7f800000u) return x;
    if (ab < 0x00800000u) return __uint_as_float(sgn);
    if (ab >= max_bits) return __uint_as_float(sgn | max_bits);
    float ax = __uint_as_float(ab);
    int e = rtf_exp(ax);
    int shift = man_bits - (e > emin ? e : emin);
    unsigned int qb = __float_as_uint(ldexpf(rintf(ldexpf(ax, shift)), -shift));
    if (qb > max_bits) qb = max_bits;
    return __uint_as_float(sgn | (qb < 0x00800000u ? 0u : qb));
}

// Bits of |x|, or 0 for NaN, so NaN never wins a maximum.
__device__ __forceinline__ unsigned int rtf_abs_bits(float x) {
    unsigned int b = __float_as_uint(x) & 0x7fffffffu;
    return b > 0x7f800000u ? 0u : b;
}

__device__ __forceinline__ int rtf_scale_exp(unsigned int amax_bits, int emax) {
    if (amax_bits == 0u || amax_bits >= 0x7f800000u) return 0;
    int e = (int)(amax_bits >> 23) - 127 - emax;
    return e < -126 ? -126 : (e > 127 ? 127 : e);
}

extern "C" __global__ void round_to_format(
    float* arena,
    unsigned int n,
    unsigned int last_dim,
    unsigned int block,
    unsigned int mode,
    int man_bits,
    int emin,
    int emax,
    unsigned int max_bits,
    unsigned int in_off,
    unsigned int amax_off,
    unsigned int out_off
) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    const float* x = arena + in_off;
    float* out = arena + out_off;
    if (mode == 2u) {
        unsigned int per_row = (last_dim + block - 1u) / block;
        unsigned int row = t / per_row;
        unsigned int start = row * last_dim + (t % per_row) * block;
        if (start >= n) return;
        unsigned int end = start + block;
        unsigned int row_end = row * last_dim + last_dim;
        if (end > row_end) end = row_end;
        unsigned int m = 0u;
        for (unsigned int i = start; i < end; i++) {
            unsigned int b = rtf_abs_bits(x[i]);
            if (b > m) m = b;
        }
        int e = rtf_scale_exp(m, emax);
        for (unsigned int i = start; i < end; i++) {
            out[i] = rtf_flush(ldexpf(rtf_round(ldexpf(x[i], -e), man_bits, emin, max_bits), e));
        }
        return;
    }
    if (t >= n) return;
    int e = mode == 1u ? rtf_scale_exp(rtf_abs_bits(arena[amax_off]), emax) : 0;
    out[t] = rtf_flush(ldexpf(rtf_round(ldexpf(x[t], -e), man_bits, emin, max_bits), e));
}
