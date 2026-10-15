// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Op::RoundToFormat: round to a binary float format emulated on f32, with an
// optional power-of-two scale. Bit-identical to
// `rlx_ir::FloatFormat::round_slice_scaled`. The exponent, NaN/inf and
// subnormal tests read the f32 bits, so relaxed float math cannot change them;
// `ldexp` by an integer and `round` (ties to even) are exact.

struct Params {
    n: u32,
    last_dim: u32,
    block: u32,
    mode: u32,        // 0 none, 1 per-tensor (amax at amax_off), 2 per block
    man_bits: i32,
    emin: i32,
    emax: i32,
    max_normal: f32,
    in_off: u32,
    amax_off: u32,
    out_off: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

fn f32_exp(a: f32) -> i32 { return i32((bitcast<u32>(a) >> 23u) & 0xffu) - 127; }

fn flush(y: f32) -> f32 {
    let b = bitcast<u32>(y);
    if ((b & 0x7fffffffu) < 0x00800000u) { return bitcast<f32>(b & 0x80000000u); }
    return y;
}

fn round_fmt(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let ab = bits & 0x7fffffffu;
    let sign = bits & 0x80000000u;
    if (ab == 0u || ab > 0x7f800000u) { return x; }
    if (ab < 0x00800000u) { return bitcast<f32>(sign); }
    let ax = bitcast<f32>(ab);
    if (ax >= params.max_normal) { return bitcast<f32>(sign | bitcast<u32>(params.max_normal)); }
    let ee = max(f32_exp(ax), params.emin);
    let shift = params.man_bits - ee;
    let q = min(ldexp(round(ldexp(ax, shift)), -shift), params.max_normal);
    let qb = bitcast<u32>(q);
    if (qb < 0x00800000u) { return bitcast<f32>(sign); }
    return bitcast<f32>(sign | qb);
}

fn scale_exp(amax: f32) -> i32 {
    if (amax == 0.0 || bitcast<u32>(amax) >= 0x7f800000u) { return 0; }
    return clamp(f32_exp(amax) - params.emax, -126, 127);
}

fn round_scaled(x: f32, e: i32) -> f32 {
    return flush(ldexp(round_fmt(ldexp(x, -e)), e));
}

@compute @workgroup_size(64)
fn round_to_format(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let t = gid.x + gid.y * ngs.x * 64u;
    if (params.mode == 2u) {
        let per_row = (params.last_dim + params.block - 1u) / params.block;
        let row = t / per_row;
        let start = row * params.last_dim + (t % per_row) * params.block;
        if (start >= params.n) { return; }
        let end = min(start + params.block, row * params.last_dim + params.last_dim);
        var m: f32 = 0.0;
        for (var i = start; i < end; i = i + 1u) {
            let v = abs(arena[params.in_off + i]);
            if (v > m) { m = v; }   // NaN never wins, as in the reference
        }
        let e = scale_exp(m);
        for (var i = start; i < end; i = i + 1u) {
            arena[params.out_off + i] = round_scaled(arena[params.in_off + i], e);
        }
        return;
    }
    if (t >= params.n) { return; }
    var e: i32 = 0;
    if (params.mode == 1u) { e = scale_exp(arena[params.amax_off]); }
    arena[params.out_off + t] = round_scaled(arena[params.in_off + t], e);
}
