// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `sgemm_general` — every transpose pair, `alpha`/`beta` mix and padded
//! leading dimension — against an f64 reference written out here.
//!
//! The reference is independent of every GEMM in the crate on purpose: in a
//! BLAS-less build `sgemm_general` IS the fallback (the loop nest, or the
//! `gemm` crate under `portable-gemm`), so comparing it to another rlx-cpu
//! entry point would compare the fallback with itself. Shapes straddle the
//! `portable-gemm` small-`m` cutover so both of its paths run.

use rlx_cpu::blas::sgemm_general;

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

#[allow(clippy::too_many_arguments)]
fn reference(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    (m, n, k): (usize, usize, usize),
    (alpha, beta): (f32, f32),
    (lda, ldb, ldc): (usize, usize, usize),
    (ta, tb): (bool, bool),
) -> Vec<f64> {
    let mut out: Vec<f64> = c.iter().map(|&v| v as f64).collect();
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f64;
            for p in 0..k {
                let av = if ta { a[p * lda + i] } else { a[i * lda + p] };
                let bv = if tb { b[j * ldb + p] } else { b[p * ldb + j] };
                acc += av as f64 * bv as f64;
            }
            let prev = if beta == 0.0 {
                0.0
            } else {
                beta as f64 * c[i * ldc + j] as f64
            };
            out[i * ldc + j] = alpha as f64 * acc + prev;
        }
    }
    out
}

#[test]
fn sgemm_general_matches_f64_reference() {
    let shapes = [
        (1, 7, 5),
        (3, 33, 17),
        (8, 64, 40),
        (37, 29, 70),
        (128, 96, 200),
    ];
    let scales = [(1.0f32, 0.0f32), (0.5, 1.0), (-1.25, 0.75)];
    let mut cases = 0;
    for &(m, n, k) in &shapes {
        for ta in [false, true] {
            for tb in [false, true] {
                for &(alpha, beta) in &scales {
                    // Padded leading dims, so a stride mix-up reads the wrong
                    // elements instead of silently landing on the right ones.
                    let lda = if ta { m } else { k } + 3;
                    let ldb = if tb { k } else { n } + 5;
                    let ldc = n + 2;
                    let a = fill(if ta { k } else { m } * lda, 1);
                    let b = fill(if tb { n } else { k } * ldb, 2);
                    let mut c = fill(m * ldc, 3);
                    if beta == 0.0 {
                        // CBLAS must not read C when beta == 0.
                        c.iter_mut().step_by(7).for_each(|v| *v = f32::NAN);
                    }
                    let want = reference(
                        &a,
                        &b,
                        &c,
                        (m, n, k),
                        (alpha, beta),
                        (lda, ldb, ldc),
                        (ta, tb),
                    );
                    // SAFETY: buffers cover the padded extents built above.
                    unsafe {
                        sgemm_general(
                            a.as_ptr(),
                            b.as_ptr(),
                            c.as_mut_ptr(),
                            m,
                            n,
                            k,
                            alpha,
                            beta,
                            lda,
                            ldb,
                            ldc,
                            ta,
                            tb,
                        )
                    };
                    for i in 0..m {
                        for j in 0..ldc {
                            let (got, exp) = (c[i * ldc + j], want[i * ldc + j]);
                            if j >= n {
                                // Padding columns are outside C: untouched.
                                assert!(
                                    got.to_bits() == (exp as f32).to_bits() || got.is_nan(),
                                    "padding written at ({i},{j})"
                                );
                                continue;
                            }
                            let tol = 1e-5 * (k as f64).sqrt() * (1.0 + exp.abs());
                            assert!(
                                (got as f64 - exp).abs() <= tol,
                                "m={m} n={n} k={k} ta={ta} tb={tb} alpha={alpha} beta={beta}: \
                                 c[{i},{j}] = {got}, want {exp}"
                            );
                        }
                    }
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, shapes.len() * 4 * scales.len());
}
