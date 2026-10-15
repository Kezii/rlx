// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Block-diagonal attention over a **packed** batch.
//!
//! A model over irregularly-sampled events has a different token count per sample. Two
//! ways to batch that: pad every sample to the longest and mask the padding, or
//! **concatenate** the samples into one flat sequence and mask across the boundaries.
//! The second wastes nothing and is what `torch_brain`'s
//! `rotary_attn_xformers_varlen_func` does, via
//! `xformers.ops.fmha.BlockDiagonalMask.from_seqlens`.
//!
//! The mask is the whole content of that path, and it is not optional. Measured on
//! unit-scale activations with two samples of 2 and 3 query tokens: masked packed
//! attention is **bit-identical** to running each sample separately, while the same
//! packed batch through an **unmasked** dense path differs by up to **2.46** in absolute
//! value. Not a rounding difference — one sample's keys contribute to another's output.
//!
//! Pairs with [`crate::rope`], which supplies the continuous-time rotary tables these
//! queries and keys are rotated by.

use rlx_ir::GraphExt;

/// Cumulative token offsets for a packed batch: `len() == seqlens.len() + 1`.
///
/// `offsets[i] .. offsets[i + 1]` is sample `i`'s slice of the flat sequence, and
/// `offsets[n]` is the total token count.
pub fn offsets(seqlens: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(seqlens.len() + 1);
    let mut acc = 0;
    out.push(0);
    for &l in seqlens {
        acc += l;
        out.push(acc);
    }
    out
}

/// Which sample each packed token belongs to.
///
/// A zero-length sample contributes no token, so it does not appear here at all — the
/// returned length is the token count, not the sample count.
pub fn sample_of_token(seqlens: &[usize]) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, &l) in seqlens.iter().enumerate() {
        out.extend(std::iter::repeat_n(i, l));
    }
    out
}

/// The additive block-diagonal attention bias — `BlockDiagonalMask::from_seqlens`.
///
/// `[sum(q_seqlen), sum(kv_seqlen)]` row-major: `0.0` where query token *i* may attend to
/// key token *j* (both in the same sample) and `-inf` everywhere else. Additive rather
/// than boolean because that is what a graph adds to the scores before the softmax, and
/// `-inf` rather than a large negative because `softmax` then yields **exactly** zero.
///
/// `kv_seqlen = None` means *the same lengths as the queries* — packed self-attention.
/// That default is upstream's and it is load-bearing: a port that required the argument
/// would make every self-attention call site pass the list twice, and one of them would
/// eventually pass the wrong one.
///
/// A sample whose **kv** length is 0 leaves its query rows wholly masked, and `softmax`
/// over a row of `-inf` is `NaN`. That is not guarded against here: it is a malformed
/// batch, the NaN propagates visibly, and silently substituting zeros would hide a sample
/// that has no keys at all. [`fully_masked_rows`] reports it for a caller that wants to
/// check first.
pub fn block_diagonal_bias(
    q_seqlen: &[usize],
    kv_seqlen: Option<&[usize]>,
) -> Result<Vec<f32>, String> {
    let kv = kv_seqlen.unwrap_or(q_seqlen);
    if kv.len() != q_seqlen.len() {
        return Err(format!(
            "q_seqlen has {} sample(s) and kv_seqlen has {}; a block-diagonal mask needs \
             one kv length per query length",
            q_seqlen.len(),
            kv.len()
        ));
    }
    let n_q: usize = q_seqlen.iter().sum();
    let n_kv: usize = kv.iter().sum();
    let mut bias = vec![f32::NEG_INFINITY; n_q * n_kv];
    let (qo, ko) = (offsets(q_seqlen), offsets(kv));
    for s in 0..q_seqlen.len() {
        for i in qo[s]..qo[s + 1] {
            for j in ko[s]..ko[s + 1] {
                bias[i * n_kv + j] = 0.0;
            }
        }
    }
    Ok(bias)
}

/// Query rows with no key to attend to — the rows whose softmax would be `NaN`.
pub fn fully_masked_rows(q_seqlen: &[usize], kv_seqlen: Option<&[usize]>) -> Vec<usize> {
    let kv = kv_seqlen.unwrap_or(q_seqlen);
    let qo = offsets(q_seqlen);
    let mut out = Vec::new();
    for s in 0..q_seqlen.len().min(kv.len()) {
        if kv[s] == 0 {
            out.extend(qo[s]..qo[s + 1]);
        }
    }
    out
}

/// Rotary attention over a packed batch, with the block-diagonal bias applied.
///
/// `q`, `k`, `v` are `[1, N, heads * dh]` — the packed `(N, H·D)` layout upstream uses,
/// carrying a leading batch of 1. `bias` is the `[1, 1, n_q, n_kv]` node holding
/// [`block_diagonal_bias`]; it broadcasts over the heads.
///
/// The rotation order is upstream's and is not interchangeable: rotate the queries and
/// keys by their own position tables, rotate the **values** too when `rotate_value`, and
/// then rotate the **output** by the *inverted* query tables — `(cos, -sin)`, which is
/// what `q_nsin` must carry. Skipping the output rotation leaves the result in the
/// rotated frame, where it is the right shape and the wrong basis.
#[allow(clippy::too_many_arguments)]
pub fn packed_rotary_attn(
    g: &mut rlx_ir::Graph,
    q: rlx_ir::NodeId,
    k: rlx_ir::NodeId,
    v: rlx_ir::NodeId,
    bias: rlx_ir::NodeId,
    scale: rlx_ir::NodeId,
    q_cos: rlx_ir::NodeId,
    q_sin: rlx_ir::NodeId,
    q_nsin: rlx_ir::NodeId,
    kv_cos: rlx_ir::NodeId,
    kv_sin: rlx_ir::NodeId,
    n_q: usize,
    n_kv: usize,
    heads: usize,
    dh: usize,
    rotate_value: bool,
) -> rlx_ir::NodeId {
    let split = |g: &mut rlx_ir::Graph, x: rlx_ir::NodeId, n: usize| {
        // [1, N, H*dh] -> [1, H, N, dh]
        let r = g.reshape_(x, vec![1, n as i64, heads as i64, dh as i64]);
        g.transpose_(r, vec![0, 2, 1, 3])
    };
    let qh = split(g, q, n_q);
    let kh = split(g, k, n_kv);
    let vh = split(g, v, n_kv);

    let qh = crate::rope::rotate_by_tables(g, qh, q_cos, q_sin, 1, heads, n_q, dh);
    let kh = crate::rope::rotate_by_tables(g, kh, kv_cos, kv_sin, 1, heads, n_kv, dh);
    let vh = if rotate_value {
        crate::rope::rotate_by_tables(g, vh, kv_cos, kv_sin, 1, heads, n_kv, dh)
    } else {
        vh
    };

    let kt = g.transpose_(kh, vec![0, 1, 3, 2]); // [1, H, dh, n_kv]
    let scores = g.mm(qh, kt); // [1, H, n_q, n_kv]
    let scores = g.mul(scores, scale);
    // The bias goes on AFTER the scale: scaling -inf is still -inf, but scaling a large
    // finite sentinel would shrink it, which is one reason the sentinel is -inf.
    let scores = g.add(scores, bias);
    let attn = g.sm(scores, -1);
    let out = g.mm(attn, vh); // [1, H, n_q, dh]
    let out = if rotate_value {
        crate::rope::rotate_by_tables(g, out, q_cos, q_nsin, 1, heads, n_q, dh)
    } else {
        out
    };
    let out = g.transpose_(out, vec![0, 2, 1, 3]); // [1, n_q, H, dh]
    g.reshape_(out, vec![1, n_q as i64, (heads * dh) as i64])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_bound_each_samples_slice() {
        assert_eq!(offsets(&[2, 3]), vec![0, 2, 5]);
        assert_eq!(offsets(&[]), vec![0]);
        // A zero-length sample repeats the previous offset, so its slice is empty.
        assert_eq!(offsets(&[2, 0, 3]), vec![0, 2, 2, 5]);
    }

    #[test]
    fn a_zero_length_sample_contributes_no_token() {
        // The returned length is the TOKEN count, not the sample count, so a caller
        // indexing by sample must use `offsets` instead.
        assert_eq!(sample_of_token(&[2, 3]), vec![0, 0, 1, 1, 1]);
        assert_eq!(sample_of_token(&[2, 0, 1]), vec![0, 0, 2]);
        assert!(sample_of_token(&[0, 0]).is_empty());
    }

    #[test]
    fn the_bias_is_zero_inside_a_block_and_negative_infinity_outside() {
        // Two samples of 2 and 3 tokens, self-attention.
        let b = block_diagonal_bias(&[2, 3], None).unwrap();
        assert_eq!(b.len(), 25);
        let at = |i: usize, j: usize| b[i * 5 + j];
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(at(i, j), 0.0, "({i},{j}) is inside sample 0");
            }
            for j in 2..5 {
                assert!(at(i, j).is_infinite(), "({i},{j}) crosses into sample 1");
            }
        }
        for i in 2..5 {
            for j in 0..2 {
                assert!(at(i, j).is_infinite(), "({i},{j}) crosses into sample 0");
            }
            for j in 2..5 {
                assert_eq!(at(i, j), 0.0, "({i},{j}) is inside sample 1");
            }
        }
        // 2*2 + 3*3 = 13 allowed pairs.
        assert_eq!(b.iter().filter(|x| **x == 0.0).count(), 13);
    }

    #[test]
    fn the_sentinel_is_negative_infinity_so_softmax_yields_exactly_zero() {
        // A large finite negative gives ~0, not 0 -- and it would also be shrunk by the
        // score scaling if the bias were applied before it.
        let b = block_diagonal_bias(&[1, 1], None).unwrap();
        assert_eq!(b[1], f32::NEG_INFINITY);
        assert!(b[1].is_infinite() && b[1].is_sign_negative());
        // exp(-inf) is exactly 0.0.
        assert_eq!(b[1].exp(), 0.0);
    }

    #[test]
    fn omitting_the_kv_lengths_means_the_query_lengths() {
        // Upstream's default. Requiring the argument would make every self-attention call
        // site pass the list twice, and one of them would eventually pass the wrong one.
        let implicit = block_diagonal_bias(&[2, 3], None).unwrap();
        let explicit = block_diagonal_bias(&[2, 3], Some(&[2, 3])).unwrap();
        assert_eq!(implicit, explicit);
    }

    #[test]
    fn cross_attention_blocks_need_not_be_square() {
        // 2 queries over 4 keys, then 3 queries over 1 key.
        let b = block_diagonal_bias(&[2, 3], Some(&[4, 1])).unwrap();
        assert_eq!(b.len(), 5 * 5);
        let at = |i: usize, j: usize| b[i * 5 + j];
        assert_eq!(at(0, 3), 0.0, "query 0 reaches key 3 of its own sample");
        assert!(at(0, 4).is_infinite(), "but not key 4, which is sample 1's");
        assert_eq!(
            at(2, 4),
            0.0,
            "query 2 is sample 1 and reaches its single key"
        );
        assert!(at(2, 0).is_infinite());
        // 2*4 + 3*1 = 11 allowed pairs.
        assert_eq!(b.iter().filter(|x| **x == 0.0).count(), 11);
    }

    #[test]
    fn a_single_sample_masks_nothing() {
        // The packed path must degenerate to the dense one, or a batch of one would take
        // a different code path from a batch of two.
        let b = block_diagonal_bias(&[5], None).unwrap();
        assert!(b.iter().all(|x| *x == 0.0), "no pair is masked");
        assert_eq!(b.len(), 25);
    }

    #[test]
    fn single_token_samples_leave_only_the_diagonal() {
        let b = block_diagonal_bias(&[1, 1, 1], None).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_eq!(b[i * 3 + j], 0.0);
                } else {
                    assert!(b[i * 3 + j].is_infinite());
                }
            }
        }
    }

    #[test]
    fn a_sample_with_no_keys_leaves_its_rows_wholly_masked_and_says_so() {
        // softmax over a row of -inf is NaN. That is a malformed batch; the NaN is left to
        // propagate visibly rather than being papered over with zeros, and the rows are
        // reported for a caller that wants to check first.
        let b = block_diagonal_bias(&[2, 2], Some(&[3, 0])).unwrap();
        assert_eq!(b.len(), 4 * 3);
        for i in 2..4 {
            for j in 0..3 {
                assert!(b[i * 3 + j].is_infinite(), "row {i} has no key");
            }
        }
        assert_eq!(fully_masked_rows(&[2, 2], Some(&[3, 0])), vec![2, 3]);
        assert!(fully_masked_rows(&[2, 3], None).is_empty());
    }

    #[test]
    fn mismatched_sample_counts_are_refused() {
        let e = block_diagonal_bias(&[2, 3], Some(&[4])).unwrap_err();
        assert!(e.contains("one kv length per query length"), "{e}");
        assert!(block_diagonal_bias(&[], None).unwrap().is_empty());
    }
}
