// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Attention-mask helpers for bucketed decode (pad-to-upper, slice-back).

/// Causal decode mask padded to bucket `upper`.
///
/// **K layout** (after `concat(past_k, new_k_rope, dim=1)` in the decode graph):
/// - `K[0..past_seq]`: real prompt K from the cache
/// - `K[past_seq..upper]`: zero padding from `pad_layers_to_upper`
/// - `K[upper]`: the newly rope-rotated K for the current token
///
/// So `mask[upper] = 1.0` (attend to self) and `mask[past_seq..upper] = 0.0`
/// (mask the padding). An earlier off-by-one version attended padding at
/// `past_seq` AND masked the new K at `upper`, which removed self-attention
/// to the current token's own K and collapsed decode to degenerate tokens —
/// `\n\n\n…` for short prompts, `attention attention attention…` for longer
/// ones.
///
/// **Convention**: the IR's `MaskKind::Custom` is a **binary keep mask** —
/// the CPU executor and Metal/MLX lowering all gate scores with a
/// `m[ki] < 0.5` threshold.
pub fn bucket_decode_mask(past_seq: usize, upper: usize) -> Vec<f32> {
    // Graph mask shape is `[batch, upper + 1]` (past keys + new token).
    (0..=upper)
        .map(|i| {
            if i < past_seq || i == upper {
                // real past K, or newly-rope'd K for the current decode position
                1.0
            } else {
                0.0 // padding
            }
        })
        .collect()
}

/// Like [`bucket_decode_mask`] but for **sliding-window** attention: the
/// current token (absolute position `past_seq`) attends only to cached keys in
/// `[past_seq - window, past_seq]` — older real-past keys are masked out too,
/// matching `MaskKind::SlidingWindow(window)` in prefill. `window == 0` or
/// `window >= past_seq` reduces to the full causal [`bucket_decode_mask`].
pub fn bucket_decode_mask_windowed(past_seq: usize, upper: usize, window: usize) -> Vec<f32> {
    let lo = if window == 0 {
        0
    } else {
        past_seq.saturating_sub(window)
    };
    (0..=upper)
        .map(|i| {
            let real_past_in_window = i >= lo && i < past_seq;
            if real_past_in_window || i == upper {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where a mask lets attention land: the indices scored 1.0.
    fn attended(mask: &[f32]) -> Vec<usize> {
        mask.iter()
            .enumerate()
            .filter(|(_, v)| **v == 1.0)
            .map(|(i, _)| i)
            .collect()
    }

    #[test]
    fn the_decode_mask_covers_the_real_past_and_the_new_token() {
        // Untested until now, though it runs on every decode token of every
        // bucketed model. The shape is `[past keys .. upper, new token]`, so
        // the slots between `past_seq` and `upper` are padding holding
        // whatever the last longer sequence left there — attending to them
        // is not an error anybody sees, just a wrong answer.
        let m = bucket_decode_mask(3, 7);
        assert_eq!(m.len(), 8, "mask is upper + 1 wide");
        assert_eq!(
            attended(&m),
            vec![0, 1, 2, 7],
            "3 real past keys and the new token"
        );
        assert_eq!(m[3..7], [0.0, 0.0, 0.0, 0.0], "padding is masked out");
    }

    #[test]
    fn a_full_bucket_leaves_no_padding_to_mask() {
        // `past_seq == upper` is the exactly-full case, where every slot is
        // real and the new token sits in the last one.
        let m = bucket_decode_mask(4, 4);
        assert_eq!(m.len(), 5);
        assert_eq!(attended(&m), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn the_first_token_attends_only_to_itself() {
        let m = bucket_decode_mask(0, 3);
        assert_eq!(attended(&m), vec![3], "nothing cached yet");
    }

    #[test]
    fn every_bucket_of_every_ladder_masks_its_padding() {
        // The seam between two modules: `compile_cache` decides the bucket
        // extents and this decides what the graph may attend to at that
        // extent. Each is tested on its own; the invariant that matters is
        // the pair — for any `past` the cache routes to a bucket, the mask
        // for that bucket's extent must attend to exactly the `past` real
        // keys plus the new token, and nothing else. An off-by-one in
        // either module breaks it, and the symptom is attention over stale
        // padding rather than a crash.
        use crate::Device;
        use crate::compile_cache::BucketedCompileCache;

        let ladders = [
            (
                "pow2 1..512",
                BucketedCompileCache::power_of_two_ladder(Device::Cpu, 1, 512),
            ),
            (
                "pow2 1..1",
                BucketedCompileCache::power_of_two_ladder(Device::Cpu, 1, 1),
            ),
            (
                "geometric 512/3",
                BucketedCompileCache::geometric_ladder(Device::Cpu, 512, 3),
            ),
            (
                "geometric 512/1",
                BucketedCompileCache::geometric_ladder(Device::Cpu, 512, 1),
            ),
            (
                "geometric 4096/5",
                BucketedCompileCache::geometric_ladder(Device::Cpu, 4096, 5),
            ),
        ];
        for (name, cache) in &ladders {
            let extents: Vec<(u64, u64)> = cache.buckets().map(|r| (r.start, r.end - 1)).collect();
            let max = extents.last().unwrap().1;
            for past in 0..=max {
                // 0 past keys still has to route somewhere: the cache keys
                // on the length being run, which is at least 1.
                let key = past.max(1);
                let idx = cache
                    .bucket_for(key)
                    .unwrap_or_else(|| panic!("{name}: past={past} routed to no bucket"));
                let upper = extents[idx].1 as usize;
                assert!(
                    past as usize <= upper,
                    "{name}: past={past} routed to a bucket ending at {upper}"
                );
                let m = bucket_decode_mask(past as usize, upper);
                assert_eq!(m.len(), upper + 1, "{name}: past={past}");
                let want: Vec<usize> = (0..past as usize).chain(std::iter::once(upper)).collect();
                assert_eq!(attended(&m), want, "{name}: past={past} upper={upper}");
            }
        }
    }

    #[test]
    fn windowed_mask_reduces_to_causal_when_wide() {
        // window >= past_seq → identical to the full causal mask.
        assert_eq!(
            bucket_decode_mask_windowed(5, 8, 100),
            bucket_decode_mask(5, 8)
        );
    }

    #[test]
    fn a_window_of_zero_means_no_window_at_all() {
        // The doc for `bucket_decode_mask_windowed` says window == 0 *and*
        // window >= past_seq both reduce to the causal mask. Only the second
        // half was tested; this is the first, and both are checked over a
        // range rather than at one point.
        for (past, upper) in [(0, 3), (1, 3), (4, 4), (5, 8), (8, 8)] {
            assert_eq!(
                bucket_decode_mask_windowed(past, upper, 0),
                bucket_decode_mask(past, upper),
                "window 0 should be causal: past={past} upper={upper}"
            );
            assert_eq!(
                bucket_decode_mask_windowed(past, upper, past + 1),
                bucket_decode_mask(past, upper),
                "a window wider than the past should be causal: past={past} upper={upper}"
            );
        }
    }

    #[test]
    fn a_window_never_attends_to_padding_either() {
        // The windowed mask shares the causal one's seam with the bucket
        // extents: whatever the window admits, the slots between `past_seq`
        // and `upper` hold keys left by a longer sequence.
        for upper in [1usize, 4, 7, 16] {
            for past in 0..=upper {
                for window in [0usize, 1, 2, 3, upper, upper + 5] {
                    let m = bucket_decode_mask_windowed(past, upper, window);
                    assert_eq!(
                        m.len(),
                        upper + 1,
                        "past={past} upper={upper} window={window}"
                    );
                    for (i, v) in m.iter().enumerate() {
                        if i >= past && i != upper {
                            assert_eq!(
                                *v, 0.0,
                                "padding at {i} attended: past={past} upper={upper} window={window}"
                            );
                        }
                    }
                    assert_eq!(m[upper], 1.0, "the new token must always be attended");
                }
            }
        }
    }

    #[test]
    fn windowed_mask_drops_old_keys() {
        // past_seq=5, window=2 → attend to keys [3,4] + the new token at upper.
        let m = bucket_decode_mask_windowed(5, 6, 2);
        assert_eq!(m, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        //                  0    1    2    3    4   pad  new(=upper)
    }
}
