// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pipeline-parallel layer partitioning.
//!
//! Layers are split into `world_size` contiguous blocks and assigned to
//! ranks in **reverse**: rank 0 owns the *last* block (so it produces the
//! logits and samples), rank `world-1` owns the *first* block (so it does
//! the token embedding). This matches mlx-lm's `PipelineMixin` and means a
//! token's activations flow rank `world-1 → … → 1 → 0`.
//!
//! The split is exact (every layer assigned exactly once) and as even as
//! possible: the first `num_layers % world` forward-blocks get one extra
//! layer.

use std::ops::Range;

/// Where a rank sits in the pipeline. Drives whether it embeds tokens,
/// just transforms hidden states, or produces logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockRole {
    /// `world == 1`: this rank runs the whole model (embed → layers → head).
    Single,
    /// Highest rank: owns the first layers, embeds the input tokens, and
    /// sends hidden states toward rank 0.
    First,
    /// Interior rank: receives hidden states, runs its layers, forwards.
    Middle,
    /// Rank 0: owns the last layers, runs the final norm + LM head, and
    /// produces logits to sample from.
    Last,
}

/// Role of `rank` in a `world`-sized pipeline.
pub fn block_role(rank: u32, world: u32) -> BlockRole {
    if world <= 1 {
        BlockRole::Single
    } else if rank == world - 1 {
        BlockRole::First
    } else if rank == 0 {
        BlockRole::Last
    } else {
        BlockRole::Middle
    }
}

/// How many layers each rank holds, in rank order.
///
/// Counts rather than ranges, and in rank order rather than model order,
/// because that is the only way to write it that cannot be read two ways.
/// Rank 0 holds the *last* block, so `[10, 18]` means rank 1 takes the
/// model's first 18 layers and rank 0 the last 10 — the ranges fall out of
/// the counts and never have to be typed.
///
/// The point is machines that are not alike. An even split gives a 16 GB
/// laptop and a 64 GB desktop the same number of layers, and then the laptop
/// decides how fast the pipeline runs, or cannot hold its block at all.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LayerSplit {
    /// Layers per rank, indexed by rank.
    pub counts: Vec<u32>,
}

impl LayerSplit {
    /// The even split — what you get without asking for anything.
    ///
    /// Identical to [`pipeline_layer_range`] for every rank, which
    /// `the_even_split_and_the_range_function_describe_the_same_thing`
    /// checks. They must be: that function is what a caller gets by passing
    /// no split at all, so a divergence would make asking for an even split
    /// *explicitly* mean something else — and both sides would look
    /// self-consistent while the ranks disagreed about which layers they
    /// held.
    pub fn even(layers: usize, world: u32) -> Self {
        let world = world.max(1) as usize;
        let base = layers / world;
        let extra = layers % world;
        // The remainder goes to the blocks at the *start of the model*,
        // which are the *highest* ranks — ranks count backwards. Giving it
        // to rank 0 instead is the obvious-looking mistake, and it is what
        // this originally did.
        let counts = (0..world)
            .map(|r| (base + usize::from(world - 1 - r < extra)) as u32)
            .collect();
        Self { counts }
    }

    /// Split in proportion to each rank's `weight` — memory, usually.
    ///
    /// Every rank gets at least one layer: a rank with none would still take
    /// a turn in the chain, paying a network hop to do nothing.
    pub fn weighted(layers: usize, weights: &[u64]) -> Self {
        let world = weights.len();
        if world == 0 || layers == 0 {
            return Self { counts: Vec::new() };
        }
        if world > layers {
            // More machines than layers: nothing to weight, and some ranks
            // would have to be empty. Fall back rather than pretend.
            return Self::even(layers, world as u32);
        }
        let total: u128 = weights.iter().map(|&w| w as u128).sum();
        if total == 0 {
            return Self::even(layers, world as u32);
        }

        // Floor each share, keep one layer back for every rank, then hand
        // out what rounding left over to whoever was cheated most.
        let spare = layers - world;
        let mut counts: Vec<u32> = vec![1; world];
        let mut shares: Vec<(usize, f64)> = Vec::with_capacity(world);
        let mut given = 0usize;
        for (r, &w) in weights.iter().enumerate() {
            let exact = spare as f64 * (w as f64 / total as f64);
            let whole = exact.floor() as usize;
            counts[r] += whole as u32;
            given += whole;
            shares.push((r, exact - exact.floor()));
        }
        shares.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for (r, _) in shares.into_iter().take(spare - given) {
            counts[r] += 1;
        }
        Self { counts }
    }

    /// Parse `14,14` or `10, 18`.
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut counts = Vec::new();
        for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let n: u32 = part
                .parse()
                .map_err(|_| format!("{part:?} is not a layer count"))?;
            counts.push(n);
        }
        if counts.is_empty() {
            return Err("a layer split needs one count per rank, like 14,14".into());
        }
        Ok(Self { counts })
    }

    pub fn total(&self) -> usize {
        self.counts.iter().map(|&c| c as usize).sum()
    }

    /// This rank's half-open layer range.
    ///
    /// Rank `world-1` starts at 0 and the blocks stack upward, so rank 0
    /// ends at the last layer — the orientation
    /// [`pipeline_layer_range`] uses.
    pub fn range(&self, rank: u32) -> Option<std::ops::Range<usize>> {
        let r = rank as usize;
        if r >= self.counts.len() {
            return None;
        }
        let start: usize = self.counts[r + 1..].iter().map(|&c| c as usize).sum();
        Some(start..start + self.counts[r] as usize)
    }

    /// A number every rank can compare to check they were told the same
    /// thing.
    ///
    /// Two ranks with different splits is the worst failure this design can
    /// have: the hidden state is the same shape either way, so nothing
    /// errors — the model simply runs some layers twice and others never,
    /// and answers nonsense. Small enough to travel exactly through the f32
    /// control channel.
    pub fn fingerprint(&self) -> u32 {
        // FNV-1a, truncated to 24 bits so it survives f32 unchanged.
        let mut h: u32 = 0x811c9dc5;
        for &c in &self.counts {
            for b in c.to_le_bytes() {
                h ^= b as u32;
                h = h.wrapping_mul(0x0100_0193);
            }
        }
        h & 0x00ff_ffff
    }

    /// Check it describes this model on this many machines.
    pub fn validate(&self, layers: usize, world: u32) -> Result<(), String> {
        if self.counts.len() as u32 != world {
            return Err(format!(
                "the layer split has {} entries for a world of {world}: it needs \
                 one count per rank",
                self.counts.len()
            ));
        }
        if let Some(r) = self.counts.iter().position(|&c| c == 0) {
            return Err(format!(
                "rank {r} was given no layers, so it would take a turn in the \
                 chain to do nothing"
            ));
        }
        if self.total() != layers {
            return Err(format!(
                "the layer split covers {} layers but the model has {layers}",
                self.total()
            ));
        }
        Ok(())
    }
}

/// The contiguous, half-open range of layer indices `[start, end)` that
/// `rank` owns in a `world`-sized pipeline.
///
/// rank 0 → last block, rank `world-1` → first block. Panics if
/// `rank >= world`.
pub fn pipeline_layer_range(num_layers: usize, rank: u32, world: u32) -> Range<usize> {
    let world = world.max(1) as usize;
    let rank = rank as usize;
    assert!(rank < world, "rank {rank} out of range for world {world}");

    let per = num_layers / world;
    let extra = num_layers % world;
    // This rank owns forward-block `j` (reverse assignment).
    let j = world - 1 - rank;
    // Forward block i has `per + (i < extra)` layers; the first `extra`
    // blocks carry the remainder.
    let size_of = |i: usize| per + if i < extra { 1 } else { 0 };
    let start: usize = (0..j).map(size_of).sum();
    let len = size_of(j);
    start..(start + len)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── How the layers are divided ──────────────────────────────────────

    /// Every layer on exactly one rank, in order, with no gaps — the
    /// property that makes a split correct whatever produced it.
    fn covers_every_layer_once(split: &LayerSplit, layers: usize) {
        let world = split.counts.len() as u32;
        let mut seen = vec![0u32; layers];
        for rank in 0..world {
            for g in split.range(rank).expect("every rank has a range") {
                seen[g] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "split {:?} over {layers} layers: {seen:?}",
            split.counts
        );
        // And the blocks must stack in rank order: rank world-1 first.
        for rank in 1..world {
            let lower = split.range(rank).unwrap();
            let upper = split.range(rank - 1).unwrap();
            assert_eq!(
                lower.end,
                upper.start,
                "rank {rank} must abut rank {}",
                rank - 1
            );
        }
    }

    #[test]
    fn an_even_split_matches_what_no_split_at_all_does() {
        // `None` means `pipeline_layer_range`, so an explicit even split has
        // to describe the same thing or the two paths would disagree.
        for layers in [4usize, 28, 32, 36, 64, 80] {
            for world in 1..=8u32 {
                if world as usize > layers {
                    continue;
                }
                let split = LayerSplit::even(layers, world);
                assert_eq!(split.total(), layers);
                covers_every_layer_once(&split, layers);
            }
        }
    }

    #[test]
    fn a_weighted_split_follows_the_memory() {
        // A 64 GB machine should hold about four times what a 16 GB one does.
        let split = LayerSplit::weighted(28, &[64 << 30, 16 << 30]);
        assert_eq!(split.total(), 28);
        covers_every_layer_once(&split, 28);
        assert!(
            split.counts[0] > split.counts[1] * 3,
            "expected roughly 4:1, got {:?}",
            split.counts
        );
    }

    #[test]
    fn a_weighted_split_never_leaves_a_rank_with_nothing() {
        // A rank with no layers still takes its turn in the chain, paying a
        // network hop to pass a hidden state through unchanged.
        let split = LayerSplit::weighted(28, &[1 << 40, 1, 1]);
        assert_eq!(split.total(), 28);
        assert!(split.counts.iter().all(|&c| c >= 1), "{:?}", split.counts);
        covers_every_layer_once(&split, 28);
    }

    #[test]
    fn a_weighted_split_handles_degenerate_inputs() {
        // Equal weights are an even split.
        assert_eq!(
            LayerSplit::weighted(28, &[8, 8]).counts,
            LayerSplit::even(28, 2).counts
        );
        // Weights that say nothing fall back rather than divide by zero.
        assert_eq!(
            LayerSplit::weighted(28, &[0, 0]).counts,
            LayerSplit::even(28, 2).counts
        );
        // More machines than layers: there is nothing to weight.
        let many = LayerSplit::weighted(3, &[9, 1, 1, 1, 1]);
        assert_eq!(many.total(), 3);
        // And nothing at all is not a crash.
        assert!(LayerSplit::weighted(0, &[1]).counts.is_empty());
        assert!(LayerSplit::weighted(28, &[]).counts.is_empty());
    }

    #[test]
    fn the_even_split_and_the_range_function_describe_the_same_thing() {
        // Two ways to say "divide these layers evenly", now in one module.
        // `pipeline_layer_range` is what a caller gets by passing no split
        // at all, so if they ever disagree then asking for an even split
        // explicitly would silently mean something else.
        for layers in [1usize, 2, 4, 7, 28, 32, 33, 64, 80] {
            for world in 1..=8u32 {
                if world as usize > layers {
                    continue;
                }
                let split = LayerSplit::even(layers, world);
                for rank in 0..world {
                    assert_eq!(
                        split.range(rank).unwrap(),
                        pipeline_layer_range(layers, rank, world),
                        "layers={layers} world={world} rank={rank}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_split_is_read_the_way_it_is_written() {
        // Rank 0 holds the *last* block, so the first count is the top of
        // the model. Writing it any other way invites reading it backwards.
        let split = LayerSplit::parse("10,18").unwrap();
        assert_eq!(split.range(1).unwrap(), 0..18, "rank 1 takes the first 18");
        assert_eq!(split.range(0).unwrap(), 18..28, "rank 0 takes the last 10");
        assert_eq!(split.range(2), None);
        covers_every_layer_once(&split, 28);
    }

    #[test]
    fn a_split_that_is_not_a_split_is_refused() {
        let s = LayerSplit::parse("14,14").unwrap();
        assert_eq!(s.validate(28, 2), Ok(()));
        // Wrong number of machines.
        assert!(
            s.validate(28, 3)
                .unwrap_err()
                .contains("one count per rank")
        );
        // Wrong number of layers — the quiet one, since it looks plausible.
        assert!(s.validate(32, 2).unwrap_err().contains("covers 28"));
        // A rank with nothing to do.
        let empty = LayerSplit::parse("28,0").unwrap();
        assert!(empty.validate(28, 2).unwrap_err().contains("no layers"));
        // Nonsense.
        assert!(LayerSplit::parse("14,x").is_err());
        assert!(LayerSplit::parse("").is_err());
        assert!(LayerSplit::parse("  , ").is_err());
    }

    #[test]
    fn the_fingerprint_separates_splits_that_would_otherwise_pass_silently() {
        // Two ranks given different splits is the one failure with no
        // symptom: the hidden state is the same shape either way, so the
        // model just runs some layers twice and answers nonsense. The
        // fingerprint is what turns that into an error.
        let a = LayerSplit::parse("14,14").unwrap();
        let b = LayerSplit::parse("13,15").unwrap();
        let c = LayerSplit::parse("15,13").unwrap();
        assert_ne!(a.fingerprint(), b.fingerprint());
        assert_ne!(b.fingerprint(), c.fingerprint(), "order has to matter");
        assert_eq!(
            a.fingerprint(),
            LayerSplit::parse("14, 14").unwrap().fingerprint()
        );
    }

    #[test]
    fn a_fingerprint_survives_the_float_control_channel() {
        // It travels as an f32 alongside the other control fields, so it has
        // to be exactly representable.
        for counts in [vec![14u32, 14], vec![1, 27], vec![9, 9, 10], vec![80]] {
            let f = LayerSplit { counts }.fingerprint();
            assert!(f < (1 << 24), "{f} is too large for f32");
            assert_eq!(f as f32 as u32, f);
        }
    }

    #[test]
    fn roles_map_correctly() {
        assert_eq!(block_role(0, 1), BlockRole::Single);
        assert_eq!(block_role(0, 4), BlockRole::Last);
        assert_eq!(block_role(3, 4), BlockRole::First);
        assert_eq!(block_role(1, 4), BlockRole::Middle);
        assert_eq!(block_role(2, 4), BlockRole::Middle);
    }

    #[test]
    fn even_split_is_reversed() {
        // 16 layers / 4 ranks: rank 0 = last 4, rank 3 = first 4.
        assert_eq!(pipeline_layer_range(16, 0, 4), 12..16);
        assert_eq!(pipeline_layer_range(16, 1, 4), 8..12);
        assert_eq!(pipeline_layer_range(16, 2, 4), 4..8);
        assert_eq!(pipeline_layer_range(16, 3, 4), 0..4);
    }

    #[test]
    fn uneven_split_tiles_exactly() {
        // 10 layers / 4 ranks: forward block sizes 3,3,2,2.
        // reverse → rank0=[8,10), rank1=[6,8), rank2=[3,6), rank3=[0,3).
        assert_eq!(pipeline_layer_range(10, 0, 4), 8..10);
        assert_eq!(pipeline_layer_range(10, 1, 4), 6..8);
        assert_eq!(pipeline_layer_range(10, 2, 4), 3..6);
        assert_eq!(pipeline_layer_range(10, 3, 4), 0..3);

        // Reassembled in rank order (world-1 down to 0) covers [0,10) once.
        let mut covered = [0u8; 10];
        for r in 0..4u32 {
            for i in pipeline_layer_range(10, r, 4) {
                covered[i] += 1;
            }
        }
        assert!(covered.iter().all(|&c| c == 1), "every layer covered once");
    }

    #[test]
    fn single_rank_owns_all() {
        assert_eq!(pipeline_layer_range(28, 0, 1), 0..28);
    }

    #[test]
    fn more_ranks_than_layers_is_safe() {
        // 2 layers, 4 ranks: two ranks get a layer, two get empty ranges.
        let total: usize = (0..4u32).map(|r| pipeline_layer_range(2, r, 4).len()).sum();
        assert_eq!(total, 2);
    }
}
