// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Op sharding** — split one op into independent pieces, price the IO, and
//! pick where to run them.
//!
//! A single op is often the whole problem: a convolution that lowers to a host
//! step stalls a GPU schedule, and a big matmul is more work than one device
//! should own. Both can be cut into pieces that have *no data dependency on
//! each other*, so they can run on separate threads, separate devices, or
//! separate nodes — but only along the right axis, and only if the cut is worth
//! what it costs to move the data.
//!
//! This module answers three questions, and deliberately answers nothing else:
//!
//! 1. **Along which axis can this op be cut, and into what pieces?**
//!    [`shardings`] enumerates the options. Every shard's piece of the output
//!    depends only on its own slice of one operand plus operands it reads
//!    whole, so shards never need to talk to each other.
//! 2. **What does each option cost in bytes moved?** [`ShardOption::moved_bytes`].
//!    The axes differ in *which* operand has to be replicated, and that is
//!    usually the entire story: cutting a conv by batch replicates the weights,
//!    cutting it by output channel replicates the activations.
//! 3. **Which option, on which workers?** [`choose`], scoring compute against
//!    transfer and refusing to rank a worker whose cost model is uncalibrated.
//!
//! ## What this module is not
//!
//! It performs no graph surgery. [`ShardOption`] describes the cut —
//! which operand, which axis, which ranges — and the caller emits narrows and
//! a concat, or ships ranges to workers, as suits it. Keeping emission out
//! means this stays an analysis usable from a backend, from the distributed
//! layer, and from a browser worker, none of which agree on how to execute.
//!
//! ## Sharding is exact, but not always bit-identical
//!
//! Every option here preserves the maths: each output element is computed from
//! the same inputs by the same reduction. Whether it is *bit-identical* depends
//! on the axis, because the backend may block a narrower shape differently:
//!
//! * [`ShardAxis::Batch`] and [`ShardAxis::Rows`] came back bit-identical.
//! * [`ShardAxis::OutChannel`] and [`ShardAxis::Cols`] agreed to ~5e-7 — the
//!   kernel reassociated its accumulation for the narrower output.
//!
//! Measured in `rlx-runtime/tests/shard_equivalence.rs`. It matters for
//! placement: a cohort that relies on every worker holding *identical* values
//! (as the browser tab cluster does) must shard along batch/rows, or accept
//! that its ranks now differ in the last bits.
//!
//! It also defines its own cost inputs ([`LinkCost`], [`WorkerCost`]) rather
//! than importing `rlx_collectives::planner::Link` or
//! `rlx_runtime::BackendCostModel`: both of those crates sit *above* this one,
//! so depending on them would be a cycle. Callers adapt — the shapes are
//! deliberately the same.

use rlx_ir::{Dim, Graph, NodeId, Op, OpKind};

/// Which axis an op is cut along. Each names the output axis that is split and,
/// by implication, the operand that has to be replicated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardAxis {
    /// Output batch axis. Slices the activations; **replicates the weights**.
    /// Exact for convolution — no halo, because each sample's output depends
    /// only on that sample.
    Batch,
    /// Output channel axis. Slices the weights (and bias); **replicates the
    /// activations**.
    OutChannel,
    /// Matmul output rows. Slices `A`; replicates `B`.
    Rows,
    /// Matmul output columns. Slices `B`; replicates `A`.
    Cols,
}

impl ShardAxis {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Batch => "batch",
            Self::OutChannel => "out-channel",
            Self::Rows => "rows",
            Self::Cols => "cols",
        }
    }
}

/// One piece of the split axis: `[start, start + len)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shard {
    pub start: usize,
    pub len: usize,
}

/// An operand that follows the shard boundary, and the axis it is cut on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceSpec {
    /// Index into the node's `inputs`.
    pub operand: usize,
    /// Axis of that operand which lines up with the split output axis.
    pub axis: usize,
}

/// Is the replicated operand already where the work will happen?
///
/// This is the difference between sharding being nearly free and sharding
/// dominating the step. In data-parallel training the weights are already on
/// every node, so a batch split moves only activations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Residency {
    /// The operands read whole are already resident on every worker.
    pub shared_resident: bool,
}

/// One way to cut an op into independent pieces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardOption {
    pub axis: ShardAxis,
    /// Output axis being split.
    pub output_axis: usize,
    /// Operands cut along the shard boundary.
    pub sliced: Vec<SliceSpec>,
    /// Operands every shard reads whole — the ones replication costs apply to.
    pub shared: Vec<usize>,
    /// Pieces of the split axis: disjoint, in order, covering it exactly.
    pub shards: Vec<Shard>,
    pub elem_bytes: u64,
    /// Elements across all sliced operands (each crosses the link once).
    pub sliced_elems: u64,
    /// Elements across all shared operands (each crosses once *per worker*).
    pub shared_elems: u64,
    pub output_elems: u64,
}

impl ShardOption {
    pub fn count(&self) -> usize {
        self.shards.len()
    }

    /// Bytes that must cross a link to run this option on `count()` workers.
    ///
    /// Sliced operands cross once in total (each element belongs to exactly one
    /// shard) and the output comes back once; a shared operand crosses once per
    /// worker unless it is already resident. That asymmetry is what makes the
    /// axis choice an IO decision rather than a style preference.
    pub fn moved_bytes(&self, residency: Residency) -> u64 {
        let workers = self.count() as u64;
        let shared = if residency.shared_resident {
            0
        } else {
            self.shared_elems.saturating_mul(workers)
        };
        self.sliced_elems
            .saturating_add(shared)
            .saturating_add(self.output_elems)
            .saturating_mul(self.elem_bytes)
    }

    /// Output elements the largest shard computes — the critical path, since
    /// shards run concurrently.
    pub fn widest_shard_elems(&self) -> u64 {
        let total: usize = self.shards.iter().map(|s| s.len).sum();
        if total == 0 {
            return 0;
        }
        let widest = self.shards.iter().map(|s| s.len).max().unwrap_or(0) as u64;
        self.output_elems.saturating_mul(widest) / total as u64
    }
}

/// Split `len` into `count` contiguous pieces, remainder spread over the first
/// ones. Never produces an empty piece, so `count` is clamped to `len`.
pub fn shard_ranges(len: usize, count: usize) -> Vec<Shard> {
    let count = count.clamp(1, len.max(1));
    let base = len / count;
    let extra = len % count;
    let mut out = Vec::with_capacity(count);
    let mut start = 0;
    for i in 0..count {
        let this = base + usize::from(i < extra);
        out.push(Shard { start, len: this });
        start += this;
    }
    out
}

fn static_dim(shape: &rlx_ir::Shape, axis: usize) -> Option<usize> {
    match shape.dims().get(axis)? {
        Dim::Static(n) => Some(*n),
        Dim::Dynamic(_) => None,
    }
}

fn elems(shape: &rlx_ir::Shape) -> Option<u64> {
    shape.num_elements().map(|n| n as u64)
}

/// Every way `node` can be cut, up to `max_shards` pieces, cheapest-to-describe
/// first. Empty when the op is not shardable, its shapes are not static, or the
/// split would not be exact.
pub fn shardings(graph: &Graph, node: NodeId, max_shards: usize) -> Vec<ShardOption> {
    let n = graph.node(node);
    let out_shape = &n.shape;
    if !out_shape.is_static() || max_shards < 2 {
        return Vec::new();
    }
    let elem_bytes = out_shape.dtype().size_bytes() as u64;
    let operand =
        |i: usize| -> Option<&rlx_ir::Shape> { n.inputs.get(i).map(|id| &graph.node(*id).shape) };

    // (axis, output_axis, sliced operands, shared operands)
    let mut layouts: Vec<(ShardAxis, usize, Vec<SliceSpec>, Vec<usize>)> = Vec::new();
    match n.op.kind() {
        OpKind::Conv | OpKind::Conv3d => {
            let groups = match &n.op {
                Op::Conv { groups, .. } => *groups,
                _ => 1,
            };
            // Activations axis 0 is the output batch axis for NCHW/NCDHW.
            let mut shared = vec![1];
            if n.inputs.len() > 2 {
                shared.push(2); // bias: read whole by every shard
            }
            layouts.push((
                ShardAxis::Batch,
                0,
                vec![SliceSpec {
                    operand: 0,
                    axis: 0,
                }],
                shared,
            ));

            // A grouped conv's output channels are tied to *input* channel
            // groups, so cutting by output channel also cuts the activations —
            // the operand this option claims is shared. Rather than model group
            // alignment, only offer the batch split, which is exact for any
            // `groups`.
            if groups <= 1 {
                let mut sliced = vec![SliceSpec {
                    operand: 1,
                    axis: 0,
                }];
                if n.inputs.len() > 2 {
                    sliced.push(SliceSpec {
                        operand: 2,
                        axis: 0,
                    });
                }
                layouts.push((ShardAxis::OutChannel, 1, sliced, vec![0]));
            }
        }
        OpKind::MatMul => {
            let rank = out_shape.rank();
            if rank < 2 {
                return Vec::new();
            }
            layouts.push((
                ShardAxis::Rows,
                rank - 2,
                vec![SliceSpec {
                    operand: 0,
                    axis: operand(0).map_or(rank - 2, |s| s.rank().saturating_sub(2)),
                }],
                vec![1],
            ));
            layouts.push((
                ShardAxis::Cols,
                rank - 1,
                vec![SliceSpec {
                    operand: 1,
                    axis: operand(1).map_or(rank - 1, |s| s.rank().saturating_sub(1)),
                }],
                vec![0],
            ));
        }
        _ => return Vec::new(),
    }

    let output_elems = match elems(out_shape) {
        Some(e) => e,
        None => return Vec::new(),
    };

    let mut options = Vec::new();
    for (axis, output_axis, sliced, shared) in layouts {
        let Some(axis_len) = static_dim(out_shape, output_axis) else {
            continue;
        };
        if axis_len < 2 {
            continue;
        }
        // Every sliced operand must have a static axis of the same length, or
        // the cut is not expressible.
        let mut sliced_elems = 0u64;
        let mut ok = true;
        for spec in &sliced {
            match operand(spec.operand) {
                Some(s) if static_dim(s, spec.axis) == Some(axis_len) => match elems(s) {
                    Some(e) => sliced_elems += e,
                    None => ok = false,
                },
                _ => ok = false,
            }
        }
        let mut shared_elems = 0u64;
        for &i in &shared {
            match operand(i).and_then(elems) {
                Some(e) => shared_elems += e,
                None => ok = false,
            }
        }
        if !ok {
            continue;
        }

        for count in 2..=max_shards.min(axis_len) {
            options.push(ShardOption {
                axis,
                output_axis,
                sliced: sliced.clone(),
                shared: shared.clone(),
                shards: shard_ranges(axis_len, count),
                elem_bytes,
                sliced_elems,
                shared_elems,
                output_elems,
            });
        }
    }
    options
}

// ── placement ─────────────────────────────────────────────────────────────

/// Transfer cost of the link the shards would cross.
///
/// Mirrors `rlx_collectives::planner::Link`, which sits above this crate, so a
/// caller converts with two field reads:
/// `LinkCost { bytes_per_s: link.bandwidth_bytes_per_s, latency_s: link.latency_s }`.
#[derive(Clone, Copy, Debug)]
pub struct LinkCost {
    pub bytes_per_s: f64,
    pub latency_s: f64,
}

impl LinkCost {
    pub fn send_time(&self, bytes: u64) -> f64 {
        if self.bytes_per_s <= 0.0 {
            return f64::INFINITY;
        }
        self.latency_s + bytes as f64 / self.bytes_per_s
    }
}

/// One worker that could take a shard.
#[derive(Clone, Debug)]
pub struct WorkerCost {
    pub label: String,
    /// Output elements per second for this op shape.
    pub elems_per_s: f64,
    /// May this worker's number be compared against another device's?
    ///
    /// Mirrors `rlx_runtime::CostCalibration::is_rankable`, which is the rule
    /// this repo already applies to device ranking: `Measured` and
    /// `ArchDefault` are comparable, `Uncalibrated` is a generic guess and is
    /// **excluded from ranking** rather than quietly trusted. Set it from that
    /// method rather than deciding here — a second, stricter rule living in
    /// this module is exactly how capability tables drift apart.
    pub rankable: bool,
}

/// The chosen option, and what was rejected.
#[derive(Clone, Debug)]
pub struct Choice {
    /// Index into the `options` passed to [`choose`]; `None` = do not shard.
    pub option: Option<usize>,
    pub score_s: f64,
    pub moved_bytes: u64,
    pub why: String,
    /// Workers left out of ranking because their cost model is not rankable.
    pub excluded: Vec<String>,
}

/// Pick the option whose slowest shard plus its transfer finishes first, or
/// `None` for "run it whole" when no split beats that.
///
/// `whole_elems_per_s` is the rate of the device that would run the unsharded
/// op, so "do not shard" is a real candidate rather than a fallback.
pub fn choose(
    options: &[ShardOption],
    workers: &[WorkerCost],
    link: LinkCost,
    residency: Residency,
    whole_elems_per_s: f64,
) -> Choice {
    let excluded: Vec<String> = workers
        .iter()
        .filter(|w| !w.rankable)
        .map(|w| w.label.clone())
        .collect();
    let usable: Vec<&WorkerCost> = workers
        .iter()
        .filter(|w| w.rankable && w.elems_per_s > 0.0)
        .collect();

    let whole_elems = options.first().map_or(0, |o| o.output_elems);
    let baseline = if whole_elems_per_s > 0.0 {
        whole_elems as f64 / whole_elems_per_s
    } else {
        f64::INFINITY
    };
    let mut best = Choice {
        option: None,
        score_s: baseline,
        moved_bytes: 0,
        why: format!("unsharded: {:.6}s, no split beat it", baseline),
        excluded: excluded.clone(),
    };
    if usable.is_empty() {
        best.why = if workers.is_empty() {
            "no workers offered".into()
        } else {
            "no worker has a rankable cost model; nothing to rank".into()
        };
        return best;
    }

    for (i, opt) in options.iter().enumerate() {
        if opt.count() > usable.len() {
            continue; // more shards than workers to run them
        }
        // Shards run concurrently, so the slowest one sets compute time; the
        // slowest *worker* is the one that matters among those used.
        let slowest_rate = usable
            .iter()
            .take(opt.count())
            .map(|w| w.elems_per_s)
            .fold(f64::INFINITY, f64::min);
        let compute_s = opt.widest_shard_elems() as f64 / slowest_rate;
        let bytes = opt.moved_bytes(residency);
        let score = compute_s + link.send_time(bytes);
        if score < best.score_s {
            best = Choice {
                option: Some(i),
                score_s: score,
                moved_bytes: bytes,
                why: format!(
                    "{} × {}: {:.6}s compute + {:.6}s moving {} bytes",
                    opt.axis.as_str(),
                    opt.count(),
                    compute_s,
                    link.send_time(bytes),
                    bytes
                ),
                excluded: excluded.clone(),
            };
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, Shape};

    fn conv_graph(
        n: usize,
        c_in: usize,
        c_out: usize,
        hw: usize,
        groups: usize,
    ) -> (Graph, NodeId) {
        let f = DType::F32;
        let mut g = Graph::new("conv");
        let x = g.input("x", Shape::new(&[n, c_in, hw, hw], f));
        let w = g.param("w", Shape::new(&[c_out, c_in / groups, 3, 3], f));
        let out = g.conv2d(x, w, [3, 3], [1, 1], [0, 0], [1, 1], groups);
        (g, out)
    }

    fn matmul_graph(m: usize, k: usize, n: usize) -> (Graph, NodeId) {
        let f = DType::F32;
        let mut g = Graph::new("mm");
        let a = g.input("a", Shape::new(&[m, k], f));
        let b = g.param("b", Shape::new(&[k, n], f));
        let out = g.matmul(a, b, Shape::new(&[m, n], f));
        (g, out)
    }

    #[test]
    fn shard_ranges_cover_the_axis_exactly() {
        for len in [1usize, 2, 7, 8, 100] {
            for count in [1usize, 2, 3, 5, 8] {
                let shards = shard_ranges(len, count);
                assert!(!shards.is_empty());
                assert!(
                    shards.iter().all(|s| s.len > 0),
                    "empty piece: {len}/{count}"
                );
                // Contiguous, in order, and covering: a gap or an overlap would
                // silently drop or double-count samples.
                let mut at = 0;
                for s in &shards {
                    assert_eq!(s.start, at, "gap/overlap at {len}/{count}");
                    at += s.len;
                }
                assert_eq!(at, len, "did not cover {len}/{count}");
                assert!(shards.len() <= count.max(1));
            }
        }
    }

    #[test]
    fn conv_offers_batch_and_channel_splits_with_the_right_operands() {
        let (g, out) = conv_graph(8, 4, 16, 8, 1);
        let opts = shardings(&g, out, 4);
        assert!(!opts.is_empty());

        let batch = opts.iter().find(|o| o.axis == ShardAxis::Batch).unwrap();
        assert_eq!(batch.output_axis, 0);
        assert_eq!(
            batch.sliced,
            vec![SliceSpec {
                operand: 0,
                axis: 0
            }]
        );
        assert_eq!(batch.shared, vec![1], "batch split must replicate weights");

        let chan = opts
            .iter()
            .find(|o| o.axis == ShardAxis::OutChannel)
            .unwrap();
        assert_eq!(chan.output_axis, 1);
        assert_eq!(
            chan.sliced,
            vec![SliceSpec {
                operand: 1,
                axis: 0
            }]
        );
        assert_eq!(
            chan.shared,
            vec![0],
            "channel split must replicate activations"
        );
    }

    /// A grouped conv ties output channels to input channel groups, so the
    /// channel split's "activations are shared" claim is false there. Offering
    /// it anyway would hand a worker the wrong input channels.
    #[test]
    fn a_grouped_conv_offers_only_the_batch_split() {
        let (g, out) = conv_graph(8, 8, 8, 8, 4);
        let opts = shardings(&g, out, 4);
        assert!(!opts.is_empty());
        assert!(
            opts.iter().all(|o| o.axis == ShardAxis::Batch),
            "grouped conv must not offer a channel split"
        );
    }

    #[test]
    fn matmul_offers_row_and_column_splits() {
        let (g, out) = matmul_graph(64, 32, 16);
        let opts = shardings(&g, out, 2);
        let rows = opts.iter().find(|o| o.axis == ShardAxis::Rows).unwrap();
        assert_eq!(rows.shared, vec![1], "row split replicates B");
        let cols = opts.iter().find(|o| o.axis == ShardAxis::Cols).unwrap();
        assert_eq!(cols.shared, vec![0], "column split replicates A");
    }

    /// The IO decision: whichever axis replicates the *smaller* operand wins,
    /// and the crossover is real in both directions.
    #[test]
    fn the_cheaper_axis_is_the_one_replicating_less() {
        // Big activations, small weights → batch split (replicate weights).
        let (g, out) = conv_graph(32, 4, 8, 32, 1);
        let opts = shardings(&g, out, 4);
        let pick = |axis| {
            opts.iter()
                .filter(|o| o.axis == axis && o.count() == 4)
                .map(|o| o.moved_bytes(Residency::default()))
                .next()
                .unwrap()
        };
        assert!(
            pick(ShardAxis::Batch) < pick(ShardAxis::OutChannel),
            "batch split should win when weights are the small operand"
        );

        // Small activations, big weights → channel split (replicate acts).
        let (g2, out2) = conv_graph(2, 64, 256, 4, 1);
        let opts2 = shardings(&g2, out2, 4);
        let pick2 = |axis| {
            opts2
                .iter()
                .filter(|o| o.axis == axis && o.count() == 2)
                .map(|o| o.moved_bytes(Residency::default()))
                .next()
                .unwrap()
        };
        assert!(
            pick2(ShardAxis::OutChannel) < pick2(ShardAxis::Batch),
            "channel split should win when activations are the small operand"
        );
    }

    /// Resident weights are the data-parallel case, and they change the answer:
    /// a batch split then moves only activations and its own output.
    #[test]
    fn residency_removes_the_replication_cost() {
        let (g, out) = conv_graph(8, 4, 16, 8, 1);
        let batch = shardings(&g, out, 4)
            .into_iter()
            .find(|o| o.axis == ShardAxis::Batch && o.count() == 4)
            .unwrap();
        let cold = batch.moved_bytes(Residency::default());
        let warm = batch.moved_bytes(Residency {
            shared_resident: true,
        });
        assert!(warm < cold, "resident weights must cost less, not the same");
        assert_eq!(
            warm,
            (batch.sliced_elems + batch.output_elems) * batch.elem_bytes
        );
    }

    #[test]
    fn a_slow_link_makes_running_it_whole_the_right_answer() {
        let (g, out) = conv_graph(8, 4, 16, 8, 1);
        let opts = shardings(&g, out, 4);
        let workers = vec![
            WorkerCost {
                label: "a".into(),
                elems_per_s: 1e6,
                rankable: true,
            },
            WorkerCost {
                label: "b".into(),
                elems_per_s: 1e6,
                rankable: true,
            },
        ];
        // 10 kB/s: moving the operands costs far more than computing locally.
        let slow = choose(
            &opts,
            &workers,
            LinkCost {
                bytes_per_s: 1.0e4,
                latency_s: 0.01,
            },
            Residency::default(),
            1e6,
        );
        assert!(slow.option.is_none(), "{}", slow.why);

        // 10 GB/s and a local device 20× slower than the workers: split.
        let fast = choose(
            &opts,
            &workers,
            LinkCost {
                bytes_per_s: 1.0e10,
                latency_s: 1e-6,
            },
            Residency::default(),
            5e4,
        );
        assert!(fast.option.is_some(), "{}", fast.why);
        assert!(fast.score_s < slow.score_s);
    }

    /// An uncalibrated cost model is excluded from ranking and said so, rather
    /// than being trusted because it was the only number available.
    #[test]
    fn unrankable_workers_are_excluded_and_reported() {
        let (g, out) = conv_graph(8, 4, 16, 8, 1);
        let opts = shardings(&g, out, 4);
        let workers = vec![
            WorkerCost {
                label: "guessed".into(),
                elems_per_s: 1e12,
                rankable: false,
            },
            WorkerCost {
                label: "measured".into(),
                elems_per_s: 1e6,
                rankable: true,
            },
        ];
        let c = choose(
            &opts,
            &workers,
            LinkCost {
                bytes_per_s: 1e10,
                latency_s: 1e-6,
            },
            Residency::default(),
            1e3,
        );
        assert_eq!(c.excluded, vec!["guessed".to_string()]);

        // With *only* an uncalibrated worker there is nothing to rank.
        let only_guess = vec![WorkerCost {
            label: "guessed".into(),
            elems_per_s: 1e12,
            rankable: false,
        }];
        let c2 = choose(
            &opts,
            &only_guess,
            LinkCost {
                bytes_per_s: 1e10,
                latency_s: 1e-6,
            },
            Residency::default(),
            1e3,
        );
        assert!(c2.option.is_none());
        assert!(c2.why.contains("rankable"), "{}", c2.why);
    }

    #[test]
    fn unshardable_and_dynamic_ops_offer_nothing() {
        let f = DType::F32;
        let mut g = Graph::new("u");
        let x = g.input("x", Shape::new(&[8, 8], f));
        let y = g.activation(rlx_ir::op::Activation::Relu, x, Shape::new(&[8, 8], f));
        assert!(shardings(&g, y, 4).is_empty(), "relu is not sharded here");

        let mut g2 = Graph::new("d");
        let a = g2.input(
            "a",
            Shape::from_dims(&[Dim::Dynamic(0), Dim::Static(32)], f),
        );
        let b = g2.param("b", Shape::new(&[32, 16], f));
        let out = g2.matmul(
            a,
            b,
            Shape::from_dims(&[Dim::Dynamic(0), Dim::Static(16)], f),
        );
        assert!(
            shardings(&g2, out, 4).is_empty(),
            "a dynamic axis cannot be split into static ranges"
        );
    }
}
