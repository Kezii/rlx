// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Run a graph as if every value were stored in a narrower float format.
//!
//! [`emulate_float_format`] inserts `Op::RoundToFormat` after the weights and
//! after every compute op that produces a float tensor, so every value a
//! consumer reads is representable in the format. Arithmetic inside an op still
//! accumulates in `f32`, as FP8/FP4 tensor cores do.
//!
//! Skipped on purpose:
//!
//! - graph inputs and constants (unless [`EmulateOptions::round_inputs`] /
//!   [`EmulateOptions::round_constants`]): they are host data crossing the
//!   boundary — attention masks holding `-inf` / `-1e9`, KV caches, positions.
//!   Rounded, a block-scaled `-inf` mask becomes `-6` in `f4` and attention
//!   leaks; values *computed* from weights (embeddings, K/V) are still rounded;
//! - pure data movement (reshape, transpose, slicing, concat, gather …): its
//!   values were already rounded where they were produced;
//! - non-float tensors (indices, masks);
//! - graph outputs, which keep the unrounded `f32` value so callers (an LM's
//!   sampler, a forecaster's quantile map) see full precision;
//! - anything matched by [`EmulateOptions::keep`].
//!
//! The inserted op has a straight-through gradient, so the same rewrite turns a
//! forward graph into a quantization-aware-training graph.

use std::collections::HashMap;

use rlx_ir::op::{Activation, ReduceOp};
use rlx_ir::*;

/// What [`emulate_float_format`] leaves alone.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EmulateOptions {
    /// Also round graph inputs (off by default; see the module docs).
    pub round_inputs: bool,
    /// Also round `Op::Constant`s (off by default; see the module docs).
    pub round_constants: bool,
    /// Also round values consumed only by a softmax (attention scores and the
    /// mask added to them). Off by default: low-precision attention kernels
    /// keep `Q·Kᵀ`, the mask and the softmax in f32, and rounding them is
    /// destructive under block scaling — a block holding a `-1e9` mask entry
    /// gets a scale near `1e9`, which flushes every real score in it to zero.
    pub round_softmax_inputs: bool,
    /// Round weights (`Op::Param`) inside the graph on every run. Off by
    /// default: the runtime rounds them once as they are uploaded
    /// (`CompiledGraph::set_param`), which is exact and does not re-read every
    /// weight per call. Training graphs whose weights never pass through the
    /// host (an in-graph optimizer) need it on.
    pub params_in_graph: bool,
    /// Weights (`Op::Param`) are already rounded on the host; don't round
    /// them again on every run. Inference callers that round once at load
    /// should set this — rounding in-graph re-reads every weight per call.
    pub params_prerounded: bool,
    /// `Op::Param` names whose values stay in `f32` (and whose consumers'
    /// outputs are still rounded). Matched by substring.
    pub keep_params: Vec<String>,
    /// Op kinds whose outputs are not rounded.
    pub keep_ops: Vec<OpKind>,
}

impl EmulateOptions {
    fn keeps_param(&self, name: &str) -> bool {
        self.keep_params.iter().any(|k| name.contains(k.as_str()))
    }
}

/// Ops that only move or select values already in the format.
fn is_data_movement(op: &Op) -> bool {
    matches!(
        op.kind(),
        OpKind::Reshape
            | OpKind::Transpose
            | OpKind::Narrow
            | OpKind::Concat
            | OpKind::Gather
            | OpKind::Expand
            | OpKind::StopGradient
            | OpKind::RoundToFormat
    )
}

/// Insert `RoundToFormat` after every float-producing node of `graph`.
pub fn emulate_float_format(
    graph: Graph,
    format: FloatFormat,
    scaling: FormatScaling,
    opts: &EmulateOptions,
) -> Graph {
    let softmax_feed = if opts.round_softmax_inputs {
        std::collections::HashSet::new()
    } else {
        softmax_feeds(&graph)
    };
    let mut out = Graph::new(&graph.name);
    // Old id → (id in `out`, whether consumers read a rounded copy).
    let mut map: HashMap<NodeId, (NodeId, bool)> = HashMap::new();
    // Rounded copies, made on first use so a value only the caller reads
    // (a graph output) never gets one.
    let mut rounded: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        let inputs: Vec<NodeId> = node
            .inputs
            .iter()
            .map(|i| {
                let (raw, round) = map[i];
                if round {
                    *rounded
                        .entry(raw)
                        .or_insert_with(|| round_node(&mut out, raw, format, scaling))
                } else {
                    raw
                }
            })
            .collect();
        let raw = out.add_node(node.op.clone(), inputs, node.shape.clone());
        let skip = !node.shape.dtype().is_float()
            || is_data_movement(&node.op)
            || softmax_feed.contains(&node.id)
            || opts.keep_ops.contains(&node.op.kind())
            || match &node.op {
                Op::Param { name } => {
                    !opts.params_in_graph || opts.params_prerounded || opts.keeps_param(name)
                }
                Op::Input { .. } => !opts.round_inputs,
                Op::Constant { .. } => !opts.round_constants,
                _ => false,
            };
        map.insert(node.id, (raw, !skip));
    }

    let outputs: Vec<NodeId> = graph.outputs.iter().map(|o| map[o].0).collect();
    out.set_outputs(outputs);
    out
}

/// Nodes whose every consumer is a `Softmax`, or another such node reached
/// through an elementwise or movement op (`scores · scale + mask → softmax`).
fn softmax_feeds(graph: &Graph) -> std::collections::HashSet<NodeId> {
    let mut consumers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for n in graph.nodes() {
        for &i in &n.inputs {
            consumers.entry(i).or_default().push(n.id);
        }
    }
    let outputs: std::collections::HashSet<NodeId> = graph.outputs.iter().copied().collect();
    let mut feeds = std::collections::HashSet::new();
    for n in graph.nodes().iter().rev() {
        let Some(users) = consumers.get(&n.id) else {
            continue;
        };
        if outputs.contains(&n.id) {
            continue;
        }
        let all_feed = users.iter().all(|&u| {
            let op = &graph.node(u).op;
            matches!(op, Op::Softmax { .. })
                || (feeds.contains(&u)
                    && (matches!(op, Op::Binary(_) | Op::Activation(_)) || is_data_movement(op)))
        });
        if all_feed {
            feeds.insert(n.id);
        }
    }
    feeds
}

/// `RoundToFormat(x)`, adding the tensor-wide max magnitude for per-tensor scaling.
pub fn round_node(g: &mut Graph, x: NodeId, format: FloatFormat, scaling: FormatScaling) -> NodeId {
    let shape = g.shape(x).clone();
    let mut inputs = vec![x];
    if scaling == FormatScaling::PerTensor {
        let abs = g.activation(Activation::Abs, x, shape.clone());
        let axes: Vec<usize> = (0..shape.rank()).collect();
        let rshape = shape::reduce_shape(&shape, &axes, false).expect("reduce shape");
        inputs.push(g.reduce(abs, ReduceOp::Max, axes, false, rshape));
    }
    g.add_node(Op::RoundToFormat { format, scaling }, inputs, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_graph() -> Graph {
        let mut g = Graph::new("t");
        let x = g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            Shape::new(&[2, 4], DType::F32),
        );
        let w = g.add_node(
            Op::Param { name: "w".into() },
            vec![],
            Shape::new(&[4, 3], DType::F32),
        );
        let y = g.add_node(Op::MatMul, vec![x, w], Shape::new(&[2, 3], DType::F32));
        let t = g.add_node(
            Op::Transpose { perm: vec![1, 0] },
            vec![y],
            Shape::new(&[3, 2], DType::F32),
        );
        let z = g.activation(Activation::Silu, t, Shape::new(&[3, 2], DType::F32));
        g.set_outputs(vec![z]);
        g
    }

    fn count_rounds(g: &Graph) -> usize {
        g.nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::RoundToFormat { .. }))
            .count()
    }

    #[test]
    fn rounds_producers_not_movement_or_outputs() {
        let g = emulate_float_format(
            tiny_graph(),
            FloatFormat::F16,
            FormatScaling::None,
            &EmulateOptions::default(),
        );
        // Only the matmul: weights are rounded on upload by default, inputs
        // stay f32, the transpose is movement, the silu is the output.
        assert_eq!(count_rounds(&g), 1);
        let with_inputs = emulate_float_format(
            tiny_graph(),
            FloatFormat::F16,
            FormatScaling::None,
            &EmulateOptions {
                round_inputs: true,
                params_in_graph: true,
                ..Default::default()
            },
        );
        assert_eq!(count_rounds(&with_inputs), 3);
        let out = g.outputs[0];
        assert!(matches!(g.node(out).op, Op::Activation(Activation::Silu)));
    }

    #[test]
    fn softmax_inputs_stay_f32() {
        let mut g = Graph::new("attn");
        let s = Shape::new(&[2, 4], DType::F32);
        let x = g.input("x", s.clone());
        let mask = g.input("mask", s.clone());
        let w = g.add_node(
            Op::Param { name: "w".into() },
            vec![],
            Shape::new(&[4, 4], DType::F32),
        );
        let scores = g.add_node(Op::MatMul, vec![x, w], s.clone());
        let masked = g.binary(rlx_ir::op::BinaryOp::Add, scores, mask, s.clone());
        let p = g.add_node(Op::Softmax { axis: -1 }, vec![masked], s.clone());
        let y = g.add_node(Op::MatMul, vec![p, w], s.clone());
        g.set_outputs(vec![y]);
        let out = emulate_float_format(
            g.clone(),
            FloatFormat::F16,
            FormatScaling::None,
            &EmulateOptions::default(),
        );
        // Only the softmax output is rounded: scores and the masked scores feed it.
        assert_eq!(count_rounds(&out), 1);
        let all = emulate_float_format(
            g,
            FloatFormat::F16,
            FormatScaling::None,
            &EmulateOptions {
                round_softmax_inputs: true,
                ..Default::default()
            },
        );
        assert_eq!(count_rounds(&all), 3);
    }

    #[test]
    fn prerounded_params_and_per_tensor_scale() {
        let opts = EmulateOptions {
            params_prerounded: true,
            ..Default::default()
        };
        let f4 = FloatFormat::parse("f4").unwrap();
        let g = emulate_float_format(tiny_graph(), f4, FormatScaling::PerTensor, &opts);
        assert_eq!(count_rounds(&g), 1);
        for n in g.nodes() {
            if let Op::RoundToFormat { .. } = n.op {
                assert_eq!(n.inputs.len(), 2, "per-tensor scaling carries amax");
            }
        }
    }
}
