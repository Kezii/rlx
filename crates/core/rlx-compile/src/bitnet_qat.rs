// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **BitNet b1.58 quantization-aware training** as a graph rewrite.
//!
//! [`inject_bitnet_qat`] rewrites every `MatMul` / `Conv` whose weight
//! operand traces back to a leaf (`Param` / `Constant` / `Input`) so the
//! forward pass sees ternary weights — and, for `MatMul`, int8 per-token
//! activations — while the gradient reaches the f32 master weights
//! unchanged:
//!
//! ```text
//!   w_q = w + stop_gradient(ternary(w) − w)       ternary = γ·clamp(round(w/γ),−1,1)
//!   x_q = x + stop_gradient(int8_tok(x) − x)      int8_tok = clamp(round(x·s),−128,127)/s
//! ```
//!
//! That is the BitNet reference STE (`w + (quant(w) − w).detach()`): the
//! backward is exactly identity, independent of `γ`. It is built from
//! existing primitives (`Abs`, `Reduce`, `Round`, `Clamp`, `StopGradient`),
//! so it trains on every backend that runs those — no new opcode.
//!
//! The numerics match [`rlx_ir::ternary`] (the export-side definition), so
//! after training, `rlx_ir::ternary::ternarize` on the master weights gives
//! exactly the codes the forward pass trained against.
//!
//! BitNet keeps the embedding and the LM head in full precision; list those
//! weight names in [`BitNetQatConfig::keep_full_precision`].

use std::collections::HashMap;

use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::ternary::{BITNET_ACT_QMAX, TernaryConfig, TernaryScale};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

/// What [`inject_bitnet_qat`] quantizes.
#[derive(Debug, Clone)]
pub struct BitNetQatConfig {
    /// Weight scale scope + `ε`. Default: BitNet per-tensor absmean.
    pub ternary: TernaryConfig,
    /// Fake-quantize `MatMul` activations to int8 per token (BitNet "A8").
    pub quantize_activations: bool,
    /// Leaf names (exact match) left in full precision — typically the
    /// embedding table and the LM head.
    pub keep_full_precision: Vec<String>,
    /// Treat `Op::Input` leaves as weights. On by default because training
    /// loops commonly feed master weights as inputs; turn off when a
    /// `MatMul` RHS can be a genuine data input.
    pub weights_from_inputs: bool,
}

impl Default for BitNetQatConfig {
    fn default() -> Self {
        Self {
            ternary: TernaryConfig::default(),
            quantize_activations: true,
            keep_full_precision: Vec::new(),
            weights_from_inputs: true,
        }
    }
}

impl BitNetQatConfig {
    pub fn keep_full_precision<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.keep_full_precision = names.into_iter().map(Into::into).collect();
        self
    }
}

/// What the rewrite did.
#[derive(Debug, Clone, Default)]
pub struct BitNetQatReport {
    /// Leaf names whose consumers now see ternary weights. Ternarize these
    /// with the same [`TernaryConfig`] at export.
    pub ternary_weights: Vec<String>,
    /// `MatMul`s whose activation input is now int8 per-token.
    pub quantized_activations: usize,
    /// `(node, reason)` for compute nodes left alone.
    pub skipped: Vec<(String, String)>,
}

/// Rewrite `graph` for BitNet b1.58 QAT. Output indices are preserved.
pub fn inject_bitnet_qat(graph: &Graph, cfg: &BitNetQatConfig) -> (Graph, BitNetQatReport) {
    let mut g = Graph::new(&graph.name);
    let mut report = BitNetQatReport::default();
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    // A weight shared by several consumers (same edge layout) is fake-
    // quantized once.
    let mut wq_cache: HashMap<(NodeId, bool), NodeId> = HashMap::new();

    for node in graph.nodes() {
        let mut inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let is_mm = matches!(node.op, Op::MatMul);
        let is_conv = matches!(node.op, Op::Conv { .. });
        if (is_mm || is_conv) && inputs.len() >= 2 {
            let label = node
                .name
                .clone()
                .unwrap_or_else(|| format!("{:?}#{}", node.op.kind(), node.id.0));
            match weight_leaf(graph, node.inputs[1], cfg) {
                Ok(leaf_name) => {
                    let key = (node.inputs[1], is_conv);
                    let wq = *wq_cache
                        .entry(key)
                        .or_insert_with(|| fake_ternary(&mut g, inputs[1], is_conv, &cfg.ternary));
                    inputs[1] = wq;
                    if !report.ternary_weights.contains(&leaf_name) {
                        report.ternary_weights.push(leaf_name);
                    }
                    if is_mm && cfg.quantize_activations {
                        inputs[0] = fake_int8_per_token(&mut g, inputs[0], cfg.ternary.eps);
                        report.quantized_activations += 1;
                    }
                }
                Err(reason) => report.skipped.push((label, reason)),
            }
        }
        let new_id = g.add_node(node.op.clone(), inputs, node.shape.clone());
        let n = g.node_mut(new_id);
        n.name = node.name.clone();
        n.origin = node.origin.clone();
        id_map.insert(node.id, new_id);
    }
    g.set_outputs(graph.outputs.iter().map(|i| id_map[i]).collect());
    (g, report)
}

/// Follow layout-only ops back to the weight leaf. `Ok(name)` when the
/// operand is a quantizable weight.
fn weight_leaf(graph: &Graph, mut id: NodeId, cfg: &BitNetQatConfig) -> Result<String, String> {
    loop {
        let n = graph.node(id);
        match &n.op {
            Op::Transpose { .. } | Op::Reshape { .. } | Op::Cast { .. } | Op::Expand { .. } => {
                id = n.inputs[0];
            }
            Op::Param { name } => return keep_check(name, cfg),
            Op::Input { name } if cfg.weights_from_inputs => return keep_check(name, cfg),
            Op::Input { name } => {
                return Err(format!(
                    "RHS is input {name:?} (weights_from_inputs = false)"
                ));
            }
            Op::Constant { .. } => {
                let name = n.name.clone().unwrap_or_else(|| format!("const#{}", id.0));
                return keep_check(&name, cfg);
            }
            other => {
                return Err(format!(
                    "RHS is computed ({:?}), not a weight",
                    other.kind()
                ));
            }
        }
    }
}

fn keep_check(name: &str, cfg: &BitNetQatConfig) -> Result<String, String> {
    if cfg.keep_full_precision.iter().any(|k| k == name) {
        Err(format!("{name:?} kept full precision"))
    } else {
        Ok(name.to_string())
    }
}

fn dims(g: &Graph, id: NodeId) -> Vec<usize> {
    let s = g.shape(id);
    (0..s.rank()).map(|i| s.dim(i).unwrap_static()).collect()
}

/// Scalar f32 constant with `like`'s rank and all-ones dims — broadcasts
/// along every axis without a rank change.
fn scalar(g: &mut Graph, v: f32, like: &Shape) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&vec![1; like.rank()], DType::F32),
    )
}

/// `x + stop_gradient(q − x)`.
fn ste(g: &mut Graph, x: NodeId, q: NodeId, shape: &Shape) -> NodeId {
    let d = g.binary(BinaryOp::Sub, q, x, shape.clone());
    let sg = g.add_node(Op::StopGradient, vec![d], shape.clone());
    g.binary(BinaryOp::Add, x, sg, shape.clone())
}

/// `γ·clamp(round(w/γ), −1, 1)` on the edge layout: MatMul RHS `[…, K, N]`
/// (output channel = last axis) or Conv `[O, I, …]` (output channel = 0).
fn fake_ternary(g: &mut Graph, w: NodeId, is_conv: bool, cfg: &TernaryConfig) -> NodeId {
    let wshape = g.shape(w).clone().with_dtype(DType::F32);
    let d = dims(g, w);
    let rank = d.len();
    let axes: Vec<usize> = match cfg.scale {
        TernaryScale::PerTensor => (0..rank).collect(),
        TernaryScale::PerChannel if is_conv => (1..rank).collect(),
        TernaryScale::PerChannel => vec![rank.saturating_sub(2)],
    };
    let mut rd = d.clone();
    for &a in &axes {
        rd[a] = 1;
    }
    let rshape = Shape::new(&rd, DType::F32);

    let abs = g.activation(Activation::Abs, w, wshape.clone());
    let mean = g.reduce(abs, ReduceOp::Mean, axes, true, rshape.clone());
    let eps = scalar(g, cfg.eps, &rshape);
    let gamma = g.binary(BinaryOp::Max, mean, eps, rshape.clone());
    let n = g.binary(BinaryOp::Div, w, gamma, wshape.clone());
    let r = g.activation(Activation::Round, n, wshape.clone());
    let c = g.add_node(
        Op::Clamp {
            min: -1.0,
            max: 1.0,
        },
        vec![r],
        wshape.clone(),
    );
    let q = g.binary(BinaryOp::Mul, c, gamma, wshape.clone());
    ste(g, w, q, &wshape)
}

/// `clamp(round(x·s), −128, 127)/s`, `s = 127/max(max|x|, ε)` per row of
/// the last axis.
fn fake_int8_per_token(g: &mut Graph, x: NodeId, eps: f32) -> NodeId {
    let xshape = g.shape(x).clone();
    let d = dims(g, x);
    if d.is_empty() {
        return x;
    }
    let last = d.len() - 1;
    let mut rd = d.clone();
    rd[last] = 1;
    let rshape = Shape::new(&rd, DType::F32);

    let abs = g.activation(Activation::Abs, x, xshape.clone());
    let amax = g.reduce(abs, ReduceOp::Max, vec![last], true, rshape.clone());
    let e = scalar(g, eps, &rshape);
    let amax = g.binary(BinaryOp::Max, amax, e, rshape.clone());
    let qmax = scalar(g, BITNET_ACT_QMAX, &rshape);
    let s = g.binary(BinaryOp::Div, qmax, amax, rshape.clone());
    let up = g.binary(BinaryOp::Mul, x, s, xshape.clone());
    let r = g.activation(Activation::Round, up, xshape.clone());
    let c = g.add_node(
        Op::Clamp {
            min: -128.0,
            max: 127.0,
        },
        vec![r],
        xshape.clone(),
    );
    let q = g.binary(BinaryOp::Div, c, s, xshape.clone());
    ste(g, x, q, &xshape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mlp() -> Graph {
        let mut g = Graph::new("mlp");
        let x = g.input("x", Shape::new(&[2, 4], DType::F32));
        let w1 = g.param("w1", Shape::new(&[4, 3], DType::F32));
        let head = g.param("head", Shape::new(&[3, 2], DType::F32));
        let h = g.matmul(x, w1, Shape::new(&[2, 3], DType::F32));
        let y = g.matmul(h, head, Shape::new(&[2, 2], DType::F32));
        g.set_outputs(vec![y]);
        g
    }

    #[test]
    fn rewrites_weights_and_activations() {
        let (q, rep) = inject_bitnet_qat(&mlp(), &BitNetQatConfig::default());
        assert_eq!(rep.ternary_weights, vec!["w1", "head"]);
        assert_eq!(rep.quantized_activations, 2);
        let n_sg = q
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::StopGradient))
            .count();
        assert_eq!(n_sg, 4);
        assert_eq!(q.outputs.len(), 1);
        assert_eq!(dims(&q, q.outputs[0]), vec![2, 2]);
    }

    #[test]
    fn keeps_named_head_full_precision() {
        let cfg = BitNetQatConfig::default().keep_full_precision(["head"]);
        let (_, rep) = inject_bitnet_qat(&mlp(), &cfg);
        assert_eq!(rep.ternary_weights, vec!["w1"]);
        assert_eq!(rep.quantized_activations, 1);
        assert_eq!(rep.skipped.len(), 1);
    }

    #[test]
    fn computed_rhs_is_not_a_weight() {
        let mut g = Graph::new("attn");
        let q = g.input("q", Shape::new(&[2, 4], DType::F32));
        let k = g.input("k", Shape::new(&[2, 4], DType::F32));
        let kt = g.add_node(
            Op::Activation(Activation::Relu),
            vec![k],
            Shape::new(&[2, 4], DType::F32),
        );
        let kt = g.add_node(
            Op::Transpose { perm: vec![1, 0] },
            vec![kt],
            Shape::new(&[4, 2], DType::F32),
        );
        let s = g.matmul(q, kt, Shape::new(&[2, 2], DType::F32));
        g.set_outputs(vec![s]);
        let (_, rep) = inject_bitnet_qat(&g, &BitNetQatConfig::default());
        assert!(rep.ternary_weights.is_empty());
        assert_eq!(rep.skipped.len(), 1);
    }
}
