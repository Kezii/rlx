// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feed real cost models into [`rlx_compile::shard`]'s chooser.
//!
//! [`rlx_compile::shard`] deliberately knows nothing about this crate — it sits
//! below it, so it takes a plain [`WorkerCost`] instead of a
//! [`BackendCostModel`]. This module is the adapter, and it exists so the rule
//! about *which* numbers may be ranked lives in exactly one place:
//! [`CostCalibration::is_rankable`].

use crate::cost::{BackendCostModel, CostCalibration};
use rlx_compile::shard::WorkerCost;
use rlx_ir::{Dim, Graph, NodeId, Op, OpKind};

/// Fallible counterpart to `Dim::unwrap_static`: a dynamic axis means the cost
/// is not knowable here, and declining beats panicking inside a planner.
fn static_of(dim: &Dim) -> Option<usize> {
    match dim {
        Dim::Static(n) => Some(*n),
        Dim::Dynamic(_) => None,
    }
}

/// Multiply-adds per output element for the ops [`rlx_compile::shard`] can cut.
///
/// This is what turns a backend's GFLOP/s into "output elements per second",
/// which is the rate the shard chooser compares. `None` for an op whose cost
/// per element is not a simple reduction width — better to decline than to
/// invent a number that then gets ranked.
pub fn flops_per_output_elem(graph: &Graph, node: NodeId) -> Option<f64> {
    let n = graph.node(node);
    let operand = |i: usize| n.inputs.get(i).map(|id| &graph.node(*id).shape);
    match n.op.kind() {
        OpKind::MatMul => {
            // Each output element is a length-K dot product: one multiply and
            // one add per step.
            let k = static_of(operand(0)?.dims().last()?)?;
            Some(2.0 * k as f64)
        }
        OpKind::Conv | OpKind::Conv3d => {
            let groups = match &n.op {
                Op::Conv { groups, .. } => (*groups).max(1),
                _ => 1,
            };
            let w = operand(1)?;
            // Weight is [c_out, c_in/groups, k…]: everything past axis 0 is the
            // per-output-element reduction.
            let per: usize = w
                .dims()
                .iter()
                .skip(1)
                .map(static_of)
                .collect::<Option<Vec<_>>>()?
                .iter()
                .product();
            let _ = groups; // already folded into the weight's channel dim
            Some(2.0 * per as f64)
        }
        _ => None,
    }
}

/// A [`WorkerCost`] for running `node` on the device `model` describes.
///
/// The rate comes from the model's `sgemm_gflops` at this op's reduction width,
/// which is the throughput number every backend cost model is obliged to
/// provide. `rankable` comes straight from [`CostCalibration::is_rankable`], so
/// an `Uncalibrated` model is reported and excluded rather than trusted for
/// being the only number on offer.
pub fn worker_cost(
    model: &dyn BackendCostModel,
    graph: &Graph,
    node: NodeId,
) -> Option<WorkerCost> {
    let flops_per_elem = flops_per_output_elem(graph, node)?;
    if flops_per_elem <= 0.0 {
        return None;
    }
    let out = &graph.node(node).shape;
    let elems = out.num_elements()?;
    // Shape the probe like the work: a square-ish GEMM of the same reduction
    // width, so a model that varies with size answers for the right regime.
    let k = (flops_per_elem / 2.0).max(1.0) as usize;
    let m = (elems as f64).sqrt().max(1.0) as usize;
    let gflops = model.sgemm_gflops(m, k, m);
    if !gflops.is_finite() || gflops <= 0.0 {
        return None;
    }
    let calibration = model.calibration();
    Some(WorkerCost {
        label: format!("{:?} ({})", model.device(), calibration.as_str()),
        elems_per_s: gflops * 1e9 / flops_per_elem,
        rankable: calibration.is_rankable(),
    })
}

/// `WorkerCost` for each candidate device, skipping any whose model cannot
/// price this op at all.
pub fn worker_costs(
    models: &[&dyn BackendCostModel],
    graph: &Graph,
    node: NodeId,
) -> Vec<WorkerCost> {
    models
        .iter()
        .filter_map(|m| worker_cost(*m, graph, node))
        .collect()
}

/// Convenience: is anything here worth ranking? Mirrors the device-ranking
/// rule so a caller does not re-derive it.
pub fn any_rankable(costs: &[WorkerCost]) -> bool {
    costs.iter().any(|c| c.rankable)
}

/// Spell out why a calibration state is or is not usable, for a report.
pub fn calibration_note(calibration: CostCalibration) -> &'static str {
    match calibration {
        CostCalibration::Measured => "measured on this machine; comparable",
        CostCalibration::ArchDefault => "arch default; indicative, still ranked",
        CostCalibration::Uncalibrated => "generic guess; excluded from ranking",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, Shape};

    struct Model {
        gflops: f64,
        calibration: CostCalibration,
    }

    impl BackendCostModel for Model {
        fn device(&self) -> rlx_driver::Device {
            rlx_driver::Device::Cpu
        }
        fn calibration(&self) -> CostCalibration {
            self.calibration
        }
        fn sgemm_gflops(&self, _m: usize, _k: usize, _n: usize) -> f64 {
            self.gflops
        }
        fn dispatch_overhead_ns(&self) -> f64 {
            0.0
        }
        fn roundtrip_overhead_ns(&self) -> f64 {
            0.0
        }
        fn memory_bw(&self) -> f64 {
            1e11
        }
        fn num_threads(&self) -> usize {
            1
        }
    }

    fn matmul(m: usize, k: usize, n: usize) -> (Graph, NodeId) {
        let f = DType::F32;
        let mut g = Graph::new("mm");
        let a = g.input("a", Shape::new(&[m, k], f));
        let b = g.param("b", Shape::new(&[k, n], f));
        let out = g.matmul(a, b, Shape::new(&[m, n], f));
        (g, out)
    }

    fn conv(n: usize, c_in: usize, c_out: usize, hw: usize) -> (Graph, NodeId) {
        let f = DType::F32;
        let mut g = Graph::new("conv");
        let x = g.input("x", Shape::new(&[n, c_in, hw, hw], f));
        let w = g.param("w", Shape::new(&[c_out, c_in, 3, 3], f));
        let out = g.conv2d(x, w, [3, 3], [1, 1], [0, 0], [1, 1], 1);
        (g, out)
    }

    #[test]
    fn flops_per_element_follows_the_reduction_width() {
        let (g, out) = matmul(8, 32, 4);
        assert_eq!(flops_per_output_elem(&g, out), Some(64.0)); // 2·K

        let (g2, out2) = conv(2, 3, 5, 8);
        // 2 · (c_in · kh · kw) = 2 · 27
        assert_eq!(flops_per_output_elem(&g2, out2), Some(54.0));
    }

    #[test]
    fn an_op_we_cannot_price_declines_instead_of_guessing() {
        let f = DType::F32;
        let mut g = Graph::new("u");
        let x = g.input("x", Shape::new(&[4, 4], f));
        let y = g.activation(rlx_ir::op::Activation::Relu, x, Shape::new(&[4, 4], f));
        assert_eq!(flops_per_output_elem(&g, y), None);
        let model = Model {
            gflops: 100.0,
            calibration: CostCalibration::Measured,
        };
        assert!(worker_cost(&model, &g, y).is_none());
    }

    #[test]
    fn element_rate_scales_with_gflops_and_inversely_with_reduction_width() {
        let (g, out) = matmul(8, 32, 4);
        let fast = worker_cost(
            &Model {
                gflops: 200.0,
                calibration: CostCalibration::Measured,
            },
            &g,
            out,
        )
        .unwrap();
        let slow = worker_cost(
            &Model {
                gflops: 100.0,
                calibration: CostCalibration::Measured,
            },
            &g,
            out,
        )
        .unwrap();
        assert!((fast.elems_per_s / slow.elems_per_s - 2.0).abs() < 1e-9);

        // Twice the reduction width at the same GFLOP/s = half the elements.
        let (g2, out2) = matmul(8, 64, 4);
        let wide = worker_cost(
            &Model {
                gflops: 100.0,
                calibration: CostCalibration::Measured,
            },
            &g2,
            out2,
        )
        .unwrap();
        assert!((slow.elems_per_s / wide.elems_per_s - 2.0).abs() < 1e-9);
    }

    /// The rule about which numbers may be ranked lives in `CostCalibration`,
    /// and this adapter must carry it rather than re-deciding.
    #[test]
    fn rankability_comes_from_the_calibration_state() {
        let (g, out) = matmul(8, 32, 4);
        for (state, expected) in [
            (CostCalibration::Measured, true),
            (CostCalibration::ArchDefault, true),
            (CostCalibration::Uncalibrated, false),
        ] {
            let c = worker_cost(
                &Model {
                    gflops: 100.0,
                    calibration: state,
                },
                &g,
                out,
            )
            .unwrap();
            assert_eq!(c.rankable, expected, "{state:?}");
            assert_eq!(
                c.rankable,
                state.is_rankable(),
                "{state:?} drifted from the rule"
            );
            assert!(
                c.label.contains(state.as_str()),
                "label should say so: {}",
                c.label
            );
        }
        assert!(!any_rankable(&[WorkerCost {
            label: "guess".into(),
            elems_per_s: 1e12,
            rankable: false,
        }]));
    }
}
