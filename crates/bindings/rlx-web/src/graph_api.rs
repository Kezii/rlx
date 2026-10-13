// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Build an RLX graph **from JavaScript**.
//!
//! Every other JS entry point in this crate hands back a model that was
//! defined in Rust — `mlp_forward` has its topology baked in, `VisionBench`
//! takes a slug. So the browser could *run* RLX but not *describe* anything to
//! it, while Rust and `pyrlx` both build graphs op by op. This closes that gap
//! with the same three steps the other two languages use: build a graph, let
//! autodiff rewrite it into a backward graph, compile and run.
//!
//! Inputs are fed by name rather than passed to `run`, because wasm-bindgen has
//! no clean way to marshal a JS map of `Float32Array`s. Outputs come back as
//! one flat array plus their sizes, which the caller slices — the same shape
//! the gradient bucket already uses elsewhere in this crate.

use rlx_ir::op::{Activation, BinaryOp, Op, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_runtime::{BrowserCompiledGraph, BrowserSession, Device};
use std::cell::RefCell;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

fn shape_of(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

/// A graph under construction. Node ids are plain integers on the JS side.
#[wasm_bindgen]
pub struct WebGraph {
    inner: Graph,
}

#[wasm_bindgen]
impl WebGraph {
    #[wasm_bindgen(constructor)]
    pub fn new(name: &str) -> WebGraph {
        WebGraph {
            inner: Graph::new(name),
        }
    }

    /// A tensor fed at run time (`feed`).
    pub fn input(&mut self, name: &str, dims: Vec<usize>) -> u32 {
        self.inner.input(name, shape_of(&dims)).0
    }

    /// A trainable tensor, set with `set_param` and differentiated with respect
    /// to when passed to [`compile_backward`].
    pub fn param(&mut self, name: &str, dims: Vec<usize>) -> u32 {
        self.inner.param(name, shape_of(&dims)).0
    }

    pub fn matmul(&mut self, a: u32, b: u32, dims: Vec<usize>) -> u32 {
        self.inner.matmul(NodeId(a), NodeId(b), shape_of(&dims)).0
    }

    pub fn add(&mut self, a: u32, b: u32, dims: Vec<usize>) -> u32 {
        self.inner
            .binary(BinaryOp::Add, NodeId(a), NodeId(b), shape_of(&dims))
            .0
    }

    pub fn mul(&mut self, a: u32, b: u32, dims: Vec<usize>) -> u32 {
        self.inner
            .binary(BinaryOp::Mul, NodeId(a), NodeId(b), shape_of(&dims))
            .0
    }

    pub fn relu(&mut self, x: u32, dims: Vec<usize>) -> u32 {
        self.inner
            .activation(Activation::Relu, NodeId(x), shape_of(&dims))
            .0
    }

    pub fn gelu(&mut self, x: u32, dims: Vec<usize>) -> u32 {
        self.inner
            .activation(Activation::Gelu, NodeId(x), shape_of(&dims))
            .0
    }

    /// Per-sample softmax cross-entropy. `labels` holds integer class indices
    /// as f32, one per row — the same convention the vision models use.
    pub fn softmax_cross_entropy(&mut self, logits: u32, labels: u32) -> u32 {
        self.inner
            .softmax_cross_entropy_with_logits(NodeId(logits), NodeId(labels))
            .0
    }

    /// Mean over `axis`, dropping it. A scalar loss is `mean(per_sample, 0)`.
    pub fn mean(&mut self, x: u32, axis: usize, dims: Vec<usize>) -> u32 {
        self.inner
            .add_node(
                Op::Reduce {
                    op: ReduceOp::Mean,
                    axes: vec![axis],
                    keep_dim: false,
                },
                vec![NodeId(x)],
                shape_of(&dims),
            )
            .0
    }

    pub fn set_outputs(&mut self, ids: Vec<u32>) {
        self.inner
            .set_outputs(ids.into_iter().map(NodeId).collect());
    }

    /// Node count — a cheap way to confirm from JS that the graph was built.
    #[wasm_bindgen(getter)]
    pub fn nodes(&self) -> usize {
        self.inner.nodes().len()
    }
}

/// A compiled graph, ready to feed and run.
#[wasm_bindgen]
pub struct WebModel {
    exec: RefCell<BrowserCompiledGraph>,
    feeds: RefCell<HashMap<String, Vec<f32>>>,
    sizes: Vec<usize>,
    backend: String,
}

/// Backend name → device, or `None` for "pick the best available".
///
/// Pure, and separate from session creation, because constructing a `JsValue`
/// **aborts** outside wasm — so an error path that returns one cannot be tested
/// natively at all. Keeping the parse in plain `Result<_, String>` means the
/// refusal and its wording stay checkable; only the conversion sits at the
/// boundary.
fn parse_backend_name(backend: &str) -> Result<Option<Device>, String> {
    match backend.to_ascii_lowercase().as_str() {
        "auto" | "" => Ok(None),
        "cpu" => Ok(Some(Device::Cpu)),
        "webgpu" | "gpu" | "wgpu" => Ok(Some(Device::WebGpu)),
        "webgl" | "opengl" | "gl" => Ok(Some(Device::OpenGl)),
        other => Err(format!(
            "unknown backend {other:?}; use auto, cpu, webgpu or webgl"
        )),
    }
}

fn resolve(backend: &str) -> Result<BrowserSession, String> {
    let session = match parse_backend_name(backend)? {
        None => BrowserSession::new_preferred(),
        Some(device) => BrowserSession::new(device),
    };
    session.map_err(|e| e.to_string())
}

fn build(graph: Graph, backend: &str) -> Result<WebModel, String> {
    let sizes = graph
        .outputs
        .iter()
        .map(|id| graph.node(*id).shape.num_elements().unwrap_or(0))
        .collect();
    let session = resolve(backend)?;
    let label = crate::cluster::device_label(session.device()).to_string();
    let exec = session.compile(graph).map_err(|e| e.to_string())?;
    Ok(WebModel {
        exec: RefCell::new(exec),
        feeds: RefCell::new(HashMap::new()),
        sizes,
        backend: label,
    })
}

/// Compile the forward graph as written.
#[wasm_bindgen]
pub fn compile_forward(graph: &WebGraph, backend: &str) -> Result<WebModel, JsValue> {
    build(
        rlx_opt::legalize_broadcast::run(graph.inner.clone()),
        backend,
    )
    .map_err(|e| JsValue::from_str(&e))
}

/// Compile the **backward** graph: autodiff rewrites it so the outputs become
/// `[loss, grad_0, grad_1, …]`, one gradient per id in `wrt`, and a seed input
/// named `d_output` appears for `∂loss/∂loss`.
#[wasm_bindgen]
pub fn compile_backward(
    graph: &WebGraph,
    wrt: Vec<u32>,
    backend: &str,
) -> Result<WebModel, JsValue> {
    let params: Vec<NodeId> = wrt.into_iter().map(NodeId).collect();
    let bwd = rlx_autodiff::grad_with_loss(&graph.inner, &params);
    build(rlx_opt::legalize_broadcast::run(bwd), backend).map_err(|e| JsValue::from_str(&e))
}

#[wasm_bindgen]
impl WebModel {
    /// Backend this model actually compiled for.
    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> String {
        self.backend.clone()
    }

    /// Element count of each output, so the caller can slice `run`'s result.
    pub fn output_sizes(&self) -> Vec<usize> {
        self.sizes.clone()
    }

    pub fn set_param(&self, name: &str, values: &[f32]) {
        self.exec.borrow_mut().set_param(name, values);
    }

    /// Stage an input. Values are kept until overwritten, so a training loop
    /// feeds the constants once and only re-feeds what changes.
    pub fn feed(&self, name: &str, values: &[f32]) {
        self.feeds
            .borrow_mut()
            .insert(name.to_string(), values.to_vec());
    }

    /// Run with whatever has been fed. Returns every output concatenated; use
    /// [`WebModel::output_sizes`] to split it.
    pub fn run(&self) -> Result<Vec<f32>, JsValue> {
        let feeds = self.feeds.borrow();
        let inputs: Vec<(&str, &[f32])> = feeds
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect();
        let outs = self.exec.borrow_mut().run(&inputs);
        Ok(outs.into_iter().flatten().collect())
    }

    /// Async sibling for WebGPU, where readback cannot block the event loop.
    #[cfg(target_arch = "wasm32")]
    pub async fn run_async(&self) -> Result<Vec<f32>, JsValue> {
        let feeds = self.feeds.borrow().clone();
        let inputs: Vec<(&str, &[f32])> = feeds
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect();
        let outs = self
            .exec
            .borrow_mut()
            .run_async(&inputs)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(outs.into_iter().flatten().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the page's example model: 784 → 128 → 10 with softmax CE.
    fn mlp(batch: usize) -> (WebGraph, Vec<u32>) {
        let mut g = WebGraph::new("mlp");
        let x = g.input("x", vec![batch, 784]);
        let labels = g.input("labels", vec![batch]);
        let w1 = g.param("w1", vec![784, 128]);
        let b1 = g.param("b1", vec![128]);
        let w2 = g.param("w2", vec![128, 10]);
        let b2 = g.param("b2", vec![10]);

        let h = g.matmul(x, w1, vec![batch, 128]);
        let h = g.add(h, b1, vec![batch, 128]);
        let h = g.relu(h, vec![batch, 128]);
        let o = g.matmul(h, w2, vec![batch, 10]);
        let logits = g.add(o, b2, vec![batch, 10]);
        let per = g.softmax_cross_entropy(logits, labels);
        let loss = g.mean(per, 0, vec![]);
        g.set_outputs(vec![loss]);
        (g, vec![w1, b1, w2, b2])
    }

    fn ramp(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.01 + seed).sin() * 0.1)
            .collect()
    }

    #[test]
    fn a_graph_described_from_js_compiles_and_runs() {
        let (g, _) = mlp(4);
        assert!(g.nodes() > 8, "graph looks empty: {} nodes", g.nodes());

        let model = compile_forward(&g, "cpu").expect("compile");
        assert_eq!(model.output_sizes(), vec![1], "loss is a scalar");

        model.set_param("w1", &ramp(784 * 128, 0.1));
        model.set_param("b1", &[0.0; 128]);
        model.set_param("w2", &ramp(128 * 10, 0.2));
        model.set_param("b2", &[0.0; 10]);
        model.feed("x", &ramp(4 * 784, 0.3));
        model.feed("labels", &[0.0, 1.0, 2.0, 3.0]);

        let out = model.run().expect("run");
        assert_eq!(out.len(), 1);
        assert!(out[0].is_finite() && out[0] > 0.0, "loss {}", out[0]);
    }

    /// The backward graph must expose one gradient per requested parameter,
    /// each the right size — that is the contract the JS example relies on.
    #[test]
    fn the_backward_graph_returns_a_gradient_per_parameter() {
        let (g, params) = mlp(4);
        let model = compile_backward(&g, params, "cpu").expect("compile backward");
        assert_eq!(
            model.output_sizes(),
            vec![1, 784 * 128, 128, 128 * 10, 10],
            "outputs should be [loss, dW1, db1, dW2, db2]"
        );
    }

    /// And training with them actually descends.
    #[test]
    fn a_js_described_model_trains() {
        let (g, params) = mlp(4);
        let model = compile_backward(&g, params, "cpu").expect("compile");

        let mut w1 = ramp(784 * 128, 0.1);
        let mut b1 = vec![0.0; 128];
        let mut w2 = ramp(128 * 10, 0.2);
        let mut b2 = vec![0.0; 10];
        model.feed("x", &ramp(4 * 784, 0.3));
        model.feed("labels", &[0.0, 1.0, 2.0, 3.0]);
        model.feed("d_output", &[1.0]);

        let mut first = f32::NAN;
        let mut last = f32::NAN;
        for step in 0..25 {
            model.set_param("w1", &w1);
            model.set_param("b1", &b1);
            model.set_param("w2", &w2);
            model.set_param("b2", &b2);
            let out = model.run().expect("run");
            let sizes = model.output_sizes();

            let loss = out[0];
            if step == 0 {
                first = loss;
            }
            last = loss;

            let mut at = sizes[0];
            for (values, size) in [
                (&mut w1, sizes[1]),
                (&mut b1, sizes[2]),
                (&mut w2, sizes[3]),
                (&mut b2, sizes[4]),
            ] {
                for i in 0..size {
                    values[i] -= 0.1 * out[at + i];
                }
                at += size;
            }
        }
        assert!(last < first, "loss did not fall: {first} → {last}");
    }

    /// An unknown backend is refused by name, with the valid ones listed.
    #[test]
    fn an_unknown_backend_is_refused_by_name() {
        let err = parse_backend_name("quantum").unwrap_err();
        assert!(err.contains("unknown backend"), "unhelpful: {err}");
        assert!(err.contains("webgpu"), "should list the options: {err}");

        assert_eq!(parse_backend_name("cpu"), Ok(Some(Device::Cpu)));
        assert_eq!(parse_backend_name("WebGPU"), Ok(Some(Device::WebGpu)));
        assert_eq!(parse_backend_name("gl"), Ok(Some(Device::OpenGl)));
        assert_eq!(
            parse_backend_name("auto"),
            Ok(None),
            "auto means 'pick one'"
        );
        assert_eq!(parse_backend_name(""), Ok(None));
    }
}
