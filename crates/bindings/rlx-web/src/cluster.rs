// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Elastic data-parallel worker** — one browser tab's rank in a tab cluster.
//!
//! Each tab runs this in a dedicated Web Worker: it owns a compiled backward
//! graph plus the current parameters, computes gradients over *its* shard of a
//! deterministic dataset, and applies an externally averaged gradient bucket.
//!
//! ## Why the reduce lives outside
//!
//! [`rlx_runtime::dist`]'s ship-graph trainer is deliberately collective-free:
//! `run_train` takes a `reduce: FnMut(&[f32]) -> Vec<f32>` closure and calls it
//! once per step with every gradient flattened into one DDP-style bucket. This
//! worker keeps that exact contract — [`ClusterWorker::grads`] produces the
//! bucket, [`ClusterWorker::apply`] consumes the averaged one — so the browser
//! can plug a `BroadcastChannel` (tabs) or an iroh relay (machines) in where a
//! native rank plugs in `rlx-collectives`.
//!
//! `run_train` itself is *not* reused here, for one reason: its reduce is a
//! barrier that requires every rank to call it the same number of times, so the
//! world size is fixed for the run. Tabs opening and closing mid-run is the
//! point of this demo, so the step loop and the per-step cohort live in the
//! coordinator (`web/cluster-worker.js`) and this type stays step-wise.
//!
//! ## Why averaging gradients is correct here
//!
//! Every vision loss reduces over the batch with [`ReduceOp::Mean`], so a
//! gradient is a per-sample mean. Equal-size shards therefore average with
//! equal weight, and unequal ones need sample-count weighting — which is what
//! the native fabric's `mean_reduce` does too. Guarded by
//! `two_shards_average_to_full_batch` below.

use crate::vision::{self, ModelMeta, VisionModelId};
use rlx_runtime::{BrowserCompiledGraph, BrowserSession, Device};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

/// Size of the deterministic synthetic dataset the cluster shards over.
///
/// Sample `i` is generated from `Philox(i)`, so every tab agrees on what
/// sample `i` *is* without any data crossing the wire — the browser analogue
/// of the native fabric's node-local `DataRef` resolution.
pub const DATASET: u32 = 60_000;

/// One tab's rank: a compiled backward graph + the current parameters.
///
/// The graph is compiled **once** in [`ClusterWorker::new`] and reused for
/// every step. Compiling per step (what the `bench_*` entry points do) would
/// dominate the measurement and make added tabs look free.
///
/// Everything routes through [`BrowserSession`], including the CPU path, so one
/// code path serves `cpu` / `webgpu` / `webgl` and the same graph compiles
/// natively for the tests below. WebGPU readback cannot block the event loop,
/// so `grads()` is async for every backend. It exists only on wasm (the
/// `#[wasm_bindgen]` async expansion needs `wasm_bindgen_futures`, a wasm-only
/// dependency); [`ClusterWorker::grads_sync`] is the host-executable path and
/// carries the native tests.
///
/// State is behind `RefCell` because wasm-bindgen async methods take `&self`
/// and the compiled graph needs `&mut` to run. Re-entry is an error rather
/// than a panic: a second step starting before the first resolves would
/// otherwise take down the worker.
#[wasm_bindgen]
pub struct ClusterWorker {
    meta: ModelMeta,
    exec: RefCell<BrowserCompiledGraph>,
    params: RefCell<Vec<f32>>,
    batch: usize,
    slug: String,
    backend: String,
    /// Why the requested backend was not used, when it was not.
    ///
    /// Without this the rank just reports `cpu` and the obvious question —
    /// "why is it only computing on CPU?" — has no answer anywhere in the UI.
    /// Every conv model hits this: its backward lowers to host steps that
    /// browser WebGPU refuses, so `auto` legitimately lands on CPU.
    fallback: String,
}

#[wasm_bindgen]
impl ClusterWorker {
    /// Compile `slug`'s backward graph for `batch` samples on `backend`
    /// (`auto` / `cpu` / `webgpu` / `webgl`) and initialise parameters from
    /// `seed`.
    ///
    /// Every rank must pass the same `seed`, so a joining tab starts from the
    /// same point it would have reached had it been there from step 0 — and a
    /// mismatch shows up as a loss discontinuity rather than silent divergence.
    ///
    /// `webgpu` requires `init_webgpu()` to have resolved first: device
    /// availability is probed, not assumed, so an unsupported browser fails
    /// here with a message instead of silently running somewhere else.
    ///
    /// `webgl` additionally requires a model that can *train* there — see
    /// [`resolve_backend`].
    #[wasm_bindgen(constructor)]
    pub fn new(
        slug: &str,
        batch: usize,
        seed: u32,
        backend: &str,
    ) -> Result<ClusterWorker, JsValue> {
        let id = VisionModelId::from_slug(slug)
            .ok_or_else(|| JsValue::from_str(&format!("unknown vision model {slug:?}")))?;
        let batch = batch.max(1);
        let meta = vision::model_meta(id);
        let graph = vision::build_train(id, batch);

        let candidates = backend_candidates(backend, id, &rlx_runtime::available_browser_devices())
            .map_err(|e| JsValue::from_str(&e))?;

        let mut device = Device::Cpu;
        let mut exec = None;
        let mut why = String::new();
        let mut skipped: Vec<String> = Vec::new();
        for &candidate in &candidates {
            match BrowserSession::new(candidate).and_then(|s| s.compile(graph.clone())) {
                Ok(compiled) => {
                    if let Some(reason) = host_execution_block(candidate, &compiled) {
                        skipped.push(reason.clone());
                        why = reason;
                        continue;
                    }
                    device = candidate;
                    exec = Some(compiled);
                    break;
                }
                Err(e) => {
                    skipped.push(format!("{}: {e}", device_label(candidate)));
                    why = e.to_string();
                }
            }
        }
        let exec = exec.ok_or_else(|| {
            JsValue::from_str(&format!(
                "no browser backend can train {} ({}): {why}",
                slug,
                candidates
                    .iter()
                    .map(|d| device_label(*d))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;
        let params = vision::init_params(&meta, seed);

        Ok(Self {
            meta,
            exec: RefCell::new(exec),
            params: RefCell::new(params),
            batch,
            slug: slug.to_string(),
            backend: device_label(device).to_string(),
            fallback: skipped.join("; "),
        })
    }

    /// Backend this rank actually resolved to — what the roster displays, so a
    /// cohort mixing WebGPU and CPU tabs is visible rather than assumed.
    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> String {
        self.backend.clone()
    }

    /// Why a faster backend was passed over, or empty when none was.
    #[wasm_bindgen(getter)]
    pub fn fallback(&self) -> String {
        self.fallback.clone()
    }

    /// Model slug this rank is training.
    #[wasm_bindgen(getter)]
    pub fn slug(&self) -> String {
        self.slug.clone()
    }

    /// Samples this rank consumes per step.
    #[wasm_bindgen(getter)]
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Length of the flat gradient bucket (= flat parameter length).
    #[wasm_bindgen(getter)]
    pub fn bucket_len(&self) -> usize {
        self.meta.param_flat_len()
    }

    /// Current parameters, flat, in `param_names()` order — the snapshot the
    /// leader sends a joining tab.
    pub fn params(&self) -> Vec<f32> {
        self.params.borrow().clone()
    }

    /// Adopt a parameter snapshot (a joining tab catching up to the cohort).
    pub fn set_params(&self, flat: &[f32]) -> Result<(), JsValue> {
        let mut params = self.params.borrow_mut();
        if flat.len() != params.len() {
            return Err(JsValue::from_str(&format!(
                "params: expected {} floats, got {}",
                params.len(),
                flat.len()
            )));
        }
        params.copy_from_slice(flat);
        Ok(())
    }

    /// Gradients over this rank's shard: samples `[start, start + batch)` of
    /// the shared synthetic dataset, wrapping at [`DATASET`].
    ///
    /// Returns `[loss, grad_0 .., grad_1 ..]` — the loss first, then the flat
    /// DDP bucket in parameter order, which is what the coordinator averages.
    #[cfg(target_arch = "wasm32")]
    pub async fn grads(&self, start: u32) -> Result<Vec<f32>, JsValue> {
        let (x, labels) = self.shard(start);
        let mut exec = self.borrow_exec()?;
        {
            let params = self.params.borrow();
            vision::apply_browser_params(&mut exec, &self.meta, &params);
        }
        let outs = exec
            .run_async(&[("x", &x), ("labels", &labels), ("d_output", &[1.0])])
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        drop(exec);
        self.bucket(outs)
    }

    /// Synchronous gradients, for backends that can run on the host thread
    /// (CPU, WebGL). Errors on WebGPU, where readback must not block the event
    /// loop — use [`ClusterWorker::grads`] there.
    pub fn grads_sync(&self, start: u32) -> Result<Vec<f32>, JsValue> {
        if self.backend == "webgpu" {
            return Err(JsValue::from_str(
                "webgpu cannot run synchronously — await grads() instead",
            ));
        }
        let (x, labels) = self.shard(start);
        let mut exec = self.borrow_exec()?;
        {
            let params = self.params.borrow();
            vision::apply_browser_params(&mut exec, &self.meta, &params);
        }
        let outs = exec.run(&[("x", &x), ("labels", &labels), ("d_output", &[1.0])]);
        drop(exec);
        self.bucket(outs)
    }

    /// SGD on the cohort-averaged bucket. Every rank applies the same averaged
    /// gradients, so parameters stay identical across tabs without shipping
    /// parameters every step.
    pub fn apply(&self, avg: &[f32], lr: f32) -> Result<(), JsValue> {
        if avg.len() != self.meta.param_flat_len() {
            return Err(JsValue::from_str(&format!(
                "averaged bucket: expected {} floats, got {}",
                self.meta.param_flat_len(),
                avg.len()
            )));
        }
        let mut off = 0;
        let slices: Vec<&[f32]> = self
            .meta
            .params
            .iter()
            .map(|p| {
                let n = p.shape.iter().product::<usize>();
                let s = &avg[off..off + n];
                off += n;
                s
            })
            .collect();
        let mut params = self.params.borrow_mut();
        *params = vision::train_step_params(&self.meta, &params, &slices, lr);
        Ok(())
    }

    /// Loss on this rank's shard without touching the parameters — what the UI
    /// plots, and what a joining tab reports before its first step.
    #[cfg(target_arch = "wasm32")]
    pub async fn loss(&self, start: u32) -> Result<f32, JsValue> {
        Ok(self.grads(start).await?[0])
    }
}

/// Backends to try for (`want`, `model`), best first, given which are live.
///
/// Pure, and separated from device bring-up, because the decision it makes is
/// the one thing about backend selection that can be verified without a
/// browser — and getting it wrong is not a compile error.
///
/// WebGL2 has no conv backward in this stack
/// ([`VisionModelId::webgl_train`]), so a WebGL rank can only train the MLP.
/// That matters more than it looks: `auto` resolves by device priority alone
/// (`WebGpu`, then `OpenGl`, then `Cpu`), so on a browser with WebGL2 but no
/// WebGPU it would otherwise hand a conv model to a backend that cannot
/// produce its gradients — and a rank that contributes wrong buckets to the
/// cohort's average is far worse than one that falls back to CPU.
///
/// So `auto` returns every backend that could work, best first, and the caller
/// walks the list — the remaining reason a backend can fail is only knowable
/// after compiling (see [`host_execution_block`]). An explicit request yields
/// exactly one candidate, so it is refused with a reason rather than silently
/// running somewhere else.
fn backend_candidates(
    want: &str,
    model: VisionModelId,
    available: &[Device],
) -> Result<Vec<Device>, String> {
    let live = |d: Device| d == Device::Cpu || available.contains(&d);
    let trains = |d: Device| d != Device::OpenGl || model.webgl_train();

    match want.to_ascii_lowercase().as_str() {
        "auto" | "" => {
            let mut ds: Vec<Device> = [Device::WebGpu, Device::OpenGl, Device::Cpu]
                .into_iter()
                .filter(|&d| live(d) && trains(d))
                .collect();
            if ds.is_empty() {
                ds.push(Device::Cpu);
            }
            Ok(ds)
        }
        "cpu" => Ok(vec![Device::Cpu]),
        "webgpu" | "gpu" | "wgpu" => {
            if live(Device::WebGpu) {
                Ok(vec![Device::WebGpu])
            } else {
                Err("webgpu is not available — call init_webgpu() first".into())
            }
        }
        "webgl" | "opengl" | "gl" => {
            if !live(Device::OpenGl) {
                Err("webgl is not available in this browser".into())
            } else if !trains(Device::OpenGl) {
                Err(format!(
                    "model {} cannot train on webgl (conv backward is unavailable there); \
                     use mnist-mlp, or another backend",
                    model.slug()
                ))
            } else {
                Ok(vec![Device::OpenGl])
            }
        }
        other => Err(format!(
            "unknown backend {other:?}; use auto, cpu, webgpu or webgl"
        )),
    }
}

/// Why `device` cannot run this compiled graph, if it cannot.
///
/// Browser WebGPU refuses any schedule with host-executed steps, and a conv
/// lowers to `Step::Conv2dHost` there — so every conv model compiles for
/// WebGPU and then fails on *every* `run_async`. Observed in a browser as a
/// rank that joined the cohort and never contributed a bucket, reporting
/// "WebGPU wasm graph contains host-executed ops … use WebGL fallback or CPU"
/// once per step. Asking at setup turns that into one clear fallback.
#[cfg(target_arch = "wasm32")]
fn host_execution_block(device: Device, compiled: &BrowserCompiledGraph) -> Option<String> {
    if matches!(device, Device::WebGpu | Device::Gpu) && compiled.requires_host_execution() {
        // The summary names the steps and counts the sync points they cost, so
        // the fallback message says which way to move rather than listing every
        // op that could ever be hosted.
        let detail = compiled
            .host_execution_summary()
            .unwrap_or_else(|| "host-executed steps".to_string());
        Some(format!(
            "{} cannot run this graph: {detail}",
            device_label(device)
        ))
    } else {
        None
    }
}

/// Native builds have no browser WebGPU path to refuse anything.
#[cfg(not(target_arch = "wasm32"))]
fn host_execution_block(_device: Device, _compiled: &BrowserCompiledGraph) -> Option<String> {
    None
}

/// Backend label for the roster. Kept here rather than reusing `Device`'s
/// `Debug` so the string matches what the UI and `exec::ExecBackend` use.
pub(crate) fn device_label(device: Device) -> &'static str {
    match device {
        Device::WebGpu | Device::Gpu => "webgpu",
        Device::OpenGl => "webgl",
        _ => "cpu",
    }
}

impl ClusterWorker {
    /// Borrow the compiled graph, turning re-entry into an error. A second step
    /// starting before the first resolves would otherwise panic the worker.
    fn borrow_exec(&self) -> Result<std::cell::RefMut<'_, BrowserCompiledGraph>, JsValue> {
        self.exec
            .try_borrow_mut()
            .map_err(|_| JsValue::from_str("a step is already in flight on this rank"))
    }

    /// `[loss, grad_0 .., grad_1 ..]` — the DDP bucket the coordinator averages.
    fn bucket(&self, outs: Vec<Vec<f32>>) -> Result<Vec<f32>, JsValue> {
        let loss = outs
            .first()
            .and_then(|o| o.first().copied())
            .ok_or_else(|| JsValue::from_str("train graph produced no loss"))?;
        let mut flat = Vec::with_capacity(1 + self.meta.param_flat_len());
        flat.push(loss);
        for g in outs.iter().skip(1) {
            flat.extend_from_slice(g);
        }
        if flat.len() != 1 + self.meta.param_flat_len() {
            return Err(JsValue::from_str(&format!(
                "gradient bucket: expected {} floats, got {}",
                self.meta.param_flat_len(),
                flat.len() - 1
            )));
        }
        Ok(flat)
    }

    /// This rank's batch: `x` as `batch × input_len`, plus one label per sample.
    fn shard(&self, start: u32) -> (Vec<f32>, Vec<f32>) {
        let mut x = Vec::with_capacity(self.batch * self.meta.input_len());
        let mut labels = Vec::with_capacity(self.batch);
        for i in 0..self.batch as u32 {
            let index = start.wrapping_add(i) % DATASET;
            let (sx, label) = vision::synthetic_input(&self.meta, index);
            x.extend_from_slice(&sx);
            labels.push(label);
        }
        (x, labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the whole cluster rests on: averaging the gradient buckets
    /// of two half-batch shards equals the gradient of the full batch. True
    /// because each loss is a batch `Mean` — if a model ever reduced with `Sum`
    /// instead, added tabs would silently scale the effective learning rate.
    #[test]
    fn two_shards_average_to_full_batch() {
        for slug in ["mnist-mlp", "mnist-cnn"] {
            let full = ClusterWorker::new(slug, 8, 42, "cpu").unwrap();
            let a = ClusterWorker::new(slug, 4, 42, "cpu").unwrap();
            let b = ClusterWorker::new(slug, 4, 42, "cpu").unwrap();

            let g_full = full.grads_sync(0).unwrap();
            let g_a = a.grads_sync(0).unwrap();
            let g_b = b.grads_sync(4).unwrap();

            assert_eq!(g_full.len(), g_a.len());
            let mut worst = 0.0f32;
            for i in 1..g_full.len() {
                let avg = 0.5 * (g_a[i] + g_b[i]);
                worst = worst.max((avg - g_full[i]).abs());
            }
            assert!(
                worst < 1e-5,
                "{slug}: sharded gradients differ from full batch by {worst}"
            );
        }
    }

    /// Ranks that apply the same averaged bucket stay bit-identical, which is
    /// what lets the cohort ship gradients instead of parameters each step.
    #[test]
    fn ranks_stay_in_sync_without_shipping_params() {
        let a = ClusterWorker::new("mnist-mlp", 4, 7, "cpu").unwrap();
        let b = ClusterWorker::new("mnist-mlp", 4, 7, "cpu").unwrap();
        assert_eq!(a.params(), b.params());

        for step in 0..5u32 {
            let ga = a.grads_sync(step * 8).unwrap();
            let gb = b.grads_sync(step * 8 + 4).unwrap();
            let avg: Vec<f32> = ga[1..]
                .iter()
                .zip(&gb[1..])
                .map(|(x, y)| 0.5 * (x + y))
                .collect();
            a.apply(&avg, 0.05).unwrap();
            b.apply(&avg, 0.05).unwrap();
            assert_eq!(a.params(), b.params(), "diverged at step {step}");
        }
    }

    /// A tab joining mid-run adopts the cohort's parameters and immediately
    /// agrees with it — the elastic-join path.
    #[test]
    fn joining_rank_adopts_cohort_params() {
        let veteran = ClusterWorker::new("mnist-mlp", 4, 7, "cpu").unwrap();
        for step in 0..3u32 {
            let g = veteran.grads_sync(step * 4).unwrap();
            veteran.apply(&g[1..], 0.05).unwrap();
        }
        let joiner = ClusterWorker::new("mnist-mlp", 4, 999, "cpu").unwrap();
        assert_ne!(joiner.params(), veteran.params());
        joiner.set_params(&veteran.params()).unwrap();
        assert_eq!(
            joiner.grads_sync(12).unwrap(),
            veteran.grads_sync(12).unwrap()
        );
    }

    /// Training actually descends, so "more tabs" has something to accelerate.
    #[test]
    fn loss_decreases_over_steps() {
        let w = ClusterWorker::new("mnist-mlp", 8, 3, "cpu").unwrap();
        let first = w.grads_sync(0).unwrap()[0];
        for step in 0..30u32 {
            let g = w.grads_sync(step * 8).unwrap();
            w.apply(&g[1..], 0.1).unwrap();
        }
        let last = w.grads_sync(0).unwrap()[0];
        assert!(last < first, "loss did not decrease: {first} → {last}");
    }

    /// WebGL cannot produce conv gradients, so backend selection has to know
    /// about the *model*, not just which devices are live. None of this is a
    /// compile error, and a rank that contributed wrong buckets would simply
    /// bias the cohort's average.
    #[test]
    fn webgl_is_only_chosen_for_a_model_it_can_train() {
        use Device::{Cpu, OpenGl, WebGpu};
        let gl_only = [OpenGl, Cpu];
        let gpu_and_gl = [WebGpu, OpenGl, Cpu];
        let cpu_only = [Cpu];

        // Explicit request: granted for the MLP, refused with a reason for conv.
        assert_eq!(
            backend_candidates("webgl", VisionModelId::MnistMlp, &gl_only),
            Ok(vec![OpenGl])
        );
        let err = backend_candidates("webgl", VisionModelId::MnistCnn, &gl_only).unwrap_err();
        assert!(err.contains("conv backward"), "unhelpful refusal: {err}");

        // `auto` would otherwise hand a conv model to WebGL on a browser that
        // has no WebGPU, purely because WebGL ranks higher in device priority.
        assert_eq!(
            backend_candidates("auto", VisionModelId::MnistCnn, &gl_only),
            Ok(vec![Cpu])
        );
        assert_eq!(
            backend_candidates("auto", VisionModelId::MnistMlp, &gl_only),
            Ok(vec![OpenGl, Cpu])
        );
        // Ordered best-first, and CPU is always the last resort — the caller
        // walks the list because WebGPU can still refuse after compiling.
        assert_eq!(
            backend_candidates("auto", VisionModelId::MnistCnn, &gpu_and_gl),
            Ok(vec![WebGpu, Cpu])
        );
        assert_eq!(
            backend_candidates("auto", VisionModelId::MnistMlp, &gpu_and_gl),
            Ok(vec![WebGpu, OpenGl, Cpu])
        );

        // Unavailable backends are refused rather than silently downgraded.
        assert!(backend_candidates("webgl", VisionModelId::MnistMlp, &cpu_only).is_err());
        assert!(backend_candidates("webgpu", VisionModelId::MnistCnn, &cpu_only).is_err());
        assert_eq!(
            backend_candidates("cpu", VisionModelId::MnistCnn, &cpu_only),
            Ok(vec![Cpu])
        );
        assert!(backend_candidates("vulkan", VisionModelId::MnistCnn, &gpu_and_gl).is_err());
    }

    /// Bucket length is the contract between wasm and the JS reduce.
    #[test]
    fn bucket_len_matches_param_flat_len() {
        let w = ClusterWorker::new("mnist-cnn", 2, 1, "cpu").unwrap();
        assert_eq!(w.bucket_len(), w.params().len());
        assert_eq!(w.grads_sync(0).unwrap().len(), w.bucket_len() + 1);
    }
}
