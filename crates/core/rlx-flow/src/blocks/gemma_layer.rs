// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Gemma / Gemma 2 decoder blocks for tier-0 [`ModelFlow`] recipes.

use std::sync::{Arc, Mutex};

use super::{
    GeGluStage, GemmaDecodeLayerSpec, GemmaDecodeLayerStage, GemmaKvTapStage, GemmaRmsNormStage,
    HeadNormSpec, LayerScaleStage, SelfAttnPrefillSpec,
};
use crate::layer::LayerStack;
use crate::stage::FlowStage;
use rlx_ir::op::MaskKind;

/// Per-architecture layer recipe (norm placement + FFN style).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmaLayerStyle {
    Gemma,
    Gemma2,
    Gemma3,
    Gemma4,
}

impl GemmaLayerStyle {
    /// Gemma 2/3/4 normalize the attention output (`post_attention_layernorm`)
    /// and the FFN output, and use `pre_feedforward_layernorm` before the FFN;
    /// Gemma 1 uses `post_attention_layernorm` as its pre-FFN norm.
    pub fn sandwich_norms(self) -> bool {
        self != GemmaLayerStyle::Gemma
    }

    /// Per-head Q/K norms (Gemma 3/4), plus the weightless V norm (Gemma 4).
    pub fn head_norm(self, prefix: &str, eps: f32) -> Option<HeadNormSpec> {
        match self {
            GemmaLayerStyle::Gemma3 => Some(HeadNormSpec::gemma(prefix, false, eps)),
            GemmaLayerStyle::Gemma4 => Some(HeadNormSpec::gemma(prefix, true, eps)),
            _ => None,
        }
    }
}

/// Build prefill self-attention spec for one layer. `n_rot` controls
/// the rotary slice — pass `head_dim` for plain RoPE, or
/// `(head_dim * partial_rotary_factor)` for Gemma 4 full-attention
/// p-RoPE.
pub fn gemma_attn_spec(
    layer: usize,
    num_heads: usize,
    head_dim: usize,
    num_kv_heads: usize,
    n_rot: usize,
    mask: MaskKind,
    score_scale: Option<f32>,
    attn_logit_softcap: Option<f32>,
) -> SelfAttnPrefillSpec {
    let prefix = format!("model.layers.{layer}");
    SelfAttnPrefillSpec {
        q_key: format!("{prefix}.self_attn.q_proj.weight"),
        k_key: format!("{prefix}.self_attn.k_proj.weight"),
        v_key: format!("{prefix}.self_attn.v_proj.weight"),
        num_heads,
        head_dim,
        num_kv_heads,
        n_rot,
        rope_table: None,
        k_eq_v: false,
        mask,
        score_scale,
        attn_logit_softcap,
        // Gemma uses HF NeoX rotate-half rope.
        rope_style: rlx_ir::op::RopeStyle::NeoX,
        // Set from the layer style by `gemma_prefill_layer_composed`.
        head_norm: None,
        bias_input: None,
    }
}

/// Sliding-window mask for Gemma 2 local-attention layers.
pub fn gemma2_layer_mask(_layer: usize, window: usize) -> MaskKind {
    MaskKind::SlidingWindow(window)
}

/// Gemma 3 / 4 strided pattern: `stride-1` layers use full causal, others sliding.
pub fn gemma_strided_layer_mask(layer: usize, window: usize, stride: usize) -> MaskKind {
    if stride > 1 && (layer + 1).is_multiple_of(stride) {
        MaskKind::Causal
    } else {
        MaskKind::SlidingWindow(window)
    }
}

/// Composed Gemma prefill decoder block.
pub fn gemma_prefill_layer_composed(
    layer_idx: usize,
    style: GemmaLayerStyle,
    mut attn: SelfAttnPrefillSpec,
    eps: f32,
    kv_sink: Option<Arc<Mutex<Vec<rlx_ir::HirNodeId>>>>,
) -> FlowStage {
    let prefix = format!("model.layers.{layer_idx}");
    if attn.head_norm.is_none() {
        attn.head_norm = style.head_norm(&prefix, eps);
    }
    let mut stack = LayerStack::named(format!("layer{layer_idx}"))
        .residual_save()
        .stage(FlowStage::GemmaRmsNorm(GemmaRmsNormStage::hf_layer(
            format!("{prefix}.input_layernorm"),
            eps,
        )));

    if let Some(sink) = kv_sink {
        // Mirror the SelfAttn block's per-layer rope + k_eq_v
        // configuration so the cached K/V match what the attention
        // step will recompute. Gemma 4 full-attention layers use
        // partial rotary (n_rot < head_dim) and a named "global"
        // rope table; without honouring both here, the prefill K
        // export drifts from the runtime attention K.
        let mut tap = GemmaKvTapStage::layer(layer_idx, attn.head_dim, eps, sink);
        if attn.n_rot != attn.head_dim {
            tap = tap.with_n_rot(attn.n_rot);
        }
        if let Some(name) = attn.rope_table.as_deref() {
            tap = tap.with_rope_table(name);
        }
        if attn.k_eq_v {
            tap = tap.with_k_eq_v();
        }
        if let Some(norm) = attn.head_norm.clone() {
            tap = tap.with_head_norm(norm);
        }
        stack = stack.stage(FlowStage::GemmaKvTap(tap));
    }

    stack = stack
        .self_attn_prefill(attn)
        .linear(format!("{prefix}.self_attn.o_proj.weight"), true);
    if style.sandwich_norms() {
        stack = stack.stage(FlowStage::GemmaRmsNorm(GemmaRmsNormStage::hf_layer(
            format!("{prefix}.post_attention_layernorm"),
            eps,
        )));
    }
    stack = stack.residual_add().residual_save();

    stack = if style.sandwich_norms() {
        stack.stage(FlowStage::GemmaRmsNorm(GemmaRmsNormStage::hf_layer(
            format!("{prefix}.pre_feedforward_layernorm"),
            eps,
        )))
    } else {
        stack.stage(FlowStage::GemmaRmsNorm(GemmaRmsNormStage::hf_layer(
            format!("{prefix}.post_attention_layernorm"),
            eps,
        )))
    };

    stack = stack.stage(FlowStage::GeGlu(GeGluStage::hf_mlp(&prefix)));

    if style.sandwich_norms() {
        stack = stack.stage(FlowStage::GemmaRmsNorm(GemmaRmsNormStage::hf_layer(
            format!("{prefix}.post_feedforward_layernorm"),
            eps,
        )));
    }

    stack = stack.residual_add();
    // Gemma 4 scales each layer's output by a stored scalar; without it the
    // residual stream grows layer over layer. Absent from Gemma 4 GGUFs (they
    // ship `output_scale`) and from earlier generations.
    if style == GemmaLayerStyle::Gemma4 {
        stack = stack.stage(FlowStage::LayerScale(LayerScaleStage::optional(format!(
            "{prefix}.layer_scalar"
        ))));
    }
    stack.build()
}

/// MoE placeholder — dense Gemma paths use [`gemma_prefill_layer_composed`].
pub fn gemma_moe_prefill_layer_composed(
    layer_idx: usize,
    style: GemmaLayerStyle,
    attn: SelfAttnPrefillSpec,
    eps: f32,
    kv_sink: Option<Arc<Mutex<Vec<rlx_ir::HirNodeId>>>>,
    _moe: super::MoeFfnStage,
) -> FlowStage {
    gemma_prefill_layer_composed(layer_idx, style, attn, eps, kv_sink)
}

pub fn gemma_moe_decode_layer_composed(
    layer_idx: usize,
    spec: GemmaDecodeLayerSpec,
    kv_out: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
    _moe: super::MoeFfnStage,
) -> FlowStage {
    FlowStage::Named {
        name: format!("layer{layer_idx}"),
        inner: Arc::new(FlowStage::GemmaDecodeLayer(GemmaDecodeLayerStage::layer(
            layer_idx, spec, kv_out,
        ))),
    }
}
