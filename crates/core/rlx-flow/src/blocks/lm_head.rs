// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, Shape};

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;
#[derive(Debug, Clone)]
pub struct LmHeadStage {
    pub weight_key: Option<String>,
    pub tie_word_embeddings: bool,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub tied_param_name: String,
}

impl LmHeadStage {
    pub fn tied(vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            weight_key: None,
            tie_word_embeddings: true,
            vocab_size,
            hidden_size,
            tied_param_name: "lm_head.tied_t".into(),
        }
    }

    pub fn separate(weight_key: impl Into<String>, vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            weight_key: Some(weight_key.into()),
            tie_word_embeddings: false,
            vocab_size,
            hidden_size,
            tied_param_name: "lm_head.tied_t".into(),
        }
    }
}

impl BlockStage for LmHeadStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        // The LM head is the single largest weight read per decoded token
        // ([hidden, vocab]); store it F16-resident to halve that bandwidth on
        // batch-1 decode. Same opt-in as the projection weights; the backend
        // converts the f32 bytes at bind. (Only the head weight — the embed
        // table used by the token gather stays F32.)
        let w_dt = if rlx_ir::env::flag("RLX_QWEN3_F16_WEIGHTS") {
            DType::F16
        } else {
            DType::F32
        };
        // Opt-in packed head (`RLX_PACKED_LM_HEAD`): project through the
        // checkpoint's quant blob with a fused `DequantMatMul` instead of an F32
        // `[hidden, vocab]` copy — ~5x fewer bytes read per decoded token, and
        // that copy is never built. Tied checkpoints use a separately stored
        // `output.weight` when the file has one (as llama.cpp does), else the
        // packed embedding table itself — available when no stage consumed it
        // as F32 (a host-gathered embedding). Otherwise, or for a dense
        // source, this falls through to the F32 head below.
        if rlx_ir::env::flag("RLX_PACKED_LM_HEAD") {
            let mut keys = vec![self.weight_key.as_deref().unwrap_or("lm_head.weight")];
            if self.tie_word_embeddings && !ctx.params.contains_key("model.embed_tokens.weight") {
                keys.push("model.embed_tokens.weight");
            }
            let found = keys
                .into_iter()
                .find_map(|key| match ctx.weights.take_packed(key) {
                    Ok(Some(p)) => Some((key, p)),
                    _ => None,
                });
            if let Some((key, packed)) = found {
                if (packed.out_dim, packed.in_dim) != (self.vocab_size, self.hidden_size) {
                    anyhow::bail!(
                        "packed {key} is [{}, {}], expected [{}, {}]",
                        packed.out_dim,
                        packed.in_dim,
                        self.vocab_size,
                        self.hidden_size
                    );
                }
                return Ok(Some(ctx.register_packed_linear(&input, key, packed)?));
            }
        }
        let lm_head_w = if self.tie_word_embeddings {
            let embed_key = "model.embed_tokens.weight";
            let vocab = self.vocab_size;
            let hidden_size = self.hidden_size;
            let mut transposed = vec![0f32; vocab * hidden_size];
            {
                // Usually the token-embed stage has already read the table, in
                // which case reuse it. But a pipeline-parallel model's *last*
                // block holds the head without embedding anything, so nothing
                // has read it — it has to be read here rather than requiring a
                // stage that this block deliberately does not have.
                //
                // Read, not loaded as a param: the table is only needed to
                // build the transposed head, and declaring it a param would
                // upload a second copy of the largest tensor in the model for
                // a graph with no node that uses it.
                let fetched: Option<Vec<f32>> = if ctx.params.contains_key(embed_key) {
                    None
                } else {
                    Some(
                        ctx.weights
                            .take(embed_key, false)
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "tied lm_head needs {embed_key}, which is \
                                     neither already loaded nor in the weights: {e}"
                                )
                            })?
                            .0,
                    )
                };
                let embed: &[f32] = match &fetched {
                    Some(v) => v,
                    None => &ctx.params[embed_key],
                };
                if embed.len() < vocab * hidden_size {
                    anyhow::bail!(
                        "{embed_key} has {} values, too few for a [{vocab}, \
                         {hidden_size}] tied lm_head",
                        embed.len()
                    );
                }
                for v in 0..vocab {
                    for hi in 0..hidden_size {
                        transposed[hi * vocab + v] = embed[v * hidden_size + hi];
                    }
                }
            }
            ctx.synth_param(
                &self.tied_param_name,
                transposed,
                Shape::new(&[hidden_size, vocab], w_dt),
            )
        } else {
            let key = self
                .weight_key
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("LmHead: weight_key required when not tied"))?;
            ctx.load_param_typed(key, true, w_dt)?
        };

        let mut gb = HirMut::new(ctx.hir());
        let id = gb.mm(input.id, lm_head_w);
        let dims = input.shape.dims();
        let out_shape = Shape::from_dims(
            &[dims[0], dims[1], rlx_ir::Dim::Static(self.vocab_size)],
            input.shape.dtype(),
        );
        Ok(Some(ctx.wrap(id, out_shape)))
    }
}
