// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bind decode graph inputs (RoPE slice, past K/V, optional mask).

use anyhow::Result;
use rlx_ir::hir::{HirModule, HirNodeId, HirOp};

use crate::context::{DecodeBindings, FlowCtx};
#[derive(Debug, Clone)]
pub struct BindDecodeInputsStage {
    pub num_layers: usize,
    pub use_custom_mask: bool,
    pub need_past_kv: bool,
    /// Rows of real history in `past_k_*`/`past_v_*` when those inputs are
    /// declared with spare capacity. See [`crate::DecodeBindings::past_len`] —
    /// this is the only place the distinction can enter the flow, because the
    /// two cache shapes are indistinguishable once built.
    pub kv_past_len: Option<usize>,
}

impl BindDecodeInputsStage {
    pub fn emit(&self, ctx: &mut FlowCtx<'_>) -> Result<()> {
        let cos = ctx
            .state
            .rope_cos
            .or_else(|| find_input(ctx.hir(), "rope_cos").ok())
            .ok_or_else(|| anyhow::anyhow!("decode flow missing rope_cos"))?;
        let sin = ctx
            .state
            .rope_sin
            .or_else(|| find_input(ctx.hir(), "rope_sin").ok())
            .ok_or_else(|| anyhow::anyhow!("decode flow missing rope_sin"))?;
        let mask = if self.use_custom_mask {
            Some(find_input(ctx.hir(), "mask")?)
        } else {
            None
        };
        let mut past_k = Vec::with_capacity(self.num_layers);
        let mut past_v = Vec::with_capacity(self.num_layers);
        if self.need_past_kv {
            for i in 0..self.num_layers {
                past_k.push(find_input(ctx.hir(), &format!("past_k_{i}"))?);
                past_v.push(find_input(ctx.hir(), &format!("past_v_{i}"))?);
            }
        }
        // Validate the CACHE CONTRACT against the shapes actually declared.
        //
        // `kv_past_len = Some(p)` promises the caches hold `p` rows of history
        // inside a buffer with room for at least one more, so a decode layer can
        // write the new row in place. Nothing else can check that promise: the
        // two cache shapes are indistinguishable once built, which is why the
        // contract is passed explicitly in the first place. If a flow declares
        // the contract but sizes its caches to exactly `p`, the layer's in-place
        // write lands on the last REAL row and returns a prefix one row short —
        // wrong logits, no error, on every backend. Fail at build instead.
        if let Some(past_len) = self.kv_past_len {
            for (i, &id) in past_k.iter().chain(past_v.iter()).enumerate() {
                let dims = ctx.hir().node(id).shape.clone();
                // Only a statically-shaped cache can be checked (and only a
                // statically-shaped one can carry a spare row at all).
                let axis1 = dims.dim(1);
                if axis1.is_static() {
                    let cap = axis1.unwrap_static();
                    anyhow::ensure!(
                        cap > past_len,
                        "decode cache contract: past_k/v[{}] is {:?} with past_len={past_len}, \
                         so there is no spare row for an in-place KV append. Size the declared \
                         inputs to `past_len + kv_spare_rows(..)` (the same value passed here) \
                         or bind with `bind_decode_inputs` instead.",
                        i % past_k.len().max(1),
                        dims.dims(),
                    );
                }
            }
        }
        ctx.state.decode = Some(DecodeBindings {
            cos,
            sin,
            mask,
            past_k,
            past_v,
            past_len: self.kv_past_len,
        });
        Ok(())
    }
}

fn find_input(hir: &HirModule, name: &str) -> Result<HirNodeId> {
    for node in hir.nodes() {
        if let HirOp::Input { name: n } = &node.op {
            if n == name {
                return Ok(node.id);
            }
        }
    }
    Err(anyhow::anyhow!("decode flow missing input: {name}"))
}
