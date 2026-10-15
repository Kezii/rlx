// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

#[derive(Debug, Clone)]
pub struct LayerScaleStage {
    pub gamma_key: String,
    /// Pass the input through unchanged when the checkpoint lacks the key.
    pub optional: bool,
}

impl LayerScaleStage {
    pub fn new(gamma_key: impl Into<String>) -> Self {
        Self {
            gamma_key: gamma_key.into(),
            optional: false,
        }
    }

    /// Scale by `gamma_key` when the checkpoint has it, else pass through.
    pub fn optional(gamma_key: impl Into<String>) -> Self {
        Self {
            gamma_key: gamma_key.into(),
            optional: true,
        }
    }
}

impl BlockStage for LayerScaleStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let gamma = match ctx.load_param(&self.gamma_key, false) {
            Ok(g) => g,
            Err(_) if self.optional => return Ok(Some(input)),
            Err(e) => return Err(e),
        };
        let mut gb = HirMut::new(ctx.hir());
        let out = gb.mul(input.id, gamma);
        Ok(Some(ctx.wrap(out, input.shape.clone())))
    }
}
