// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The decode cache contract has TWO halves, and both are build errors.
//!
//! `bind_decode_inputs_with_capacity(.., past_len)` promises that `past_k_*` /
//! `past_v_*` hold `past_len` rows of history inside a buffer with room for one
//! more, so a decode layer can write the new token's row in place. Nothing but
//! the caller can know that — the two cache shapes are indistinguishable once
//! built — and getting either half wrong is SILENT on every backend:
//!
//! * contract WITHOUT a spare row — the in-place write lands on the last real
//!   row and the returned prefix is one row short;
//! * spare row WITHOUT the append — the `concat` puts the new token at
//!   `past_len + 1` and leaves row `past_len` zeroed mid-history.
//!
//! The second is not hypothetical: it is how the in-place append became dead
//! code once, and it is what migrating a flow whose decode layer is crate-local
//! and hand-rolls the `concat` does (rlx-models' unlimited-ocr hit exactly this,
//! caught by its chunk-decode test rather than by the build). So the binder
//! checks the shapes and `ModelFlow::build` checks the finished graph.

use rlx_flow::blocks::{CustomStage, append_kv_row};
use rlx_flow::{FlowStage, MapWeights, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};

/// How the test's decode layer grows its K/V cache.
#[derive(Clone, Copy, PartialEq)]
enum Grow {
    /// Through the shared helper — honours the contract, emits `Op::KvAppend`.
    Shared,
    /// Hand-rolled, ignoring the contract — what a crate-local layer does.
    Concat,
    /// Through the shared helper, but appending a 2-row chunk in one step.
    SharedChunk,
    /// No layer at all.
    None,
}

/// A decode flow with `cache_rows`-row K/V caches and a one-row-per-step layer.
fn build_flow(
    cache_rows: usize,
    past_len: Option<usize>,
    grow: Grow,
) -> anyhow::Result<rlx_flow::BuiltModel> {
    let f = DType::F32;
    let mut weights = MapWeights::default();
    let mut flow = ModelFlow::new("contract")
        .input("rope_cos", Shape::new(&[1, 4], f))
        .input("rope_sin", Shape::new(&[1, 4], f))
        .input("past_k_0", Shape::new(&[1, cache_rows, 8], f))
        .input("past_v_0", Shape::new(&[1, cache_rows, 8], f));
    flow = match past_len {
        Some(p) => flow.bind_decode_inputs_with_capacity(1, false, p),
        None => flow.bind_decode_inputs(1, false, true),
    };
    if grow != Grow::None {
        flow = flow.stage(FlowStage::Custom(CustomStage::named(
            "layer0",
            move |emit, _input| {
                let decode = emit
                    .state
                    .decode
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("layer requires BindDecodeInputs"))?;
                let past_k = decode.past_k[0];
                let rows = if grow == Grow::SharedChunk { 2 } else { 1 };
                let row =
                    emit.synth_param("row", vec![0.0; rows * 8], Shape::new(&[1, rows, 8], f));
                let mut gb = HirMut::new(emit.hir());
                let new_k = match grow {
                    Grow::Shared | Grow::SharedChunk => {
                        append_kv_row(&mut gb, past_k, row, 1, decode.past_len)
                    }
                    _ => gb.concat_(vec![past_k, row], 1),
                };
                let shape = gb.shape(new_k).clone();
                Ok(Some(emit.wrap(new_k, shape)))
            },
        )));
    }
    flow.build(&mut weights)
}

#[test]
fn a_contract_without_a_spare_row_is_refused() {
    let err = build_flow(4, Some(4), Grow::Shared)
        .expect_err("a cache sized exactly to past_len has no spare row to append into")
        .to_string();
    assert!(
        err.contains("no spare row"),
        "expected the contract error, got: {err}"
    );
}

#[test]
fn a_contract_with_a_spare_row_and_an_append_builds() {
    build_flow(5, Some(4), Grow::Shared)
        .expect("one spare row plus the append satisfies both halves");
}

#[test]
fn a_spare_row_with_no_append_is_refused() {
    // The silent-corruption case: shapes are legal, the op is simply absent.
    // Reachable by migrating a flow while its decode layer still `concat`s.
    let err = build_flow(5, Some(4), Grow::Concat)
        .expect_err("a declared spare row that nothing writes into is a misconfiguration")
        .to_string();
    assert!(
        err.contains("no `Op::KvAppend`"),
        "expected the missing-append error, got: {err}"
    );
}

#[test]
fn a_contract_with_no_decode_layer_at_all_is_refused() {
    // Same failure, reached from the other side: the binder ran but no layer
    // ever touched the cache, so the spare row is pure dead weight.
    let err = build_flow(5, Some(4), Grow::None)
        .expect_err("binding the contract with no layer emits no append")
        .to_string();
    assert!(
        err.contains("no `Op::KvAppend`"),
        "expected the missing-append error, got: {err}"
    );
}

#[test]
fn more_capacity_than_asked_for_is_fine() {
    // A wider cache than the contract needs is correct — the append position is
    // `past_len`, not "the end" — and runners rely on it: they keep a graph that
    // is already wide enough rather than recompiling for a shorter prompt.
    build_flow(64, Some(4), Grow::Shared).expect("extra capacity is allowed");
}

#[test]
fn the_plain_binder_imposes_no_requirements() {
    // Without the contract there is no promise to check: an exactly-sized
    // history cache grown by `concat` is the normal case, and `append_kv_row`
    // emits that same `concat` when `past_len` is `None`.
    build_flow(4, None, Grow::Concat).expect("plain binding accepts a concat layer");
    build_flow(4, None, Grow::Shared).expect("the shared helper degrades to concat");
}

#[test]
fn a_multi_token_step_cannot_use_the_contract() {
    // `Op::KvAppend` writes one row, so chunked / speculative decode must stay
    // on `concat` — and a flow that declared a spare row for it is wrong, not
    // merely slow. The helper declines the in-place path (rather than tripping
    // `kv_append`'s shape check with a panic from deep inside the IR) and the
    // build reports the contract error that names the fix.
    let err = build_flow(5, Some(4), Grow::SharedChunk)
        .expect_err("a 2-row step cannot append in place")
        .to_string();
    assert!(
        err.contains("no `Op::KvAppend`"),
        "expected the missing-append error, got: {err}"
    );
}
