// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! No WGSL kernel may `return` from non-uniform control flow before a
//! `workgroupBarrier()`.
//!
//! WGSL requires a barrier to be reached in uniform control flow. A `return`
//! gated on `global_invocation_id` breaks that, and the consequences are
//! platform-dependent in the worst way:
//!
//! * **naga → Metal** accepts it, so native wgpu tests pass.
//! * **Dawn (every browser)** rejects the whole shader module: *"'workgroupBarrier'
//!   must only be called from uniform control flow"*. Every pipeline built from
//!   it is invalid, every dispatch fails, and `run_async` still resolves — so a
//!   browser rank reported `webgpu`, trained on a loss of exactly 0.0000 and
//!   poisoned a cluster's gradient average. No panic, no error, just zeros.
//! * **FXC (DirectX)** rejects it as X4026, which is why `matmul.wgsl` already
//!   carries the fix and the comment explaining it.
//!
//! `matmul_f16w` and `matmul_f16_compute` were written later and missed it, and
//! `reduce` acquired it when the tree-reduction path was added. A comment in one
//! shader did not stop that, so this checks all of them.
//!
//! The rule applied here: within a function that contains a barrier, a `return`
//! before the first barrier is an error if any **enclosing `if` condition**
//! depends on `global_invocation_id` (directly or through a `let`). Conditions
//! on `workgroup_id`, `num_workgroups` or uniform-buffer fields are fine —
//! those are uniform across a workgroup, which is the granularity barriers care
//! about.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Strip `//` and `/* */` comments: the word "return" appears in several
/// comments that explain this very rule, and flagging those would be absurd.
fn strip_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i..].starts_with(b"//") {
            while i < bytes.len() && bytes[i] != b'\n' {
                out.push(' ');
                i += 1;
            }
        } else if bytes[i..].starts_with(b"/*") {
            while i < bytes.len() && !bytes[i..].starts_with(b"*/") {
                out.push(if bytes[i] == b'\n' { '\n' } else { ' ' });
                i += 1;
            }
            i = (i + 2).min(bytes.len());
            out.push_str("  ");
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

/// `(name, body)` for each top-level `fn`.
fn functions(src: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let bytes = src.as_bytes();
    let mut search = 0;
    while let Some(rel) = src[search..].find("fn ") {
        let at = search + rel;
        search = at + 3;
        // A real declaration, not a substring of an identifier.
        if at > 0 && (bytes[at - 1].is_ascii_alphanumeric() || bytes[at - 1] == b'_') {
            continue;
        }
        let rest = &src[at + 3..];
        let name: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        let Some(open_rel) = rest.find('{') else {
            continue;
        };
        let open = at + 3 + open_rel;
        let mut depth = 0usize;
        for (k, c) in src[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        out.push((name.clone(), src[open..open + k].to_string()));
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    out
}

/// Names bound from `global_invocation_id` (directly or transitively).
fn non_uniform_names(body: &str) -> HashSet<String> {
    let mut names: HashSet<String> = HashSet::new();
    names.insert("gid".into());
    names.insert("global_invocation_id".into());
    // Two passes catch `let a = gid.x; let b = a * 2u;`.
    for _ in 0..2 {
        for line in body.lines() {
            let t = line.trim();
            let Some(rest) = t.strip_prefix("let ").or_else(|| t.strip_prefix("var ")) else {
                continue;
            };
            let Some((lhs, rhs)) = rest.split_once('=') else {
                continue;
            };
            let name: String = lhs
                .trim()
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if name.is_empty() {
                continue;
            }
            if names.iter().any(|n| mentions(rhs, n)) {
                names.insert(name);
            }
        }
    }
    names
}

/// Whole-identifier match, so `grid` does not count as `gid`.
fn mentions(haystack: &str, ident: &str) -> bool {
    let bytes = haystack.as_bytes();
    let mut from = 0;
    while let Some(rel) = haystack[from..].find(ident) {
        let at = from + rel;
        let before_ok =
            at == 0 || !(bytes[at - 1].is_ascii_alphanumeric() || bytes[at - 1] == b'_');
        let end = at + ident.len();
        let after_ok =
            end >= bytes.len() || !(bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_');
        if before_ok && after_ok {
            return true;
        }
        from = at + ident.len();
    }
    false
}

/// Offending returns: `(enclosing condition, line)`.
fn bad_returns(body: &str) -> Vec<(String, usize)> {
    let Some(first_barrier) = body.find("workgroupBarrier(") else {
        return Vec::new();
    };
    let head = &body[..first_barrier];
    let suspect = non_uniform_names(body);

    let mut stack: Vec<Option<String>> = Vec::new(); // condition per open brace
    let mut found = Vec::new();
    let bytes = head.as_bytes();
    let mut i = 0;
    let mut line = 1usize;
    while i < bytes.len() {
        match bytes[i] {
            b'\n' => line += 1,
            b'{' => {
                // Attribute this brace to the nearest preceding `if (...)`.
                let before = &head[..i];
                let cond = before.rfind("if").and_then(|at| {
                    let after = &before[at + 2..];
                    if after.trim_start().starts_with('(') && !after.contains('{') {
                        let start = after.find('(')? + 1;
                        let end = after.rfind(')')?;
                        (start <= end).then(|| after[start..end].to_string())
                    } else {
                        None
                    }
                });
                stack.push(cond);
            }
            b'}' => {
                stack.pop();
            }
            b'r' if head[i..].starts_with("return")
                && (i == 0 || !bytes[i - 1].is_ascii_alphanumeric()) =>
            {
                if let Some(cond) = stack
                    .iter()
                    .flatten()
                    .find(|c| suspect.iter().any(|n| mentions(c, n)))
                {
                    found.push((cond.clone(), line));
                }
                i += 6;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    found
}

fn kernels_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/kernels")
}

#[test]
fn no_non_uniform_return_before_a_workgroup_barrier() {
    let mut offenders = Vec::new();
    let dir = kernels_dir();
    let entries = std::fs::read_dir(&dir).expect("kernels dir");
    let mut checked = 0usize;
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("wgsl") {
            continue;
        }
        let src = strip_comments(&std::fs::read_to_string(&path).expect("read wgsl"));
        for (name, body) in functions(&src) {
            if !body.contains("workgroupBarrier(") {
                continue;
            }
            checked += 1;
            for (cond, line) in bad_returns(&body) {
                offenders.push(format!(
                    "{}: fn {name} returns at line ~{line} under `if ({})`, which depends on \
                     global_invocation_id, before a workgroupBarrier",
                    path.file_name().unwrap().to_string_lossy(),
                    cond.trim()
                ));
            }
        }
    }

    assert!(
        checked > 10,
        "scanned only {checked} barrier functions — the walk is broken"
    );
    assert!(
        offenders.is_empty(),
        "non-uniform return before a barrier — Dawn will reject these shaders and the \
         backend will compute zeros without erroring:\n  {}\n\nFix as `matmul.wgsl` does: \
         derive the index from workgroup_id where possible, or replace the early return with \
         an `in_range` flag and guard the stores.",
        offenders.join("\n  ")
    );
}

/// The scanner has to actually catch the pattern, or it is decoration. This is
/// the shape `matmul_f16w` had when Chrome computed zeros.
#[test]
fn the_scanner_catches_the_pattern_it_exists_for() {
    let bad = r#"
        @compute @workgroup_size(8, 8)
        fn broken(@builtin(global_invocation_id) gid: vec3<u32>) {
            let bz = gid.z;
            if (bz >= params.batch) { return; }
            tile[0] = 1.0;
            workgroupBarrier();
        }
    "#;
    let fns = functions(&strip_comments(bad));
    assert_eq!(fns.len(), 1, "function walk failed");
    assert!(
        !bad_returns(&fns[0].1).is_empty(),
        "scanner missed a gid-gated return"
    );

    // And must not flag the legal forms: workgroup-uniform guards, or a
    // uniform-buffer-gated return (both appear in `reduce.wgsl`).
    let ok = r#"
        @compute @workgroup_size(64)
        fn fine(@builtin(workgroup_id) wid: vec3<u32>,
                @builtin(global_invocation_id) gid: vec3<u32>) {
            if (params.reduce_dim == 0u) { return; }
            if (params.reduce_dim <= 64u) {
                let cell = gid.x;
                if (cell < total) { out[cell] = 1.0; }
                return;
            }
            let row = wid.x;
            if (row >= total) { return; }
            scratch[0] = 1.0;
            workgroupBarrier();
        }
    "#;
    let fns = functions(&strip_comments(ok));
    let flagged = bad_returns(&fns[0].1);
    assert!(
        flagged.is_empty(),
        "false positive on legal shader: {flagged:?}"
    );
}
