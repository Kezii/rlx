// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The syntax highlighter for the page's code examples.
//
//   node --test crates/bindings/rlx-web/tests/cluster_highlight.test.mjs
//
// Hand-written rather than a CDN highlighter because the page is a static site
// that should work offline — and because being pure string → string, it can be
// checked properly. The invariant that matters is not the colours: it is that
// the code a reader copies is byte-for-byte the code that was written.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { highlight, highlightLines } from "../web/highlight.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));

/// Strip the markup the highlighter added and decode entities — the result must
/// equal the input exactly.
function plain(html) {
  return html
    .replace(/<[^>]*>/g, "")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, "&");
}

const SAMPLES = {
  rust: `use rlx_runtime::{Device, Session};

/// Doc comment with a "quoted" word and <angle> brackets.
#[derive(Clone)]
pub struct Rank { batch: usize }

fn main() {
    let mut sgd = Sgd::new(0.05);
    let s = "a string with // not a comment and a 'quote'";
    println!("{}", s);
    for i in 0..100u32 { let _ = i as f32; }
}`,
  javascript: `import init, { VisionBench } from "./pkg/rlx_web.js";

// A comment containing "a string" and <html>.
const model = new VisionBench("mnist-mlp");
let params = model.init_params(42);
const tpl = \`template \${params.length} literal\`;
if (params.length > 0 && true !== false) { console.log(tpl); }`,
  python: `import numpy as np
import pyrlx as rlx

# A comment with "quotes" and <brackets>.
def train(steps=100):
    """Docstring with a # hash and 'quotes'."""
    W = np.zeros((784, 10), dtype=np.float32)
    for i in range(steps):
        if i % 20 == 0 and W is not None:
            print(i, float(W.sum()))
    return W`,
};

test("highlighting never changes the code", () => {
  // The whole point. A highlighter that eats a character, drops a line, or
  // leaves an entity undecoded hands the reader something that will not run.
  for (const [lang, code] of Object.entries(SAMPLES)) {
    assert.equal(plain(highlight(code, lang)), code, `${lang} was altered`);
  }
});

test("markup in the source is escaped, not interpreted", () => {
  // Output goes through `innerHTML`; an unescaped `<` would be parsed as a tag
  // and silently swallow code.
  const code = 'let v: Vec<u8> = vec![]; // a & b < c > d "quoted"';
  const html = highlight(code, "rust");
  assert.ok(!/<(?!\/?span)/.test(html), `raw markup leaked: ${html}`);
  // The entities need not be adjacent — `Vec<u8>` tokenizes as `Vec`, `&lt;`,
  // `u8`, `&gt;` — so check they are present and that nothing raw survived.
  assert.ok(html.includes("&lt;") && html.includes("&gt;"), "angle brackets should be entities");
  assert.ok(html.includes("&amp;"), "ampersand should be an entity");
  assert.ok(!/[^&](<|>)[^s/]/.test(html.replace(/<\/?span[^>]*>/g, "")), "raw angle bracket left");
  assert.equal(plain(html), code);
});

test("comments and strings win over keywords inside them", () => {
  // Ordering is the grammar: a keyword inside a comment must stay a comment,
  // or the colours actively mislead.
  const rust = highlight('// let x = 1\nlet y = "let z = 2";', "rust");
  const comments = [...rust.matchAll(/<span class="tok-comment">([^<]*)</g)].map((m) => m[1]);
  assert.deepEqual(comments, ["// let x = 1"]);
  // The `let` inside the string must not be its own keyword span.
  const keywords = [...rust.matchAll(/<span class="tok-keyword">([^<]*)</g)].map((m) => m[1]);
  assert.deepEqual(keywords, ["let"], `keyword leaked into a string: ${keywords}`);

  const py = highlight(`x = "# not a comment"\n# real comment`, "python");
  assert.equal(
    [...py.matchAll(/<span class="tok-comment">([^<]*)</g)].length,
    1,
    "a hash inside a string is not a comment",
  );
});

test("each language colours its own constructs", () => {
  const rust = highlight('#[derive(Clone)]\nfn f<\'a>(x: u32) -> bool { true }', "rust");
  assert.match(rust, /tok-attribute/, "Rust attributes");
  assert.match(rust, /tok-lifetime/, "Rust lifetimes");
  assert.match(rust, /tok-type">u32/, "primitive types");

  const js = highlight("const x = `tpl`; class A extends B {}", "javascript");
  assert.match(js, /tok-string">`tpl`/, "template literals are strings");
  assert.match(js, /tok-type">A/, "classes read as types");

  const py = highlight('@decorator\ndef f():\n    return None', "python");
  assert.match(py, /tok-decorator/, "Python decorators");
  assert.match(py, /tok-constant">None/, "None is a constant");
});

test("an unknown language loses colour, not content", () => {
  const code = 'fn main() { /* <b>hi</b> */ }';
  const html = highlight(code, "brainfuck");
  assert.ok(!html.includes("<span"), "should not invent tokens");
  assert.equal(plain(html), code, "content must survive");
  assert.ok(html.includes("&lt;b&gt;"), "and still be escaped");
});

test("every example on the page declares a language the highlighter knows", () => {
  // A `<pre data-lang>` with a grammar we do not have renders plain, which is
  // easy to miss in review and obvious here.
  const html = readFileSync(path.join(HERE, "..", "web", "cluster.html"), "utf8");
  const langs = [...html.matchAll(/<pre data-lang="([^"]+)"/g)].map((m) => m[1]);
  assert.deepEqual(langs.sort(), ["javascript", "python", "rust"]);
  for (const lang of langs) {
    assert.ok(highlight("x = 1", lang).includes("<span"), `no grammar for ${lang}`);
  }
});

// ── per-line output (what the page actually uses) ──────────────────────────
// `highlightAll` emits one `<span class="line">` per source line, because a
// block can only hang-indent *continuation* lines if each line is its own box.
// Without that, a line wrapped on a phone restarts at column zero and reads as
// new code.

test("per-line output preserves the code exactly", () => {
  for (const [lang, code] of Object.entries(SAMPLES)) {
    assert.equal(plain(highlightLines(code, lang)), code, `${lang} was altered`);
  }
});

test("there is exactly one line span per source line", () => {
  for (const [lang, code] of Object.entries(SAMPLES)) {
    const spans = (highlightLines(code, lang).match(/class="line"/g) ?? []).length;
    assert.equal(spans, code.split("\n").length, `${lang}: line count mismatch`);
  }
});

test("a token spanning newlines keeps its colour on every line", () => {
  // A block comment or triple-quoted string crosses lines; splitting it must
  // not drop the class on the continuation, or half a comment renders as code.
  const rust = highlightLines("/* one\ntwo */ let z = 3;", "rust");
  const commentSpans = (rust.match(/tok-comment/g) ?? []).length;
  assert.equal(commentSpans, 2, "both halves of the comment should be coloured");
  assert.equal(plain(rust), "/* one\ntwo */ let z = 3;");

  const py = highlightLines('x = """a\nb"""\ny = 1', "python");
  assert.equal((py.match(/tok-string/g) ?? []).length, 2);
  assert.equal(plain(py), 'x = """a\nb"""\ny = 1');
});

test("the page's examples fit a phone after wrapping", () => {
  // ~52 characters fit at 390px; wrapping handles the rest, but the source
  // should not rely on it for most lines or every line wraps.
  const html = readFileSync(path.join(HERE, "..", "web", "cluster.html"), "utf8");
  const blocks = [...html.matchAll(/<pre data-lang="([^"]+)">([\s\S]*?)<\/pre>/g)];
  assert.equal(blocks.length, 3);
  for (const [, lang, body] of blocks) {
    const lines = body
      .replace(/&lt;/g, "<").replace(/&gt;/g, ">")
      .replace(/&quot;/g, '"').replace(/&amp;/g, "&")
      .split("\n");
    const longest = Math.max(...lines.map((l) => l.length));
    assert.ok(longest <= 72, `${lang}: longest line is ${longest} chars`);
    const wide = lines.filter((l) => l.length > 56).length;
    assert.ok(wide / lines.length < 0.35, `${lang}: ${wide}/${lines.length} lines over 56`);
  }
});
