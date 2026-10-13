// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Syntax highlighting for the code examples, in about a hundred lines.
//
// Not highlight.js or Prism from a CDN: the page is a static site that should
// work offline, and a third-party script that rewrites `innerHTML` is a poor
// trade for colouring three fixed snippets. This is pure string → string, so
// unlike the WebRTC pieces it can be tested directly
// (`tests/cluster_highlight.test.mjs`), including the invariant that actually
// matters — the code you read is byte-for-byte the code that was written.

/// Order is the grammar: whatever matches first wins, so comments and strings
/// must precede keywords or a keyword inside a comment gets coloured.
const GRAMMARS = {
  rust: [
    ["comment", String.raw`//[^\n]*|/\*[\s\S]*?\*/`],
    ["string", String.raw`"(?:[^"\\]|\\[\s\S])*"|'(?:[^'\\]|\\[\s\S])'`],
    ["attribute", String.raw`#!?\[[^\]]*\]`],
    ["lifetime", String.raw`'[a-z_][a-z0-9_]*\b`],
    ["number", String.raw`\b\d[\d_]*(?:\.\d+)?(?:[iuf](?:8|16|32|64|size))?\b`],
    [
      "keyword",
      String.raw`\b(?:as|async|await|break|const|continue|crate|dyn|else|enum|extern|fn|for|if|impl|in|let|loop|match|mod|move|mut|pub|ref|return|self|Self|static|struct|trait|type|unsafe|use|where|while)\b`,
    ],
    [
      "type",
      String.raw`\b(?:[A-Z][A-Za-z0-9_]*|u8|u16|u32|u64|usize|i8|i16|i32|i64|isize|f32|f64|bool|str|char)\b`,
    ],
    ["macro", String.raw`\b[a-z_][a-z0-9_]*!`],
    ["function", String.raw`\b[a-z_][a-z0-9_]*(?=\s*\()`],
  ],
  javascript: [
    ["comment", String.raw`//[^\n]*|/\*[\s\S]*?\*/`],
    ["string", String.raw`"(?:[^"\\]|\\[\s\S])*"|'(?:[^'\\]|\\[\s\S])*'|` + "`(?:[^`\\\\]|\\\\[\\s\\S])*`"],
    ["number", String.raw`\b\d[\d_]*(?:\.\d+)?\b`],
    [
      "keyword",
      String.raw`\b(?:async|await|break|case|catch|class|const|continue|default|delete|do|else|export|extends|finally|for|from|function|if|import|in|instanceof|let|new|of|return|static|super|switch|this|throw|try|typeof|var|void|while|yield)\b`,
    ],
    ["constant", String.raw`\b(?:true|false|null|undefined|NaN|Infinity)\b`],
    ["type", String.raw`\b(?:[A-Z][A-Za-z0-9_]*)\b`],
    ["function", String.raw`\b[a-z_$][\w$]*(?=\s*\()`],
  ],
  python: [
    ["comment", String.raw`#[^\n]*`],
    [
      "string",
      String.raw`[rbfu]{0,2}"""[\s\S]*?"""|[rbfu]{0,2}'''[\s\S]*?'''|[rbfu]{0,2}"(?:[^"\\]|\\[\s\S])*"|[rbfu]{0,2}'(?:[^'\\]|\\[\s\S])*'`,
    ],
    ["decorator", String.raw`@[A-Za-z_][\w.]*`],
    ["number", String.raw`\b\d[\d_]*(?:\.\d+)?\b`],
    [
      "keyword",
      String.raw`\b(?:and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield)\b`,
    ],
    ["constant", String.raw`\b(?:True|False|None)\b`],
    ["type", String.raw`\b(?:[A-Z][A-Za-z0-9_]*)\b`],
    ["function", String.raw`\b[a-z_][\w]*(?=\s*\()`],
  ],
};

const compiled = new Map();

function grammar(lang) {
  if (!compiled.has(lang)) {
    const rules = GRAMMARS[lang];
    compiled.set(
      lang,
      rules
        ? new RegExp(rules.map(([name, src]) => `(?<${name}>${src})`).join("|"), "g")
        : null,
    );
  }
  return compiled.get(lang);
}

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };

/// Escape every run of plain text. Not optional and not cosmetic: the output
/// goes through `innerHTML`, so an unescaped `<` in a snippet would be parsed
/// as markup — silently dropping code, or worse.
const escape = (text) => text.replace(/[&<>"]/g, (c) => ESCAPES[c]);

/// Split `code` into `{ kind, text }` pieces. `kind` is null for plain text.
function tokenize(code, lang) {
  const re = grammar(lang);
  if (!re) return [{ kind: null, text: code }];

  const pieces = [];
  let last = 0;
  for (const m of code.matchAll(re)) {
    const kind = Object.entries(m.groups).find(([, v]) => v !== undefined)?.[0];
    if (m.index > last) pieces.push({ kind: null, text: code.slice(last, m.index) });
    pieces.push({ kind, text: m[0] });
    last = m.index + m[0].length;
  }
  if (last < code.length) pieces.push({ kind: null, text: code.slice(last) });
  return pieces;
}

const wrap = (kind, text) =>
  kind ? `<span class="tok-${kind}">${escape(text)}</span>` : escape(text);

/// `code` → HTML with `<span class="tok-…">` around each token.
///
/// An unknown language is escaped and returned unchanged rather than throwing:
/// a missing grammar should cost colour, not content.
export function highlight(code, lang) {
  return tokenize(code, lang)
    .map(({ kind, text }) => wrap(kind, text))
    .join("");
}

/// Same, but each source line is its own `<span class="line">`.
///
/// This is what makes wrapped code readable on a phone: a block can only give
/// a *hanging* indent to continuation lines if each line is its own box, so
/// without this a wrapped line restarts at column zero and reads as new code.
///
/// Tokens that span newlines — a block comment, a triple-quoted string — are
/// split at the newline and each fragment keeps the token class, so the colour
/// survives the split and the text is still returned byte-for-byte.
export function highlightLines(code, lang) {
  const lines = [[]];
  for (const { kind, text } of tokenize(code, lang)) {
    const parts = text.split("\n");
    parts.forEach((part, i) => {
      if (i > 0) lines.push([]);
      if (part) lines[lines.length - 1].push(wrap(kind, part));
    });
  }
  return lines.map((line) => `<span class="line">${line.join("")}</span>`).join("\n");
}

/// Highlight every `<pre data-lang="…">` under `root`, reading `textContent`
/// so the source is the already-decoded text rather than markup.
export function highlightAll(root = document) {
  for (const pre of root.querySelectorAll("pre[data-lang]")) {
    pre.innerHTML = highlightLines(pre.textContent, pre.dataset.lang);
  }
}
