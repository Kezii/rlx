// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The join QR. Getting the room onto a second device is the one thing the
// cluster page cannot do without, so the encoder is vendored (no CDN at
// runtime) and checked here — it has no dependencies and runs in Node.
//
//   node --test crates/bindings/rlx-web/tests/cluster_qr.test.mjs

import { test } from "node:test";
import assert from "node:assert/strict";
import qrcode from "../web/vendor/qrcode-generator.mjs";

/// Encode and return the module matrix.
function encode(text, level = "M") {
  const qr = qrcode(0, level);
  qr.addData(text);
  qr.make();
  const n = qr.getModuleCount();
  const grid = [];
  for (let r = 0; r < n; r++) {
    const row = [];
    for (let c = 0; c < n; c++) row.push(qr.isDark(r, c) ? 1 : 0);
    grid.push(row);
  }
  return { n, grid };
}

test("the vendored encoder survives being vendored", () => {
  // The vendoring adds exactly one line (`export default qrcode;`). If anything
  // else drifted, the API disappears rather than misbehaving quietly.
  assert.equal(typeof qrcode, "function");
  const qr = qrcode(0, "M");
  for (const fn of ["addData", "make", "getModuleCount", "isDark"]) {
    assert.equal(typeof qr[fn], "function", `missing ${fn}`);
  }
});

test("a join URL encodes to a plausible QR", () => {
  const { n, grid } = encode("https://10.0.0.185:8443/cluster.html?room=ABC123");
  // Versions are 21, 25, 29, 33 … modules: always 21 + 4k.
  assert.ok(n >= 21 && (n - 21) % 4 === 0, `implausible module count ${n}`);

  // Roughly half the modules are dark in a real QR; all-dark or all-light means
  // the matrix was never populated.
  const dark = grid.flat().reduce((a, b) => a + b, 0);
  const ratio = dark / (n * n);
  assert.ok(ratio > 0.25 && ratio < 0.75, `dark ratio ${ratio.toFixed(2)} is not a QR`);
});

test("the three finder patterns are present", () => {
  // A 7×7 finder sits in each corner except bottom-right: dark ring, light
  // ring, 3×3 dark core. If the matrix were transposed or offset, this fails.
  const { n, grid } = encode("https://example.test/cluster.html?room=ZZZZZZ");
  const finderAt = (r0, c0) => {
    for (let r = 0; r < 7; r++) {
      for (let c = 0; c < 7; c++) {
        const onBorder = r === 0 || r === 6 || c === 0 || c === 6;
        const inCore = r >= 2 && r <= 4 && c >= 2 && c <= 4;
        const expected = onBorder || inCore ? 1 : 0;
        if (grid[r0 + r][c0 + c] !== expected) return false;
      }
    }
    return true;
  };
  assert.ok(finderAt(0, 0), "top-left finder missing");
  assert.ok(finderAt(0, n - 7), "top-right finder missing");
  assert.ok(finderAt(n - 7, 0), "bottom-left finder missing");
});

test("a longer URL grows the matrix rather than failing", () => {
  const short = encode("https://a.test/c.html?room=AAAAAA").n;
  const long = encode(
    "https://10.0.0.185:8443/cluster.html?room=ABC123&backend=webgpu&model=mnist-mlp" +
      "&batch=256&turn=turn:relay.example.test:3478|username|credential",
  ).n;
  assert.ok(long > short, `long URL did not need a bigger matrix (${short} vs ${long})`);
});

test("error-correction level M is used, not L", () => {
  // Level M tolerates ~15% damage against L's ~7%, which is the difference
  // between a phone scanning this off a screen at an angle and not.
  const m = encode("https://10.0.0.185:8443/cluster.html?room=ABC123", "M").n;
  const l = encode("https://10.0.0.185:8443/cluster.html?room=ABC123", "L").n;
  assert.ok(m >= l, "M should need at least as many modules as L for the same data");
});
