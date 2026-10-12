#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Build each crate's rustdoc the way docs.rs will, and fail on any warning.

Why per-crate and not `cargo doc --workspace`: a workspace doc build uses ONE
unified feature set (the default one), so every intra-doc link into a
feature-gated item reads as "unresolved" even though docs.rs resolves it
perfectly. That mismatch produced dozens of phantom warnings and hid the real
ones. docs.rs builds one crate at a time with that crate's
`[package.metadata.docs.rs]` features, so this does the same.

`publish = false` crates are skipped: docs.rs never builds them.

Usage:
  python3 scripts/check-rustdoc.py              # all publishable crates
  python3 scripts/check-rustdoc.py rlx-tensor   # just these
  python3 scripts/check-rustdoc.py --keep-going # report every crate, not just the first bad one
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Warnings rustdoc emits that are not the crate's fault. A build-script `cargo:warning`
# is printed by rustdoc runs too, and ours say "kernels not built on this host".
NOISE = re.compile(r"^warning: [a-z0-9-]+@\d")


def metadata() -> dict:
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def docsrs_args(pkg: dict) -> list[str] | None:
    """The feature flags docs.rs would use, or None to skip this crate."""
    if pkg.get("publish") == []:  # `publish = false`
        return None
    meta = (pkg.get("metadata") or {}).get("docs", {}).get("rs", {}) or {}
    args: list[str] = []
    if meta.get("all-features"):
        args.append("--all-features")
    else:
        if meta.get("no-default-features"):
            args.append("--no-default-features")
        feats = meta.get("features") or []
        if feats:
            args += ["--features", ",".join(feats)]
    return args


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("crates", nargs="*", help="only these packages (default: all publishable)")
    ap.add_argument("--keep-going", action="store_true", help="do not stop at the first bad crate")
    args = ap.parse_args()

    md = metadata()
    targets: list[tuple[str, list[str]]] = []
    skipped: list[str] = []
    for pkg in sorted(md["packages"], key=lambda p: p["name"]):
        if args.crates and pkg["name"] not in args.crates:
            continue
        flags = docsrs_args(pkg)
        if flags is None:
            skipped.append(pkg["name"])
            continue
        targets.append((pkg["name"], flags))

    if not targets:
        print("no matching publishable crates", file=sys.stderr)
        return 1

    env = dict(os.environ)
    # `--cfg docsrs` matches the `rustdoc-args` every crate here declares.
    env["RUSTDOCFLAGS"] = f"{env.get('RUSTDOCFLAGS', '')} -D warnings --cfg docsrs".strip()

    bad: list[tuple[str, list[str]]] = []
    for i, (name, flags) in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {name} {' '.join(flags)}", flush=True)
        run = subprocess.run(
            ["cargo", "doc", "-p", name, "--no-deps", *flags],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        if run.returncode == 0:
            continue
        lines = [
            l for l in run.stderr.splitlines()
            if (l.startswith("warning:") or l.startswith("error:")) and not NOISE.match(l)
        ]
        if not lines:
            # Failed to *compile* under its docs.rs features — also a docs.rs failure.
            lines = [l for l in run.stderr.splitlines() if l.startswith("error")] or ["build failed"]
        bad.append((name, lines))
        for l in lines:
            print(f"    {l}", flush=True)
        if not args.keep_going:
            break

    print()
    if skipped:
        print(f"skipped (publish = false): {', '.join(skipped)}")
    if bad:
        total = sum(len(l) for _, l in bad)
        print(f"rustdoc: {total} diagnostic(s) across {len(bad)} crate(s): "
              f"{', '.join(n for n, _ in bad)}", file=sys.stderr)
        return 1
    print(f"rustdoc OK — {len(targets)} crate(s) clean under their docs.rs features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
