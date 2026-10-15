#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Check rlx-mlx-sys's CUDA toolkit selection against NVIDIA's CUDA images.

Each scenario builds `Dockerfile` on an `nvidia/cuda:<ver>-cudnn-devel` base
(the host's architecture — arm64 on Apple Silicon, amd64 elsewhere) and runs
the rlx-mlx CUDA build in it with `RLX_MLX_CUDA=1`. No GPU is needed:

- `select` stops once the build script has chosen a toolkit and CMake has
  configured MLX's CUDA backend (or the build script has refused CUDA);
- `full` also compiles MLX's CUDA kernels and links every rlx-mlx test
  binary (`cargo test --no-run`) — tens of minutes per image.

The build script names its choice in a `cargo:warning` (streamed live by
`cargo -vv`); each scenario states which choice it expects.

    python3 cuda_matrix.py                  # every scenario
    python3 cuda_matrix.py 12.8 multi-root  # just these
    python3 cuda_matrix.py --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LOGS = REPO / "target" / "cuda-matrix"


@dataclass
class Scenario:
    name: str
    base: str
    mode: str  # "select" | "full"
    expect: list[str]  # substrings the build-script output must contain
    env: dict[str, str] = field(default_factory=dict)
    extra_apt: str = ""


def image(ver: str) -> str:
    return f"nvidia/cuda:{ver}-cudnn-devel-ubuntu24.04"


SCENARIOS = [
    Scenario("12.8", image("12.8.1"), "full", ["building MLX CUDA backend", "with CUDA 12.8"]),
    Scenario("13.0", image("13.0.2"), "full", ["building MLX CUDA backend", "with CUDA 13.0"]),
    Scenario("13.2", image("13.2.0"), "full", ["building MLX CUDA backend", "with CUDA 13.2"]),
    Scenario(
        "13.1-refused",
        image("13.1.1"),
        "select",
        ["MLX does not build with 13.1", "building the CPU-only MLX backend"],
    ),
    # Two toolkits: the image's 13.0 behind /usr/local/cuda, plus 12.8.
    Scenario(
        "multi-default",
        image("13.0.2"),
        "select",
        ["with CUDA 13.0"],
        extra_apt="cuda-toolkit-12-8",
    ),
    Scenario(
        "multi-root",
        image("13.0.2"),
        "select",
        ["with CUDA 12.8 at /usr/local/cuda-12.8"],
        env={"RLX_MLX_CUDA_ROOT": "/usr/local/cuda-12.8"},
        extra_apt="cuda-toolkit-12-8",
    ),
]


def build_image(s: Scenario) -> str:
    tag = "rlx-mlx-cuda:" + s.base.split(":")[1] + ("-multi" if s.extra_apt else "")
    subprocess.run(
        [
            "docker", "build", "-q", "-t", tag,
            "--build-arg", f"BASE={s.base}",
            "--build-arg", f"EXTRA_APT={s.extra_apt}",
            str(HERE),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return tag


def run(s: Scenario) -> tuple[bool, str]:
    tag = build_image(s)
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{s.name}.log"
    cargo = (
        "cargo test --release -p rlx-mlx --features cuda --no-run -vv"
        if s.mode == "full"
        else "cargo build --release -p rlx-mlx-sys -vv"
    )
    script = (
        # No `.git` anywhere: a submodule's `.git` file points into the
        # parent's `.git/modules`, which is not copied, and `git apply` then
        # refuses the tree; without them `vendor/mlx` is a plain tree.
        "tar -C /src --exclude=./target --exclude='*/target' --exclude=.git -cf - . "
        "| tar -C /work -xf - && cd /work && "
        f"RLX_MLX_CUDA=1 RLX_MLX_CUDA_ARCH=86 {cargo}"
    )
    name = f"rlx-mlx-cuda-{s.name}-{int(time.time())}"
    cmd = [
        "docker", "run", "--rm", "--name", name,
        "-v", f"{REPO}:/src:ro",
        "-v", "rlx-mlx-cuda-cargo:/root/.cargo/registry",
        "-v", f"rlx-mlx-cuda-target-{s.name}:/work/target",
    ]
    for k, v in s.env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [tag, "bash", "-c", script]
    seen: list[str] = []
    configured = False
    with open(log_path, "w") as log, subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
    ) as p:
        assert p.stdout is not None
        for line in p.stdout:
            log.write(line)
            if "cargo:warning=rlx-mlx-sys:" in line:
                seen.append(line.split("cargo:warning=", 1)[1].strip())
            if "CMake Error" in line or "panicked at" in line:
                seen.append(line.strip())
            refused = any("building the CPU-only MLX backend" in w for w in seen)
            configured = configured or "-- Generating done" in line
            if s.mode == "select" and (refused or (configured and seen)):
                subprocess.run(["docker", "kill", name], stdout=subprocess.DEVNULL)
                break
        p.wait()
    text = "\n".join(seen)
    missing = [e for e in s.expect if e not in text]
    built = s.mode == "select" or p.returncode == 0
    ok = not missing and built and "CMake Error" not in text and "panicked at" not in text
    detail = text or "(no build-script warning)"
    if missing:
        detail += f"\n    missing: {missing}"
    if not built:
        detail += f"\n    cargo exit {p.returncode} — see {log_path}"
    return ok, detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="scenarios to run (default: all)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for s in SCENARIOS:
            print(f"{s.name:14} {s.mode:6} {s.base}{' + ' + s.extra_apt if s.extra_apt else ''}")
        return 0
    chosen = [s for s in SCENARIOS if not args.names or s.name in args.names]
    unknown = set(args.names) - {s.name for s in SCENARIOS}
    if unknown:
        print(f"unknown scenario(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    failed = 0
    for s in chosen:
        t = time.time()
        ok, detail = run(s)
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {s.name} ({s.mode}, {time.time() - t:.0f} s)\n    {detail}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
