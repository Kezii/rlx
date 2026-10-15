#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""One command for the PS5 payloads: fetch the SDK, build, send, read output.

    python3 scripts/ps5.py setup                  # SDK v0.43 into the cache, checksum-pinned
    python3 scripts/ps5.py build                  # all payloads → target/ps5/*.elf (stripped)
    python3 scripts/ps5.py send probe --host IP   # send to the ELF loader, stream its stdout
    python3 scripts/ps5.py run rlx --host IP      # build, then send

    python3 scripts/ps5.py build --manifest-path ../my-payload/Cargo.toml
                                                  # any crate that depends on rlx-ps5

`--host` defaults to $PS5_HOST, the port to $PS5_PORT or 9021 (elfldr).
The SDK is cached in $PS5_SDK_CACHE, else ~/.cache/rlx-ps5; an existing
SDK is used as-is when $PS5_PAYLOAD_SDK is set. macOS needs Homebrew
`llvm@21` and `lld`; Linux needs clang/lld 18–21 (the SDK drives the host
LLVM). The SDK's wrappers are bash, so Windows builds run under WSL.
"""

import argparse
import hashlib
import json
import re
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

SDK_VERSION = "v0.43"
SDK_URL = f"https://github.com/ps5-payload-dev/sdk/releases/download/{SDK_VERSION}/ps5-payload-sdk.zip"
SDK_SHA256 = "a9cc9929f21b2b2c5d5b309f3bab4997067c45281c0622cf4838b1aecba66fcb"

CRATE = Path(__file__).resolve().parent.parent
CARGO_CONFIG = CRATE / ".cargo" / "config.toml"
TARGET = "x86_64-unknown-freebsd"
PAYLOADS = {"probe": "ps5-probe", "rlx": "ps5-rlx", "node": "ps5-node"}


def cache_dir() -> Path:
    if d := os.environ.get("PS5_SDK_CACHE"):
        return Path(d)
    return Path.home() / ".cache" / "rlx-ps5"


def sdk_dir() -> Path:
    if d := os.environ.get("PS5_PAYLOAD_SDK"):
        return Path(d)
    return cache_dir() / "ps5-payload-sdk"


def die(msg: str) -> "None":
    sys.exit(f"ps5: {msg}")


def setup(_args) -> None:
    sdk = sdk_dir()
    if (sdk / "bin" / "prospero-clang").exists():
        print(f"ps5: SDK present at {sdk}")
    else:
        cache = cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        zpath = cache / f"ps5-payload-sdk-{SDK_VERSION}.zip"
        if not zpath.exists():
            print(f"ps5: downloading {SDK_URL}")
            part = zpath.with_suffix(".part")
            urllib.request.urlretrieve(SDK_URL, part)
            part.rename(zpath)
        digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
        if digest != SDK_SHA256:
            zpath.unlink()
            die(f"checksum mismatch for the SDK zip ({digest}); removed it, re-run setup")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(cache)
        # zipfile drops the executable bits the SDK's wrappers need.
        for f in (cache / "ps5-payload-sdk" / "bin").iterdir():
            f.chmod(f.stat().st_mode | 0o111)
        print(f"ps5: SDK {SDK_VERSION} in {sdk}")
    if platform.system() == "Darwin" and llvm_config() is None:
        print("ps5: macOS also needs `brew install llvm@21 lld`")


def llvm_config() -> "str | None":
    if v := os.environ.get("LLVM_CONFIG"):
        return v
    if platform.system() == "Darwin":
        brew = Path("/opt/homebrew/opt/llvm@21/bin/llvm-config")
        return str(brew) if brew.exists() else None
    return None  # the SDK finds llvm-config-18..21 on Linux by itself


def toolchain() -> str:
    """The nightly pinned in rust-toolchain.toml, passed explicitly so a crate
    outside this directory builds with it too (rustup only reads the file
    from the working directory upward)."""
    text = (CRATE / "rust-toolchain.toml").read_text()
    m = re.search(r'^channel\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else "nightly"


def build_env() -> dict:
    sdk = sdk_dir()
    if not (sdk / "bin" / "prospero-clang").exists():
        die(f"no SDK at {sdk}; run `python3 scripts/ps5.py setup`")
    env = dict(os.environ, PS5_PAYLOAD_SDK=str(sdk))
    if cfg := llvm_config():
        env["LLVM_CONFIG"] = cfg
    return env


def build(args) -> None:
    env = build_env()
    manifest = Path(args.manifest_path).resolve() if getattr(args, "manifest_path", None) else CRATE / "Cargo.toml"
    tc = toolchain()
    if shutil.which("rustup"):
        # build-std compiles std from rust-src; the toolchain file asks for it
        # here, but a crate elsewhere does not carry that file.
        subprocess.run(["rustup", "component", "add", "rust-src", "--toolchain", tc],
                       check=True, stdout=subprocess.DEVNULL)
    # The target, linker shim, build-std and C-dependency settings come from
    # this crate's .cargo/config.toml; its relative paths resolve against this
    # crate, so the same file serves any manifest.
    cmd = ["cargo", f"+{tc}", "build", "--release", "--manifest-path", str(manifest),
           "--config", str(CARGO_CONFIG)]
    subprocess.run(cmd, env=env, check=True)
    meta = json.loads(subprocess.run(
        ["cargo", f"+{tc}", "metadata", "--no-deps", "--format-version", "1",
         "--manifest-path", str(manifest)], env=env, check=True, capture_output=True).stdout)
    bins = [t["name"] for pkg in meta["packages"] if Path(pkg["manifest_path"]) == manifest
            for t in pkg["targets"] if "bin" in t["kind"]]
    target_dir = Path(meta["target_directory"])
    out = target_dir / "ps5"
    out.mkdir(parents=True, exist_ok=True)
    strip = Path(env["PS5_PAYLOAD_SDK"]) / "bin" / "prospero-strip"
    for name in bins:
        src = target_dir / TARGET / "release" / name
        dst = out / f"{name}.elf"
        # Debug info stays in the unstripped binary for symbolizing a crash.
        subprocess.run([str(strip), "-o", str(dst), str(src)], env=env, check=True)
        print(f"ps5: {dst}  {dst.stat().st_size / 1e6:.1f} MB")


def send(args) -> None:
    host = args.host or os.environ.get("PS5_HOST")
    if not host:
        die("no console address: pass --host or set PS5_HOST")
    port = args.port or int(os.environ.get("PS5_PORT", "9021"))
    name = f"{PAYLOADS.get(args.payload, args.payload)}.elf"
    dirs = [CRATE / "target" / "ps5"]
    if getattr(args, "manifest_path", None):
        dirs.insert(0, Path(args.manifest_path).resolve().parent / "target" / "ps5")
    candidates = [Path(args.payload)] + [d / name for d in dirs]
    elf = next((c for c in candidates if c.is_file()), candidates[-1])
    if not elf.exists():
        die(f"no payload {args.payload!r}; run `build` first (known: {', '.join(PAYLOADS)})")
    data = elf.read_bytes()
    print(f"ps5: sending {elf.name} ({len(data) / 1e6:.1f} MB) to {host}:{port}", file=sys.stderr)
    # elfldr runs the ELF it reads and writes the payload's stdout back on the
    # same connection until the payload exits.
    with socket.create_connection((host, port), timeout=30) as s:
        s.sendall(data)
        s.shutdown(socket.SHUT_WR)
        s.settimeout(None)
        while chunk := s.recv(65536):
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()


def run(args) -> None:
    build(args)
    send(args)


def main() -> None:
    if shutil.which("cargo") is None:
        die("cargo not found")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="fetch and verify the SDK").set_defaults(fn=setup)
    b = sub.add_parser("build", help="build and strip every payload")
    b.add_argument("--manifest-path", help="a payload crate elsewhere (it must depend on rlx-ps5)")
    b.set_defaults(fn=build)
    for name, fn in (("send", send), ("run", run)):
        p = sub.add_parser(name, help=f"{name} a payload ({', '.join(PAYLOADS)} or a path)")
        p.add_argument("payload")
        if name == "run":
            p.add_argument("--manifest-path", help="build this crate first, then send its payload")
        p.add_argument("--host")
        p.add_argument("--port", type=int)
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
