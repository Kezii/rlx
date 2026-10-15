# rlx-ps5

RLX on a jailbroken PS5 (firmware 7.00–13.60, an ELF loader such as
[elfldr](https://github.com/ps5-payload-dev/elfldr) on port 9021), built with
the [ps5-payload-dev SDK](https://github.com/ps5-payload-dev/sdk). CPU only.

## How the build works

- **Target:** Rust's built-in `x86_64-unknown-freebsd`; the SDK presents the
  console as FreeBSD 11 on Zen 2 (`-C target-cpu=znver2`).
- **Nightly `build-std` with `-Ztls-model=emulated`:** the prebuilt std uses
  native TLS (`PT_TLS`), which the payload loader never sets up (elfldr maps
  only `PT_LOAD`). Rebuilt with emulated TLS, std matches the SDK's
  `-femulated-tls`.
- **`scripts/ps5-cc`:** rustc's linker and the C compiler for `cc`-built
  dependencies. Links through `prospero-clang` (Sony's libc/libkernel stubs and
  the payload linker script), drops FreeBSD system libraries the console lacks,
  and maps `-lgcc_s` to the SDK's libunwind.
- **`compat/compat.c`:** `pthread_setname_np` and `dl_iterate_phdr`, which std
  imports and Sony does not export, plus `ld --wrap` shims that widen Sony's
  FreeBSD 11 `stat`/`dirent` into the FreeBSD 12 layouts std is built for.
- **Thread count:** `rlx_ps5::init()` sizes the RLX CPU pool to the payload's
  affinity mask minus two (`RLX_WORKERS` overrides).
- **Exit:** `rlx_ps5::exit()` flushes and `_exit`s, skipping the SDK start
  code's Sony `exit()` (static destructors with the pool still running).
- **GEMM:** no BLAS links for the console, so rlx-cpu's `portable-gemm` feature
  puts the fallback on the `gemm` crate's blocked AVX2/FMA kernels. Measured on
  three AVX2 cores (i9-12900H, same no-BLAS build): prefill GEMMs 36–41 →
  283–335 GF/s, attention QKᵀ 5.6 → 202 GF/s; `ps5-rlx` end to end 160–190
  GF/s at 1024³.

## Build and send

```sh
python3 scripts/ps5.py setup                    # SDK v0.43, sha256-pinned, into ~/.cache/rlx-ps5
python3 scripts/ps5.py build                    # → target/ps5/{ps5-probe,ps5-rlx,ps5-node}.elf
python3 scripts/ps5.py send probe --host <ps5>  # to elfldr :9021, streams the payload's stdout
python3 scripts/ps5.py run rlx --host <ps5>     # build + send
```

macOS needs `brew install llvm@21 lld`; Linux needs clang/lld 18–21. The SDK's
wrappers are bash, so on Windows run it under WSL. `PS5_PAYLOAD_SDK` points at
an existing SDK instead of the cache; `PS5_HOST` / `PS5_PORT` replace the flags.

| Payload | Checks |
|---|---|
| `ps5-probe` | std on the console: threads + TLS, condvar, files/`stat`/`readdir`, mmap, 1 GiB alloc, timing, unwinding, CPU mask, SIMD |
| `ps5-rlx` | `rlx-runtime` on `Device::Cpu`: `softmax(x@w+b)` vs an f64 reference, square GEMM timings |
| `ps5-node` | the console as a dist node: dials the coordinator and serves shipped stages on its CPU |

Each prints its results and writes them to `/data/rlx-ps5/{probe,rlx,node}.txt`.
`ps5-rlx` and `ps5-node` also run on the host (`--target <host>`).

## Your own payload

A model payload (a chat app, a benchmark) lives in its own crate — models stay
out of rlx core — and depends on this one:

```toml
[dependencies]
rlx-ps5 = { path = "<rlx>/crates/backends/rlx-ps5" }
rlx-runtime = { path = "<rlx>/crates/core/rlx-runtime", default-features = false, features = ["cpu"] }
```

```rust
fn main() {
    rlx_ps5::init();                 // CPU pool sized to the payload's cores
    // … load weights, compile, then keep them resident:
    // rlx_ps5::lock(&weights)?;     // refuses past LOCK_BUDGET_BYTES (1.6 GB)
    rlx_ps5::exit(0);                // _exit, not Sony's exit()
}
```

```sh
just ps5 build --manifest-path /abs/path/to/payload/Cargo.toml   # → <payload>/target/ps5/*.elf
just ps5 run <bin> --manifest-path /abs/path/to/payload/Cargo.toml --host <ps5>
```

The dependency brings the compat shims and the `portable-gemm` GEMM along;
the build applies this crate's toolchain and `.cargo/config.toml`, so the
payload crate needs no PS5 configuration of its own.

## Dist node

A payload starts with no arguments, so `ps5-node` reads `NodeConfig::from_env`
keys from `/data/rlx-ps5/node.conf` (upload it over FTP). Use a star: the
console dials out and nothing has to reach it.

```text
RANK=1
WORLD=2
PEERS=<mac-ip>:29500,<ps5-ip>:29501
TOPOLOGY=star
```

then on the Mac:

```sh
cargo run --release -p rlx-ffi --example node_coordinator -- \
    --world 2 --peers <mac-ip>:29500,<ps5-ip>:29501
```

It ships `gelu(x * 3 + 1)` and checks every step against a CPU reference.
