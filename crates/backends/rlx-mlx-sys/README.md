# rlx-mlx-sys

Vendored [MLX](https://github.com/ml-explore/mlx) C++ (git submodule at
`vendor/mlx`) plus the `rlx_mlx_shim` C ABI built via CMake + `cc` in
`build.rs`. Consumed by [`rlx-mlx`](../rlx-mlx); not meant for direct use
outside RLX unless you accept the shim API stability policy (none yet).

After clone:

```sh
git submodule update --init rlx-mlx-sys/vendor/mlx
```

The submodule is marked `shallow = true` in `.gitmodules` (depth-1 clone of
the pinned MLX commit) so local `.git` stays small. A full MLX history is not
required to build.

## Linux compile times

| Backend | First build (approx.) | Opt-in |
|---------|----------------------|--------|
| **CPU only** (default) | ~5–10 min | always |
| **CUDA** (nvcc kernels) | ~45–90 min | `RLX_MLX_CUDA=1` or `--features cuda` |

**CPU-only is the default** even when CUDA/cuDNN are installed. Auto-building
the CUDA backend added an hour+ to every clean `cargo build` on WSL rigs.

### Faster Linux builds

1. Install system LAPACK headers (avoids bootstrapping OpenBLAS into `OUT_DIR`):

   ```sh
   sudo apt-get install libblas-dev liblapack-dev liblapacke-dev
   ```

2. Use **debug** cargo profile for iteration — `build.rs` maps `cargo build`
   → CMake `Debug` (much faster nvcc when CUDA is enabled).

3. Install **sccache** or **ccache**. `build.rs` uses the first one it finds
   (sccache first) for MLX's C++ and CUDA sources, so a rebuilt `OUT_DIR` — a
   new target dir, a feature or profile change, a CI runner — recompiles from
   the cache. `RLX_MLX_CCACHE=sccache|ccache|<path>|off` picks one; not used on
   Windows (the Visual Studio generator ignores compiler launchers).

   ```sh
   sudo apt-get install ccache        # or: cargo install sccache
   ```

4. Pin parallel nvcc jobs if WSL OOMs or thrashes:

   ```sh
   RLX_MLX_JOBS=4 cargo build -p rlx-mlx-sys --features cuda
   ```

5. Pin GPU arch when cross-compiling or CI has no GPU:

   ```sh
   RLX_MLX_CUDA_ARCH=89 RLX_MLX_CUDA=1 cargo build -p rlx-mlx --features cuda
   ```

6. Runtime device selection (after build): `RLX_MLX_DEVICE=cpu|gpu`.

### Which CUDA toolkit

With CUDA on, `build.rs` picks the toolkit itself and names it (and any it
skipped) in a build warning:

- `RLX_MLX_CUDA_ROOT` (a directory with `bin/nvcc`) chooses one outright.
  Otherwise: `CUDA_HOME`, `CUDA_PATH`, `/usr/local/cuda`, `/usr/local/cuda-*`
  (newest first), `/opt/cuda`, then Debian/Ubuntu's `nvidia-cuda-toolkit`.
- Versions MLX refuses are skipped — read from MLX's own `CMakeLists.txt`
  (currently 13.1), so the list follows the submodule.
- Ubuntu's toolkit splits itself across `/usr/lib/nvidia-cuda-toolkit/bin`,
  `/usr/include` and the multiarch lib dir; the build gives CMake a shim root
  in `OUT_DIR` for it.
- The CUDA host compiler is the toolkit's own `bin/g++` wrapper when it has
  one, else the newest `g++-N` within the toolkit's GCC limit when the default
  `c++` is too new. `CUDAHOSTCXX` overrides.
- cuDNN comes from `CUDNN_ROOT` / `CUDNN_PATH`, the toolkit, or the system dirs.
- The chosen toolkit is also the JIT's default header root at run time
  (`CUDA_HOME` / `CUDA_PATH` still override it), and the JIT uses the CCCL
  headers MLX installed into `OUT_DIR` (patch 0003).
- When CMake finds NCCL, MLX builds its NCCL distributed backend and the build
  links the same `libnccl`.

If CUDA is requested but nothing fits, the build says why per toolkit and
builds the CPU-only MLX. Binaries need the chosen toolkit's libraries on the
loader path (`LD_LIBRARY_PATH` for a `/usr/local/cuda-*` install).

`docker/cuda_matrix.py` checks all of this against NVIDIA's CUDA images
(12.8, 13.0, 13.1 refused, 13.2, two toolkits side by side), on the host's
architecture and without a GPU: toolkit selection, then the MLX CUDA compile
and the link of every rlx-mlx test binary. `--list` shows the scenarios.

Full guide: [`docs/benchmarks/mlx-linux.md`](../../../docs/benchmarks/mlx-linux.md).

### macOS / Windows

Requires macOS + Xcode (`xcrun metal`) for Metal kernels. CPU MLX builds on
Linux and Windows; on Linux, `lapacke.h` must be present (`liblapacke-dev`) or
`build.rs` bootstraps OpenBLAS into `OUT_DIR`.

## License

MIT OR Apache-2.0.
