// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA toolkit discovery for MLX's Linux CUDA backend (`build.rs`).
//!
//! A machine can carry several toolkits: `/usr/local/cuda` (a symlink to one of
//! `/usr/local/cuda-X.Y`), others beside it, `/opt/cuda`, and Debian/Ubuntu's
//! `nvidia-cuda-toolkit`, which cuDNN's distro package pulls in and which
//! splits itself across `/usr/lib/nvidia-cuda-toolkit/bin`, `/usr/include` and
//! the multiarch lib dir. This picks one MLX can build with:
//!
//! - `RLX_MLX_CUDA_ROOT` names a toolkit outright; otherwise `CUDA_HOME`,
//!   `CUDA_PATH`, then the standard locations, newest version first.
//! - Versions MLX refuses are skipped. The ranges are read from MLX's own
//!   `CMakeLists.txt` (`CUDA Toolkit 13.1 is not supported`), so they follow
//!   the submodule.
//! - A split layout gets a shim root in `OUT_DIR` (a copy of `nvcc` and its
//!   profile, symlinks to the headers and libraries): CMake derives the toolkit
//!   root from where `nvcc` lives, and finds no headers next to the distro one.
//! - The CUDA host compiler is the toolkit's own `bin/g++` wrapper when it has
//!   one, else the newest `g++-N` within the GCC limit in the toolkit's
//!   `crt/host_config.h` when the default `c++` is newer. `CUDAHOSTCXX` wins.
//! - cuDNN is looked for under `CUDNN_ROOT` / `CUDNN_PATH`, the toolkit, and
//!   the system include / multiarch lib dirs.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The toolkit MLX builds and links against.
pub struct CudaToolkit {
    /// What CMake sees as the toolkit root: the install, or the shim.
    pub root: PathBuf,
    pub nvcc: PathBuf,
    pub version: (u32, u32),
    pub lib_dirs: Vec<PathBuf>,
    /// `CMAKE_CUDA_HOST_COMPILER`, when the default would not do.
    pub host_cxx: Option<PathBuf>,
    /// Where it was found (differs from `root` for a shimmed layout).
    pub origin: PathBuf,
    pub cudnn_include: PathBuf,
    pub cudnn_lib: Option<PathBuf>,
    /// Toolkits passed over on the way, with the reason.
    pub skipped: Vec<String>,
}

impl CudaToolkit {
    pub fn describe(&self) -> String {
        let mut s = format!(
            "CUDA {}.{} at {}",
            self.version.0,
            self.version.1,
            self.origin.display()
        );
        if self.root != self.origin {
            s += &format!(" (split layout, shim {})", self.root.display());
        }
        if let Some(h) = &self.host_cxx {
            s += &format!(", host compiler {}", h.display());
        }
        s += &format!(", cuDNN {}", self.cudnn_include.display());
        if !self.skipped.is_empty() {
            s += &format!(" (skipped {})", self.skipped.join("; "));
        }
        s
    }
}

pub const ENV_VARS: &[&str] = &[
    "RLX_MLX_CUDA_ROOT",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDNN_ROOT",
    "CUDNN_PATH",
    "CUDAHOSTCXX",
];

/// Pick a toolkit, or say why none fits (each candidate and what was wrong
/// with it).
pub fn select(mlx_src: &Path, out_dir: &Path) -> Result<CudaToolkit, String> {
    let refused = mlx_refused_versions(mlx_src);
    let explicit = env::var_os("RLX_MLX_CUDA_ROOT").filter(|r| !r.is_empty());
    let candidates = match &explicit {
        Some(root) => vec![PathBuf::from(root)],
        None => candidate_roots(),
    };
    let mut rejected = Vec::new();
    let mut seen = Vec::new();
    for root in candidates {
        let Some(nvcc) = Some(root.join("bin/nvcc")).filter(|n| n.is_file()) else {
            if explicit.is_some() {
                rejected.push(format!("{}: no bin/nvcc", root.display()));
            }
            continue;
        };
        // `/usr/local/cuda` and the `cuda-X.Y` it points to are one toolkit.
        let canon = nvcc.canonicalize().unwrap_or_else(|_| nvcc.clone());
        if seen.contains(&canon) {
            continue;
        }
        seen.push(canon);
        match inspect(&root, &nvcc, &refused, out_dir) {
            Ok(tk) => {
                return Ok(CudaToolkit {
                    skipped: rejected,
                    ..tk
                });
            }
            Err(why) => rejected.push(format!("{}: {why}", root.display())),
        }
    }
    if rejected.is_empty() {
        Err(
            "no CUDA toolkit found (looked for bin/nvcc under RLX_MLX_CUDA_ROOT, \
             CUDA_HOME, CUDA_PATH, /usr/local/cuda*, /opt/cuda, /usr/lib/nvidia-cuda-toolkit)"
                .into(),
        )
    } else {
        Err(format!("no usable CUDA toolkit — {}", rejected.join("; ")))
    }
}

fn candidate_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = ["CUDA_HOME", "CUDA_PATH"]
        .iter()
        .filter_map(|v| env::var_os(v).filter(|p| !p.is_empty()).map(PathBuf::from))
        .collect();
    roots.push(PathBuf::from("/usr/local/cuda"));
    let mut versioned: Vec<(Vec<u32>, PathBuf)> = std::fs::read_dir("/usr/local")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let v = name.strip_prefix("cuda-")?;
            let parts: Vec<u32> = v.split('.').map_while(|p| p.parse().ok()).collect();
            (!parts.is_empty()).then(|| (parts, e.path()))
        })
        .collect();
    versioned.sort_by(|a, b| b.0.cmp(&a.0));
    roots.extend(versioned.into_iter().map(|(_, p)| p));
    roots.push(PathBuf::from("/opt/cuda"));
    roots.push(PathBuf::from("/usr/lib/nvidia-cuda-toolkit"));
    roots
}

fn inspect(
    root: &Path,
    nvcc: &Path,
    refused: &[((u32, u32), (u32, u32))],
    out_dir: &Path,
) -> Result<CudaToolkit, String> {
    let version = nvcc_version(nvcc).ok_or("could not read `nvcc --version`")?;
    if let Some((lo, hi)) = refused
        .iter()
        .find(|(lo, hi)| version >= *lo && version < *hi)
    {
        return Err(format!(
            "CUDA {}.{} — MLX does not build with {}.{}..{}.{}",
            version.0, version.1, lo.0, lo.1, hi.0, hi.1
        ));
    }
    let triple = multiarch_triple();
    let target = target_dir_name();
    let include = [
        root.join("include"),
        root.join(format!("targets/{target}/include")),
        PathBuf::from("/usr/include"),
    ]
    .into_iter()
    .find(|d| d.join("cuda_runtime.h").is_file())
    .ok_or("no cuda_runtime.h")?;
    let lib_dirs: Vec<PathBuf> = [
        root.join("lib64"),
        root.join(format!("targets/{target}/lib")),
        PathBuf::from(format!("/usr/lib/{triple}")),
    ]
    .into_iter()
    .filter(|d| d.join("libcudart.so").exists())
    .collect();
    if lib_dirs.is_empty() {
        return Err("no libcudart.so".into());
    }
    let (cudnn_include, cudnn_lib) =
        find_cudnn(&include, &lib_dirs, &triple).ok_or("no cuDNN (cudnn.h)")?;
    // Headers outside the root: CMake will not find them from `nvcc`'s
    // location, so give it a root that has both.
    let split = !include.starts_with(root);
    let (cmake_root, cmake_nvcc) = if split {
        let shim = shim_root(root, nvcc, &include, &lib_dirs[0], version, out_dir)?;
        let nvcc = shim.join("bin/nvcc");
        (shim, nvcc)
    } else {
        (root.to_path_buf(), nvcc.to_path_buf())
    };
    Ok(CudaToolkit {
        host_cxx: host_compiler(&root.join("bin"), &include),
        root: cmake_root,
        nvcc: cmake_nvcc,
        version,
        lib_dirs,
        origin: root.to_path_buf(),
        cudnn_include,
        cudnn_lib,
        skipped: Vec::new(),
    })
}

fn nvcc_version(nvcc: &Path) -> Option<(u32, u32)> {
    let out = Command::new(nvcc).arg("--version").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let rest = text.split("release ").nth(1)?;
    let ver = rest.split(|c: char| c == ',' || c.is_whitespace()).next()?;
    let mut it = ver.split('.').map(|p| p.parse::<u32>().ok());
    Some((it.next()??, it.next()??))
}

/// `[lo, hi)` version ranges MLX's CMake rejects, from
/// `CUDAToolkit_VERSION VERSION_GREATER_EQUAL "lo" AND CUDAToolkit_VERSION
/// VERSION_LESS "hi"` followed by a `FATAL_ERROR`.
fn mlx_refused_versions(mlx_src: &Path) -> Vec<((u32, u32), (u32, u32))> {
    let text = std::fs::read_to_string(mlx_src.join("CMakeLists.txt")).unwrap_or_default();
    let quoted_after = |s: &str, key: &str| -> Option<(u32, u32)> {
        let rest = &s[s.find(key)? + key.len()..];
        let v = rest.trim_start().strip_prefix('"')?.split('"').next()?;
        let mut it = v.split('.').map(|p| p.parse::<u32>().ok());
        Some((it.next()??, it.next().flatten().unwrap_or(0)))
    };
    let mut out = Vec::new();
    let mut rest = text.as_str();
    while let Some(i) = rest.find("VERSION_GREATER_EQUAL") {
        let chunk = &rest[i..rest.len().min(i + 400)];
        let cond_end = chunk.find(')').unwrap_or(chunk.len());
        let cond = &chunk[..cond_end];
        if chunk[cond_end..]
            .trim_start_matches(')')
            .trim_start()
            .starts_with("message(FATAL_ERROR")
            && let (Some(lo), Some(hi)) = (
                quoted_after(cond, "VERSION_GREATER_EQUAL"),
                quoted_after(cond, "VERSION_LESS"),
            )
        {
            out.push((lo, hi));
        }
        rest = &rest[i + "VERSION_GREATER_EQUAL".len()..];
    }
    out
}

fn find_cudnn(
    cuda_include: &Path,
    cuda_libs: &[PathBuf],
    triple: &str,
) -> Option<(PathBuf, Option<PathBuf>)> {
    let roots: Vec<PathBuf> = ["CUDNN_ROOT", "CUDNN_PATH"]
        .iter()
        .filter_map(|v| env::var_os(v).filter(|p| !p.is_empty()).map(PathBuf::from))
        .collect();
    let mut includes: Vec<PathBuf> = roots.iter().map(|r| r.join("include")).collect();
    includes.push(cuda_include.to_path_buf());
    includes.push(PathBuf::from(format!("/usr/include/{triple}")));
    includes.push(PathBuf::from("/usr/include"));
    let include = includes.into_iter().find(|d| d.join("cudnn.h").is_file())?;
    let mut libs: Vec<PathBuf> = roots
        .iter()
        .flat_map(|r| [r.join("lib64"), r.join("lib")])
        .collect();
    libs.extend(cuda_libs.iter().cloned());
    libs.push(PathBuf::from(format!("/usr/lib/{triple}")));
    let lib = libs.into_iter().find(|d| d.join("libcudnn.so").exists());
    Some((include, lib))
}

/// A root CMake accepts for a split install: `bin/nvcc` (+ profile) copied,
/// so `nvcc` reports this directory as its home, and `include` / `lib64` /
/// `bin/crt` symlinked. Rebuilt only when the source `nvcc` changes size.
fn shim_root(
    root: &Path,
    nvcc: &Path,
    include: &Path,
    lib: &Path,
    version: (u32, u32),
    out_dir: &Path,
) -> Result<PathBuf, String> {
    let shim = out_dir.join(format!("cuda-{}.{}-shim", version.0, version.1));
    let bin = shim.join("bin");
    let err = |e: std::io::Error| format!("shim {}: {e}", shim.display());
    std::fs::create_dir_all(&bin).map_err(err)?;
    let same_size = |a: &Path, b: &Path| match (a.metadata(), b.metadata()) {
        (Ok(x), Ok(y)) => x.len() == y.len(),
        _ => false,
    };
    let dst = bin.join("nvcc");
    if !same_size(nvcc, &dst) {
        std::fs::copy(nvcc, &dst).map_err(err)?;
    }
    let profile = root.join("bin/nvcc.profile");
    if profile.exists() {
        std::fs::copy(&profile, bin.join("nvcc.profile")).map_err(err)?;
    }
    let link = |target: &Path, at: PathBuf| -> Result<(), String> {
        if !target.exists() {
            return Ok(());
        }
        let _ = std::fs::remove_file(&at);
        // The build script also compiles on Windows hosts; only Linux
        // (where MLX's CUDA backend builds) ever reaches this.
        #[cfg(unix)]
        return std::os::unix::fs::symlink(target, &at).map_err(err);
        #[cfg(not(unix))]
        return Err(format!("cannot symlink {} on this host", at.display()));
    };
    link(&root.join("bin/crt"), bin.join("crt"))?;
    link(include, shim.join("include"))?;
    link(lib, shim.join("lib64"))?;
    Ok(shim)
}

/// The CUDA host compiler, when CMake's default (`c++`) would not do.
fn host_compiler(bin: &Path, include: &Path) -> Option<PathBuf> {
    if env::var_os("CUDAHOSTCXX").is_some() {
        return None;
    }
    let wrapper = bin.join("g++");
    if wrapper.is_file() {
        return Some(wrapper);
    }
    let config = std::fs::read_to_string(include.join("crt/host_config.h")).ok()?;
    let max: u32 = config
        .split("#if __GNUC__ > ")
        .nth(1)?
        .split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()?;
    let out = Command::new("c++").arg("-dumpversion").output().ok()?;
    let default: u32 = String::from_utf8_lossy(&out.stdout)
        .trim()
        .split('.')
        .next()?
        .parse()
        .ok()?;
    if default <= max {
        return None;
    }
    (max.saturating_sub(4)..=max)
        .rev()
        .map(|v| PathBuf::from(format!("/usr/bin/g++-{v}")))
        .find(|p| p.is_file())
}

fn multiarch_triple() -> String {
    match env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") => "aarch64-linux-gnu".into(),
        _ => "x86_64-linux-gnu".into(),
    }
}

fn target_dir_name() -> &'static str {
    match env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") => "sbsa-linux",
        _ => "x86_64-linux",
    }
}
