// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! PS5 bring-up probe: exercises every part of the prebuilt FreeBSD std that
//! rlx-cpu relies on (threads, TLS, sync, files + `stat`/`dirent` layouts,
//! mmap, timing, panics/unwinding) and prints one `ok`/`FAIL` line per check.
//! Results also go to `/data/rlx-ps5/probe.txt` on the console.

use std::cell::Cell;
use std::fmt::Write as _;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

const DIR: &str = "/data/rlx-ps5";

thread_local! {
    static TLS: Cell<usize> = const { Cell::new(0) };
}

fn check(out: &mut String, name: &str, r: Result<String, String>) {
    let line = match r {
        Ok(v) => format!("ok   {name}: {v}"),
        Err(e) => format!("FAIL {name}: {e}"),
    };
    println!("{line}");
    let _ = writeln!(out, "{line}");
}

fn threads_and_tls() -> Result<String, String> {
    let hits = Arc::new(AtomicUsize::new(0));
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let hits = hits.clone();
            std::thread::Builder::new()
                .name(format!("probe-{i}"))
                .spawn(move || {
                    TLS.with(|t| t.set(i + 1));
                    std::thread::yield_now();
                    let mine = TLS.with(|t| t.get());
                    if mine == i + 1 {
                        hits.fetch_add(1, Ordering::SeqCst);
                    }
                })
                .map_err(|e| e.to_string())
        })
        .collect::<Result<_, _>>()?;
    for h in handles {
        h.join().map_err(|_| "join panicked".to_string())?;
    }
    let main_tls = TLS.with(|t| t.get());
    match (hits.load(Ordering::SeqCst), main_tls) {
        (4, 0) => Ok("4 threads, per-thread TLS distinct".into()),
        (n, m) => Err(format!("{n}/4 threads saw own TLS, main TLS = {m}")),
    }
}

fn condvar() -> Result<String, String> {
    let pair = Arc::new((Mutex::new(false), Condvar::new()));
    let p2 = pair.clone();
    let t = std::thread::spawn(move || {
        *p2.0.lock().unwrap() = true;
        p2.1.notify_one();
    });
    let (lock, cv) = &*pair;
    let mut ready = lock.lock().unwrap();
    while !*ready {
        ready = cv.wait(ready).unwrap();
    }
    t.join().map_err(|_| "join".to_string())?;
    Ok("mutex + condvar handoff".into())
}

fn files() -> Result<String, String> {
    std::fs::create_dir_all(DIR).map_err(|e| format!("mkdir: {e}"))?;
    let path = format!("{DIR}/probe.bin");
    let data: Vec<u8> = (0..123_457u32).map(|i| (i * 31) as u8).collect();
    std::fs::File::create(&path)
        .and_then(|mut f| f.write_all(&data))
        .map_err(|e| format!("write: {e}"))?;
    // `metadata` goes through `stat`: a struct-layout mismatch shows up as a
    // wrong length, not an error.
    let len = std::fs::metadata(&path)
        .map_err(|e| format!("stat: {e}"))?
        .len();
    if len != data.len() as u64 {
        return Err(format!("stat len {len} != {}", data.len()));
    }
    let mut back = Vec::new();
    std::fs::File::open(&path)
        .and_then(|mut f| f.read_to_end(&mut back))
        .map_err(|e| format!("read: {e}"))?;
    if back != data {
        return Err("read-back differs".into());
    }
    // `read_dir` goes through `dirent`: check the name comes back intact.
    let names: Vec<String> = std::fs::read_dir(DIR)
        .map_err(|e| format!("readdir: {e}"))?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    if !names.iter().any(|n| n == "probe.bin") {
        return Err(format!("readdir names {names:?}"));
    }
    Ok(format!("write/stat/read/readdir, {len} B"))
}

fn mmap() -> Result<String, String> {
    let path = format!("{DIR}/probe.bin");
    let f = std::fs::File::open(&path).map_err(|e| e.to_string())?;
    let len = f.metadata().map_err(|e| e.to_string())?.len() as usize;
    use std::os::fd::AsRawFd;
    // SAFETY: read-only private map of a file we just wrote; unmapped below.
    let p = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_PRIVATE,
            f.as_raw_fd(),
            0,
        )
    };
    if p == libc::MAP_FAILED {
        return Err(format!("mmap: {}", std::io::Error::last_os_error()));
    }
    // SAFETY: `p` maps `len` readable bytes.
    let s = unsafe { std::slice::from_raw_parts(p as *const u8, len) };
    let ok = s
        .iter()
        .enumerate()
        .all(|(i, &b)| b == (i as u32 * 31) as u8);
    // SAFETY: matches the mmap above.
    unsafe { libc::munmap(p, len) };
    if ok {
        Ok(format!("{len} B file map matches"))
    } else {
        Err("mapped bytes differ".into())
    }
}

fn big_alloc() -> Result<String, String> {
    // A model-sized block, touched page by page (rlx arenas are one block).
    let n = 1usize << 30;
    let mut v: Vec<u8> = Vec::new();
    v.try_reserve_exact(n).map_err(|e| e.to_string())?;
    v.resize(n, 0);
    for i in (0..n).step_by(4096) {
        v[i] = i as u8;
    }
    Ok("1 GiB allocated and touched".into())
}

fn cpus() -> Result<String, String> {
    let ap = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|e| format!("err({e})"));
    let mask = rlx_ps5::affinity_mask().ok_or("scePthreadGetaffinity failed")?;
    Ok(format!(
        "available_parallelism={ap} affinity={mask:#x} ({} cpus)",
        mask.count_ones()
    ))
}

fn simd() -> Result<String, String> {
    let f = [
        ("avx2", std::arch::is_x86_feature_detected!("avx2")),
        ("fma", std::arch::is_x86_feature_detected!("fma")),
        ("f16c", std::arch::is_x86_feature_detected!("f16c")),
        ("avx512f", std::arch::is_x86_feature_detected!("avx512f")),
    ];
    Ok(f.iter()
        .map(|(n, b)| format!("{n}={b}"))
        .collect::<Vec<_>>()
        .join(" "))
}

fn timing() -> Result<String, String> {
    let t = Instant::now();
    let mut acc = 0f32;
    for i in 0..10_000_000u32 {
        acc = acc.mul_add(1.000_000_1, i as f32 * 1e-9);
    }
    let dt = t.elapsed();
    if dt.is_zero() {
        return Err("Instant did not advance".into());
    }
    Ok(format!(
        "10M fma in {:.2} ms (acc {acc:.3}), sin(1)={:.6}",
        dt.as_secs_f64() * 1e3,
        1f64.sin()
    ))
}

fn unwind() -> Result<String, String> {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(|| panic!("probe"));
    std::panic::set_hook(prev);
    match r {
        Err(_) => Ok("panic caught by catch_unwind".into()),
        Ok(()) => Err("no panic?".into()),
    }
}

fn main() {
    let mut out = String::new();
    check(&mut out, "cpus", cpus());
    check(&mut out, "simd", simd());
    check(&mut out, "threads+tls", threads_and_tls());
    check(&mut out, "condvar", condvar());
    check(&mut out, "files", files());
    check(&mut out, "mmap", mmap());
    check(&mut out, "alloc", big_alloc());
    check(&mut out, "timing", timing());
    check(&mut out, "unwind", unwind());
    let _ = std::fs::write(format!("{DIR}/probe.txt"), &out);
    println!("done");
    rlx_ps5::exit(i32::from(out.contains("FAIL")));
}
