// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! PS5 payload support for RLX: the platform layer a payload links.
//!
//! Call [`init`] first thing in `main`, before any RLX compile or run starts
//! the CPU thread pool; lock model weights with [`lock`]; leave through
//! [`exit`]. Depending on this crate also links `compat/compat.c` (the libc
//! functions std needs that Sony's libraries lack), so a payload crate
//! outside this directory builds with `just ps5 build --manifest-path …`.

use std::sync::atomic::{AtomicUsize, Ordering};

/// The CPUs this thread may run on (Sony's affinity mask). A payload gets 5
/// of the 16 (`0xea00` on firmware 13.60), shared with system services.
/// `None` off the console, where the payloads also build for host checks.
#[cfg(target_os = "freebsd")]
pub fn affinity_mask() -> Option<u64> {
    unsafe extern "C" {
        fn scePthreadGetaffinity(thread: libc::pthread_t, mask: *mut u64) -> i32;
    }
    let mut mask = 0u64;
    // SAFETY: Sony libkernel export; `mask` is a valid out-pointer.
    let r = unsafe { scePthreadGetaffinity(libc::pthread_self(), &mut mask) };
    (r == 0 && mask != 0).then_some(mask)
}

#[cfg(not(target_os = "freebsd"))]
pub fn affinity_mask() -> Option<u64> {
    None
}

/// Size the RLX CPU pool for the console: the allowed CPUs minus two, so the
/// system services that share them keep running (PS5LM found that using all
/// of them shut the console down). An explicit `RLX_WORKERS` is kept.
/// Returns the worker count in effect.
pub fn init() -> usize {
    if let Some(n) = rlx_ir::env::var("RLX_WORKERS")
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
    {
        return n;
    }
    let n = affinity_mask().map_or(2, |m| (m.count_ones() as usize).saturating_sub(2).max(1));
    rlx_ir::env::set("RLX_WORKERS", n.to_string());
    n
}

/// Set an RLX option unless the environment (or an earlier override) already
/// has it. The console starts a payload with no environment, so this is how a
/// payload chooses its defaults while a host run can still override them.
pub fn set_default(key: &str, value: &str) {
    if rlx_ir::env::is_unset(key) {
        rlx_ir::env::set(key, value);
    }
}

/// Leave the payload: flush stdout/stderr, then `_exit` without running
/// `atexit` handlers or static destructors.
///
/// Returning from `main` hands control to the SDK's start code, which unloads
/// the payload's Sony modules and calls Sony's `exit()` while the RLX pool's
/// threads are still alive. PS5LM traced `[SceLibc] A heap error is detected`
/// to that path and leaves through `_Exit` the same way (their issue #3).
pub fn exit(code: i32) -> ! {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // SAFETY: `_exit` never returns and touches no Rust-managed state.
    unsafe { libc::_exit(code) }
}

/// Most memory a payload can keep locked without starving the system. PS5LM
/// measured it on firmware 13.60: 1.4 GB locked ran, 2.7 GB froze the console
/// (the locked pages come out of the pool the home screen needs). [`lock`]
/// refuses past this.
pub const LOCK_BUDGET_BYTES: usize = 1_600_000_000;

static LOCKED: AtomicUsize = AtomicUsize::new(0);

/// Bytes locked so far through [`lock`].
pub fn locked_bytes() -> usize {
    LOCKED.load(Ordering::Relaxed)
}

/// Keep `data` resident (`mlock`). Without it the system pages a payload's
/// memory out to disk while the browser is in front, and the next token pays
/// for paging it back in — PS5LM measured 10–65× slower decode. Lock the
/// weights once they are loaded. Fails without locking anything when it would
/// take the total past the budget ([`LOCK_BUDGET_BYTES`]) or `mlock` refuses.
pub fn lock<T>(data: &[T]) -> Result<(), String> {
    let len = std::mem::size_of_val(data);
    if len == 0 {
        return Ok(());
    }
    let budget = LOCK_BUDGET_BYTES;
    let prev = LOCKED.fetch_add(len, Ordering::Relaxed);
    if prev + len > budget {
        LOCKED.fetch_sub(len, Ordering::Relaxed);
        return Err(format!(
            "locking {} MB would hold {} MB, past the {} MB a payload can lock",
            len >> 20,
            (prev + len) >> 20,
            budget >> 20
        ));
    }
    // SAFETY: `data` is a live borrow covering `len` bytes.
    if unsafe { libc::mlock(data.as_ptr().cast(), len) } != 0 {
        LOCKED.fetch_sub(len, Ordering::Relaxed);
        return Err(format!(
            "mlock {} MB: {}",
            len >> 20,
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

/// Flexible memory the system can still hand this process, in bytes. A
/// payload has no direct memory (its budget is 0), so this is the ceiling for
/// weights, KV cache and buffers together. `None` off the console.
#[cfg(target_os = "freebsd")]
pub fn available_memory() -> Option<usize> {
    unsafe extern "C" {
        fn sceKernelAvailableFlexibleMemorySize(size: *mut usize) -> i32;
    }
    let mut n = 0usize;
    // SAFETY: Sony libkernel export; `n` is a valid out-pointer.
    (unsafe { sceKernelAvailableFlexibleMemorySize(&mut n) } == 0).then_some(n)
}

#[cfg(not(target_os = "freebsd"))]
pub fn available_memory() -> Option<usize> {
    None
}

/// Peak resident memory of this process, in bytes (`getrusage`).
pub fn peak_resident_bytes() -> usize {
    // SAFETY: plain out-struct for getrusage.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `ru` is a valid out-pointer.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } != 0 {
        return 0;
    }
    // FreeBSD (and Linux) report kilobytes; macOS reports bytes.
    let rss = ru.ru_maxrss.max(0) as usize;
    if cfg!(target_vendor = "apple") {
        rss
    } else {
        rss << 10
    }
}

/// Lock everything resident now (`mlockall(MCL_CURRENT)`), for when the
/// weights live in buffers the payload cannot name — a runner that owns its
/// arenas. Call once loading is done. Refuses when the process's peak resident
/// size is past [`LOCK_BUDGET_BYTES`], which is conservative: the peak counts
/// load-time scratch that is no longer mapped.
pub fn lock_resident() -> Result<usize, String> {
    let rss = peak_resident_bytes();
    if rss > LOCK_BUDGET_BYTES {
        return Err(format!(
            "{} MB resident is past the {} MB a payload can lock; not locking (expect paging \
             slowdowns while the browser is in front)",
            rss >> 20,
            LOCK_BUDGET_BYTES >> 20
        ));
    }
    mlock_all()?;
    LOCKED.fetch_max(rss, Ordering::Relaxed);
    Ok(rss)
}

#[cfg(target_os = "freebsd")]
fn mlock_all() -> Result<(), String> {
    // SAFETY: no pointers involved.
    if unsafe { libc::mlockall(libc::MCL_CURRENT) } != 0 {
        return Err(format!("mlockall: {}", std::io::Error::last_os_error()));
    }
    Ok(())
}

#[cfg(not(target_os = "freebsd"))]
fn mlock_all() -> Result<(), String> {
    Err("lock_resident is console-only".into())
}
