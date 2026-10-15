// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compiles `compat/compat.c` with the SDK's `prospero-clang` and links it.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=compat/compat.c");
    println!("cargo:rerun-if-env-changed=PS5_PAYLOAD_SDK");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("freebsd") {
        return;
    }
    let sdk = PathBuf::from(
        std::env::var("PS5_PAYLOAD_SDK").expect("set PS5_PAYLOAD_SDK to the ps5-payload-dev SDK"),
    );
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let obj = out.join("compat.o");
    let lib = out.join("librlxps5compat.a");
    let run = |cmd: &mut Command| {
        let st = cmd.status().expect("spawn SDK tool");
        assert!(st.success(), "{cmd:?} failed");
    };
    run(Command::new(sdk.join("bin/prospero-clang"))
        .args([
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-c",
            "compat/compat.c",
            "-o",
        ])
        .arg(&obj));
    let _ = std::fs::remove_file(&lib);
    run(Command::new(sdk.join("bin/prospero-ar"))
        .arg("rcs")
        .arg(&lib)
        .arg(&obj));
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=rlxps5compat");
}
