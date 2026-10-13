#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Serve the rlx-web static demo (no wasm rebuild).

ES modules + wasm require http:// — do not open HTML as file://.

Examples:
    python3 crates/bindings/rlx-web/serve.py
    python3 crates/bindings/rlx-web/serve.py --backend npx
    python3 crates/bindings/rlx-web/serve.py --backend miniserve --port 8080
    python3 crates/bindings/rlx-web/serve.py --open vision-bench.html
"""

from __future__ import annotations

import argparse
import functools
import http.server
import shutil
import socket
import socketserver
import ssl
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB_DIR = HERE / "web"
PKG = WEB_DIR / "pkg" / "rlx_web.js"

BACKENDS = ("python", "npx", "miniserve", "basic-http-server")


def ensure_bundle() -> None:
    if not PKG.exists():
        raise SystemExit(
            f"wasm bundle missing: {PKG}\n"
            "  Build first:\n"
            "    just build-web --all\n"
            "    python3 crates/bindings/rlx-web/build.py --all"
        )


def lan_address() -> str | None:
    """This host's LAN IP, for opening the demo on a second device.

    Connecting a UDP socket sends nothing; it just asks the routing table which
    local address would be used, which beats `gethostbyname` on a multi-homed
    machine.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return str(s.getsockname()[0])
    except OSError:
        return None


def ensure_cert(lan: str | None) -> tuple[Path, Path]:
    """A self-signed cert for `--tls`, generated once into a temp dir.

    `RTCPeerConnection` is only available in a **secure context**, so the tab
    cluster's WebRTC mesh cannot come up over plain http on a LAN address — it
    falls back to same-browser tabs, which defeats the point of binding
    0.0.0.0. localhost is exempt, which is why this only bites on a second
    device. A self-signed cert makes the context secure; the browser will warn
    once and then behave.

    The cert covers the LAN IP as a SAN, because browsers reject a certificate
    whose subject does not match the address typed in.
    """
    out = Path(tempfile.gettempdir()) / "rlx-web-tls"
    out.mkdir(parents=True, exist_ok=True)
    cert, key = out / "cert.pem", out / "key.pem"
    if cert.exists() and key.exists():
        return cert, key

    openssl = shutil.which("openssl")
    if openssl is None:
        raise SystemExit(
            "--tls needs the `openssl` CLI to generate a self-signed cert.\n"
            "  Either install it, or serve over http and accept that the\n"
            "  WebRTC mesh will not start on devices other than this one."
        )
    sans = ["DNS:localhost", "IP:127.0.0.1"]
    if lan:
        sans.append(f"IP:{lan}")
    subprocess.run(
        [
            openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(cert), "-days", "365",
            "-subj", "/CN=rlx-web",
            "-addext", f"subjectAltName={','.join(sans)}",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return cert, key


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    """Static files with caching off.

    The demo is ES modules plus a wasm bundle, and a browser will happily reuse
    a cached module after a rebuild — which turns "did my fix land?" into an
    unanswerable question. Correctness of the dev loop beats a warm cache.
    """

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()


class ThreadingServer(socketserver.ThreadingTCPServer):
    """One thread per connection, and a reusable address.

    The single-threaded default serialises requests, which a page that pulls a
    5 MB wasm bundle plus a module worker notices.
    """

    daemon_threads = True
    allow_reuse_address = True


def serve_python(port: int, host: str, tls: bool = False) -> None:
    handler = functools.partial(NoCacheHandler, directory=str(WEB_DIR))
    with ThreadingServer((host, port), handler) as httpd:
        if tls:
            cert, key = ensure_cert(lan_address())
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(certfile=cert, keyfile=key)
            httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        httpd.serve_forever()


def serve_npx(port: int, host: str) -> None:
    if shutil.which("npx") is None:
        raise SystemExit("npx not found — install Node.js or use --backend python")
    subprocess.run(
        ["npx", "--yes", "serve", str(WEB_DIR), "-l", f"tcp://{host}:{port}"],
        check=True,
    )


def serve_miniserve(port: int, host: str) -> None:
    exe = shutil.which("miniserve")
    if exe is None:
        raise SystemExit(
            "miniserve not found.\n"
            "  Install: cargo install miniserve\n"
            "  Or use:  --backend python | npx"
        )
    subprocess.run([exe, "-i", host, "-p", str(port), str(WEB_DIR)], check=True)


def serve_basic_http(port: int, host: str) -> None:
    exe = shutil.which("basic-http-server")
    if exe is None:
        raise SystemExit(
            "basic-http-server not found.\n"
            "  Install: cargo install basic-http-server\n"
            "  Or use:  --backend python | npx"
        )
    subprocess.run([exe, str(WEB_DIR), "--addr", f"{host}:{port}"], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve rlx-web static files.")
    ap.add_argument(
        "--backend",
        choices=BACKENDS,
        default="python",
        help="static file server (default: python stdlib)",
    )
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--host",
        default="0.0.0.0",  # noqa: S104 - a LAN demo is the point
        help="interface to bind (default: 0.0.0.0, reachable from other devices)",
    )
    ap.add_argument(
        "--open",
        default="vision-bench.html",
        help="page to open in browser (default: vision-bench.html)",
    )
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument(
        "--tls",
        action="store_true",
        help=(
            "serve https with a self-signed cert (python backend only). "
            "Needed for the tab cluster's WebRTC mesh on any device other than "
            "this one: RTCPeerConnection requires a secure context."
        ),
    )
    ap.add_argument("--skip-check", action="store_true", help="do not require pkg/ bundle")
    args = ap.parse_args()

    if not args.skip_check:
        ensure_bundle()

    if args.tls and args.backend != "python":
        raise SystemExit(f"--tls is only implemented for --backend python, not {args.backend}")
    scheme = "https" if args.tls else "http"
    url = f"{scheme}://127.0.0.1:{args.port}/{args.open}"
    print(f"Serving {WEB_DIR}")
    print(f"  backend: {args.backend}")
    print(f"  bind:    {args.host}:{args.port}")
    print(f"  URL:     {url}")
    if args.host in ("0.0.0.0", "::", ""):
        lan = lan_address()
        if lan:
            print(f"  LAN:     {scheme}://{lan}:{args.port}/{args.open}")
            if not args.tls:
                print("           (the cluster's WebRTC mesh needs a secure context —")
                print("            add --tls to reach it from another device)")
    print("  Pages:   index.html (MLP), vision-bench.html (vision), cluster.html (tab cluster)")
    print("  Ctrl-C to stop\n")

    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    runners = {
        "python": serve_python,
        "npx": serve_npx,
        "miniserve": serve_miniserve,
        "basic-http-server": serve_basic_http,
    }
    try:
        if args.backend == "python":
            serve_python(args.port, args.host, args.tls)
        else:
            runners[args.backend](args.port, args.host)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
