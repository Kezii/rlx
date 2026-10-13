#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Run a TURN relay for the tab cluster, and print the `?turn=` parameter to use.

Why this exists: **there is no usable no-signup public TURN any more.** Checked
2026-10-12 —

  * `openrelay.metered.ca` (the Open Relay Project, the one every list still
    recommends) resolves to `192.0.2.1`, an RFC 5737 documentation address. It
    is parked; all ports closed. Its successor `relay.metered.ca` is alive but
    rejects the old open credentials.
  * `freeturn.net` no longer resolves at all.
  * `turn.anyfirewall.com` and `numb.viagenie.ca` are gone.
  * `turn.cloudflare.com` is alive but issues short-lived credentials through an
    authenticated API, so it needs an account. `--cloudflare` below mints a set
    for you if you have a TURN key; the free allowance is 1,000 GB/month of
    egress shared with Realtime SFU, which at this demo's measured ~30 MB/s is
    about nine hours of cluster time.

Public **STUN** is still plentiful and is configured by default — that is the
cheap half, since STUN only reports an address and relays nothing. TURN carries
every byte, which is exactly why nobody gives it away.

For this cluster TURN is a *connectivity test*, not a transport: a round ships
the whole parameter vector, so even a 20 GB monthly allowance would be gone in
minutes. Use it to prove the peers can talk, then move both devices onto one
network for real runs.

Usage:
    python3 crates/bindings/rlx-web/tools/turn.py
    # → prints the ?turn=… to append to the room URL
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

CF_ENDPOINT = (
    "https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate-ice-servers"
)

# Which of Cloudflare's six returned URLs to put in the share link. UDP first
# because it is the fast path; TLS-over-443 second because it is the one that
# survives networks that block everything else. The rest only lengthen a URL
# that has to fit in a QR code.
CF_PREFERRED = (
    "turn:turn.cloudflare.com:3478?transport=udp",
    "turns:turn.cloudflare.com:443?transport=tcp",
)


def cloudflare_credentials(key_id: str, token: str, ttl: int) -> list[dict]:
    """Mint short-lived TURN credentials.

    Cloudflare has no long-lived option by design: the API derives temporary
    credentials from a long-term key that never leaves your account. That is
    the right shape for a static page — the token stays here, and only the
    expiring credential travels — but it does mean the `?turn=` link goes stale
    after `ttl` seconds, and until then **anyone holding it can relay through
    your account**. Keep the TTL near the length of the test.
    """
    request = urllib.request.Request(
        CF_ENDPOINT.format(key_id=key_id),
        data=json.dumps({"ttl": ttl}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:400]
        raise SystemExit(
            f"Cloudflare returned {e.code}: {detail}\n"
            "  A 403 usually means the token lacks the TURN permission, or the\n"
            "  key id belongs to a different account."
        ) from e
    except urllib.error.URLError as e:
        raise SystemExit(f"could not reach Cloudflare: {e.reason}") from e

    servers = payload.get("iceServers")
    if not servers:
        raise SystemExit(f"unexpected response shape: {json.dumps(payload)[:400]}")
    return servers if isinstance(servers, list) else [servers]


def cloudflare_mode(ttl: int) -> int:
    key_id = os.environ.get("TURN_KEY_ID", "")
    token = os.environ.get("TURN_KEY_API_TOKEN", "")
    if not key_id or not token:
        raise SystemExit(
            "--cloudflare needs a TURN key from the Cloudflare dashboard:\n"
            "    export TURN_KEY_ID=...\n"
            "    export TURN_KEY_API_TOKEN=...\n"
            "  Realtime → TURN → create a key. Free allowance is 1,000 GB/month\n"
            "  of egress, shared with Realtime SFU."
        )

    entries = []
    stun = []
    for server in cloudflare_credentials(key_id, token, ttl):
        urls = server.get("urls") or []
        urls = [urls] if isinstance(urls, str) else urls
        user, cred = server.get("username"), server.get("credential")
        if not user or not cred:
            stun.extend(u for u in urls if u.startswith("stun:"))
            continue
        for url in CF_PREFERRED:
            if url in urls:
                entries.append(f"{url}|{user}|{cred}")
    if not entries:
        raise SystemExit("response carried no credentialed TURN urls")

    print(f"Cloudflare TURN credentials, valid {ttl}s (~{ttl / 3600:.1f}h)")
    print()
    print("Append to the room URL on *both* devices:")
    print("  &turn=" + ";".join(entries))
    if stun:
        print(f"  (their STUN is also available: &ice={','.join(stun)})")
    print()
    print("These are bearer credentials until they expire — anyone with the link")
    print("can relay through your account, so keep the TTL near the length of the")
    print("test. Revoke early with:")
    print("  curl --request POST \\")
    print("    https://rtc.live.cloudflare.com/v1/turn/keys/$TURN_KEY_ID/credentials/<USERNAME>/revoke \\")
    print('    --header "Authorization: Bearer $TURN_KEY_API_TOKEN"')
    return 0

INSTALL_HINT = """turnserver (coturn) not found.

  macOS:   brew install coturn
  Debian:  sudo apt install coturn
  Fedora:  sudo dnf install coturn

A relay on this machine fixes the case where the two devices *can* reach each
other but NAT will not let them pair directly. It does **not** help when the
network blocks client-to-client traffic outright (Wi-Fi "client isolation"),
because the relay is just another client on that same network — put the devices
on a network without isolation, or host the relay somewhere both can reach.
"""


def lan_address() -> str:
    """This host's LAN IP — the address the other device must dial."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return str(s.getsockname()[0])
    except OSError:
        return "127.0.0.1"


def config(realm: str, user: str, password: str, port: int, relay_ip: str) -> str:
    """A minimal long-term-credential coturn config.

    No TLS: this is a LAN debugging relay, and a self-signed cert would add a
    second trust prompt on the phone for nothing. The credentials are generated
    per run so a stale URL cannot keep using it.
    """
    return "\n".join(
        [
            f"listening-port={port}",
            f"listening-ip={relay_ip}",
            f"relay-ip={relay_ip}",
            f"realm={realm}",
            "fingerprint",
            "lt-cred-mech",
            f"user={user}:{password}",
            # Keep it to what the demo needs; no TCP relay, no quota games.
            "no-tls",
            "no-dtls",
            "no-cli",
            "simple-log",
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=3478)
    ap.add_argument("--user", default="rlx")
    ap.add_argument("--password", default="", help="default: a fresh random one")
    ap.add_argument("--host", default="", help="address to advertise (default: this LAN IP)")
    ap.add_argument("--print-only", action="store_true", help="show the config and URL, do not run")
    ap.add_argument(
        "--cloudflare",
        action="store_true",
        help="mint short-lived Cloudflare TURN credentials instead of running a local relay",
    )
    ap.add_argument("--ttl", type=int, default=3600, help="Cloudflare credential lifetime (seconds)")
    args = ap.parse_args()

    if args.cloudflare:
        return cloudflare_mode(args.ttl)

    exe = shutil.which("turnserver")
    if exe is None and not args.print_only:
        raise SystemExit(INSTALL_HINT)

    host = args.host or lan_address()
    password = args.password or secrets.token_urlsafe(9)
    conf = config(realm="rlx.local", user=args.user, password=password, port=args.port, relay_ip=host)

    turn_param = f"turn:{host}:{args.port}|{args.user}|{password}"
    print(f"TURN relay   {host}:{args.port}  (user {args.user})")
    print()
    print("Append this to the room URL on *both* devices:")
    print(f"  &turn={turn_param}")
    print()
    print("The page's ICE preflight line should then show `relay ✓`. If it does")
    print("not, the devices cannot reach this machine on that port at all.")
    print()

    if args.print_only:
        print("--- coturn config ---")
        print(conf)
        return 0

    path = Path(tempfile.mkdtemp(prefix="rlx-turn-")) / "turnserver.conf"
    path.write_text(conf + "\n")
    print(f"config: {path}")
    print("Ctrl-C to stop\n")
    try:
        subprocess.run([exe, "-c", str(path)], check=True)
    except KeyboardInterrupt:
        print("\nstopped.")
    except subprocess.CalledProcessError as e:
        print(f"turnserver exited with {e.returncode}", file=sys.stderr)
        return e.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
