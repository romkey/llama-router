"""WireGuard key helpers and wg-quick config rendering for dashboard-managed tunnels."""

from __future__ import annotations

import base64
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

_WG_KEY_RE = re.compile(r"^[A-Za-z0-9+/]{43}=$")


def clamp_wg_private_key(raw_32: bytes) -> bytes:
    """Apply WireGuard / Curve25519 clamping to 32 raw secret bytes."""
    k = bytearray(raw_32)
    k[0] &= 248
    k[31] &= 127
    k[31] |= 64
    return bytes(k)


def generate_wireguard_private_key() -> str:
    """Return a base64-encoded WireGuard private key (compatible with `wg genkey`)."""
    priv = X25519PrivateKey.generate()
    raw = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    clamped = clamp_wg_private_key(raw)
    return base64.b64encode(clamped).decode("ascii")


def public_key_from_private(private_key_b64: str) -> str:
    """Derive WireGuard public key from private key (base64)."""
    raw = base64.b64decode(private_key_b64.strip())
    if len(raw) != 32:
        raise ValueError("Invalid WireGuard private key length")
    clamped = clamp_wg_private_key(raw)
    priv = X25519PrivateKey.from_private_bytes(clamped)
    pub = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(pub).decode("ascii")


def is_valid_wg_key_b64(key: str) -> bool:
    s = key.strip()
    if not _WG_KEY_RE.match(s):
        return False
    try:
        raw = base64.b64decode(s, validate=True)
    except Exception:
        return False
    return len(raw) == 32


def render_wg_quick_config(
    interface: dict[str, Any],
    peers: list[dict[str, Any]],
) -> str:
    """Build wg-quick(8) configuration text.

    When disabled or missing private key, returns a comment-only file so the
    sidecar can tear down the interface.
    """
    enabled = bool(interface.get("enabled"))
    priv = (interface.get("private_key") or "").strip()
    if not enabled or not priv:
        return (
            "# llama-router: WireGuard is disabled or no private key is set.\n"
            "# Add a key in the dashboard and enable the tunnel to generate "
            "a full configuration.\n"
        )

    if not is_valid_wg_key_b64(priv):
        raise ValueError("Invalid interface private key (expected 32-byte base64 key)")

    address = (interface.get("address_cidr") or "").strip()
    if not address:
        raise ValueError("Tunnel address (CIDR) is required when WireGuard is enabled")

    listen_port = int(interface.get("listen_port") or 51820)
    if listen_port < 1 or listen_port > 65535:
        raise ValueError("Listen port must be between 1 and 65535")

    lines = [
        "[Interface]",
        f"PrivateKey = {priv}",
        f"Address = {address}",
        f"ListenPort = {listen_port}",
    ]
    mtu = interface.get("mtu")
    if mtu is not None:
        try:
            mtu_int = int(mtu)
            if mtu_int > 0:
                lines.append(f"MTU = {mtu_int}")
        except (TypeError, ValueError):
            pass

    for peer in peers:
        if not peer.get("enabled", True):
            continue
        pk = (peer.get("public_key") or "").strip()
        if not pk:
            continue
        if not is_valid_wg_key_b64(pk):
            raise ValueError(
                f"Invalid peer public key: {peer.get('name') or peer.get('id')}"
            )
        allowed = (peer.get("allowed_ips") or "").strip()
        if not allowed:
            raise ValueError(
                f"Peer {peer.get('name') or peer.get('id')!r} needs AllowedIPs"
            )
        lines.append("")
        lines.append("[Peer]")
        lines.append(f"PublicKey = {pk}")
        psk = (peer.get("preshared_key") or "").strip()
        if psk:
            if not is_valid_wg_key_b64(psk):
                raise ValueError(
                    f"Invalid preshared key for peer {peer.get('name') or peer.get('id')!r}"
                )
            lines.append(f"PresharedKey = {psk}")
        lines.append(f"AllowedIPs = {allowed}")
        endpoint = (peer.get("endpoint") or "").strip()
        if endpoint:
            lines.append(f"Endpoint = {endpoint}")
        keepalive = peer.get("persistent_keepalive")
        if keepalive is not None:
            try:
                ka = int(keepalive)
                if ka > 0:
                    lines.append(f"PersistentKeepalive = {ka}")
            except (TypeError, ValueError):
                pass

    lines.append("")
    return "\n".join(lines)


def write_wg_config_atomic(path: str, content: str) -> None:
    """Write config atomically; create parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(p.parent), prefix=".wg0-", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
