"""Apply WireGuard configuration on the host (wg-quick / wg syncconf) or write-only legacy mode."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
from pathlib import Path

from .config import settings
from .database import Database
from .wireguard_config import render_wg_quick_config, write_wg_config_atomic

logger = logging.getLogger(__name__)

_WG_QUICK = shutil.which("wg-quick")
_WG = shutil.which("wg")


def _interface_name(config_path: str) -> str:
    return Path(config_path).stem or "wg0"


async def is_wireguard_available() -> bool:
    """Return True if wg-quick is present on the host."""
    return _WG_QUICK is not None


async def _run(cmd: list[str], timeout: float = 30.0) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "", "timeout"
    return (
        proc.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


async def get_tunnel_status() -> dict:
    """Parse ``wg show <iface> dump`` into running flag and peer stats."""
    if not _WG:
        return {"running": False, "peers": []}
    path = (settings.wireguard_config_path or "").strip()
    if not path:
        return {"running": False, "peers": []}
    iface = _interface_name(path)
    code, out, _err = await _run([_WG, "show", iface, "dump"], timeout=5.0)
    if code != 0 or not out.strip():
        return {"running": False, "peers": []}
    lines = [ln for ln in out.strip().split("\n") if ln.strip()]
    if not lines:
        return {"running": True, "peers": []}
    # First line: interface (private_key, public_key, listen_port, fwmark)
    peers_out: list[dict] = []
    now = int(time.time())
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) < 8:
            continue
        pub, _psk, endpoint, allowed_ips, last_hs, rx_b, tx_b, _ka = parts[:8]
        try:
            last_hs_i = int(last_hs)
        except ValueError:
            last_hs_i = 0
        if last_hs_i > 0:
            ago = max(0, now - last_hs_i)
        else:
            ago = None  # never
        try:
            rx = int(rx_b)
            tx = int(tx_b)
        except ValueError:
            rx, tx = 0, 0
        peers_out.append(
            {
                "public_key": pub.strip(),
                "endpoint": endpoint if endpoint != "(none)" else None,
                "allowed_ips": allowed_ips,
                "last_handshake_seconds_ago": ago,
                "rx_bytes": rx,
                "tx_bytes": tx,
            }
        )
    return {"running": True, "peers": peers_out}


async def bring_up(config_path: str) -> tuple[bool, str]:
    """Run ``wg-quick up <config_path>``."""
    if not _WG_QUICK:
        return False, "wg-quick not found on PATH"
    code, out, err = await _run([_WG_QUICK, "up", config_path], timeout=60.0)
    msg = (out + err).strip() or ("ok" if code == 0 else "failed")
    return code == 0, msg


async def bring_down(config_path: str) -> tuple[bool, str]:
    """Run ``wg-quick down <config_path>``."""
    if not _WG_QUICK:
        return False, "wg-quick not found on PATH"
    code, out, err = await _run([_WG_QUICK, "down", config_path], timeout=60.0)
    msg = (out + err).strip() or ("ok" if code == 0 else "failed")
    return code == 0, msg


async def apply_config(db: Database) -> tuple[bool, str]:
    """Render config from DB and apply to the host tunnel or write-only (legacy)."""
    path = (settings.wireguard_config_path or "").strip()
    if not path:
        return True, "WireGuard config path not set; skipping"

    iface = await db.get_wireguard_interface()
    peers = await db.list_wireguard_peers()
    try:
        text = render_wg_quick_config(iface, peers)
        write_wg_config_atomic(path, text)
    except Exception as exc:
        logger.warning("WireGuard config write failed: %s", exc)
        return False, str(exc)

    if settings.wireguard_legacy_volume:
        logger.info(
            "WireGuard config written to %s (legacy volume mode; no wg-quick)", path
        )
        return True, f"Written {path} (legacy volume mode)"

    if not _WG_QUICK or not _WG:
        logger.info("WireGuard config written to %s (wg-quick/wg not available)", path)
        return True, f"Written {path} (wg-quick not available on PATH)"

    if_name = _interface_name(path)
    # Try syncconf for live reload without dropping connections
    strip_code, strip_out, strip_err = await _run(
        [_WG_QUICK, "strip", path], timeout=10.0
    )
    if strip_code != 0:
        logger.debug("wg-quick strip failed: %s", strip_err)
        ok, msg = await bring_up(path)
        return ok, msg if not ok else f"Applied {path} (wg-quick up)"

    stripped = strip_out
    if not stripped.strip():
        ok, msg = await bring_up(path)
        return ok, msg if not ok else f"Applied {path} (wg-quick up)"

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".conf",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(stripped)
            tmp_path = tmp.name
    except OSError as exc:
        logger.warning("temp file for wg syncconf: %s", exc)
        ok, msg = await bring_up(path)
        return ok, msg if not ok else f"Applied {path} (wg-quick up)"

    try:
        sync_code, _so, sync_err = await _run(
            [_WG, "syncconf", if_name, tmp_path], timeout=15.0
        )
        if sync_code == 0:
            logger.info("WireGuard syncconf applied for %s", if_name)
            return True, f"Synced {if_name}"
        logger.debug(
            "wg syncconf failed (%s), trying wg-quick up: %s", sync_code, sync_err
        )
        await bring_down(path)
        ok, msg = await bring_up(path)
        return ok, (
            msg if not ok else f"Applied {path} (wg-quick up after syncconf failure)"
        )
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass
