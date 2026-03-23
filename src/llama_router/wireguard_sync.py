"""Write WireGuard wg0.conf from database state to a shared volume (Docker sidecar)."""

from __future__ import annotations

import logging

from .config import settings
from .database import Database
from .wireguard_config import render_wg_quick_config, write_wg_config_atomic

logger = logging.getLogger(__name__)


async def sync_wireguard_config_to_disk(db: Database) -> tuple[bool, str]:
    """Render DB WireGuard settings and atomically write wg0.conf.

    Returns (ok, message). Skips quietly when ``wireguard_config_path`` is unset.
    """
    path = (settings.wireguard_config_path or "").strip()
    if not path:
        return True, "WireGuard config path not set; skipping file write"

    iface = await db.get_wireguard_interface()
    peers = await db.list_wireguard_peers()
    try:
        text = render_wg_quick_config(iface, peers)
        write_wg_config_atomic(path, text)
    except Exception as exc:
        logger.warning("WireGuard config write failed: %s", exc)
        return False, str(exc)
    logger.info("WireGuard config written to %s", path)
    return True, f"Written {path}"


async def try_sync_wireguard_config_on_startup(db: Database) -> None:
    """Best-effort sync after DB connect (e.g. sidecar picks up last saved config)."""
    ok, msg = await sync_wireguard_config_to_disk(db)
    if not ok:
        logger.warning("Startup WireGuard sync: %s", msg)
