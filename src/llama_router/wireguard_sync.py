"""Apply WireGuard configuration from database state to disk and optional host tunnel."""

from __future__ import annotations

import logging

from .config import settings
from .database import Database
from .wireguard_manager import apply_config, is_wireguard_available

logger = logging.getLogger(__name__)


async def sync_wireguard_config_to_disk(db: Database) -> tuple[bool, str]:
    """Render DB WireGuard settings, write config, and apply on the host when configured."""
    return await apply_config(db)


async def try_sync_wireguard_config_on_startup(db: Database) -> None:
    """Apply WireGuard on startup when enabled and tools are available."""
    if not settings.wireguard_enabled:
        return
    if not await is_wireguard_available():
        logger.warning(
            "LLAMA_ROUTER_WIREGUARD_ENABLED is true but wg-quick was not found on PATH; "
            "the tunnel will not be applied automatically at startup."
        )
        return
    ok, msg = await apply_config(db)
    if not ok:
        logger.warning("Startup WireGuard apply: %s", msg)
    else:
        logger.info("Startup WireGuard: %s", msg)
