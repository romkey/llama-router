"""Run Alembic upgrades synchronously (safe from async startup; no nested event loops)."""

from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config

import llama_router

logger = logging.getLogger(__name__)


def run_upgrade_sync(sync_database_url: str) -> None:
    """Apply all pending Alembic revisions using a synchronous engine URL."""
    pkg = Path(llama_router.__file__).resolve().parent
    ini = pkg / "alembic.ini"
    cfg = Config(str(ini) if ini.exists() else None)
    cfg.set_main_option("script_location", str(pkg / "alembic"))
    cfg.set_main_option("sqlalchemy.url", sync_database_url)
    command.upgrade(cfg, "head")
