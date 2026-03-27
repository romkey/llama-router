"""Run Alembic upgrades synchronously (safe from async startup; no nested event loops)."""

from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

import llama_router

logger = logging.getLogger(__name__)

# Must match revision in alembic/versions/001_initial_schema.py
_INITIAL_REVISION = "001_initial"


def _stamp_if_schema_exists_without_alembic_revision(
    cfg: Config, sync_database_url: str
) -> None:
    """If the DB was created outside Alembic (or alembic_version was lost), stamping
    avoids re-running the initial migration and ``table already exists`` errors.
    """
    engine = create_engine(sync_database_url)
    try:
        insp = inspect(engine)
        table_names = {t.lower() for t in insp.get_table_names()}
        has_revision_row = False
        if "alembic_version" in table_names:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM alembic_version"))
                has_revision_row = (result.scalar_one() or 0) > 0
        if has_revision_row:
            return
        schema_present = (
            "wireguard_interface" in table_names or "providers" in table_names
        )
        if not schema_present:
            return
        logger.warning(
            "Database has application tables but no Alembic revision recorded; "
            "stamping %s. If the schema is incomplete, repair or recreate the database.",
            _INITIAL_REVISION,
        )
        command.stamp(cfg, _INITIAL_REVISION)
    finally:
        engine.dispose()


def run_upgrade_sync(sync_database_url: str) -> None:
    """Apply all pending Alembic revisions using a synchronous engine URL."""
    pkg = Path(llama_router.__file__).resolve().parent
    ini = pkg / "alembic.ini"
    cfg = Config(str(ini) if ini.exists() else None)
    cfg.set_main_option("script_location", str(pkg / "alembic"))
    cfg.set_main_option("sqlalchemy.url", sync_database_url)
    _stamp_if_schema_exists_without_alembic_revision(cfg, sync_database_url)
    command.upgrade(cfg, "head")
