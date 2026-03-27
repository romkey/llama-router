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

# All tables created by 001_initial (lowercase names as returned by SQLite inspect).
_INITIAL_SCHEMA_TABLES = frozenset(
    {
        "wireguard_interface",
        "wireguard_peers",
        "providers",
        "provider_models",
        "benchmarks",
        "request_log",
        "model_fallbacks",
        "provider_addresses",
        "api_keys",
        "api_key_model_pins",
        "app_settings",
        "dashboard_users",
    }
)


def _clear_revision_if_marked_head_but_schema_incomplete(
    sync_database_url: str,
) -> None:
    """Remove a false ``001_initial`` stamp when new tables were added after a legacy DB.

    Older releases stamped ``001_initial`` whenever *any* app table existed, which
    skipped migration and left newer tables (e.g. ``dashboard_users``) missing.
    """
    engine = create_engine(sync_database_url)
    try:
        insp = inspect(engine)
        table_names = {t.lower() for t in insp.get_table_names()}
        if "alembic_version" not in table_names:
            return
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT version_num FROM alembic_version LIMIT 1")
            ).fetchone()
            if not row:
                return
        revision = row[0]
        missing = sorted(_INITIAL_SCHEMA_TABLES - table_names)
        if not missing:
            return
        logger.warning(
            "Alembic revision is %s but expected tables are missing: %s. "
            "Clearing the revision so the migration can create missing objects "
            "(existing tables use IF NOT EXISTS).",
            revision,
            ", ".join(missing),
        )
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM alembic_version"))
            conn.commit()
    finally:
        engine.dispose()


def _stamp_if_full_schema_exists_without_alembic_revision(
    cfg: Config, sync_database_url: str
) -> None:
    """If every 001 table already exists (e.g. restored dump) but ``alembic_version`` is empty, stamp.

    Partial legacy schemas must *not* be stamped: ``upgrade`` runs with IF NOT EXISTS and fills gaps.
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
        if _INITIAL_SCHEMA_TABLES - table_names:
            return
        logger.info(
            "Full application schema detected without Alembic revision; stamping %s.",
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
    _clear_revision_if_marked_head_but_schema_incomplete(sync_database_url)
    _stamp_if_full_schema_exists_without_alembic_revision(cfg, sync_database_url)
    command.upgrade(cfg, "head")
