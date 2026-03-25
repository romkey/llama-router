"""Alembic sync migration entrypoint (used before the async engine starts)."""

from __future__ import annotations

import sqlite3

import pytest

from llama_router import config
from llama_router.db_migrate import run_upgrade_sync


def test_run_upgrade_sync_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    dbfile = tmp_path / "migrate_twice.db"
    monkeypatch.setattr(config.settings, "database_url", "")
    monkeypatch.setattr(config.settings, "database_path", str(dbfile))
    sync_url = config.settings.sync_database_url_for_alembic()
    run_upgrade_sync(sync_url)
    run_upgrade_sync(sync_url)
    con = sqlite3.connect(dbfile)
    try:
        n = con.execute("SELECT COUNT(*) FROM alembic_version").fetchone()[0]
        assert n == 1
        assert (
            con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='providers'"
            ).fetchone()
            is not None
        )
    finally:
        con.close()
