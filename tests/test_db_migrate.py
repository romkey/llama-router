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


def test_run_upgrade_sync_after_dropped_alembic_version_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Schema present but no revision (e.g. lost alembic_version) must not loop on CREATE."""
    dbfile = tmp_path / "legacy_no_revision.db"
    monkeypatch.setattr(config.settings, "database_url", "")
    monkeypatch.setattr(config.settings, "database_path", str(dbfile))
    sync_url = config.settings.sync_database_url_for_alembic()
    run_upgrade_sync(sync_url)
    con = sqlite3.connect(dbfile)
    try:
        con.execute("DROP TABLE alembic_version")
        con.commit()
    finally:
        con.close()

    run_upgrade_sync(sync_url)

    con = sqlite3.connect(dbfile)
    try:
        ver = con.execute("SELECT version_num FROM alembic_version").fetchone()
        assert ver is not None
        assert ver[0] == "002_peering_key_expiry"
    finally:
        con.close()


def test_clear_false_stamp_when_table_missing_then_recreate(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Stale 001_initial stamp with a missing new-era table is repaired on startup."""
    dbfile = tmp_path / "false_stamp.db"
    monkeypatch.setattr(config.settings, "database_url", "")
    monkeypatch.setattr(config.settings, "database_path", str(dbfile))
    sync_url = config.settings.sync_database_url_for_alembic()
    run_upgrade_sync(sync_url)
    con = sqlite3.connect(dbfile)
    try:
        con.execute("DROP TABLE dashboard_users")
        con.commit()
    finally:
        con.close()

    run_upgrade_sync(sync_url)

    con = sqlite3.connect(dbfile)
    try:
        assert (
            con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='dashboard_users'"
            ).fetchone()
            is not None
        )
        ver = con.execute("SELECT version_num FROM alembic_version").fetchone()
        assert ver is not None
        assert ver[0] == "002_peering_key_expiry"
    finally:
        con.close()
