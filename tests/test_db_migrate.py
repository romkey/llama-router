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
        assert ver[0] == "003_workers"
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
        assert ver[0] == "003_workers"
    finally:
        con.close()


def _alembic_cfg(sync_url: str):
    from pathlib import Path

    from alembic.config import Config

    import llama_router

    pkg = Path(llama_router.__file__).resolve().parent
    cfg = Config(str(pkg / "alembic.ini"))
    cfg.set_main_option("script_location", str(pkg / "alembic"))
    cfg.set_main_option("sqlalchemy.url", sync_url)
    return cfg


def test_upgrade_from_wireguard_schema_keeps_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A 0.15-era database with WireGuard peers upgrades cleanly to workers."""
    from alembic import command

    dbfile = tmp_path / "legacy_wg.db"
    monkeypatch.setattr(config.settings, "database_url", "")
    monkeypatch.setattr(config.settings, "database_path", str(dbfile))
    sync_url = config.settings.sync_database_url_for_alembic()
    command.upgrade(_alembic_cfg(sync_url), "002_peering_key_expiry")

    con = sqlite3.connect(dbfile)
    try:
        con.execute("PRAGMA foreign_keys = ON")
        con.execute(
            "INSERT INTO wireguard_peers (name, public_key, allowed_ips) "
            "VALUES ('peer', 'pk', '10.8.0.2/32')"
        )
        con.execute(
            "INSERT INTO providers (name, url, wireguard_peer_id) "
            "VALUES ('gpu-box', 'http://10.8.0.2:11434', 1)"
        )
        con.execute(
            "INSERT INTO provider_models (provider_id, name) VALUES (1, 'llama3:8b')"
        )
        con.commit()
    finally:
        con.close()

    run_upgrade_sync(sync_url)

    con = sqlite3.connect(dbfile)
    try:
        tables = {
            r[0]
            for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "workers" in tables
        assert "wireguard_peers" not in tables
        assert "wireguard_interface" not in tables
        cols = {r[1] for r in con.execute("PRAGMA table_info(providers)")}
        assert "wireguard_peer_id" not in cols
        assert con.execute("SELECT name, url FROM providers").fetchall() == [
            ("gpu-box", "http://10.8.0.2:11434")
        ]
        assert con.execute("SELECT COUNT(*) FROM provider_models").fetchone()[0] == 1
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []
        ver = con.execute("SELECT version_num FROM alembic_version").fetchone()[0]
        assert ver == "003_workers"
    finally:
        con.close()

    # A second start must not re-run 001 and recreate the WireGuard tables.
    run_upgrade_sync(sync_url)
    con = sqlite3.connect(dbfile)
    try:
        tables = {
            r[0]
            for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "wireguard_peers" not in tables
    finally:
        con.close()
