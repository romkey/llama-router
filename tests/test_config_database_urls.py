"""Tests for database URL derivation (async app URL vs sync Alembic URL)."""

from __future__ import annotations

import pytest
from sqlalchemy.engine import make_url

from llama_router import config


def test_effective_database_url_uses_sqlite_when_path_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    dbfile = tmp_path / "custom.db"
    monkeypatch.setattr(config.settings, "database_url", "")
    monkeypatch.setattr(config.settings, "database_path", str(dbfile))
    url = config.settings.effective_database_url()
    assert "sqlite+aiosqlite" in url
    assert str(dbfile.resolve()) in url


def test_effective_database_url_prefers_explicit_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit = "postgresql+asyncpg://u:p@db.internal:5432/app"
    monkeypatch.setattr(config.settings, "database_url", explicit)
    monkeypatch.setattr(config.settings, "database_path", "ignored.db")
    assert config.settings.effective_database_url() == explicit


def test_sync_url_strips_aiosqlite_driver(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(config.settings, "database_url", "")
    monkeypatch.setattr(config.settings, "database_path", str(tmp_path / "x.db"))
    sync_url = config.settings.sync_database_url_for_alembic()
    sync = make_url(sync_url)
    assert sync.drivername == "sqlite"


def test_sync_url_maps_asyncpg_to_psycopg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        config.settings,
        "database_url",
        "postgresql+asyncpg://user:pass@localhost:5432/mydb",
    )
    sync_url = config.settings.sync_database_url_for_alembic()
    sync = make_url(sync_url)
    assert sync.drivername == "postgresql+psycopg"
    assert sync.username == "user"
    assert sync.host == "localhost"
    assert sync.port == 5432
    assert sync.database == "mydb"


def test_sync_url_maps_asyncmy_to_pymysql(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        config.settings,
        "database_url",
        "mysql+asyncmy://root:secret@127.0.0.1:3306/router",
    )
    sync_url = config.settings.sync_database_url_for_alembic()
    assert sync_url.startswith("mysql+pymysql://")
    assert "asyncmy" not in sync_url


def test_sync_url_maps_mariadb_async_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        config.settings,
        "database_url",
        "mariadb+asyncmy://u:p@h/db",
    )
    sync_url = config.settings.sync_database_url_for_alembic()
    assert sync_url.startswith("mysql+pymysql://")


def test_sync_url_accepts_explicit_async_url_argument() -> None:
    out = config.settings.sync_database_url_for_alembic(
        "postgresql+asyncpg://localhost/db"
    )
    assert "postgresql+psycopg" in out
