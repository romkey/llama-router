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


@pytest.mark.parametrize(
    "async_url",
    [
        "postgresql+asyncpg://user:s3cr%40t@db:5432/app",
        "mysql+asyncmy://root:s3cr%40t@db:3306/router",
        "mariadb+asyncmy://root:s3cr%40t@db:3306/router",
    ],
)
def test_sync_url_keeps_the_password(
    monkeypatch: pytest.MonkeyPatch, async_url: str
) -> None:
    """Alembic must receive the real password, not SQLAlchemy's "***" mask."""
    monkeypatch.setattr(config.settings, "database_url", async_url)
    sync_url = config.settings.sync_database_url_for_alembic()
    assert "***" not in sync_url
    assert make_url(sync_url).password == "s3cr@t"  # noqa: S105  # test fixture value


def test_upgrade_handles_percent_in_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Percent-encoded characters must survive Alembic's ConfigParser interpolation."""
    from llama_router.db_migrate import run_upgrade_sync

    dbdir = tmp_path / "pct%40dir"
    dbdir.mkdir()
    monkeypatch.setattr(config.settings, "database_url", "")
    monkeypatch.setattr(config.settings, "database_path", str(dbdir / "x.db"))
    run_upgrade_sync(config.settings.sync_database_url_for_alembic())
    assert (dbdir / "x.db").exists()


def test_alembic_config_round_trips_encoded_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The URL Alembic reads back is the one we passed, %-escapes and all."""
    from alembic import command

    from llama_router import db_migrate

    seen = {}

    def fake_upgrade(cfg, rev):
        seen["main"] = cfg.get_main_option("sqlalchemy.url")
        seen["section"] = cfg.get_section(cfg.config_ini_section)["sqlalchemy.url"]

    monkeypatch.setattr(command, "upgrade", fake_upgrade)
    monkeypatch.setattr(
        db_migrate,
        "_clear_revision_if_marked_head_but_schema_incomplete",
        lambda u: None,
    )
    monkeypatch.setattr(
        db_migrate,
        "_stamp_if_full_schema_exists_without_alembic_revision",
        lambda c, u: None,
    )
    url = "postgresql+psycopg://user:s3cr%40t@db:5432/app"
    db_migrate.run_upgrade_sync(url)
    assert seen == {"main": url, "section": url}
