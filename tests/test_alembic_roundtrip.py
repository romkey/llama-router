"""The Alembic migration chain as a whole: one head, every downgrade works,
data survives a round trip, and the SQLite-to-PostgreSQL copier knows every
table and column."""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

import llama_router
from llama_router.db_migrate import run_upgrade_sync
from llama_router.migrate_sqlite_to_postgres import _TABLES

_PKG = Path(llama_router.__file__).resolve().parent

# Tables the copier leaves out on purpose: Alembic's own bookkeeping, and
# sign-in flows that only live a few minutes.
_NOT_COPIED = {"alembic_version", "dashboard_auth_flows"}


def _cfg(dbfile: Path) -> Config:
    cfg = Config(str(_PKG / "alembic.ini"))
    cfg.set_main_option("script_location", str(_PKG / "alembic"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{dbfile}")
    return cfg


def _rows(dbfile: Path, sql: str) -> list[tuple[object, ...]]:
    with contextlib.closing(sqlite3.connect(dbfile)) as con:
        return con.execute(sql).fetchall()


def _schema(dbfile: Path) -> list[tuple[object, ...]]:
    return _rows(
        dbfile,
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name",
    )


def test_migrations_have_a_single_head() -> None:
    assert len(ScriptDirectory(str(_PKG / "alembic")).get_heads()) == 1


def test_migration_files_are_named_in_chain_order() -> None:
    """Sorting the files by name lists the revisions oldest first."""
    script = ScriptDirectory(str(_PKG / "alembic"))
    chain = [rev.revision for rev in script.walk_revisions()][::-1]
    by_name = {Path(rev.path).name: rev.revision for rev in script.walk_revisions()}
    assert [by_name[name] for name in sorted(by_name)] == chain


def test_downgrade_to_base_and_back_keeps_data(tmp_path: Path) -> None:
    dbfile = tmp_path / "roundtrip.db"
    cfg = _cfg(dbfile)
    command.upgrade(cfg, "head")
    with contextlib.closing(sqlite3.connect(dbfile)) as con, con:
        con.execute(
            "INSERT INTO providers (id, name, protocol) VALUES (1, 'gpu', 'ollama')"
        )
        con.execute(
            "INSERT INTO api_keys (id, key_prefix, key_hash) "
            "VALUES (1, 'lr-abc', 'hash')"
        )
        con.execute(
            "INSERT INTO request_log (protocol, endpoint, provider_id, api_key_id) "
            "VALUES ('ollama', '/api/chat', 1, 1)"
        )
    schema = _schema(dbfile)

    # Every downgrade runs against tables that hold rows.
    command.downgrade(cfg, "base")
    assert (
        _rows(
            dbfile,
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' AND name != 'alembic_version'",
        )
        == []
    )

    with contextlib.closing(sqlite3.connect(dbfile)) as con, con:
        con.execute("DROP TABLE IF EXISTS alembic_version")
    command.upgrade(cfg, "head")
    assert _schema(dbfile) == schema


def test_data_survives_a_round_trip_to_the_first_revision(tmp_path: Path) -> None:
    """Downgrading to the initial schema (which already had these tables)
    and upgrading again keeps their rows."""
    dbfile = tmp_path / "partial.db"
    cfg = _cfg(dbfile)
    command.upgrade(cfg, "head")
    with contextlib.closing(sqlite3.connect(dbfile)) as con, con:
        con.execute(
            "INSERT INTO providers (id, name, protocol) VALUES (1, 'gpu', 'ollama')"
        )
        con.execute(
            "INSERT INTO api_keys (id, key_prefix, key_hash) "
            "VALUES (1, 'lr-abc', 'hash')"
        )
        con.execute(
            "INSERT INTO request_log (protocol, endpoint, provider_id, api_key_id) "
            "VALUES ('ollama', '/api/chat', 1, 1)"
        )

    command.downgrade(cfg, "001_initial")
    command.upgrade(cfg, "head")

    assert _rows(dbfile, "SELECT id, name, protocol FROM providers") == [
        (1, "gpu", "ollama")
    ]
    assert _rows(dbfile, "SELECT id, key_prefix, key_hash FROM api_keys") == [
        (1, "lr-abc", "hash")
    ]
    # request_log.api_key_id came after 001, so only the older columns stay.
    assert _rows(dbfile, "SELECT endpoint, provider_id FROM request_log") == [
        ("/api/chat", 1)
    ]


def test_upgrade_on_a_current_schema_changes_nothing(tmp_path: Path) -> None:
    dbfile = tmp_path / "current.db"
    url = f"sqlite:///{dbfile}"
    run_upgrade_sync(url)
    schema = _schema(dbfile)
    version = _rows(dbfile, "SELECT version_num FROM alembic_version")
    run_upgrade_sync(url)
    assert _schema(dbfile) == schema
    assert _rows(dbfile, "SELECT version_num FROM alembic_version") == version


def test_copier_tables_match_the_migrated_schema(tmp_path: Path) -> None:
    dbfile = tmp_path / "schema.db"
    run_upgrade_sync(f"sqlite:///{dbfile}")
    engine = sa.create_engine(f"sqlite:///{dbfile}")
    try:
        insp = sa.inspect(engine)
        migrated = {
            table: {c["name"] for c in insp.get_columns(table)}
            for table in insp.get_table_names()
            if table not in _NOT_COPIED
        }
    finally:
        engine.dispose()
    copied = {table: set(columns) for table, columns in _TABLES}
    assert sorted(copied) == sorted(migrated)
    for table, columns in migrated.items():
        assert copied[table] == columns, table
