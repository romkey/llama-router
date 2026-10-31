"""Unit tests for the SQLite → PostgreSQL migration CLI helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llama_router.migrate_sqlite_to_postgres import (
    _drop_orphans,
    _normalize_postgres_url,
    _sqlite_url,
)


def test_sqlite_url_is_absolute_file_path(tmp_path: Path) -> None:
    p = tmp_path / "a.db"
    p.write_bytes(b"")
    u = _sqlite_url(p)
    assert u.startswith("sqlite:///")
    assert str(p.resolve()) in u


@pytest.mark.parametrize(
    ("raw", "expect_substr"),
    [
        (
            "postgresql+asyncpg://u:p@h/db",
            "postgresql+psycopg://",
        ),
        (
            "postgres://u:p@h/db",
            "postgresql+psycopg://",
        ),
    ],
)
def test_normalize_postgres_url_maps_async_and_aliases(
    raw: str, expect_substr: str
) -> None:
    out = _normalize_postgres_url(raw)
    assert out.startswith(expect_substr)
    assert "asyncpg" not in out.split("://", 1)[0]


def test_normalize_postgres_url_rejects_asyncpg_substring_in_nonstandard_scheme() -> (
    None
):
    """Driver names like ``postgresql+xasyncpg://`` are not accepted."""
    with pytest.raises(SystemExit, match="psycopg"):
        _normalize_postgres_url("postgresql+xasyncpg://localhost/db")


def test_normalize_postgres_url_accepts_psycopg() -> None:
    u = "postgresql+psycopg://user:pass@localhost:5432/app"
    assert _normalize_postgres_url(u) == u


def test_normalize_postgres_url_rejects_non_postgresql() -> None:
    with pytest.raises(SystemExit, match="postgresql"):
        _normalize_postgres_url("mysql+pymysql://localhost/db")


_PROVIDER_FK = {
    "column": "provider_id",
    "referred_table": "providers",
    "referred_column": "id",
    "ondelete": "CASCADE",
}


def test_drop_orphans_skips_rows_whose_cascade_parent_is_gone() -> None:
    rows = [{"id": 1, "provider_id": 1}, {"id": 2, "provider_id": 2}]
    kept = _drop_orphans(
        "provider_models", rows, [_PROVIDER_FK], {("providers", "id"): {2}}
    )
    assert kept == [{"id": 2, "provider_id": 2}]


def test_drop_orphans_clears_set_null_references() -> None:
    fk = {**_PROVIDER_FK, "ondelete": "SET NULL"}
    rows: list[dict[str, Any]] = [
        {"id": 1, "provider_id": 1},
        {"id": 2, "provider_id": None},
    ]
    kept = _drop_orphans("chat_conversations", rows, [fk], {("providers", "id"): {2}})
    assert kept == [{"id": 1, "provider_id": None}, {"id": 2, "provider_id": None}]


def test_drop_orphans_leaves_keys_to_tables_not_copied() -> None:
    rows = [{"id": 1, "provider_id": 7}]
    assert _drop_orphans("provider_models", rows, [_PROVIDER_FK], {}) == rows
