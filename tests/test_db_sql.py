"""Unit tests for SQL placeholder translation used by the SQLAlchemy layer."""

from __future__ import annotations

import pytest

from llama_router.db_sql import qmark


def test_qmark_converts_placeholders_to_named_binds() -> None:
    stmt, bind = qmark("SELECT * FROM t WHERE a = ? AND b = ?", 10, "x")
    assert bind == {"p0": 10, "p1": "x"}
    assert ":p0" in stmt.text
    assert ":p1" in stmt.text


def test_qmark_no_placeholders_returns_empty_bind() -> None:
    stmt, bind = qmark("SELECT 1")
    assert bind == {}
    assert stmt.text.strip() == "SELECT 1"


def test_qmark_placeholder_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="2 placeholders"):
        qmark("SELECT ? AND ?", 1)
    with pytest.raises(ValueError, match="1 placeholders"):
        qmark("SELECT ?", 1, 2)


def test_qmark_single_placeholder() -> None:
    stmt, bind = qmark("DELETE FROM t WHERE id = ?", 99)
    assert bind == {"p0": 99}


def test_qmark_inserts_named_param_in_values_clause() -> None:
    stmt, bind = qmark("INSERT INTO t (c) VALUES (?)", None)
    assert bind == {"p0": None}
    assert ":p0" in stmt.text
