"""SQLAlchemy helpers: translate sqlite-style ``?`` placeholders to named binds."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

from sqlalchemy import CursorResult, Result, TextClause, text


def qmark(sql: str, *args: Any) -> tuple[TextClause, dict[str, Any]]:
    """Convert ``'SELECT * FROM t WHERE a = ? AND b = ?'`` to named parameters."""
    if "?" not in sql:
        return text(sql), {}
    parts = sql.split("?")
    n = len(parts) - 1
    if n != len(args):
        raise ValueError(
            f"SQL has {n} placeholders but {len(args)} arguments were supplied"
        )
    bind: dict[str, Any] = {}
    out: list[str] = []
    for i, part in enumerate(parts[:-1]):
        k = f"p{i}"
        bind[k] = args[i]
        out.append(part + f":{k}")
    out.append(parts[-1])
    return text("".join(out)), bind


def row_to_mapping(row: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    return row


def result_mappings(result: Result[Any]) -> Sequence[Mapping[str, Any]]:
    """Return all rows as read-only mappings (like sqlite Row)."""
    return cast(Sequence[Mapping[str, Any]], result.mappings().all())


def result_first(result: Result[Any]) -> Mapping[str, Any] | None:
    return cast(Mapping[str, Any] | None, result.mappings().first())


def cursor_rowcount(result: Result[Any]) -> int:
    cr = cast(CursorResult[Any], result)
    return int(cr.rowcount or 0)


def cursor_lastrowid(result: Result[Any]) -> int | None:
    cr = cast(CursorResult[Any], result)
    if cr.inserted_primary_key and cr.inserted_primary_key[0] is not None:
        return int(cr.inserted_primary_key[0])
    if hasattr(cr, "lastrowid") and cr.lastrowid is not None:
        return int(cr.lastrowid)
    return None
