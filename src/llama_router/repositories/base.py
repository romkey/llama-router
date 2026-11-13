"""Shared query helpers and change notification for the repositories."""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..db_sql import cursor_rowcount, qmark, result_first, result_mappings

logger = logging.getLogger(__name__)


def _tristate(value: Any) -> bool | None:
    """A nullable integer flag: None, False or True."""
    return None if value is None else bool(value)


class _Keep:
    """Marks an optional update argument the caller left out."""


_KEEP = _Keep()


def _bind_params(*args: Any) -> tuple[Any, ...]:
    """Flatten legacy ``(single_tuple,)`` call style to match ``qmark`` placeholders."""
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        return tuple(args[0])
    return args


def _utc_datetime(value: Any) -> datetime | None:
    """A stored UTC timestamp as an aware datetime.

    An aggregate such as ``MAX(created_at)`` comes back from SQLite as text,
    which says nothing about its zone; the database writes UTC.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.strip().replace(" ", "T", 1))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _json_or_none(raw: Any) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        val = json.loads(raw)
    except TypeError, ValueError:
        return None
    return val if isinstance(val, dict) else None


def _json_list_or_none(raw: Any) -> list[Any] | None:
    if not raw:
        return None
    try:
        val = json.loads(raw)
    except TypeError, ValueError:
        return None
    return val if isinstance(val, list) else None


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


class Repository:
    """Query helpers every repository shares; ``Database`` supplies the engine."""

    _session_factory: async_sessionmaker[AsyncSession] | None
    _dialect_name: str
    _listeners: list[Callable[[str], None]]

    def add_change_listener(self, fn: Callable[[str], None]) -> None:
        """Call *fn(reason)* after writes that can change routing or auth."""
        self._listeners.append(fn)

    def remove_change_listener(self, fn: Callable[[str], None]) -> None:
        with contextlib.suppress(ValueError):
            self._listeners.remove(fn)

    def _changed(self, reason: str) -> None:
        for fn in list(self._listeners):
            try:
                fn(reason)
            except Exception:
                logger.exception("Change listener failed for %s", reason)

    def _bind_time(self, when: datetime | None) -> Any:
        """*when* bound for a timestamp column: UTC without a zone (asyncpg
        refuses an aware one), as text on SQLite."""
        if when is None:
            return None
        if when.tzinfo is not None:
            when = when.astimezone(UTC).replace(tzinfo=None)
        # Whole seconds: MySQL would round a fraction up (past a limit).
        when = when.replace(microsecond=0)
        if getattr(self, "_dialect_name", "") == "sqlite":
            return when.strftime("%Y-%m-%d %H:%M:%S")
        return when

    def _require_session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            raise RuntimeError("Database not connected")
        return self._session_factory

    async def _select_all(self, sql: str, *args: Any) -> list[Mapping[str, Any]]:
        stmt, bind = qmark(sql, *_bind_params(*args))
        sf = self._require_session_factory()
        async with sf() as session:
            r = await session.execute(stmt, bind)
            return list(result_mappings(r))

    async def _select_one(self, sql: str, *args: Any) -> Mapping[str, Any] | None:
        stmt, bind = qmark(sql, *_bind_params(*args))
        sf = self._require_session_factory()
        async with sf() as session:
            r = await session.execute(stmt, bind)
            return result_first(r)

    async def _execute(self, sql: str, *args: Any) -> None:
        stmt, bind = qmark(sql, *_bind_params(*args))
        sf = self._require_session_factory()
        async with sf() as session, session.begin():
            await session.execute(stmt, bind)

    async def _execute_rowcount(self, sql: str, *args: Any) -> int:
        stmt, bind = qmark(sql, *_bind_params(*args))
        sf = self._require_session_factory()
        async with sf() as session, session.begin():
            r = await session.execute(stmt, bind)
            return cursor_rowcount(r)

    async def _insert_get_id(self, insert_sql: str, *args: Any) -> int:
        """Run INSERT and return primary key ``id`` (portable across dialects)."""
        sf = self._require_session_factory()
        async with sf() as session, session.begin():
            if self._dialect_name in ("mysql", "mariadb"):
                st, bd = qmark(insert_sql, *_bind_params(*args))
                await session.execute(st, bd)
                r = await session.execute(text("SELECT LAST_INSERT_ID() AS id"))
                row = r.mappings().one()
                return int(row["id"])
            st, bd = qmark(insert_sql + " RETURNING id", *_bind_params(*args))
            r = await session.execute(st, bd)
            val = r.scalar_one()
            return int(val)
