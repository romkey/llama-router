from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine.url import URL, make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .config import settings
from .db_events import (
    AUTH_CHANGES,
    CHANGE_ADDRESSES,
    CHANGE_ALIASES,
    CHANGE_API_KEYS,
    CHANGE_AUTH_SETTINGS,
    CHANGE_BENCHMARKS,
    CHANGE_CAPABILITIES,
    CHANGE_DECISIONS,
    CHANGE_FALLBACKS,
    CHANGE_GROUPS,
    CHANGE_MODELS,
    CHANGE_PRICES,
    CHANGE_PROVIDERS,
    CHANGE_STATUS,
    ROUTING_CHANGES,
)
from .db_migrate import run_upgrade_sync
from .repositories import (
    ApiKeyRepository,
    AuditRepository,
    BenchmarkRepository,
    ChatRepository,
    DashboardUserRepository,
    DecisionRepository,
    ManagementKeyRepository,
    ModelConfigRepository,
    ProviderRepository,
    RequestLogRepository,
    WorkerRepository,
)
from .repositories.base import _bind_params, _utc_datetime

# Callers import the change reasons and these helpers from here.
__all__ = [
    "AUTH_CHANGES",
    "CHANGE_ADDRESSES",
    "CHANGE_ALIASES",
    "CHANGE_API_KEYS",
    "CHANGE_AUTH_SETTINGS",
    "CHANGE_BENCHMARKS",
    "CHANGE_CAPABILITIES",
    "CHANGE_DECISIONS",
    "CHANGE_FALLBACKS",
    "CHANGE_GROUPS",
    "CHANGE_MODELS",
    "CHANGE_PRICES",
    "CHANGE_PROVIDERS",
    "CHANGE_STATUS",
    "ROUTING_CHANGES",
    "Database",
    "_bind_params",
    "_utc_datetime",
]

logger = logging.getLogger(__name__)


def _sqlite_foreign_keys_on(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    # WAL lets the dashboard read while the request log writes; NORMAL is
    # safe under WAL and skips an fsync per commit; busy_timeout waits for
    # a lock instead of failing with "database is locked".
    cursor.execute("PRAGMA busy_timeout=5000")
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError as exc:
        # Another connection is switching the file to WAL right now (two
        # routers starting together); the mode is kept in the file, so that
        # switch covers this connection too.
        logger.debug("SQLite journal mode left to another connection: %s", exc)
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def _normalize_database_url(database_url: str | None) -> str:
    if database_url is None or database_url == "":
        return settings.effective_database_url()
    if "://" in database_url:
        return database_url
    path = Path(database_url).expanduser().resolve()
    return URL.create("sqlite+aiosqlite", database=str(path)).render_as_string(
        hide_password=False
    )


class Database(
    AuditRepository,
    ProviderRepository,
    BenchmarkRepository,
    RequestLogRepository,
    ModelConfigRepository,
    DecisionRepository,
    ChatRepository,
    WorkerRepository,
    ApiKeyRepository,
    ManagementKeyRepository,
    DashboardUserRepository,
):
    """The router's store: one object over every repository."""

    def __init__(self, database_url: str | None = None):
        self._url = _normalize_database_url(database_url)
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._dialect_name: str = "sqlite"
        self._listeners: list[Callable[[str], None]] = []

    async def connect(self) -> None:
        sync_url = settings.sync_database_url_for_alembic(self._url)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: run_upgrade_sync(sync_url))
        # A SQLite file cannot drop the connection, so a ping per checkout
        # only costs time.
        is_sqlite = make_url(self._url).get_backend_name() == "sqlite"
        self._engine = create_async_engine(self._url, pool_pre_ping=not is_sqlite)
        self._session_factory = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
        self._dialect_name = self._engine.sync_engine.dialect.name

        if self._dialect_name == "sqlite":
            # SQLite enforces foreign keys (and ON DELETE) only on connections
            # that ask, so ask on every connection the pool opens.
            event.listen(self._engine.sync_engine, "connect", _sqlite_foreign_keys_on)

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
        self._engine = None
        self._session_factory = None
