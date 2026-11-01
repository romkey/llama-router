from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Concatenate

from sqlalchemy import event, text
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .config import settings
from .db_migrate import run_upgrade_sync
from .db_sql import cursor_rowcount, qmark, result_first, result_mappings
from .models import (
    BenchmarkResult,
    ModelGroup,
    ModelGroupMember,
    Provider,
    ProviderAddress,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
    RequestLog,
)

logger = logging.getLogger(__name__)


def _sqlite_foreign_keys_on(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.close()


# Reasons passed to change listeners. The router rebuilds its table on the
# routing ones; the auth cache drops entries on the auth ones.
CHANGE_PROVIDERS = "providers"
CHANGE_STATUS = "status"
CHANGE_ADDRESSES = "addresses"
CHANGE_MODELS = "models"
CHANGE_BENCHMARKS = "benchmarks"
CHANGE_FALLBACKS = "fallbacks"
CHANGE_ALIASES = "aliases"
CHANGE_GROUPS = "groups"
CHANGE_PRICES = "prices"
CHANGE_CAPABILITIES = "capabilities"
CHANGE_API_KEYS = "api_keys"
CHANGE_AUTH_SETTINGS = "auth_settings"
CHANGE_DECISIONS = "decision_calibrations"
ROUTING_CHANGES = frozenset(
    {
        CHANGE_PROVIDERS,
        CHANGE_STATUS,
        CHANGE_ADDRESSES,
        CHANGE_MODELS,
        CHANGE_BENCHMARKS,
        CHANGE_FALLBACKS,
        CHANGE_ALIASES,
        CHANGE_GROUPS,
        CHANGE_PRICES,
        CHANGE_CAPABILITIES,
    }
)
AUTH_CHANGES = frozenset({CHANGE_API_KEYS, CHANGE_AUTH_SETTINGS})


def _tristate(value) -> bool | None:
    """A nullable integer flag: None, False or True."""
    return None if value is None else bool(value)


def _notifies[**P, R](
    reason: str,
) -> Callable[
    [Callable[Concatenate[Database, P], Awaitable[R]]],
    Callable[Concatenate[Database, P], Awaitable[R]],
]:
    """Tell change listeners about *reason* after the wrapped write succeeds."""

    def wrap(
        fn: Callable[Concatenate[Database, P], Awaitable[R]],
    ) -> Callable[Concatenate[Database, P], Awaitable[R]]:
        async def inner(self: Database, *args: P.args, **kwargs: P.kwargs) -> R:
            result = await fn(self, *args, **kwargs)
            self._changed(reason)
            return result

        functools.update_wrapper(inner, fn)
        return inner

    return wrap


class _Keep:
    """Marks an optional update argument the caller left out."""


_KEEP = _Keep()


def _bind_params(*args: Any) -> tuple[Any, ...]:
    """Flatten legacy ``(single_tuple,)`` call style to match ``qmark`` placeholders."""
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        return tuple(args[0])
    return args


def _normalize_database_url(database_url: str | None) -> str:
    if database_url is None or database_url == "":
        return settings.effective_database_url()
    if "://" in database_url:
        return database_url
    path = Path(database_url).expanduser().resolve()
    return URL.create("sqlite+aiosqlite", database=str(path)).render_as_string(
        hide_password=False
    )


class Database:
    def __init__(self, database_url: str | None = None):
        self._url = _normalize_database_url(database_url)
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._dialect_name: str = "sqlite"
        self._listeners: list[Callable[[str], None]] = []

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

    async def connect(self) -> None:
        sync_url = settings.sync_database_url_for_alembic(self._url)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: run_upgrade_sync(sync_url))
        self._engine = create_async_engine(self._url, pool_pre_ping=True)
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

    # --- Providers ---

    @_notifies(CHANGE_PROVIDERS)
    async def add_provider(
        self,
        name: str,
        protocol: ProviderProtocol = ProviderProtocol.OLLAMA,
        *,
        machine_type: str | None = None,
        gpu_type: str | None = None,
        gpu_ram: str | None = None,
        worker_name: str | None = None,
        worker_parent_id: int | None = None,
        worker_remote_id: int | None = None,
    ) -> Provider:
        """Create a provider. Addresses are added separately with add_address."""
        pid = await self._insert_get_id(
            "INSERT INTO providers (name, protocol, machine_type, gpu_type, "
            "gpu_ram, worker_name, worker_parent_id, worker_remote_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                name,
                protocol.value,
                machine_type,
                gpu_type,
                gpu_ram,
                worker_name,
                worker_parent_id,
                worker_remote_id,
            ),
        )
        return Provider(
            id=pid,
            name=name,
            protocol=protocol,
            machine_type=machine_type,
            gpu_type=gpu_type,
            gpu_ram=gpu_ram,
            worker_name=worker_name,
            worker_parent_id=worker_parent_id,
            worker_remote_id=worker_remote_id,
        )

    async def list_worker_children(self, parent_id: int) -> list[Provider]:
        """Providers a worker shares individually, by its own provider id."""
        rows = await self._select_all(
            "SELECT * FROM providers WHERE worker_parent_id = ? ORDER BY id",
            (parent_id,),
        )
        return [_row_to_provider(r) for r in rows]

    @_notifies(CHANGE_PROVIDERS)
    async def remove_provider(self, provider_id: int) -> None:
        await self._execute("DELETE FROM providers WHERE id = ?", (provider_id,))

    async def get_provider(self, provider_id: int) -> Provider | None:
        row = await self._select_one(
            "SELECT * FROM providers WHERE id = ?", (provider_id,)
        )
        return _row_to_provider(row) if row else None

    async def get_provider_by_name(self, name: str) -> Provider | None:
        row = await self._select_one("SELECT * FROM providers WHERE name = ?", (name,))
        return _row_to_provider(row) if row else None

    async def list_providers(self) -> list[Provider]:
        rows = await self._select_all("SELECT * FROM providers ORDER BY name")
        return [_row_to_provider(r) for r in rows]

    @_notifies(CHANGE_PROVIDERS)
    async def update_provider(
        self,
        provider_id: int,
        name: str,
        protocol: ProviderProtocol | None = None,
        *,
        machine_type: str | None = None,
        gpu_type: str | None = None,
        gpu_ram: str | None = None,
    ) -> None:
        if protocol is not None:
            await self._execute(
                "UPDATE providers SET name = ?, protocol = ?, machine_type = ?, "
                "gpu_type = ?, gpu_ram = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ?",
                (
                    name,
                    protocol.value,
                    machine_type,
                    gpu_type,
                    gpu_ram,
                    provider_id,
                ),
            )
        else:
            await self._execute(
                "UPDATE providers SET name = ?, machine_type = ?, gpu_type = ?, "
                "gpu_ram = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (name, machine_type, gpu_type, gpu_ram, provider_id),
            )

    @_notifies(CHANGE_CAPABILITIES)
    async def set_provider_capabilities(
        self, provider_id: int, capabilities: dict | None
    ) -> None:
        """Store the last probe result (or clear it with None), and its flavor."""
        await self._execute(
            "UPDATE providers SET capabilities = ?, flavor = ?, probed_at = "  # noqa: S608 - column names are whitelisted / fixed SQL fragments; values are bound
            + ("CURRENT_TIMESTAMP" if capabilities is not None else "NULL")
            + " WHERE id = ?",
            (
                json.dumps(capabilities) if capabilities is not None else None,
                (capabilities or {}).get("flavor"),
                provider_id,
            ),
        )

    # Columns update_provider_settings may write, and whether each holds JSON.
    _PROVIDER_SETTINGS: ClassVar[dict[str, bool]] = {
        "preset": False,
        "extra_headers": True,
        "manual_models": True,
        "locality": False,
        "model_filter": False,
        "read_timeout_s": False,
        "auto_benchmark": False,
        "hub_shared": False,
    }

    @_notifies(CHANGE_PROVIDERS)
    async def update_provider_settings(self, provider_id: int, **fields: Any) -> None:
        """Write optional provider settings; None clears one."""
        if not fields:
            return
        unknown = set(fields) - set(self._PROVIDER_SETTINGS)
        if unknown:
            raise ValueError(f"unknown provider settings: {sorted(unknown)}")
        sets = []
        params: list[Any] = []
        for name, value in fields.items():
            sets.append(f"{name} = ?")
            if self._PROVIDER_SETTINGS[name] and value is not None:
                value = json.dumps(value) if value else None
            elif isinstance(value, bool):
                value = int(value)  # integer columns; asyncpg refuses a bool
            params.append(value)
        params.append(provider_id)
        await self._execute(
            f"UPDATE providers SET {', '.join(sets)}, "  # noqa: S608 - column names are whitelisted / fixed SQL fragments; values are bound
            "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            tuple(params),
        )

    @_notifies(CHANGE_PROVIDERS)
    async def set_hub_shared_providers(self, provider_ids: Iterable[int]) -> None:
        """Share exactly *provider_ids* with the hub this router works for."""
        ids = sorted({int(i) for i in provider_ids})
        if not ids:
            await self._execute("UPDATE providers SET hub_shared = 0")
            return
        # One statement, so a failure or a concurrent save cannot leave a
        # half-applied set.
        marks = ", ".join("?" for _ in ids)
        await self._execute(
            "UPDATE providers SET hub_shared = "  # noqa: S608 - placeholders only; values are bound
            f"CASE WHEN id IN ({marks}) THEN 1 ELSE 0 END",
            tuple(ids),
        )

    async def set_provider_api_key(
        self, provider_id: int, api_key_enc: str | None, header: str | None
    ) -> None:
        """Store (or with None, clear) the encrypted upstream API key."""
        await self._execute(
            "UPDATE providers SET api_key_enc = ?, api_key_header = ?, "
            "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (api_key_enc, header, provider_id),
        )

    async def update_provider_status(
        self, provider_id: int, status: ProviderStatus
    ) -> None:
        changed = await self._execute_rowcount(
            "UPDATE providers SET status = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE id = ? AND status != ?",
            (status.value, provider_id, status.value),
        )
        if changed:
            # Only a real transition matters to routing; health checks write
            # the same status every cycle.
            self._changed(CHANGE_STATUS)
        else:
            await self._execute(
                "UPDATE providers SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (provider_id,),
            )

    async def get_providers_for_model(
        self, model_name: str, protocol: str | None = None
    ) -> list[Provider]:
        query = (
            "SELECT p.* FROM providers p "
            "JOIN provider_models pm ON p.id = pm.provider_id "
            "WHERE (pm.name = ? OR pm.raw_name = ?) AND p.status != 'offline'"
        )
        params: tuple[Any, ...] = (model_name, model_name)
        if protocol is not None:
            query += " AND p.protocol = ?"
            params += (ProviderProtocol.parse(protocol).value,)
        rows = await self._select_all(query, params)
        return [_row_to_provider(r) for r in rows]

    # --- Addresses ---

    @_notifies(CHANGE_ADDRESSES)
    async def add_address(
        self,
        provider_id: int,
        url: str,
        is_preferred: bool = False,
    ) -> ProviderAddress:
        url = url.rstrip("/")
        aid = await self._insert_get_id(
            "INSERT INTO provider_addresses (provider_id, url, is_preferred) "
            "VALUES (?, ?, ?)",
            (provider_id, url, int(is_preferred)),
        )
        return ProviderAddress(
            id=aid,
            provider_id=provider_id,
            url=url,
            is_preferred=is_preferred,
        )

    @_notifies(CHANGE_ADDRESSES)
    async def update_address(
        self,
        address_id: int,
        url: str,
        is_preferred: bool | None = None,
    ) -> None:
        url = url.rstrip("/")
        if is_preferred is not None:
            await self._execute(
                "UPDATE provider_addresses SET url = ?, is_preferred = ? WHERE id = ?",
                (url, int(is_preferred), address_id),
            )
        else:
            await self._execute(
                "UPDATE provider_addresses SET url = ? WHERE id = ?",
                (url, address_id),
            )

    @_notifies(CHANGE_ADDRESSES)
    async def remove_address(self, address_id: int) -> None:
        await self._execute(
            "DELETE FROM provider_addresses WHERE id = ?", (address_id,)
        )

    async def set_address_preferred(self, address_id: int, is_preferred: bool) -> None:
        await self._execute(
            "UPDATE provider_addresses SET is_preferred = ? WHERE id = ?",
            (int(is_preferred), address_id),
        )

    @_notifies(CHANGE_ADDRESSES)
    async def set_address_live(self, address_id: int, is_live: bool) -> None:
        await self._execute(
            "UPDATE provider_addresses SET is_live = ? WHERE id = ?",
            (int(is_live), address_id),
        )

    async def get_addresses(self, provider_id: int) -> list[ProviderAddress]:
        rows = await self._select_all(
            "SELECT * FROM provider_addresses WHERE provider_id = ? "
            "ORDER BY is_preferred DESC, id ASC",
            (provider_id,),
        )
        return [_row_to_address(r) for r in rows]

    async def get_address(self, address_id: int) -> ProviderAddress | None:
        row = await self._select_one(
            "SELECT * FROM provider_addresses WHERE id = ?", (address_id,)
        )
        return _row_to_address(row) if row else None

    # --- Models ---

    @_notifies(CHANGE_MODELS)
    async def set_provider_models(
        self, provider_id: int, models: list[ProviderModel]
    ) -> None:
        sf = self._require_session_factory()
        async with sf() as session, session.begin():
            st_del, bd_del = qmark(
                "DELETE FROM provider_models WHERE provider_id = ?",
                provider_id,
            )
            await session.execute(st_del, bd_del)
            for m in models:
                st_ins, bd_ins = qmark(
                    "INSERT INTO provider_models "
                    "(provider_id, name, raw_name, size, digest, modified_at, details) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    provider_id,
                    m.name,
                    m.raw_name,
                    m.size,
                    m.digest,
                    m.modified_at,
                    json.dumps(m.details) if m.details else None,
                )
                await session.execute(st_ins, bd_ins)

    async def get_backend_model_name(self, provider_id: int, model_name: str) -> str:
        row = await self._select_one(
            "SELECT name, raw_name FROM provider_models "
            "WHERE provider_id = ? AND (name = ? OR raw_name = ?) "
            "LIMIT 1",
            (provider_id, model_name, model_name),
        )
        if row:
            return row["raw_name"] or row["name"]
        return model_name

    async def get_provider_models(self, provider_id: int) -> list[ProviderModel]:
        rows = await self._select_all(
            "SELECT * FROM provider_models WHERE provider_id = ?", (provider_id,)
        )
        return [_row_to_model(r) for r in rows]

    async def get_all_provider_models(self) -> list[ProviderModel]:
        """Every provider's models in one query, for the routing table."""
        rows = await self._select_all(
            "SELECT * FROM provider_models ORDER BY provider_id, id"
        )
        return [_row_to_model(r) for r in rows]

    async def get_all_addresses(self) -> list[ProviderAddress]:
        rows = await self._select_all(
            "SELECT * FROM provider_addresses "
            "ORDER BY provider_id, is_preferred DESC, id ASC"
        )
        return [_row_to_address(r) for r in rows]

    async def list_all_models(
        self,
        *,
        hub_only: bool = False,
        exclude_ids: Iterable[int] | None = None,
        only_id: int | None = None,
    ) -> list[dict]:
        """Models of online providers, one row per name.

        With *hub_only*, only providers shared with the hub (see
        ``Provider.hub_shared``): what a hub sees through the worker link.
        Providers in *exclude_ids* are left out; with *only_id*, only that
        provider's models are listed.
        """
        shared = "AND p.hub_shared != 0" if hub_only else ""
        # Integers only, so they can go in the text.
        skip = sorted({int(i) for i in exclude_ids or ()})
        if skip:
            shared += f" AND p.id NOT IN ({', '.join(str(i) for i in skip)})"
        if only_id is not None:
            shared += f" AND p.id = {int(only_id)}"
        q = text(f"""
            SELECT name, size, digest, modified_at, details FROM (
                SELECT pm.name, pm.size, pm.digest, pm.modified_at, pm.details,
                    ROW_NUMBER() OVER (
                        PARTITION BY pm.name ORDER BY pm.id
                    ) AS rn
                FROM provider_models pm
                INNER JOIN providers p ON p.id = pm.provider_id
                WHERE p.status != 'offline' {shared}
            ) ranked
            WHERE rn = 1
            ORDER BY name
            """)  # noqa: S608 - the only interpolation is a fixed clause
        sf = self._require_session_factory()
        async with sf() as session:
            r = await session.execute(q)
            rows = result_mappings(r)
        return [
            {
                "name": row["name"],
                "size": row["size"],
                "digest": row["digest"],
                "modified_at": row["modified_at"],
                "details": json.loads(row["details"]) if row["details"] else {},
            }
            for row in rows
        ]

    # --- Benchmarks ---

    @_notifies(CHANGE_BENCHMARKS)
    async def save_benchmark(self, result: BenchmarkResult) -> None:
        await self._execute(
            "INSERT INTO benchmarks "
            "(provider_id, model_name, protocol, startup_time_ms, tokens_per_second, "
            "source) VALUES (?, ?, ?, ?, ?, ?)",
            (
                result.provider_id,
                result.model_name,
                result.protocol,
                result.startup_time_ms,
                result.tokens_per_second,
                result.source,
            ),
        )

    async def get_latest_benchmark(
        self, provider_id: int, model_name: str, protocol: str | None = None
    ) -> BenchmarkResult | None:
        if protocol:
            query = (
                "SELECT * FROM benchmarks WHERE provider_id = ? AND model_name = ? "
                "AND protocol = ? ORDER BY created_at DESC, id DESC LIMIT 1"
            )
            params: tuple[Any, ...] = (provider_id, model_name, protocol)
        else:
            query = (
                "SELECT * FROM benchmarks WHERE provider_id = ? AND model_name = ? "
                "ORDER BY created_at DESC, id DESC LIMIT 1"
            )
            params = (provider_id, model_name)
        row = await self._select_one(query, params)
        return _row_to_benchmark(row) if row else None

    async def get_all_benchmark_results(self) -> list[BenchmarkResult]:
        """Every benchmark, newest first, for the routing table."""
        rows = await self._select_all(
            "SELECT * FROM benchmarks ORDER BY created_at DESC, id DESC"
        )
        return [_row_to_benchmark(r) for r in rows]

    async def get_benchmarks_for_provider(
        self, provider_id: int
    ) -> list[BenchmarkResult]:
        rows = await self._select_all(
            "SELECT * FROM benchmarks WHERE provider_id = ? "
            "ORDER BY created_at DESC, id DESC",
            (provider_id,),
        )
        return [_row_to_benchmark(r) for r in rows]

    async def get_all_benchmarks(self) -> list[dict]:
        rows = await self._select_all(
            "SELECT b.*, p.name AS provider_name "
            "FROM benchmarks b "
            "JOIN providers p ON p.id = b.provider_id "
            "ORDER BY b.model_name ASC, b.tokens_per_second ASC"
        )
        return [
            {
                "id": r["id"],
                "provider_id": r["provider_id"],
                "provider_name": r["provider_name"],
                "model_name": r["model_name"],
                "protocol": r["protocol"],
                "startup_time_ms": r["startup_time_ms"],
                "tokens_per_second": r["tokens_per_second"],
                "source": r.get("source"),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    @_notifies(CHANGE_BENCHMARKS)
    async def delete_benchmark(self, benchmark_id: int) -> None:
        await self._execute("DELETE FROM benchmarks WHERE id = ?", (benchmark_id,))

    @_notifies(CHANGE_BENCHMARKS)
    async def delete_benchmarks_for_model(self, model_name: str) -> int:
        return await self._execute_rowcount(
            "DELETE FROM benchmarks WHERE model_name = ?", (model_name,)
        )

    @_notifies(CHANGE_BENCHMARKS)
    async def delete_all_benchmarks(self) -> int:
        return await self._execute_rowcount("DELETE FROM benchmarks")

    # --- Request Log ---

    async def save_request_log(self, entry: RequestLog) -> None:
        await self._execute(
            "INSERT INTO request_log "
            "(provider_id, provider_name, protocol, endpoint, source_ip, model, "
            "request_size, response_size, request_meta, duration_ms, ttfb_ms, "
            "router_ms, status, streamed, error_detail, locality, api_key_id, "
            "prompt_tokens, completion_tokens, cost_usd, routed_model) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.provider_id,
                entry.provider_name,
                entry.protocol,
                entry.endpoint,
                entry.source_ip,
                entry.model,
                entry.request_size,
                entry.response_size,
                entry.request_meta,
                entry.duration_ms,
                entry.ttfb_ms,
                entry.router_ms,
                entry.status,
                int(entry.streamed),
                entry.error_detail,
                entry.locality,
                entry.api_key_id,
                entry.prompt_tokens,
                entry.completion_tokens,
                entry.cost_usd,
                entry.routed_model,
            ),
        )

    async def delete_request_logs_before(self, before: datetime) -> int:
        """Delete request log rows created before *before* (UTC); the count."""
        when: Any = before
        if self._dialect_name == "sqlite":
            when = before.strftime("%Y-%m-%d %H:%M:%S")
        return await self._execute_rowcount(
            "DELETE FROM request_log WHERE created_at < ?", (when,)
        )

    async def spend_since(self, key_id: int, since: datetime) -> float:
        """USD a key has spent on requests logged since *since* (UTC)."""
        when: Any = since
        if self._dialect_name == "sqlite":
            when = since.strftime("%Y-%m-%d %H:%M:%S")
        row = await self._select_one(
            "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM request_log "
            "WHERE api_key_id = ? AND created_at >= ?",
            (key_id, when),
        )
        return float(row["total"] or 0) if row else 0.0

    async def spend_by_key_since(self, since: datetime) -> dict[int, float]:
        when: Any = since
        if self._dialect_name == "sqlite":
            when = since.strftime("%Y-%m-%d %H:%M:%S")
        rows = await self._select_all(
            "SELECT api_key_id, COALESCE(SUM(cost_usd), 0) AS total FROM request_log "
            "WHERE api_key_id IS NOT NULL AND created_at >= ? GROUP BY api_key_id",
            (when,),
        )
        return {int(r["api_key_id"]): float(r["total"] or 0) for r in rows}

    async def usage_by_key_and_provider(
        self, since: datetime | None = None, key_id: int | None = None
    ) -> list[dict]:
        """Requests, tokens and cost per API key and provider, busiest first.

        Counts requests logged since *since* (UTC; None for all), only for
        *key_id* when given. Requests with no key have ``api_key_id`` None;
        requests that never reached a provider, and the router's own entries
        (health and probe events, benchmarks, model pulls), are left out.
        Names come from the current rows, or from the log for a provider
        since deleted.
        """
        where = [
            "r.provider_id IS NOT NULL",
            "r.protocol <> 'system'",
            "COALESCE(r.source_ip, '') <> 'internal'",
        ]
        params: list[Any] = []
        if since is not None:
            where.append("r.created_at >= ?")
            params.append(
                since.strftime("%Y-%m-%d %H:%M:%S")
                if self._dialect_name == "sqlite"
                else since
            )
        if key_id is not None:
            where.append("r.api_key_id = ?")
            params.append(key_id)
        rows = await self._select_all(
            "SELECT r.api_key_id, MAX(k.key_prefix) AS key_prefix, "  # noqa: S608 - fixed clauses; values are bound
            "r.provider_id, COALESCE(MAX(p.name), MAX(r.provider_name)) AS provider, "
            "MAX(CASE WHEN p.id IS NULL THEN 0 ELSE 1 END) AS provider_exists, "
            "COUNT(*) AS requests, "
            "SUM(CASE WHEN r.status = 'error' THEN 1 ELSE 0 END) AS errors, "
            "COALESCE(SUM(r.prompt_tokens), 0) AS prompt_tokens, "
            "COALESCE(SUM(r.completion_tokens), 0) AS completion_tokens, "
            "COALESCE(SUM(r.cost_usd), 0) AS cost_usd, "
            "MAX(r.created_at) AS last_request_at "
            "FROM request_log r "
            "LEFT JOIN api_keys k ON k.id = r.api_key_id "
            "LEFT JOIN providers p ON p.id = r.provider_id "
            f"WHERE {' AND '.join(where)} "
            "GROUP BY r.api_key_id, r.provider_id "
            "ORDER BY requests DESC, r.api_key_id, r.provider_id",
            tuple(params),
        )
        return [
            {
                "api_key_id": (
                    int(r["api_key_id"]) if r["api_key_id"] is not None else None
                ),
                "key_prefix": r["key_prefix"],
                "provider_id": int(r["provider_id"]),
                "provider": r["provider"],
                "provider_exists": bool(r["provider_exists"]),
                "requests": int(r["requests"]),
                "errors": int(r["errors"] or 0),
                "prompt_tokens": int(r["prompt_tokens"] or 0),
                "completion_tokens": int(r["completion_tokens"] or 0),
                "cost_usd": float(r["cost_usd"] or 0),
                "last_request_at": _utc_datetime(r["last_request_at"]),
            }
            for r in rows
        ]

    # ── Model prices ──────────────────────────────────────────────────

    async def list_model_prices(self, provider_id: int | None = None) -> list[dict]:
        if provider_id is None:
            rows = await self._select_all(
                "SELECT * FROM model_prices ORDER BY provider_id, model_pattern"
            )
        else:
            rows = await self._select_all(
                "SELECT * FROM model_prices WHERE provider_id = ? "
                "ORDER BY model_pattern",
                (provider_id,),
            )
        return [dict(r) for r in rows]

    @_notifies(CHANGE_PRICES)
    async def set_model_price(
        self,
        provider_id: int,
        model_pattern: str,
        input_per_mtok: float,
        output_per_mtok: float,
    ) -> None:
        if self._dialect_name in ("mysql", "mariadb"):
            await self._execute(
                "INSERT INTO model_prices (provider_id, model_pattern, input_per_mtok, "
                "output_per_mtok) VALUES (?, ?, ?, ?) ON DUPLICATE KEY UPDATE "
                "input_per_mtok = VALUES(input_per_mtok), "
                "output_per_mtok = VALUES(output_per_mtok)",
                (provider_id, model_pattern, input_per_mtok, output_per_mtok),
            )
        else:
            await self._execute(
                "INSERT INTO model_prices (provider_id, model_pattern, input_per_mtok, "
                "output_per_mtok) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (provider_id, model_pattern) DO UPDATE SET "
                "input_per_mtok = ?, output_per_mtok = ?",
                (
                    provider_id,
                    model_pattern,
                    input_per_mtok,
                    output_per_mtok,
                    input_per_mtok,
                    output_per_mtok,
                ),
            )

    @_notifies(CHANGE_PRICES)
    async def remove_model_price(self, provider_id: int, price_id: int) -> bool:
        return bool(
            await self._execute_rowcount(
                "DELETE FROM model_prices WHERE id = ? AND provider_id = ?",
                (price_id, provider_id),
            )
        )

    async def get_request_logs(
        self, limit: int = 200, offset: int = 0
    ) -> list[RequestLog]:
        rows = await self._select_all(
            # id breaks ties: MySQL TIMESTAMP has whole-second precision.
            "SELECT * FROM request_log ORDER BY created_at DESC, id DESC "
            "LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [_row_to_request_log(r) for r in rows]

    async def get_model_request_counts(self) -> dict[str, int]:
        rows = await self._select_all(
            "SELECT model, COUNT(*) AS cnt FROM request_log "
            "WHERE model IS NOT NULL GROUP BY model"
        )
        return {r["model"]: r["cnt"] for r in rows}

    async def count_request_logs(self) -> int:
        row = await self._select_one("SELECT COUNT(*) AS cnt FROM request_log")
        return row["cnt"] if row else 0

    # ── Model fallbacks ───────────────────────────────────────────────

    @_notifies(CHANGE_FALLBACKS)
    async def set_model_fallback(self, model_name: str, fallback_model: str) -> None:
        if self._dialect_name in ("mysql", "mariadb"):
            await self._execute(
                "INSERT INTO model_fallbacks (model_name, fallback_model) "
                "VALUES (?, ?) "
                "ON DUPLICATE KEY UPDATE fallback_model = VALUES(fallback_model)",
                (model_name, fallback_model),
            )
        else:
            await self._execute(
                "INSERT INTO model_fallbacks (model_name, fallback_model) "
                "VALUES (?, ?) "
                "ON CONFLICT (model_name) DO UPDATE SET fallback_model = ?",
                (model_name, fallback_model, fallback_model),
            )

    @_notifies(CHANGE_FALLBACKS)
    async def remove_model_fallback(self, model_name: str) -> None:
        await self._execute(
            "DELETE FROM model_fallbacks WHERE model_name = ?", (model_name,)
        )

    async def get_model_fallback(self, model_name: str) -> str | None:
        row = await self._select_one(
            "SELECT fallback_model FROM model_fallbacks WHERE model_name = ?",
            (model_name,),
        )
        return row["fallback_model"] if row else None

    async def get_all_model_fallbacks(self) -> dict[str, str]:
        rows = await self._select_all(
            "SELECT model_name, fallback_model FROM model_fallbacks ORDER BY model_name"
        )
        return {r["model_name"]: r["fallback_model"] for r in rows}

    # ── Decision calibrations ─────────────────────────────────────────

    @_notifies(CHANGE_DECISIONS)
    async def set_decision_calibration(
        self,
        model_name: str,
        question_key: str,
        question_type: str,
        question: dict,
        temperature: float,
        offsets: dict[str, float],
        examples: int | None = None,
    ) -> None:
        """Store the calibration for *model_name* and one question wording."""
        q = json.dumps(question, ensure_ascii=False)
        o = json.dumps(offsets)
        if self._dialect_name in ("mysql", "mariadb"):
            await self._execute(
                "INSERT INTO decision_calibrations (model_name, question_key, "
                "question_type, question, temperature, offsets, examples) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON DUPLICATE KEY UPDATE "
                "question_type = VALUES(question_type), question = VALUES(question), "
                "temperature = VALUES(temperature), offsets = VALUES(offsets), "
                "examples = VALUES(examples), updated_at = CURRENT_TIMESTAMP",
                (model_name, question_key, question_type, q, temperature, o, examples),
            )
        else:
            await self._execute(
                "INSERT INTO decision_calibrations (model_name, question_key, "
                "question_type, question, temperature, offsets, examples) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (model_name, question_key) DO UPDATE SET "
                "question_type = ?, question = ?, temperature = ?, offsets = ?, "
                "examples = ?, updated_at = CURRENT_TIMESTAMP",
                (
                    model_name,
                    question_key,
                    question_type,
                    q,
                    temperature,
                    o,
                    examples,
                    question_type,
                    q,
                    temperature,
                    o,
                    examples,
                ),
            )

    async def list_decision_calibrations(self) -> list[dict[str, Any]]:
        rows = await self._select_all(
            "SELECT * FROM decision_calibrations ORDER BY model_name, id"
        )
        return [_row_to_calibration(r) for r in rows]

    @_notifies(CHANGE_DECISIONS)
    async def delete_decision_calibration(self, calibration_id: int) -> bool:
        n = await self._execute_rowcount(
            "DELETE FROM decision_calibrations WHERE id = ?", (calibration_id,)
        )
        return n > 0

    # ── Dashboard chat ────────────────────────────────────────────────
    # Each conversation belongs to one dashboard user; ``user_id`` None is
    # the conversations made while the dashboard had no users.

    @staticmethod
    def _chat_owner(user_id: int | None) -> tuple[str, tuple[Any, ...]]:
        if user_id is None:
            return "user_id IS NULL", ()
        return "user_id = ?", (user_id,)

    async def create_chat_conversation(
        self,
        user_id: int | None,
        title: str,
        provider_id: int | None,
        model_name: str | None,
    ) -> int:
        return await self._insert_get_id(
            "INSERT INTO chat_conversations (user_id, title, provider_id, "
            "model_name) VALUES (?, ?, ?, ?)",
            (user_id, title, provider_id, model_name),
        )

    async def list_chat_conversations(
        self, user_id: int | None
    ) -> list[dict[str, Any]]:
        """The user's conversations, most recently used first, without
        their messages."""
        owner, params = self._chat_owner(user_id)
        rows = await self._select_all(
            "SELECT * FROM chat_conversations "  # noqa: S608 - fixed clause; values are bound
            f"WHERE {owner} ORDER BY updated_at DESC, id DESC",
            params,
        )
        return [_row_to_chat_conversation(r) for r in rows]

    async def get_chat_conversation(
        self, conversation_id: int, user_id: int | None
    ) -> dict[str, Any] | None:
        """The conversation with its messages, if it is the user's."""
        owner, params = self._chat_owner(user_id)
        row = await self._select_one(
            "SELECT * FROM chat_conversations "  # noqa: S608 - fixed clause; values are bound
            f"WHERE id = ? AND {owner}",
            (conversation_id, *params),
        )
        if row is None:
            return None
        conv = _row_to_chat_conversation(row)
        rows = await self._select_all(
            "SELECT * FROM chat_messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        )
        conv["messages"] = [_row_to_chat_message(r) for r in rows]
        return conv

    async def add_chat_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        *,
        image: str | None = None,
        model_name: str | None = None,
        provider_name: str | None = None,
        duration_ms: float | None = None,
    ) -> dict[str, Any]:
        mid = await self._insert_get_id(
            "INSERT INTO chat_messages (conversation_id, role, content, image, "
            "model_name, provider_name, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                role,
                content,
                image,
                model_name,
                provider_name,
                duration_ms,
            ),
        )
        row = await self._select_one("SELECT * FROM chat_messages WHERE id = ?", (mid,))
        if row is None:  # pragma: no cover - just inserted
            raise RuntimeError("chat message vanished")
        return _row_to_chat_message(row)

    async def touch_chat_conversation(
        self, conversation_id: int, provider_id: int | None, model_name: str | None
    ) -> None:
        """Remember the model last used and move the conversation to the top."""
        await self._execute(
            "UPDATE chat_conversations SET provider_id = ?, model_name = ?, "
            "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (provider_id, model_name, conversation_id),
        )

    async def delete_chat_conversation(
        self, conversation_id: int, user_id: int | None
    ) -> bool:
        owner, params = self._chat_owner(user_id)
        await self._execute(
            "DELETE FROM chat_messages WHERE conversation_id IN ("  # noqa: S608 - fixed clause; values are bound
            f"SELECT id FROM chat_conversations WHERE id = ? AND {owner})",
            (conversation_id, *params),
        )
        n = await self._execute_rowcount(
            f"DELETE FROM chat_conversations WHERE id = ? AND {owner}",  # noqa: S608 - fixed clause; values are bound
            (conversation_id, *params),
        )
        return n > 0

    # ── Model aliases ─────────────────────────────────────────────────

    @_notifies(CHANGE_ALIASES)
    async def set_model_alias(self, alias: str, canonical: str) -> None:
        """Make *alias* another name for *canonical*."""
        if self._dialect_name in ("mysql", "mariadb"):
            await self._execute(
                "INSERT INTO model_aliases (alias, canonical) VALUES (?, ?) "
                "ON DUPLICATE KEY UPDATE canonical = VALUES(canonical)",
                (alias, canonical),
            )
        else:
            await self._execute(
                "INSERT INTO model_aliases (alias, canonical) VALUES (?, ?) "
                "ON CONFLICT (alias) DO UPDATE SET canonical = ?",
                (alias, canonical, canonical),
            )

    @_notifies(CHANGE_ALIASES)
    async def remove_model_alias(self, alias: str) -> bool:
        return bool(
            await self._execute_rowcount(
                "DELETE FROM model_aliases WHERE alias = ?", (alias,)
            )
        )

    async def get_model_aliases(self) -> dict[str, str]:
        """alias -> canonical name."""
        rows = await self._select_all(
            "SELECT alias, canonical FROM model_aliases ORDER BY canonical, alias"
        )
        return {r["alias"]: r["canonical"] for r in rows}

    # ── Model groups ──────────────────────────────────────────────────

    async def list_model_groups(self) -> list[ModelGroup]:
        """Every group with its members, in name then position order."""
        groups = await self._select_all(
            "SELECT id, name, description FROM model_groups ORDER BY name"
        )
        members = await self._select_all(
            "SELECT id, group_id, model_name, provider_id, position "
            "FROM model_group_members ORDER BY group_id, position, id"
        )
        by_group: dict[int, list[ModelGroupMember]] = {}
        for m in members:
            by_group.setdefault(int(m["group_id"]), []).append(_row_to_member(m))
        return [
            ModelGroup(
                id=int(g["id"]),
                name=g["name"],
                description=g["description"],
                members=by_group.get(int(g["id"]), []),
            )
            for g in groups
        ]

    async def get_model_group(self, name: str) -> ModelGroup | None:
        for g in await self.list_model_groups():
            if g.name == name:
                return g
        return None

    @_notifies(CHANGE_GROUPS)
    async def create_model_group(
        self, name: str, description: str | None = None
    ) -> ModelGroup:
        gid = await self._insert_get_id(
            "INSERT INTO model_groups (name, description) VALUES (?, ?)",
            (name, description),
        )
        return ModelGroup(id=gid, name=name, description=description)

    @_notifies(CHANGE_GROUPS)
    async def update_model_group(
        self,
        name: str,
        *,
        description: str | _Keep | None = _KEEP,
        new_name: str | None = None,
    ) -> bool:
        """Rename the group and/or set its description. A field left out is
        unchanged; ``description=None`` clears it."""
        sets: list[str] = []
        params: list[Any] = []
        if not isinstance(description, _Keep):
            sets.append("description = ?")
            params.append(description)
        if new_name is not None and new_name != name:
            sets.append("name = ?")
            params.append(new_name)
        if not sets:
            return await self.get_model_group(name) is not None
        n = await self._execute_rowcount(
            f"UPDATE model_groups SET {', '.join(sets)} WHERE name = ?",  # noqa: S608 - fixed column names; values are bound
            (*params, name),
        )
        return n > 0

    @_notifies(CHANGE_GROUPS)
    async def delete_model_group(self, name: str) -> bool:
        row = await self._select_one(
            "SELECT id FROM model_groups WHERE name = ?", (name,)
        )
        if row is None:
            return False
        # Explicit, so MySQL without enforced foreign keys and SQLite with
        # them off end up the same.
        await self._execute(
            "DELETE FROM model_group_members WHERE group_id = ?", (row["id"],)
        )
        await self._execute("DELETE FROM model_groups WHERE id = ?", (row["id"],))
        return True

    @_notifies(CHANGE_GROUPS)
    async def add_model_group_member(
        self, group_id: int, model_name: str, provider_id: int | None = None
    ) -> ModelGroupMember:
        """Add a member, or return the existing one with the same model and
        provider."""
        rows = await self._select_all(
            "SELECT id, group_id, model_name, provider_id, position "
            "FROM model_group_members WHERE group_id = ?",
            (group_id,),
        )
        for r in rows:
            if r["model_name"] == model_name and r["provider_id"] == provider_id:
                return _row_to_member(r)
        position = max((int(r["position"]) for r in rows), default=-1) + 1
        mid = await self._insert_get_id(
            "INSERT INTO model_group_members "
            "(group_id, model_name, provider_id, position) VALUES (?, ?, ?, ?)",
            (group_id, model_name, provider_id, position),
        )
        return ModelGroupMember(
            id=mid,
            group_id=group_id,
            model_name=model_name,
            provider_id=provider_id,
            position=position,
        )

    @_notifies(CHANGE_GROUPS)
    async def remove_model_group_member(self, group_id: int, member_id: int) -> bool:
        n = await self._execute_rowcount(
            "DELETE FROM model_group_members WHERE id = ? AND group_id = ?",
            (member_id, group_id),
        )
        return n > 0

    @_notifies(CHANGE_GROUPS)
    async def set_model_group_members(
        self, group_id: int, members: list[tuple[str, int | None]]
    ) -> None:
        """Make the group's members exactly ``members`` (model, provider id)
        in one transaction. Members that stay keep their id and position; new
        ones go after them in the order given."""
        wanted = list(dict.fromkeys(members))
        sf = self._require_session_factory()
        async with sf() as session, session.begin():
            st, bd = qmark(
                "SELECT id, model_name, provider_id, position "
                "FROM model_group_members WHERE group_id = ?",
                group_id,
            )
            rows = list(result_mappings(await session.execute(st, bd)))
            have = {(r["model_name"], r["provider_id"]): r for r in rows}
            for key, r in have.items():
                if key not in wanted:
                    st, bd = qmark(
                        "DELETE FROM model_group_members WHERE id = ?", r["id"]
                    )
                    await session.execute(st, bd)
            position = max((int(r["position"]) for r in rows), default=-1)
            for model_name, provider_id in wanted:
                if (model_name, provider_id) in have:
                    continue
                position += 1
                st, bd = qmark(
                    "INSERT INTO model_group_members "
                    "(group_id, model_name, provider_id, position) "
                    "VALUES (?, ?, ?, ?)",
                    group_id,
                    model_name,
                    provider_id,
                    position,
                )
                await session.execute(st, bd)

    # --- Workers (hub side) ---

    async def create_worker(
        self, name: str, token_hash: str, token_prefix: str, provider_id: int
    ) -> int:
        return await self._insert_get_id(
            "INSERT INTO workers (name, token_hash, token_prefix, provider_id) "
            "VALUES (?, ?, ?, ?)",
            (name, token_hash, token_prefix, provider_id),
        )

    async def list_workers(self) -> list[dict]:
        rows = await self._select_all("SELECT * FROM workers ORDER BY name")
        return [_row_to_worker(r) for r in rows]

    async def get_worker(self, worker_id: int) -> dict | None:
        row = await self._select_one("SELECT * FROM workers WHERE id = ?", (worker_id,))
        return _row_to_worker(row) if row else None

    async def get_worker_by_token_hash(self, token_hash: str) -> dict | None:
        row = await self._select_one(
            "SELECT * FROM workers WHERE token_hash = ?", (token_hash,)
        )
        return _row_to_worker(row) if row else None

    async def get_worker_by_provider(self, provider_id: int) -> dict | None:
        row = await self._select_one(
            "SELECT * FROM workers WHERE provider_id = ?", (provider_id,)
        )
        return _row_to_worker(row) if row else None

    async def set_worker_token(
        self, worker_id: int, token_hash: str, token_prefix: str
    ) -> None:
        await self._execute(
            "UPDATE workers SET token_hash = ?, token_prefix = ? WHERE id = ?",
            (token_hash, token_prefix, worker_id),
        )

    async def set_worker_enabled(self, worker_id: int, enabled: bool) -> None:
        await self._execute(
            "UPDATE workers SET enabled = ? WHERE id = ?", (int(enabled), worker_id)
        )

    async def touch_worker_connected(
        self, worker_id: int, remote_addr: str | None
    ) -> None:
        await self._execute(
            "UPDATE workers SET last_connected_at = CURRENT_TIMESTAMP, "
            "last_remote_addr = ? WHERE id = ?",
            (remote_addr, worker_id),
        )

    async def delete_worker(self, worker_id: int) -> None:
        await self._execute("DELETE FROM workers WHERE id = ?", (worker_id,))

    async def resolve_fallback_chain(
        self, model_name: str, max_depth: int = 10
    ) -> list[str]:
        query = """
        WITH RECURSIVE chain(depth, resolved_name) AS (
            SELECT 0, ?
            UNION ALL
            SELECT c.depth + 1, mf.fallback_model
            FROM chain c
            INNER JOIN model_fallbacks mf ON mf.model_name = c.resolved_name
            WHERE c.depth < ? AND mf.fallback_model IS NOT NULL
        )
        SELECT resolved_name FROM chain ORDER BY depth
        """
        rows = await self._select_all(query, (model_name, max_depth))
        chain: list[str] = []
        seen: set[str] = set()
        for row in rows:
            name = row["resolved_name"]
            if name in seen:
                break
            seen.add(name)
            chain.append(name)
        return chain

    # --- API Keys / Auth settings ---

    @_notifies(CHANGE_API_KEYS)
    async def create_api_key(
        self,
        key_prefix: str,
        key_hash: str,
        routing_mode: str,
        allow_fallback: bool,
        session_affinity: bool | None = None,
        locality: str | None = None,
        daily_spend_cap_usd: float | None = None,
        owner_user_id: int | None = None,
    ) -> int:
        """*owner_user_id* is the dashboard user who issued the key to
        themselves; None for a key an admin issued."""
        return await self._insert_get_id(
            "INSERT INTO api_keys "
            "(key_prefix, key_hash, routing_mode, allow_fallback, session_affinity, "
            "locality, daily_spend_cap_usd, owner_user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key_prefix,
                key_hash,
                routing_mode,
                int(allow_fallback),
                None if session_affinity is None else int(session_affinity),
                locality,
                daily_spend_cap_usd,
                owner_user_id,
            ),
        )

    @_notifies(CHANGE_API_KEYS)
    async def set_api_key_spend_cap(self, key_id: int, cap_usd: float | None) -> bool:
        """USD a key may spend per UTC day before it is limited to local."""
        return bool(
            await self._execute_rowcount(
                "UPDATE api_keys SET daily_spend_cap_usd = ? WHERE id = ?",
                (cap_usd, key_id),
            )
        )

    @_notifies(CHANGE_API_KEYS)
    async def set_api_key_locality(self, key_id: int, locality: str | None) -> bool:
        """Limit a key to local or cloud providers (None: either)."""
        return bool(
            await self._execute_rowcount(
                "UPDATE api_keys SET locality = ? WHERE id = ?", (locality, key_id)
            )
        )

    async def list_api_keys(self, owner_user_id: int | None = None) -> list[dict]:
        """All keys, or only those *owner_user_id* issued to themselves."""
        where = "WHERE k.owner_user_id = ? " if owner_user_id is not None else ""
        rows = await self._select_all(
            "SELECT "
            "k.id, k.key_prefix, k.routing_mode, k.allow_fallback, k.session_affinity, "
            "k.locality, k.daily_spend_cap_usd, k.created_at, k.last_used_at, "
            "k.owner_user_id, u.username AS owner_username, "
            "COUNT(p.id) AS pin_count "
            "FROM api_keys k "
            "LEFT JOIN api_key_model_pins p ON p.api_key_id = k.id "
            "LEFT JOIN dashboard_users u ON u.id = k.owner_user_id "
            + where
            + "GROUP BY k.id, k.key_prefix, k.routing_mode, k.allow_fallback, "
            "k.session_affinity, k.locality, k.daily_spend_cap_usd, k.created_at, "
            "k.last_used_at, k.owner_user_id, u.username "
            "ORDER BY k.created_at DESC, k.id DESC",
            (owner_user_id,) if owner_user_id is not None else (),
        )
        return [
            {
                "id": int(r["id"]),
                "key_prefix": r["key_prefix"],
                "routing_mode": r["routing_mode"],
                "allow_fallback": bool(r["allow_fallback"]),
                "session_affinity": _tristate(r["session_affinity"]),
                "locality": r["locality"],
                "daily_spend_cap_usd": r["daily_spend_cap_usd"],
                "created_at": r["created_at"],
                "last_used_at": r["last_used_at"],
                "owner_user_id": (
                    None if r["owner_user_id"] is None else int(r["owner_user_id"])
                ),
                "owner_username": r["owner_username"],
                "pin_count": int(r["pin_count"] or 0),
            }
            for r in rows
        ]

    @_notifies(CHANGE_API_KEYS)
    async def delete_api_key(self, key_id: int) -> None:
        await self._execute("DELETE FROM api_keys WHERE id = ?", (key_id,))

    @_notifies(CHANGE_API_KEYS)
    async def delete_owned_api_key(self, key_id: int, owner_user_id: int) -> bool:
        """Delete *key_id* only if *owner_user_id* issued it to themselves."""
        return bool(
            await self._execute_rowcount(
                "DELETE FROM api_keys WHERE id = ? AND owner_user_id = ?",
                (key_id, owner_user_id),
            )
        )

    async def set_api_key_hash(self, key_id: int, key_hash: str) -> None:
        """Replace a key's stored fingerprint (re-hashing with a new pepper, or
        upgrading a legacy SHA-256 one)."""
        await self._execute(
            "UPDATE api_keys SET key_hash = ? WHERE id = ?", (key_hash, key_id)
        )

    async def lookup_api_key(self, key_hash: str) -> dict | None:
        row = await self._select_one(
            "SELECT id, key_prefix, routing_mode, allow_fallback, session_affinity, "
            "locality, daily_spend_cap_usd FROM api_keys "
            "WHERE key_hash = ?",
            (key_hash,),
        )
        if not row:
            return None
        await self._execute(
            "UPDATE api_keys SET last_used_at = CURRENT_TIMESTAMP WHERE id = ?",
            (row["id"],),
        )
        return {
            "id": int(row["id"]),
            "key_prefix": row["key_prefix"],
            "routing_mode": row["routing_mode"],
            "allow_fallback": bool(row["allow_fallback"]),
            "session_affinity": _tristate(row["session_affinity"]),
            "locality": row["locality"],
            "daily_spend_cap_usd": row["daily_spend_cap_usd"],
        }

    async def touch_api_key(self, key_id: int) -> None:
        await self._execute(
            "UPDATE api_keys SET last_used_at = CURRENT_TIMESTAMP WHERE id = ?",
            (key_id,),
        )

    @_notifies(CHANGE_API_KEYS)
    async def set_api_key_model_pin(
        self, key_id: int, model_name: str, provider_id: int
    ) -> None:
        if self._dialect_name in ("mysql", "mariadb"):
            await self._execute(
                "INSERT INTO api_key_model_pins (api_key_id, model_name, provider_id) "
                "VALUES (?, ?, ?) "
                "ON DUPLICATE KEY UPDATE provider_id = VALUES(provider_id)",
                (key_id, model_name, provider_id),
            )
        elif self._dialect_name == "postgresql":
            await self._execute(
                "INSERT INTO api_key_model_pins (api_key_id, model_name, provider_id) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT (api_key_id, model_name) "
                "DO UPDATE SET provider_id = EXCLUDED.provider_id",
                (key_id, model_name, provider_id),
            )
        else:
            await self._execute(
                "INSERT INTO api_key_model_pins (api_key_id, model_name, provider_id) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT (api_key_id, model_name) "
                "DO UPDATE SET provider_id = excluded.provider_id",
                (key_id, model_name, provider_id),
            )

    @_notifies(CHANGE_API_KEYS)
    async def remove_api_key_model_pin(self, key_id: int, model_name: str) -> None:
        await self._execute(
            "DELETE FROM api_key_model_pins WHERE api_key_id = ? AND model_name = ?",
            (key_id, model_name),
        )

    async def get_api_key_decision_policy(self, key_id: int) -> dict | None:
        """The key's decision policy as stored (a JSON object), or None."""
        row = await self._select_one(
            "SELECT policy FROM api_key_decision_policies WHERE api_key_id = ?",
            (key_id,),
        )
        if not row:
            return None
        try:
            policy = json.loads(row["policy"])
        except TypeError, ValueError:
            logger.warning("Decision policy of API key %s is not JSON", key_id)
            return None
        return policy if isinstance(policy, dict) else None

    @_notifies(CHANGE_API_KEYS)
    async def set_api_key_decision_policy(self, key_id: int, policy: dict) -> None:
        data = json.dumps(policy, ensure_ascii=False)
        if self._dialect_name in ("mysql", "mariadb"):
            await self._execute(
                "INSERT INTO api_key_decision_policies (api_key_id, policy) "
                "VALUES (?, ?) ON DUPLICATE KEY UPDATE policy = VALUES(policy), "
                "updated_at = CURRENT_TIMESTAMP",
                (key_id, data),
            )
        else:
            await self._execute(
                "INSERT INTO api_key_decision_policies (api_key_id, policy) "
                "VALUES (?, ?) ON CONFLICT (api_key_id) DO UPDATE SET "
                "policy = ?, updated_at = CURRENT_TIMESTAMP",
                (key_id, data, data),
            )

    @_notifies(CHANGE_API_KEYS)
    async def delete_api_key_decision_policy(self, key_id: int) -> bool:
        n = await self._execute_rowcount(
            "DELETE FROM api_key_decision_policies WHERE api_key_id = ?", (key_id,)
        )
        return n > 0

    async def list_api_key_model_pins(self, key_id: int) -> list[dict]:
        rows = await self._select_all(
            "SELECT p.model_name, p.provider_id, pr.name AS provider_name "
            "FROM api_key_model_pins p "
            "JOIN providers pr ON pr.id = p.provider_id "
            "WHERE p.api_key_id = ? "
            "ORDER BY p.model_name ASC",
            (key_id,),
        )
        return [
            {
                "model_name": r["model_name"],
                "provider_id": int(r["provider_id"]),
                "provider_name": r["provider_name"],
            }
            for r in rows
        ]

    async def get_allow_unauthenticated(self) -> bool:
        row = await self._select_one(
            # Qualified so MySQL accepts the reserved word without quoting.
            "SELECT value FROM app_settings "
            "WHERE app_settings.key = 'allow_unauthenticated'"
        )
        if not row:
            return True
        return str(row["value"]).lower() in ("1", "true", "yes", "on")

    @_notifies(CHANGE_AUTH_SETTINGS)
    async def set_allow_unauthenticated(self, allow: bool) -> None:
        val = "true" if allow else "false"
        if self._dialect_name in ("mysql", "mariadb"):
            await self._execute(
                "INSERT INTO app_settings (`key`, value) "
                "VALUES ('allow_unauthenticated', ?) "
                "ON DUPLICATE KEY UPDATE value = VALUES(value)",
                (val,),
            )
        elif self._dialect_name == "postgresql":
            await self._execute(
                'INSERT INTO app_settings ("key", value) '
                "VALUES ('allow_unauthenticated', ?) "
                'ON CONFLICT ("key") DO UPDATE SET value = EXCLUDED.value',
                (val,),
            )
        else:
            await self._execute(
                'INSERT INTO app_settings ("key", value) '
                "VALUES ('allow_unauthenticated', ?) "
                'ON CONFLICT ("key") DO UPDATE SET value = excluded.value',
                (val,),
            )

    async def get_app_setting(self, key: str) -> str | None:
        row = await self._select_one(
            "SELECT value FROM app_settings WHERE app_settings.key = ?", (key,)
        )
        return str(row["value"]) if row else None

    async def set_app_setting(self, key: str, value: str) -> None:
        if self._dialect_name in ("mysql", "mariadb"):
            await self._execute(
                "INSERT INTO app_settings (`key`, value) VALUES (?, ?) "
                "ON DUPLICATE KEY UPDATE value = VALUES(value)",
                (key, value),
            )
        elif self._dialect_name == "postgresql":
            await self._execute(
                'INSERT INTO app_settings ("key", value) VALUES (?, ?) '
                'ON CONFLICT ("key") DO UPDATE SET value = EXCLUDED.value',
                (key, value),
            )
        else:
            await self._execute(
                'INSERT INTO app_settings ("key", value) VALUES (?, ?) '
                'ON CONFLICT ("key") DO UPDATE SET value = excluded.value',
                (key, value),
            )

    # --- Management keys ---

    async def create_management_key(
        self,
        *,
        name: str,
        key_prefix: str,
        key_hash: str,
        privileges: Iterable[str],
        created_by: str | None,
        expires_at: int | None,
    ) -> int:
        return await self._insert_get_id(
            "INSERT INTO management_keys "
            "(name, key_prefix, key_hash, privileges, created_by, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                name,
                key_prefix,
                key_hash,
                " ".join(sorted(privileges)),
                created_by,
                expires_at,
            ),
        )

    async def list_management_keys(self) -> list[dict]:
        rows = await self._select_all(
            f"SELECT {_MANAGEMENT_KEY_COLUMNS} FROM management_keys "  # noqa: S608 - fixed column list
            "ORDER BY created_at DESC, id DESC"
        )
        return [_row_to_management_key(r) for r in rows]

    async def lookup_management_key(self, key_hash: str) -> dict | None:
        row = await self._select_one(
            f"SELECT {_MANAGEMENT_KEY_COLUMNS} FROM management_keys "  # noqa: S608 - fixed column list
            "WHERE key_hash = ?",
            (key_hash,),
        )
        return _row_to_management_key(row) if row else None

    async def touch_management_key(self, key_id: int) -> None:
        await self._execute(
            "UPDATE management_keys SET last_used_at = CURRENT_TIMESTAMP WHERE id = ?",
            (key_id,),
        )

    async def delete_management_key(self, key_id: int) -> bool:
        return bool(
            await self._execute_rowcount(
                "DELETE FROM management_keys WHERE id = ?", (key_id,)
            )
        )

    async def count_dashboard_users(self) -> int:
        row = await self._select_one("SELECT COUNT(*) AS c FROM dashboard_users")
        return int(row["c"]) if row else 0

    async def list_dashboard_users(self) -> list[dict]:
        if self._dialect_name == "sqlite":
            order_sql = "ORDER BY username COLLATE NOCASE ASC"
        else:
            order_sql = "ORDER BY LOWER(username) ASC"
        rows = await self._select_all(
            f"SELECT {_DASHBOARD_USER_COLUMNS} FROM dashboard_users {order_sql}"  # noqa: S608 - column names are whitelisted / fixed SQL fragments; values are bound
        )
        users = [_row_to_dashboard_user(r) for r in rows]
        for u in users:
            u["has_password"] = bool(u.pop("password_hash"))
        return users

    async def get_dashboard_user_by_username(self, username: str) -> dict | None:
        u = username.strip()
        if self._dialect_name == "sqlite":
            row = await self._select_one(
                f"SELECT {_DASHBOARD_USER_COLUMNS} "  # noqa: S608 - column names are whitelisted / fixed SQL fragments; values are bound
                "FROM dashboard_users WHERE username = ? COLLATE NOCASE",
                (u,),
            )
        else:
            row = await self._select_one(
                f"SELECT {_DASHBOARD_USER_COLUMNS} "  # noqa: S608 - column names are whitelisted / fixed SQL fragments; values are bound
                "FROM dashboard_users WHERE LOWER(username) = LOWER(?)",
                (u,),
            )
        return _row_to_dashboard_user(row) if row else None

    async def get_dashboard_user_by_id(self, user_id: int) -> dict | None:
        row = await self._select_one(
            f"SELECT {_DASHBOARD_USER_COLUMNS} FROM dashboard_users WHERE id = ?",  # noqa: S608 - column names are whitelisted / fixed SQL fragments; values are bound
            (user_id,),
        )
        return _row_to_dashboard_user(row) if row else None

    async def get_dashboard_user_by_oidc(
        self, issuer: str, subject: str
    ) -> dict | None:
        row = await self._select_one(
            f"SELECT {_DASHBOARD_USER_COLUMNS} FROM dashboard_users "  # noqa: S608 - column names are whitelisted / fixed SQL fragments; values are bound
            "WHERE oidc_issuer = ? AND oidc_subject = ?",
            (issuer, subject),
        )
        return _row_to_dashboard_user(row) if row else None

    async def create_dashboard_user(
        self, username: str, password_hash: str, is_admin: bool
    ) -> int:
        return await self._insert_get_id(
            "INSERT INTO dashboard_users (username, password_hash, is_admin) "
            "VALUES (?, ?, ?)",
            (username.strip(), password_hash, int(is_admin)),
        )

    async def create_oidc_dashboard_user(
        self,
        username: str,
        issuer: str,
        subject: str,
        *,
        email: str | None,
        display_name: str | None,
        is_admin: bool,
    ) -> int:
        """Create an account that signs in only through OpenID Connect."""
        return await self._insert_get_id(
            "INSERT INTO dashboard_users (username, password_hash, is_admin, "
            "auth_source, oidc_issuer, oidc_subject, email, display_name, "
            "last_login_at) VALUES (?, NULL, ?, 'oidc', ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (username.strip(), int(is_admin), issuer, subject, email, display_name),
        )

    async def update_oidc_dashboard_user(
        self,
        user_id: int,
        *,
        email: str | None,
        display_name: str | None,
        is_admin: bool,
    ) -> None:
        """Refresh an OIDC account from the identity provider at sign-in."""
        await self._execute(
            "UPDATE dashboard_users SET email = ?, display_name = ?, is_admin = ?, "
            "last_login_at = CURRENT_TIMESTAMP WHERE id = ?",
            (email, display_name, int(is_admin), user_id),
        )

    async def touch_dashboard_user_login(self, user_id: int) -> None:
        await self._execute(
            "UPDATE dashboard_users SET last_login_at = CURRENT_TIMESTAMP WHERE id = ?",
            (user_id,),
        )

    async def update_dashboard_user_password(
        self, user_id: int, password_hash: str
    ) -> None:
        await self._execute(
            "UPDATE dashboard_users SET password_hash = ? WHERE id = ?",
            (password_hash, user_id),
        )

    @_notifies(CHANGE_API_KEYS)
    async def delete_dashboard_user(self, user_id: int) -> None:
        """Delete the user and the API keys they issued to themselves."""
        await self._execute("DELETE FROM api_keys WHERE owner_user_id = ?", (user_id,))
        await self._execute("DELETE FROM dashboard_users WHERE id = ?", (user_id,))

    async def count_dashboard_admins(self) -> int:
        row = await self._select_one(
            "SELECT COUNT(*) AS c FROM dashboard_users WHERE is_admin = 1"
        )
        return int(row["c"]) if row else 0


_MANAGEMENT_KEY_COLUMNS = (
    "id, name, key_prefix, privileges, created_by, created_at, expires_at, last_used_at"
)


def _row_to_management_key(row: Mapping[str, Any]) -> dict:
    expires = row["expires_at"]
    return {
        "id": int(row["id"]),
        "name": row["name"],
        "key_prefix": row["key_prefix"],
        "privileges": str(row["privileges"] or "").split(),
        "created_by": row["created_by"],
        "created_at": row["created_at"],
        "expires_at": None if expires is None else int(expires),
        "last_used_at": row["last_used_at"],
    }


_DASHBOARD_USER_COLUMNS = (
    "id, username, password_hash, is_admin, created_at, auth_source, "
    "oidc_issuer, oidc_subject, email, display_name, last_login_at"
)


def _row_to_dashboard_user(row: Mapping[str, Any]) -> dict:
    return {
        "id": int(row["id"]),
        "username": row["username"],
        "password_hash": row["password_hash"],
        "is_admin": bool(row["is_admin"]),
        "created_at": row["created_at"],
        "auth_source": row["auth_source"] or "local",
        "oidc_issuer": row["oidc_issuer"],
        "oidc_subject": row["oidc_subject"],
        "email": row["email"],
        "display_name": row["display_name"],
        "last_login_at": row["last_login_at"],
    }


def _row_to_worker(row: Mapping[str, Any]) -> dict:
    pid = row.get("provider_id")
    return {
        "id": int(row["id"]),
        "name": row["name"],
        "token_prefix": row["token_prefix"],
        "provider_id": int(pid) if pid is not None else None,
        "enabled": bool(row["enabled"]),
        "created_at": row["created_at"],
        "last_connected_at": row["last_connected_at"],
        "last_remote_addr": row["last_remote_addr"],
    }


def _row_to_provider(row: Mapping[str, Any]) -> Provider:
    return Provider(
        id=row["id"],
        name=row["name"],
        protocol=ProviderProtocol(row["protocol"]),
        status=ProviderStatus(row["status"]),
        machine_type=row["machine_type"],
        gpu_type=row["gpu_type"],
        gpu_ram=row["gpu_ram"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        worker_name=row["worker_name"],
        api_key_enc=row["api_key_enc"],
        api_key_header=row["api_key_header"],
        capabilities=_json_or_none(row["capabilities"]),
        probed_at=row["probed_at"],
        flavor=row["flavor"],
        preset=row["preset"],
        extra_headers=_json_or_none(row["extra_headers"]),
        manual_models=_json_list_or_none(row["manual_models"]),
        locality=row["locality"] or "local",
        model_filter=row["model_filter"],
        read_timeout_s=row["read_timeout_s"],
        auto_benchmark=_tristate(row["auto_benchmark"]),
        hub_shared=row["hub_shared"] is None or bool(row["hub_shared"]),
        worker_parent_id=row["worker_parent_id"],
        worker_remote_id=row["worker_remote_id"],
    )


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


def _json_or_none(raw: Any) -> dict | None:
    if not raw:
        return None
    try:
        val = json.loads(raw)
    except TypeError, ValueError:
        return None
    return val if isinstance(val, dict) else None


def _json_list_or_none(raw: Any) -> list | None:
    if not raw:
        return None
    try:
        val = json.loads(raw)
    except TypeError, ValueError:
        return None
    return val if isinstance(val, list) else None


def _row_to_model(row: Mapping[str, Any]) -> ProviderModel:
    return ProviderModel(
        id=row["id"],
        provider_id=row["provider_id"],
        name=row["name"],
        raw_name=row.get("raw_name"),
        size=row["size"],
        digest=row["digest"],
        modified_at=row["modified_at"],
        details=json.loads(row["details"]) if row["details"] else None,
    )


def _row_to_member(row: Mapping[str, Any]) -> ModelGroupMember:
    pid = row["provider_id"]
    return ModelGroupMember(
        id=int(row["id"]),
        group_id=int(row["group_id"]),
        model_name=row["model_name"],
        provider_id=None if pid is None else int(pid),
        position=int(row["position"]),
    )


def _row_to_benchmark(row: Mapping[str, Any]) -> BenchmarkResult:
    return BenchmarkResult(
        id=row["id"],
        provider_id=row["provider_id"],
        model_name=row["model_name"],
        protocol=row["protocol"],
        startup_time_ms=row["startup_time_ms"],
        tokens_per_second=row["tokens_per_second"],
        source=row.get("source"),
        created_at=row["created_at"],
    )


def _row_to_address(row: Mapping[str, Any]) -> ProviderAddress:
    return ProviderAddress(
        id=row["id"],
        provider_id=row["provider_id"],
        url=row["url"],
        is_preferred=bool(row["is_preferred"]),
        is_live=bool(row["is_live"]),
        created_at=row["created_at"],
    )


def _row_to_calibration(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        offsets = json.loads(row["offsets"] or "{}")
    except TypeError, ValueError:
        offsets = {}
    return {
        "id": row["id"],
        "model_name": row["model_name"],
        "question_key": row["question_key"],
        "question_type": row["question_type"],
        "question": _json_or_none(row["question"]),
        "temperature": float(row["temperature"]),
        "offsets": {str(k): float(v) for k, v in offsets.items()},
        "examples": row["examples"],
        "created_at": _iso(row["created_at"]),
        "updated_at": _iso(row["updated_at"]),
    }


def _row_to_chat_conversation(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "provider_id": row["provider_id"],
        "model_name": row["model_name"],
        "created_at": _iso(row["created_at"]),
        "updated_at": _iso(row["updated_at"]),
    }


def _row_to_chat_message(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "role": row["role"],
        "content": row["content"],
        "image": row["image"],
        "model_name": row["model_name"],
        "provider_name": row["provider_name"],
        "duration_ms": row["duration_ms"],
        "created_at": _iso(row["created_at"]),
    }


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _row_to_request_log(row: Mapping[str, Any]) -> RequestLog:
    return RequestLog(
        id=row["id"],
        provider_id=row["provider_id"],
        provider_name=row["provider_name"],
        protocol=row["protocol"],
        endpoint=row["endpoint"],
        source_ip=row["source_ip"],
        model=row["model"],
        routed_model=row.get("routed_model"),
        request_size=row["request_size"],
        response_size=row["response_size"],
        request_meta=row.get("request_meta"),
        duration_ms=row["duration_ms"],
        ttfb_ms=row.get("ttfb_ms"),
        router_ms=row.get("router_ms"),
        status=row["status"],
        streamed=bool(row["streamed"]),
        error_detail=row["error_detail"],
        locality=row.get("locality"),
        api_key_id=row.get("api_key_id"),
        prompt_tokens=row.get("prompt_tokens"),
        completion_tokens=row.get("completion_tokens"),
        cost_usd=row.get("cost_usd"),
        created_at=row["created_at"],
    )
