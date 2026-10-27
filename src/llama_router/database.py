from __future__ import annotations

import asyncio
import functools
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import text
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
    Provider,
    ProviderAddress,
    ProviderModel,
    ProviderStatus,
    ProviderType,
    RequestLog,
)

logger = logging.getLogger(__name__)

# Reasons passed to change listeners. The router rebuilds its table on the
# routing ones; the auth cache drops entries on the auth ones.
CHANGE_PROVIDERS = "providers"
CHANGE_STATUS = "status"
CHANGE_ADDRESSES = "addresses"
CHANGE_MODELS = "models"
CHANGE_BENCHMARKS = "benchmarks"
CHANGE_FALLBACKS = "fallbacks"
CHANGE_CAPABILITIES = "capabilities"
CHANGE_API_KEYS = "api_keys"
CHANGE_AUTH_SETTINGS = "auth_settings"
ROUTING_CHANGES = frozenset(
    {
        CHANGE_PROVIDERS,
        CHANGE_STATUS,
        CHANGE_ADDRESSES,
        CHANGE_MODELS,
        CHANGE_BENCHMARKS,
        CHANGE_FALLBACKS,
        CHANGE_CAPABILITIES,
    }
)
AUTH_CHANGES = frozenset({CHANGE_API_KEYS, CHANGE_AUTH_SETTINGS})


def _notifies(reason: str):
    """Tell change listeners about *reason* after the wrapped write succeeds."""

    def wrap(fn):
        @functools.wraps(fn)
        async def inner(self: "Database", *args: Any, **kwargs: Any):
            result = await fn(self, *args, **kwargs)
            self._changed(reason)
            return result

        return inner

    return wrap


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
        try:
            self._listeners.remove(fn)
        except ValueError:
            pass

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
            async with self._engine.begin() as conn:
                await conn.execute(text("PRAGMA foreign_keys = ON"))

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
        self._engine = None
        self._session_factory = None

    def _require_session_factory(self) -> async_sessionmaker[AsyncSession]:
        assert self._session_factory is not None, "Database not connected"
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
        async with sf() as session:
            async with session.begin():
                await session.execute(stmt, bind)

    async def _execute_rowcount(self, sql: str, *args: Any) -> int:
        stmt, bind = qmark(sql, *_bind_params(*args))
        sf = self._require_session_factory()
        async with sf() as session:
            async with session.begin():
                r = await session.execute(stmt, bind)
                return cursor_rowcount(r)

    async def _insert_get_id(self, insert_sql: str, *args: Any) -> int:
        """Run INSERT and return primary key ``id`` (portable across SQLite / PG / MySQL)."""
        sf = self._require_session_factory()
        async with sf() as session:
            async with session.begin():
                if self._dialect_name in ("mysql", "mariadb"):
                    st, bd = qmark(insert_sql, *_bind_params(*args))
                    await session.execute(st, bd)
                    r = await session.execute(text("SELECT LAST_INSERT_ID() AS id"))
                    row = r.mappings().first()
                    assert row is not None
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
        provider_type: ProviderType = ProviderType.OLLAMA,
        *,
        machine_type: str | None = None,
        gpu_type: str | None = None,
        gpu_ram: str | None = None,
        worker_name: str | None = None,
    ) -> Provider:
        """Create a provider. Addresses are added separately with add_address."""
        pid = await self._insert_get_id(
            "INSERT INTO providers (name, provider_type, machine_type, gpu_type, "
            "gpu_ram, worker_name) VALUES (?, ?, ?, ?, ?, ?)",
            (
                name,
                provider_type.value,
                machine_type,
                gpu_type,
                gpu_ram,
                worker_name,
            ),
        )
        return Provider(
            id=pid,
            name=name,
            provider_type=provider_type,
            machine_type=machine_type,
            gpu_type=gpu_type,
            gpu_ram=gpu_ram,
            worker_name=worker_name,
        )

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
        provider_type: ProviderType | None = None,
        *,
        machine_type: str | None = None,
        gpu_type: str | None = None,
        gpu_ram: str | None = None,
    ) -> None:
        if provider_type is not None:
            await self._execute(
                "UPDATE providers SET name = ?, provider_type = ?, machine_type = ?, "
                "gpu_type = ?, gpu_ram = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ?",
                (
                    name,
                    provider_type.value,
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
        """Store the last probe result (or clear it with None)."""
        await self._execute(
            "UPDATE providers SET capabilities = ?, probed_at = "
            + ("CURRENT_TIMESTAMP" if capabilities is not None else "NULL")
            + " WHERE id = ?",
            (
                json.dumps(capabilities) if capabilities is not None else None,
                provider_id,
            ),
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
        if protocol == "ollama":
            query = (
                "SELECT p.* FROM providers p "
                "JOIN provider_models pm ON p.id = pm.provider_id "
                "WHERE (pm.name = ? OR pm.raw_name = ?) AND p.status != 'offline' "
                "AND p.provider_type IN ('ollama', 'both')"
            )
        elif protocol == "llamacpp":
            query = (
                "SELECT p.* FROM providers p "
                "JOIN provider_models pm ON p.id = pm.provider_id "
                "WHERE (pm.name = ? OR pm.raw_name = ?) AND p.status != 'offline' "
                "AND p.provider_type IN ('llamacpp', 'both')"
            )
        else:
            query = (
                "SELECT p.* FROM providers p "
                "JOIN provider_models pm ON p.id = pm.provider_id "
                "WHERE (pm.name = ? OR pm.raw_name = ?) AND p.status != 'offline'"
            )
        rows = await self._select_all(query, (model_name, model_name))
        return [_row_to_provider(r) for r in rows]

    # --- Addresses ---

    @_notifies(CHANGE_ADDRESSES)
    async def add_address(
        self,
        provider_id: int,
        url: str,
        llamacpp_url: str | None = None,
        is_preferred: bool = False,
    ) -> ProviderAddress:
        url = url.rstrip("/")
        if llamacpp_url:
            llamacpp_url = llamacpp_url.rstrip("/")
        aid = await self._insert_get_id(
            "INSERT INTO provider_addresses (provider_id, url, llamacpp_url, is_preferred) "
            "VALUES (?, ?, ?, ?)",
            (provider_id, url, llamacpp_url, int(is_preferred)),
        )
        return ProviderAddress(
            id=aid,
            provider_id=provider_id,
            url=url,
            llamacpp_url=llamacpp_url,
            is_preferred=is_preferred,
        )

    @_notifies(CHANGE_ADDRESSES)
    async def update_address(
        self,
        address_id: int,
        url: str,
        llamacpp_url: str | None = None,
        is_preferred: bool | None = None,
    ) -> None:
        url = url.rstrip("/")
        if llamacpp_url:
            llamacpp_url = llamacpp_url.rstrip("/")
        if is_preferred is not None:
            await self._execute(
                "UPDATE provider_addresses SET url = ?, llamacpp_url = ?, "
                "is_preferred = ? WHERE id = ?",
                (url, llamacpp_url, int(is_preferred), address_id),
            )
        else:
            await self._execute(
                "UPDATE provider_addresses SET url = ?, llamacpp_url = ? WHERE id = ?",
                (url, llamacpp_url, address_id),
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
        async with sf() as session:
            async with session.begin():
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

    async def list_all_models(self) -> list[dict]:
        q = text("""
            SELECT name, size, digest, modified_at, details FROM (
                SELECT pm.name, pm.size, pm.digest, pm.modified_at, pm.details,
                    ROW_NUMBER() OVER (
                        PARTITION BY pm.name ORDER BY pm.id
                    ) AS rn
                FROM provider_models pm
                INNER JOIN providers p ON p.id = pm.provider_id
                WHERE p.status != 'offline'
            ) ranked
            WHERE rn = 1
            ORDER BY name
            """)
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
                "AND protocol = ? ORDER BY created_at DESC LIMIT 1"
            )
            params: tuple[Any, ...] = (provider_id, model_name, protocol)
        else:
            query = (
                "SELECT * FROM benchmarks WHERE provider_id = ? AND model_name = ? "
                "ORDER BY created_at DESC LIMIT 1"
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
            "SELECT * FROM benchmarks WHERE provider_id = ? ORDER BY created_at DESC",
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
            "router_ms, status, streamed, error_detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            ),
        )

    async def get_request_logs(
        self, limit: int = 200, offset: int = 0
    ) -> list[RequestLog]:
        rows = await self._select_all(
            "SELECT * FROM request_log ORDER BY created_at DESC LIMIT ? OFFSET ?",
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
                "INSERT INTO model_fallbacks (model_name, fallback_model) VALUES (?, ?) "
                "ON DUPLICATE KEY UPDATE fallback_model = VALUES(fallback_model)",
                (model_name, fallback_model),
            )
        else:
            await self._execute(
                "INSERT INTO model_fallbacks (model_name, fallback_model) VALUES (?, ?) "
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
        self, key_prefix: str, key_hash: str, routing_mode: str, allow_fallback: bool
    ) -> int:
        return await self._insert_get_id(
            "INSERT INTO api_keys (key_prefix, key_hash, routing_mode, allow_fallback) "
            "VALUES (?, ?, ?, ?)",
            (key_prefix, key_hash, routing_mode, int(allow_fallback)),
        )

    async def list_api_keys(self) -> list[dict]:
        rows = await self._select_all(
            "SELECT "
            "k.id, k.key_prefix, k.routing_mode, k.allow_fallback, k.created_at, k.last_used_at, "
            "COUNT(p.id) AS pin_count "
            "FROM api_keys k "
            "LEFT JOIN api_key_model_pins p ON p.api_key_id = k.id "
            "GROUP BY k.id, k.key_prefix, k.routing_mode, k.allow_fallback, k.created_at, k.last_used_at "
            "ORDER BY k.created_at DESC"
        )
        return [
            {
                "id": int(r["id"]),
                "key_prefix": r["key_prefix"],
                "routing_mode": r["routing_mode"],
                "allow_fallback": bool(r["allow_fallback"]),
                "created_at": r["created_at"],
                "last_used_at": r["last_used_at"],
                "pin_count": int(r["pin_count"] or 0),
            }
            for r in rows
        ]

    @_notifies(CHANGE_API_KEYS)
    async def delete_api_key(self, key_id: int) -> None:
        await self._execute("DELETE FROM api_keys WHERE id = ?", (key_id,))

    async def lookup_api_key(self, key_hash: str) -> dict | None:
        row = await self._select_one(
            "SELECT id, key_prefix, routing_mode, allow_fallback FROM api_keys "
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
            "SELECT value FROM app_settings WHERE key = 'allow_unauthenticated'"
        )
        if not row:
            return True
        return str(row["value"]).lower() in ("1", "true", "yes", "on")

    @_notifies(CHANGE_AUTH_SETTINGS)
    async def set_allow_unauthenticated(self, allow: bool) -> None:
        val = "true" if allow else "false"
        if self._dialect_name in ("mysql", "mariadb"):
            await self._execute(
                "INSERT INTO app_settings (`key`, value) VALUES ('allow_unauthenticated', ?) "
                "ON DUPLICATE KEY UPDATE value = VALUES(value)",
                (val,),
            )
        elif self._dialect_name == "postgresql":
            await self._execute(
                "INSERT INTO app_settings (\"key\", value) VALUES ('allow_unauthenticated', ?) "
                'ON CONFLICT ("key") DO UPDATE SET value = EXCLUDED.value',
                (val,),
            )
        else:
            await self._execute(
                "INSERT INTO app_settings (\"key\", value) VALUES ('allow_unauthenticated', ?) "
                'ON CONFLICT ("key") DO UPDATE SET value = excluded.value',
                (val,),
            )

    async def get_app_setting(self, key: str) -> str | None:
        row = await self._select_one(
            "SELECT value FROM app_settings WHERE key = ?", (key,)
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

    async def count_dashboard_users(self) -> int:
        row = await self._select_one("SELECT COUNT(*) AS c FROM dashboard_users")
        return int(row["c"]) if row else 0

    async def list_dashboard_users(self) -> list[dict]:
        if self._dialect_name == "sqlite":
            order_sql = "ORDER BY username COLLATE NOCASE ASC"
        else:
            order_sql = "ORDER BY LOWER(username) ASC"
        rows = await self._select_all(
            f"SELECT id, username, is_admin, created_at FROM dashboard_users {order_sql}"
        )
        return [
            {
                "id": int(r["id"]),
                "username": r["username"],
                "is_admin": bool(r["is_admin"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    async def get_dashboard_user_by_username(self, username: str) -> dict | None:
        u = username.strip()
        if self._dialect_name == "sqlite":
            row = await self._select_one(
                "SELECT id, username, password_hash, is_admin, created_at "
                "FROM dashboard_users WHERE username = ? COLLATE NOCASE",
                (u,),
            )
        else:
            row = await self._select_one(
                "SELECT id, username, password_hash, is_admin, created_at "
                "FROM dashboard_users WHERE LOWER(username) = LOWER(?)",
                (u,),
            )
        if not row:
            return None
        return {
            "id": int(row["id"]),
            "username": row["username"],
            "password_hash": row["password_hash"],
            "is_admin": bool(row["is_admin"]),
            "created_at": row["created_at"],
        }

    async def get_dashboard_user_by_id(self, user_id: int) -> dict | None:
        row = await self._select_one(
            "SELECT id, username, password_hash, is_admin, created_at "
            "FROM dashboard_users WHERE id = ?",
            (user_id,),
        )
        if not row:
            return None
        return {
            "id": int(row["id"]),
            "username": row["username"],
            "password_hash": row["password_hash"],
            "is_admin": bool(row["is_admin"]),
            "created_at": row["created_at"],
        }

    async def create_dashboard_user(
        self, username: str, password_hash: str, is_admin: bool
    ) -> int:
        return await self._insert_get_id(
            "INSERT INTO dashboard_users (username, password_hash, is_admin) "
            "VALUES (?, ?, ?)",
            (username.strip(), password_hash, int(is_admin)),
        )

    async def update_dashboard_user_password(
        self, user_id: int, password_hash: str
    ) -> None:
        await self._execute(
            "UPDATE dashboard_users SET password_hash = ? WHERE id = ?",
            (password_hash, user_id),
        )

    async def delete_dashboard_user(self, user_id: int) -> None:
        await self._execute("DELETE FROM dashboard_users WHERE id = ?", (user_id,))

    async def count_dashboard_admins(self) -> int:
        row = await self._select_one(
            "SELECT COUNT(*) AS c FROM dashboard_users WHERE is_admin = 1"
        )
        return int(row["c"]) if row else 0


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
        provider_type=ProviderType(row["provider_type"]),
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
    )


def _json_or_none(raw: Any) -> dict | None:
    if not raw:
        return None
    try:
        val = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return val if isinstance(val, dict) else None


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
        llamacpp_url=row["llamacpp_url"],
        is_preferred=bool(row["is_preferred"]),
        is_live=bool(row["is_live"]),
        created_at=row["created_at"],
    )


def _row_to_request_log(row: Mapping[str, Any]) -> RequestLog:
    return RequestLog(
        id=row["id"],
        provider_id=row["provider_id"],
        provider_name=row["provider_name"],
        protocol=row["protocol"],
        endpoint=row["endpoint"],
        source_ip=row["source_ip"],
        model=row["model"],
        request_size=row["request_size"],
        response_size=row["response_size"],
        request_meta=row.get("request_meta"),
        duration_ms=row["duration_ms"],
        ttfb_ms=row.get("ttfb_ms"),
        router_ms=row.get("router_ms"),
        status=row["status"],
        streamed=bool(row["streamed"]),
        error_detail=row["error_detail"],
        created_at=row["created_at"],
    )
