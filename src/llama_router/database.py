from __future__ import annotations

import json

import aiosqlite

from .config import settings
from .models import (
    BenchmarkResult,
    Provider,
    ProviderAddress,
    ProviderModel,
    ProviderStatus,
    ProviderType,
    RequestLog,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS providers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    url TEXT NOT NULL,
    llamacpp_url TEXT,
    provider_type TEXT NOT NULL DEFAULT 'ollama',
    status TEXT NOT NULL DEFAULT 'unknown',
    machine_type TEXT,
    gpu_type TEXT,
    gpu_ram TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS provider_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id INTEGER NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    size INTEGER,
    digest TEXT,
    modified_at TEXT,
    details TEXT,
    UNIQUE(provider_id, name)
);

CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id INTEGER NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    startup_time_ms REAL,
    tokens_per_second REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS request_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id INTEGER,
    provider_name TEXT,
    protocol TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    source_ip TEXT,
    model TEXT,
    request_size INTEGER NOT NULL DEFAULT 0,
    response_size INTEGER NOT NULL DEFAULT 0,
    duration_ms REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'ok',
    streamed INTEGER NOT NULL DEFAULT 0,
    error_detail TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_request_log_created ON request_log(created_at DESC);

CREATE TABLE IF NOT EXISTS model_fallbacks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL UNIQUE,
    fallback_model TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provider_addresses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id INTEGER NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    llamacpp_url TEXT,
    is_preferred INTEGER NOT NULL DEFAULT 0,
    is_live INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_MIGRATIONS = [
    (
        "add_llamacpp_columns",
        [
            "ALTER TABLE providers ADD COLUMN llamacpp_url TEXT",
            "ALTER TABLE providers ADD COLUMN provider_type TEXT NOT NULL DEFAULT 'ollama'",
        ],
    ),
    (
        "add_benchmark_protocol",
        [
            "ALTER TABLE benchmarks ADD COLUMN protocol TEXT",
        ],
    ),
    (
        "add_provider_hw_fields",
        [
            "ALTER TABLE providers ADD COLUMN machine_type TEXT",
            "ALTER TABLE providers ADD COLUMN gpu_type TEXT",
            "ALTER TABLE providers ADD COLUMN gpu_ram TEXT",
        ],
    ),
    (
        "add_model_raw_name",
        [
            "ALTER TABLE provider_models ADD COLUMN raw_name TEXT",
        ],
    ),
]


class Database:
    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or settings.database_path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.executescript(_SCHEMA)
        await self._run_migrations()
        await self._db.commit()

    async def _run_migrations(self) -> None:
        """Apply migrations idempotently by checking column existence."""
        table_columns: dict[str, set[str]] = {}

        async def _cols(table: str) -> set[str]:
            if table not in table_columns:
                async with self.db.execute(f"PRAGMA table_info({table})") as cur:
                    table_columns[table] = {r["name"] for r in await cur.fetchall()}
            return table_columns[table]

        for _name, statements in _MIGRATIONS:
            for stmt in statements:
                if "ADD COLUMN" in stmt:
                    table = stmt.split("ALTER TABLE ")[1].split()[0]
                    col = stmt.split("ADD COLUMN ")[1].split()[0]
                    if col in await _cols(table):
                        continue
                try:
                    await self.db.execute(stmt)
                except Exception:
                    pass

        await self._seed_addresses()

    async def _seed_addresses(self) -> None:
        """Migrate existing provider url/llamacpp_url into provider_addresses."""
        async with self.db.execute(
            "SELECT id, url, llamacpp_url FROM providers"
        ) as cursor:
            rows = await cursor.fetchall()
        for row in rows:
            async with self.db.execute(
                "SELECT COUNT(*) AS cnt FROM provider_addresses WHERE provider_id = ?",
                (row["id"],),
            ) as cursor:
                count_row = await cursor.fetchone()
            if count_row["cnt"] == 0 and row["url"]:
                await self.db.execute(
                    "INSERT INTO provider_addresses (provider_id, url, llamacpp_url, is_preferred) "
                    "VALUES (?, ?, ?, 1)",
                    (row["id"], row["url"], row["llamacpp_url"]),
                )
        await self.db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None, "Database not connected"
        return self._db

    # --- Providers ---

    async def add_provider(
        self,
        name: str,
        url: str,
        provider_type: ProviderType = ProviderType.OLLAMA,
        llamacpp_url: str | None = None,
        machine_type: str | None = None,
        gpu_type: str | None = None,
        gpu_ram: str | None = None,
    ) -> Provider:
        url = url.rstrip("/")
        if llamacpp_url:
            llamacpp_url = llamacpp_url.rstrip("/")
        cursor = await self.db.execute(
            "INSERT INTO providers (name, url, llamacpp_url, provider_type, machine_type, gpu_type, gpu_ram) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                name,
                url,
                llamacpp_url,
                provider_type.value,
                machine_type,
                gpu_type,
                gpu_ram,
            ),
        )
        await self.db.commit()
        return Provider(
            id=cursor.lastrowid,
            name=name,
            url=url,
            llamacpp_url=llamacpp_url,
            provider_type=provider_type,
            machine_type=machine_type,
            gpu_type=gpu_type,
            gpu_ram=gpu_ram,
        )

    async def remove_provider(self, provider_id: int) -> None:
        await self.db.execute("DELETE FROM providers WHERE id = ?", (provider_id,))
        await self.db.commit()

    async def get_provider(self, provider_id: int) -> Provider | None:
        async with self.db.execute(
            "SELECT * FROM providers WHERE id = ?", (provider_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return _row_to_provider(row) if row else None

    async def get_provider_by_name(self, name: str) -> Provider | None:
        async with self.db.execute(
            "SELECT * FROM providers WHERE name = ?", (name,)
        ) as cursor:
            row = await cursor.fetchone()
            return _row_to_provider(row) if row else None

    async def list_providers(self) -> list[Provider]:
        async with self.db.execute("SELECT * FROM providers ORDER BY name") as cursor:
            rows = await cursor.fetchall()
            return [_row_to_provider(r) for r in rows]

    async def update_provider(
        self,
        provider_id: int,
        name: str,
        url: str,
        provider_type: ProviderType | None = None,
        llamacpp_url: str | None = None,
        machine_type: str | None = None,
        gpu_type: str | None = None,
        gpu_ram: str | None = None,
    ) -> None:
        url = url.rstrip("/")
        if llamacpp_url:
            llamacpp_url = llamacpp_url.rstrip("/")
        if provider_type is not None:
            await self.db.execute(
                "UPDATE providers SET name = ?, url = ?, llamacpp_url = ?, provider_type = ?, "
                "machine_type = ?, gpu_type = ?, gpu_ram = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (
                    name,
                    url,
                    llamacpp_url,
                    provider_type.value,
                    machine_type,
                    gpu_type,
                    gpu_ram,
                    provider_id,
                ),
            )
        else:
            await self.db.execute(
                "UPDATE providers SET name = ?, url = ?, llamacpp_url = ?, "
                "machine_type = ?, gpu_type = ?, gpu_ram = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (name, url, llamacpp_url, machine_type, gpu_type, gpu_ram, provider_id),
            )
        await self.db.commit()

    async def update_provider_status(
        self, provider_id: int, status: ProviderStatus
    ) -> None:
        await self.db.execute(
            "UPDATE providers SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status.value, provider_id),
        )
        await self.db.commit()

    async def get_providers_for_model(
        self, model_name: str, protocol: str | None = None
    ) -> list[Provider]:
        """Find providers that have a model. Matches both clean name and raw_name."""
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
        async with self.db.execute(query, (model_name, model_name)) as cursor:
            rows = await cursor.fetchall()
            return [_row_to_provider(r) for r in rows]

    # --- Addresses ---

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
        cursor = await self.db.execute(
            "INSERT INTO provider_addresses (provider_id, url, llamacpp_url, is_preferred) "
            "VALUES (?, ?, ?, ?)",
            (provider_id, url, llamacpp_url, int(is_preferred)),
        )
        await self.db.commit()
        return ProviderAddress(
            id=cursor.lastrowid,
            provider_id=provider_id,
            url=url,
            llamacpp_url=llamacpp_url,
            is_preferred=is_preferred,
        )

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
            await self.db.execute(
                "UPDATE provider_addresses SET url = ?, llamacpp_url = ?, is_preferred = ? WHERE id = ?",
                (url, llamacpp_url, int(is_preferred), address_id),
            )
        else:
            await self.db.execute(
                "UPDATE provider_addresses SET url = ?, llamacpp_url = ? WHERE id = ?",
                (url, llamacpp_url, address_id),
            )
        await self.db.commit()

    async def remove_address(self, address_id: int) -> None:
        await self.db.execute(
            "DELETE FROM provider_addresses WHERE id = ?", (address_id,)
        )
        await self.db.commit()

    async def set_address_preferred(self, address_id: int, is_preferred: bool) -> None:
        await self.db.execute(
            "UPDATE provider_addresses SET is_preferred = ? WHERE id = ?",
            (int(is_preferred), address_id),
        )
        await self.db.commit()

    async def set_address_live(self, address_id: int, is_live: bool) -> None:
        await self.db.execute(
            "UPDATE provider_addresses SET is_live = ? WHERE id = ?",
            (int(is_live), address_id),
        )
        await self.db.commit()

    async def get_addresses(self, provider_id: int) -> list[ProviderAddress]:
        async with self.db.execute(
            "SELECT * FROM provider_addresses WHERE provider_id = ? "
            "ORDER BY is_preferred DESC, id ASC",
            (provider_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [_row_to_address(r) for r in rows]

    async def get_address(self, address_id: int) -> ProviderAddress | None:
        async with self.db.execute(
            "SELECT * FROM provider_addresses WHERE id = ?", (address_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return _row_to_address(row) if row else None

    # --- Models ---

    async def set_provider_models(
        self, provider_id: int, models: list[ProviderModel]
    ) -> None:
        await self.db.execute(
            "DELETE FROM provider_models WHERE provider_id = ?", (provider_id,)
        )
        for m in models:
            await self.db.execute(
                "INSERT INTO provider_models (provider_id, name, raw_name, size, digest, modified_at, details) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    provider_id,
                    m.name,
                    m.raw_name,
                    m.size,
                    m.digest,
                    m.modified_at,
                    json.dumps(m.details) if m.details else None,
                ),
            )
        await self.db.commit()

    async def get_backend_model_name(self, provider_id: int, model_name: str) -> str:
        """Return the raw backend name for a model on a specific provider.

        Matches on both the clean name and raw_name columns. Returns the
        raw_name if set, otherwise the clean name.
        """
        async with self.db.execute(
            "SELECT name, raw_name FROM provider_models "
            "WHERE provider_id = ? AND (name = ? OR raw_name = ?) "
            "LIMIT 1",
            (provider_id, model_name, model_name),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return row["raw_name"] or row["name"]
            return model_name

    async def get_provider_models(self, provider_id: int) -> list[ProviderModel]:
        async with self.db.execute(
            "SELECT * FROM provider_models WHERE provider_id = ?", (provider_id,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [_row_to_model(r) for r in rows]

    async def list_all_models(self) -> list[dict]:
        """Return deduplicated model list across all online providers."""
        async with self.db.execute(
            "SELECT pm.name, pm.size, pm.digest, pm.modified_at, pm.details "
            "FROM provider_models pm "
            "JOIN providers p ON p.id = pm.provider_id "
            "WHERE p.status != 'offline' "
            "GROUP BY pm.name "
            "ORDER BY pm.name"
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "name": r["name"],
                    "size": r["size"],
                    "digest": r["digest"],
                    "modified_at": r["modified_at"],
                    "details": json.loads(r["details"]) if r["details"] else {},
                }
                for r in rows
            ]

    # --- Benchmarks ---

    async def save_benchmark(self, result: BenchmarkResult) -> None:
        await self.db.execute(
            "INSERT INTO benchmarks (provider_id, model_name, protocol, startup_time_ms, tokens_per_second) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                result.provider_id,
                result.model_name,
                result.protocol,
                result.startup_time_ms,
                result.tokens_per_second,
            ),
        )
        await self.db.commit()

    async def get_latest_benchmark(
        self, provider_id: int, model_name: str
    ) -> BenchmarkResult | None:
        async with self.db.execute(
            "SELECT * FROM benchmarks WHERE provider_id = ? AND model_name = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (provider_id, model_name),
        ) as cursor:
            row = await cursor.fetchone()
            return _row_to_benchmark(row) if row else None

    async def get_benchmarks_for_provider(
        self, provider_id: int
    ) -> list[BenchmarkResult]:
        async with self.db.execute(
            "SELECT * FROM benchmarks WHERE provider_id = ? ORDER BY created_at DESC",
            (provider_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [_row_to_benchmark(r) for r in rows]

    async def get_all_benchmarks(self) -> list[dict]:
        """Return all benchmarks with provider names, ordered by model then slowest first."""
        async with self.db.execute(
            "SELECT b.*, p.name AS provider_name "
            "FROM benchmarks b "
            "JOIN providers p ON p.id = b.provider_id "
            "ORDER BY b.model_name ASC, b.tokens_per_second ASC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": r["id"],
                    "provider_id": r["provider_id"],
                    "provider_name": r["provider_name"],
                    "model_name": r["model_name"],
                    "protocol": r["protocol"],
                    "startup_time_ms": r["startup_time_ms"],
                    "tokens_per_second": r["tokens_per_second"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    async def delete_benchmark(self, benchmark_id: int) -> None:
        await self.db.execute("DELETE FROM benchmarks WHERE id = ?", (benchmark_id,))
        await self.db.commit()

    async def delete_benchmarks_for_model(self, model_name: str) -> int:
        """Delete all benchmarks for a model. Returns the number of rows deleted."""
        cursor = await self.db.execute(
            "DELETE FROM benchmarks WHERE model_name = ?", (model_name,)
        )
        await self.db.commit()
        return cursor.rowcount

    async def delete_all_benchmarks(self) -> int:
        """Delete all benchmark results. Returns the number of rows deleted."""
        cursor = await self.db.execute("DELETE FROM benchmarks")
        await self.db.commit()
        return cursor.rowcount

    # --- Request Log ---

    async def save_request_log(self, entry: RequestLog) -> None:
        await self.db.execute(
            "INSERT INTO request_log "
            "(provider_id, provider_name, protocol, endpoint, source_ip, model, "
            "request_size, response_size, duration_ms, status, streamed, error_detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.provider_id,
                entry.provider_name,
                entry.protocol,
                entry.endpoint,
                entry.source_ip,
                entry.model,
                entry.request_size,
                entry.response_size,
                entry.duration_ms,
                entry.status,
                int(entry.streamed),
                entry.error_detail,
            ),
        )
        await self.db.commit()

    async def get_request_logs(
        self, limit: int = 200, offset: int = 0
    ) -> list[RequestLog]:
        async with self.db.execute(
            "SELECT * FROM request_log ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cursor:
            rows = await cursor.fetchall()
            return [_row_to_request_log(r) for r in rows]

    async def get_model_request_counts(self) -> dict[str, int]:
        """Return {model_name: request_count} for all models in the log."""
        async with self.db.execute(
            "SELECT model, COUNT(*) AS cnt FROM request_log "
            "WHERE model IS NOT NULL GROUP BY model"
        ) as cursor:
            rows = await cursor.fetchall()
            return {r["model"]: r["cnt"] for r in rows}

    async def count_request_logs(self) -> int:
        async with self.db.execute("SELECT COUNT(*) AS cnt FROM request_log") as cursor:
            row = await cursor.fetchone()
            return row["cnt"] if row else 0

    # ── Model fallbacks ───────────────────────────────────────────────

    async def set_model_fallback(self, model_name: str, fallback_model: str) -> None:
        """Set or replace the fallback for a model."""
        await self.db.execute(
            "INSERT INTO model_fallbacks (model_name, fallback_model) "
            "VALUES (?, ?) ON CONFLICT(model_name) DO UPDATE SET fallback_model = ?",
            (model_name, fallback_model, fallback_model),
        )
        await self.db.commit()

    async def remove_model_fallback(self, model_name: str) -> None:
        await self.db.execute(
            "DELETE FROM model_fallbacks WHERE model_name = ?", (model_name,)
        )
        await self.db.commit()

    async def get_model_fallback(self, model_name: str) -> str | None:
        async with self.db.execute(
            "SELECT fallback_model FROM model_fallbacks WHERE model_name = ?",
            (model_name,),
        ) as cursor:
            row = await cursor.fetchone()
            return row["fallback_model"] if row else None

    async def get_all_model_fallbacks(self) -> dict[str, str]:
        """Return {model_name: fallback_model} for all configured fallbacks."""
        async with self.db.execute(
            "SELECT model_name, fallback_model FROM model_fallbacks ORDER BY model_name"
        ) as cursor:
            rows = await cursor.fetchall()
            return {r["model_name"]: r["fallback_model"] for r in rows}

    async def resolve_fallback_chain(
        self, model_name: str, max_depth: int = 10
    ) -> list[str]:
        """Walk the fallback chain starting from model_name.

        Returns the ordered list of models to try (including the original).
        Stops at max_depth or when there is no further fallback (or a cycle).
        """
        chain = [model_name]
        seen: set[str] = {model_name}
        current = model_name
        for _ in range(max_depth):
            fb = await self.get_model_fallback(current)
            if fb is None or fb in seen:
                break
            chain.append(fb)
            seen.add(fb)
            current = fb
        return chain


def _row_to_provider(row: aiosqlite.Row) -> Provider:
    return Provider(
        id=row["id"],
        name=row["name"],
        url=row["url"],
        llamacpp_url=row["llamacpp_url"],
        provider_type=ProviderType(row["provider_type"]),
        status=ProviderStatus(row["status"]),
        machine_type=row["machine_type"],
        gpu_type=row["gpu_type"],
        gpu_ram=row["gpu_ram"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_model(row: aiosqlite.Row) -> ProviderModel:
    return ProviderModel(
        id=row["id"],
        provider_id=row["provider_id"],
        name=row["name"],
        raw_name=row["raw_name"] if "raw_name" in row.keys() else None,
        size=row["size"],
        digest=row["digest"],
        modified_at=row["modified_at"],
        details=json.loads(row["details"]) if row["details"] else None,
    )


def _row_to_benchmark(row: aiosqlite.Row) -> BenchmarkResult:
    return BenchmarkResult(
        id=row["id"],
        provider_id=row["provider_id"],
        model_name=row["model_name"],
        protocol=row["protocol"],
        startup_time_ms=row["startup_time_ms"],
        tokens_per_second=row["tokens_per_second"],
        created_at=row["created_at"],
    )


def _row_to_address(row: aiosqlite.Row) -> ProviderAddress:
    return ProviderAddress(
        id=row["id"],
        provider_id=row["provider_id"],
        url=row["url"],
        llamacpp_url=row["llamacpp_url"],
        is_preferred=bool(row["is_preferred"]),
        is_live=bool(row["is_live"]),
        created_at=row["created_at"],
    )


def _row_to_request_log(row: aiosqlite.Row) -> RequestLog:
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
        duration_ms=row["duration_ms"],
        status=row["status"],
        streamed=bool(row["streamed"]),
        error_detail=row["error_detail"],
        created_at=row["created_at"],
    )
