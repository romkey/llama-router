from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from .config import settings
from .models import BenchmarkResult, Provider, ProviderModel, ProviderStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS providers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    url TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'unknown',
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
"""


class Database:
    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or settings.database_path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None, "Database not connected"
        return self._db

    # --- Providers ---

    async def add_provider(self, name: str, url: str) -> Provider:
        url = url.rstrip("/")
        cursor = await self.db.execute(
            "INSERT INTO providers (name, url) VALUES (?, ?)",
            (name, url),
        )
        await self.db.commit()
        return Provider(id=cursor.lastrowid, name=name, url=url)

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

    async def update_provider_status(
        self, provider_id: int, status: ProviderStatus
    ) -> None:
        await self.db.execute(
            "UPDATE providers SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status.value, provider_id),
        )
        await self.db.commit()

    # --- Models ---

    async def set_provider_models(
        self, provider_id: int, models: list[ProviderModel]
    ) -> None:
        await self.db.execute(
            "DELETE FROM provider_models WHERE provider_id = ?", (provider_id,)
        )
        for m in models:
            await self.db.execute(
                "INSERT INTO provider_models (provider_id, name, size, digest, modified_at, details) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    provider_id,
                    m.name,
                    m.size,
                    m.digest,
                    m.modified_at,
                    json.dumps(m.details) if m.details else None,
                ),
            )
        await self.db.commit()

    async def get_provider_models(self, provider_id: int) -> list[ProviderModel]:
        async with self.db.execute(
            "SELECT * FROM provider_models WHERE provider_id = ?", (provider_id,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [_row_to_model(r) for r in rows]

    async def get_providers_for_model(self, model_name: str) -> list[Provider]:
        async with self.db.execute(
            "SELECT p.* FROM providers p "
            "JOIN provider_models pm ON p.id = pm.provider_id "
            "WHERE pm.name = ? AND p.status != 'offline'",
            (model_name,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [_row_to_provider(r) for r in rows]

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
            "INSERT INTO benchmarks (provider_id, model_name, startup_time_ms, tokens_per_second) "
            "VALUES (?, ?, ?, ?)",
            (
                result.provider_id,
                result.model_name,
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


def _row_to_provider(row: aiosqlite.Row) -> Provider:
    return Provider(
        id=row["id"],
        name=row["name"],
        url=row["url"],
        status=ProviderStatus(row["status"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_model(row: aiosqlite.Row) -> ProviderModel:
    return ProviderModel(
        id=row["id"],
        provider_id=row["provider_id"],
        name=row["name"],
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
        startup_time_ms=row["startup_time_ms"],
        tokens_per_second=row["tokens_per_second"],
        created_at=row["created_at"],
    )
