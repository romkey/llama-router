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
    request_meta TEXT,
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

CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    routing_mode TEXT NOT NULL DEFAULT 'latency',
    allow_fallback INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS api_key_model_pins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    api_key_id INTEGER NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    provider_id INTEGER NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(api_key_id, model_name)
);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
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
    (
        "add_api_keys_and_settings",
        [
            "CREATE TABLE IF NOT EXISTS api_keys ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "key_prefix TEXT NOT NULL, "
            "key_hash TEXT NOT NULL UNIQUE, "
            "routing_mode TEXT NOT NULL DEFAULT 'latency', "
            "allow_fallback INTEGER NOT NULL DEFAULT 1, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            "last_used_at TIMESTAMP"
            ")",
            "CREATE TABLE IF NOT EXISTS app_settings ("
            "key TEXT PRIMARY KEY, "
            "value TEXT NOT NULL"
            ")",
            "CREATE TABLE IF NOT EXISTS api_key_model_pins ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "api_key_id INTEGER NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE, "
            "model_name TEXT NOT NULL, "
            "provider_id INTEGER NOT NULL REFERENCES providers(id) ON DELETE CASCADE, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            "UNIQUE(api_key_id, model_name)"
            ")",
            "INSERT OR IGNORE INTO app_settings (key, value) VALUES ('allow_unauthenticated', 'true')",
        ],
    ),
    (
        "add_request_meta",
        [
            "ALTER TABLE request_log ADD COLUMN request_meta TEXT",
        ],
    ),
    (
        "add_wireguard_tables",
        [
            """CREATE TABLE IF NOT EXISTS wireguard_interface (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                enabled INTEGER NOT NULL DEFAULT 0,
                listen_port INTEGER NOT NULL DEFAULT 51820,
                private_key TEXT NOT NULL DEFAULT '',
                address_cidr TEXT NOT NULL DEFAULT '10.8.0.1/24',
                mtu INTEGER,
                endpoint_public TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            "INSERT OR IGNORE INTO wireguard_interface (id) VALUES (1)",
            """CREATE TABLE IF NOT EXISTS wireguard_peers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL DEFAULT '',
                public_key TEXT NOT NULL,
                preshared_key TEXT,
                allowed_ips TEXT NOT NULL,
                endpoint TEXT,
                persistent_keepalive INTEGER,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
        ],
    ),
    (
        "add_wireguard_peer_id_to_providers",
        [
            "ALTER TABLE providers ADD COLUMN wireguard_peer_id INTEGER REFERENCES wireguard_peers(id) ON DELETE SET NULL",
        ],
    ),
    (
        "add_wireguard_peering_fields",
        [
            "ALTER TABLE wireguard_interface ADD COLUMN peering_api_key TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE wireguard_interface ADD COLUMN peering_enabled INTEGER NOT NULL DEFAULT 0",
        ],
    ),
    (
        "add_dashboard_users",
        [
            """CREATE TABLE IF NOT EXISTS dashboard_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
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
        wireguard_peer_id: int | None = None,
    ) -> Provider:
        url = url.rstrip("/")
        if llamacpp_url:
            llamacpp_url = llamacpp_url.rstrip("/")
        cursor = await self.db.execute(
            "INSERT INTO providers (name, url, llamacpp_url, provider_type, machine_type, gpu_type, gpu_ram, wireguard_peer_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                name,
                url,
                llamacpp_url,
                provider_type.value,
                machine_type,
                gpu_type,
                gpu_ram,
                wireguard_peer_id,
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
            wireguard_peer_id=wireguard_peer_id,
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
        self, provider_id: int, model_name: str, protocol: str | None = None
    ) -> BenchmarkResult | None:
        if protocol:
            query = (
                "SELECT * FROM benchmarks WHERE provider_id = ? AND model_name = ? AND protocol = ? "
                "ORDER BY created_at DESC LIMIT 1"
            )
            params = (provider_id, model_name, protocol)
        else:
            query = (
                "SELECT * FROM benchmarks WHERE provider_id = ? AND model_name = ? "
                "ORDER BY created_at DESC LIMIT 1"
            )
            params = (provider_id, model_name)
        async with self.db.execute(query, params) as cursor:
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
            "request_size, response_size, request_meta, duration_ms, status, streamed, error_detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

    # --- WireGuard (dashboard-managed wg0.conf for Docker sidecar) ---

    async def get_wireguard_interface(self) -> dict:
        """Return singleton interface row as dict (includes derived public_key)."""
        from .wireguard_config import public_key_from_private

        async with self.db.execute(
            "SELECT * FROM wireguard_interface WHERE id = 1"
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            await self.db.execute(
                "INSERT OR IGNORE INTO wireguard_interface (id) VALUES (1)"
            )
            await self.db.commit()
            return await self.get_wireguard_interface()
        d = dict(row)
        d["peering_enabled"] = bool(d.get("peering_enabled"))
        d["peering_api_key"] = d.get("peering_api_key") or ""
        priv = (d.get("private_key") or "").strip()
        if priv:
            try:
                d["public_key"] = public_key_from_private(priv)
            except Exception:
                d["public_key"] = ""
        else:
            d["public_key"] = ""
        d["enabled"] = bool(d.get("enabled"))
        return d

    async def update_wireguard_interface(
        self,
        *,
        enabled: bool,
        listen_port: int,
        address_cidr: str,
        mtu: int | None = None,
        endpoint_public: str | None = None,
        new_private_key: str | None = None,
    ) -> None:
        """Update interface. If ``new_private_key`` is None, keep existing private key."""
        if new_private_key is not None:
            await self.db.execute(
                "UPDATE wireguard_interface SET enabled = ?, listen_port = ?, "
                "address_cidr = ?, mtu = ?, endpoint_public = ?, private_key = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE id = 1",
                (
                    int(enabled),
                    listen_port,
                    address_cidr.strip(),
                    mtu,
                    (endpoint_public or "").strip() or None,
                    new_private_key.strip(),
                ),
            )
        else:
            await self.db.execute(
                "UPDATE wireguard_interface SET enabled = ?, listen_port = ?, "
                "address_cidr = ?, mtu = ?, endpoint_public = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE id = 1",
                (
                    int(enabled),
                    listen_port,
                    address_cidr.strip(),
                    mtu,
                    (endpoint_public or "").strip() or None,
                ),
            )
        await self.db.commit()

    async def set_wireguard_private_key(self, private_key_b64: str) -> None:
        await self.db.execute(
            "UPDATE wireguard_interface SET private_key = ?, "
            "updated_at = CURRENT_TIMESTAMP WHERE id = 1",
            (private_key_b64.strip(),),
        )
        await self.db.commit()

    async def list_wireguard_peers(self) -> list[dict]:
        async with self.db.execute(
            "SELECT * FROM wireguard_peers ORDER BY id ASC"
        ) as cursor:
            rows = await cursor.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["enabled"] = bool(d.get("enabled"))
            out.append(d)
        return out

    async def get_wireguard_peer(self, peer_id: int) -> dict | None:
        async with self.db.execute(
            "SELECT * FROM wireguard_peers WHERE id = ?", (peer_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        d["enabled"] = bool(d.get("enabled"))
        return d

    async def add_wireguard_peer(
        self,
        *,
        name: str,
        public_key: str,
        allowed_ips: str,
        preshared_key: str | None = None,
        endpoint: str | None = None,
        persistent_keepalive: int | None = None,
        enabled: bool = True,
    ) -> int:
        cursor = await self.db.execute(
            "INSERT INTO wireguard_peers "
            "(name, public_key, preshared_key, allowed_ips, endpoint, "
            "persistent_keepalive, enabled) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                name.strip(),
                public_key.strip(),
                (preshared_key or "").strip() or None,
                allowed_ips.strip(),
                (endpoint or "").strip() or None,
                persistent_keepalive,
                int(enabled),
            ),
        )
        await self.db.commit()
        return int(cursor.lastrowid)

    async def update_wireguard_peer(
        self,
        peer_id: int,
        *,
        name: str,
        public_key: str,
        allowed_ips: str,
        preshared_key: str | None = None,
        endpoint: str | None = None,
        persistent_keepalive: int | None = None,
        enabled: bool = True,
    ) -> None:
        await self.db.execute(
            "UPDATE wireguard_peers SET name = ?, public_key = ?, preshared_key = ?, "
            "allowed_ips = ?, endpoint = ?, persistent_keepalive = ?, enabled = ? "
            "WHERE id = ?",
            (
                name.strip(),
                public_key.strip(),
                (preshared_key or "").strip() or None,
                allowed_ips.strip(),
                (endpoint or "").strip() or None,
                persistent_keepalive,
                int(enabled),
                peer_id,
            ),
        )
        await self.db.commit()

    async def remove_wireguard_peer(self, peer_id: int) -> None:
        await self.db.execute("DELETE FROM wireguard_peers WHERE id = ?", (peer_id,))
        await self.db.commit()

    async def set_provider_wireguard_peer(
        self, provider_id: int, peer_id: int | None
    ) -> None:
        await self.db.execute(
            "UPDATE providers SET wireguard_peer_id = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (peer_id, provider_id),
        )
        await self.db.commit()

    async def get_providers_by_peer_id(self, peer_id: int) -> list[Provider]:
        async with self.db.execute(
            "SELECT * FROM providers WHERE wireguard_peer_id = ? ORDER BY name",
            (peer_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_provider(r) for r in rows]

    async def get_wireguard_peering_config(self) -> dict:
        iface = await self.get_wireguard_interface()
        return {
            "peering_enabled": bool(iface.get("peering_enabled")),
            "peering_api_key": iface.get("peering_api_key") or "",
        }

    async def set_wireguard_peering_config(self, enabled: bool, api_key: str) -> None:
        await self.db.execute(
            "UPDATE wireguard_interface SET peering_enabled = ?, peering_api_key = ?, "
            "updated_at = CURRENT_TIMESTAMP WHERE id = 1",
            (int(enabled), api_key.strip()),
        )
        await self.db.commit()

    async def find_wireguard_peer_by_public_key(self, public_key: str) -> dict | None:
        key = public_key.strip()
        async with self.db.execute(
            "SELECT * FROM wireguard_peers WHERE public_key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        d["enabled"] = bool(d.get("enabled"))
        return d

    async def unlink_providers_from_wireguard_peer(self, peer_id: int) -> None:
        await self.db.execute(
            "UPDATE providers SET wireguard_peer_id = NULL, updated_at = CURRENT_TIMESTAMP "
            "WHERE wireguard_peer_id = ?",
            (peer_id,),
        )
        await self.db.commit()

    async def resolve_fallback_chain(
        self, model_name: str, max_depth: int = 10
    ) -> list[str]:
        """Walk the fallback chain starting from model_name.

        Returns the ordered list of models to try (including the original).
        Stops at max_depth or when there is no further fallback (or a cycle).
        """
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
        async with self.db.execute(query, (model_name, max_depth)) as cursor:
            rows = await cursor.fetchall()
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

    async def create_api_key(
        self, key_prefix: str, key_hash: str, routing_mode: str, allow_fallback: bool
    ) -> int:
        cursor = await self.db.execute(
            "INSERT INTO api_keys (key_prefix, key_hash, routing_mode, allow_fallback) VALUES (?, ?, ?, ?)",
            (key_prefix, key_hash, routing_mode, int(allow_fallback)),
        )
        await self.db.commit()
        return int(cursor.lastrowid)

    async def list_api_keys(self) -> list[dict]:
        async with self.db.execute(
            "SELECT "
            "k.id, k.key_prefix, k.routing_mode, k.allow_fallback, k.created_at, k.last_used_at, "
            "COUNT(p.id) AS pin_count "
            "FROM api_keys k "
            "LEFT JOIN api_key_model_pins p ON p.api_key_id = k.id "
            "GROUP BY k.id, k.key_prefix, k.routing_mode, k.allow_fallback, k.created_at, k.last_used_at "
            "ORDER BY k.created_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
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

    async def delete_api_key(self, key_id: int) -> None:
        await self.db.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
        await self.db.commit()

    async def lookup_api_key(self, key_hash: str) -> dict | None:
        async with self.db.execute(
            "SELECT id, key_prefix, routing_mode, allow_fallback FROM api_keys WHERE key_hash = ?",
            (key_hash,),
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        await self.db.execute(
            "UPDATE api_keys SET last_used_at = CURRENT_TIMESTAMP WHERE id = ?",
            (row["id"],),
        )
        await self.db.commit()
        return {
            "id": int(row["id"]),
            "key_prefix": row["key_prefix"],
            "routing_mode": row["routing_mode"],
            "allow_fallback": bool(row["allow_fallback"]),
        }

    async def set_api_key_model_pin(
        self, key_id: int, model_name: str, provider_id: int
    ) -> None:
        await self.db.execute(
            "INSERT INTO api_key_model_pins (api_key_id, model_name, provider_id) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(api_key_id, model_name) DO UPDATE SET provider_id = excluded.provider_id",
            (key_id, model_name, provider_id),
        )
        await self.db.commit()

    async def remove_api_key_model_pin(self, key_id: int, model_name: str) -> None:
        await self.db.execute(
            "DELETE FROM api_key_model_pins WHERE api_key_id = ? AND model_name = ?",
            (key_id, model_name),
        )
        await self.db.commit()

    async def list_api_key_model_pins(self, key_id: int) -> list[dict]:
        async with self.db.execute(
            "SELECT p.model_name, p.provider_id, pr.name AS provider_name "
            "FROM api_key_model_pins p "
            "JOIN providers pr ON pr.id = p.provider_id "
            "WHERE p.api_key_id = ? "
            "ORDER BY p.model_name ASC",
            (key_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "model_name": r["model_name"],
                    "provider_id": int(r["provider_id"]),
                    "provider_name": r["provider_name"],
                }
                for r in rows
            ]

    async def get_allow_unauthenticated(self) -> bool:
        async with self.db.execute(
            "SELECT value FROM app_settings WHERE key = 'allow_unauthenticated'"
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return True
        return str(row["value"]).lower() in ("1", "true", "yes", "on")

    async def set_allow_unauthenticated(self, allow: bool) -> None:
        await self.db.execute(
            "INSERT INTO app_settings (key, value) VALUES ('allow_unauthenticated', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("true" if allow else "false",),
        )
        await self.db.commit()

    async def get_app_setting(self, key: str) -> str | None:
        async with self.db.execute(
            "SELECT value FROM app_settings WHERE key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
        return str(row["value"]) if row else None

    async def set_app_setting(self, key: str, value: str) -> None:
        await self.db.execute(
            "INSERT INTO app_settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        await self.db.commit()

    async def count_dashboard_users(self) -> int:
        async with self.db.execute(
            "SELECT COUNT(*) AS c FROM dashboard_users"
        ) as cursor:
            row = await cursor.fetchone()
        return int(row["c"]) if row else 0

    async def list_dashboard_users(self) -> list[dict]:
        async with self.db.execute(
            "SELECT id, username, is_admin, created_at FROM dashboard_users "
            "ORDER BY username COLLATE NOCASE ASC"
        ) as cursor:
            rows = await cursor.fetchall()
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
        async with self.db.execute(
            "SELECT id, username, password_hash, is_admin, created_at "
            "FROM dashboard_users WHERE username = ? COLLATE NOCASE",
            (username.strip(),),
        ) as cursor:
            row = await cursor.fetchone()
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
        async with self.db.execute(
            "SELECT id, username, password_hash, is_admin, created_at "
            "FROM dashboard_users WHERE id = ?",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
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
        cursor = await self.db.execute(
            "INSERT INTO dashboard_users (username, password_hash, is_admin) "
            "VALUES (?, ?, ?)",
            (username.strip(), password_hash, int(is_admin)),
        )
        await self.db.commit()
        return int(cursor.lastrowid)

    async def update_dashboard_user_password(self, user_id: int, password_hash: str) -> None:
        await self.db.execute(
            "UPDATE dashboard_users SET password_hash = ? WHERE id = ?",
            (password_hash, user_id),
        )
        await self.db.commit()

    async def delete_dashboard_user(self, user_id: int) -> None:
        await self.db.execute("DELETE FROM dashboard_users WHERE id = ?", (user_id,))
        await self.db.commit()

    async def count_dashboard_admins(self) -> int:
        async with self.db.execute(
            "SELECT COUNT(*) AS c FROM dashboard_users WHERE is_admin = 1"
        ) as cursor:
            row = await cursor.fetchone()
        return int(row["c"]) if row else 0


def _row_to_provider(row: aiosqlite.Row) -> Provider:
    wgid = row["wireguard_peer_id"] if "wireguard_peer_id" in row.keys() else None
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
        wireguard_peer_id=int(wgid) if wgid is not None else None,
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
        request_meta=row["request_meta"] if "request_meta" in row.keys() else None,
        duration_ms=row["duration_ms"],
        status=row["status"],
        streamed=bool(row["streamed"]),
        error_detail=row["error_detail"],
        created_at=row["created_at"],
    )
