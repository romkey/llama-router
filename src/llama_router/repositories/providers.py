"""Providers, their addresses and the models they serve."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, cast

from sqlalchemy import text

from ..alerts import alerts
from ..db_events import (
    CHANGE_ADDRESSES,
    CHANGE_CAPABILITIES,
    CHANGE_MODELS,
    CHANGE_PROVIDERS,
    CHANGE_STATUS,
    _notifies,
)
from ..db_sql import qmark, result_mappings
from ..models import (
    Provider,
    ProviderAddress,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from .base import Repository, _json_list_or_none, _json_or_none, _tristate


class ProviderRepository(Repository):
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
        self, provider_id: int, capabilities: dict[str, Any] | None
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
        "max_concurrent": False,
    }

    @_notifies(CHANGE_PROVIDERS)
    async def update_provider_settings(
        self, /, provider_id: int, **fields: Any
    ) -> None:
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
        # Busy and idle only say whether a request is in flight: routing
        # treats both as online, so a flip between them must not reload it.
        online = (ProviderStatus.BUSY.value, ProviderStatus.IDLE.value)
        changed = await self._execute_rowcount(
            "UPDATE providers SET status = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE id = ? AND status != ? AND NOT (status IN (?, ?) AND ? IN (?, ?))",
            (status.value, provider_id, status.value, *online, status.value, *online),
        )
        if changed:
            # Only a real transition matters to routing; health checks write
            # the same status every cycle.
            self._changed(CHANGE_STATUS)
            if alerts.configured:
                row = await self._select_one(
                    "SELECT name FROM providers WHERE id = ?", (provider_id,)
                )
                alerts.provider_status(
                    provider_id,
                    str(row["name"]) if row else str(provider_id),
                    status not in (ProviderStatus.OFFLINE, ProviderStatus.UNKNOWN),
                )
        else:
            # Record a busy/idle flip quietly, in one statement, so a
            # concurrent change to offline is never overwritten.
            await self._execute(
                "UPDATE providers SET status = CASE WHEN status IN (?, ?) "
                "AND ? IN (?, ?) THEN ? ELSE status END, "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (*online, status.value, *online, status.value, provider_id),
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
            return cast(str, row["raw_name"] or row["name"])
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
    ) -> list[dict[str, Any]]:
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
        max_concurrent=row.get("max_concurrent") or None,
        worker_parent_id=row["worker_parent_id"],
        worker_remote_id=row["worker_remote_id"],
    )


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


def _row_to_address(row: Mapping[str, Any]) -> ProviderAddress:
    return ProviderAddress(
        id=row["id"],
        provider_id=row["provider_id"],
        url=row["url"],
        is_preferred=bool(row["is_preferred"]),
        is_live=bool(row["is_live"]),
        created_at=row["created_at"],
    )
