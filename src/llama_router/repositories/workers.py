"""Workers (hub side)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import Repository


class WorkerRepository(Repository):
    async def create_worker(
        self,
        name: str,
        token_hash: str,
        token_prefix: str,
        provider_id: int,
        token_expires_at: int | None = None,
    ) -> int:
        return await self._insert_get_id(
            "INSERT INTO workers "
            "(name, token_hash, token_prefix, provider_id, token_expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, token_hash, token_prefix, provider_id, token_expires_at),
        )

    async def list_workers(self) -> list[dict[str, Any]]:
        rows = await self._select_all("SELECT * FROM workers ORDER BY name")
        return [_row_to_worker(r) for r in rows]

    async def get_worker(self, worker_id: int) -> dict[str, Any] | None:
        row = await self._select_one("SELECT * FROM workers WHERE id = ?", (worker_id,))
        return _row_to_worker(row) if row else None

    async def get_worker_by_token_hash(self, token_hash: str) -> dict[str, Any] | None:
        row = await self._select_one(
            "SELECT * FROM workers WHERE token_hash = ?", (token_hash,)
        )
        return _row_to_worker(row) if row else None

    async def get_worker_by_provider(self, provider_id: int) -> dict[str, Any] | None:
        row = await self._select_one(
            "SELECT * FROM workers WHERE provider_id = ?", (provider_id,)
        )
        return _row_to_worker(row) if row else None

    async def set_worker_token(
        self,
        worker_id: int,
        token_hash: str,
        token_prefix: str,
        token_expires_at: int | None = None,
    ) -> None:
        await self._execute(
            "UPDATE workers SET token_hash = ?, token_prefix = ?, "
            "token_expires_at = ? WHERE id = ?",
            (token_hash, token_prefix, token_expires_at, worker_id),
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


def _row_to_worker(row: Mapping[str, Any]) -> dict[str, Any]:
    pid = row.get("provider_id")
    expires = row.get("token_expires_at")
    return {
        "id": int(row["id"]),
        "name": row["name"],
        "token_prefix": row["token_prefix"],
        "token_expires_at": None if expires is None else int(expires),
        "provider_id": int(pid) if pid is not None else None,
        "enabled": bool(row["enabled"]),
        "created_at": row["created_at"],
        "last_connected_at": row["last_connected_at"],
        "last_remote_addr": row["last_remote_addr"],
    }
