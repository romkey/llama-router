"""Management keys."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .base import Repository


class ManagementKeyRepository(Repository):
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

    async def list_management_keys(self) -> list[dict[str, Any]]:
        rows = await self._select_all(
            f"SELECT {_MANAGEMENT_KEY_COLUMNS} FROM management_keys "  # noqa: S608 - fixed column list
            "ORDER BY created_at DESC, id DESC"
        )
        return [_row_to_management_key(r) for r in rows]

    async def lookup_management_key(self, key_hash: str) -> dict[str, Any] | None:
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


_MANAGEMENT_KEY_COLUMNS = (
    "id, name, key_prefix, privileges, created_by, created_at, expires_at, last_used_at"
)


def _row_to_management_key(row: Mapping[str, Any]) -> dict[str, Any]:
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
