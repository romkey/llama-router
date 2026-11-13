"""The audit trail (``audit_log``)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .base import Repository, _utc_datetime

_COLUMNS = (
    "id, at, actor_kind, actor_id, actor_name, action, target_kind, target_id, "
    "summary, client_ip"
)
# Filters list_audit_log accepts, and the column each compares.
_FILTERS = {
    "actor_kind": "actor_kind = ?",
    "actor_name": "actor_name = ?",
    "action": "action = ?",
    "action_prefix": "action LIKE ?",
    "target_kind": "target_kind = ?",
    "target_id": "target_id = ?",
}


class AuditRepository(Repository):
    async def add_audit_entry(
        self,
        *,
        actor_kind: str,
        actor_id: int | None,
        actor_name: str | None,
        action: str,
        target_kind: str | None,
        target_id: str | None,
        summary: str | None,
        client_ip: str | None,
    ) -> None:
        await self._execute(
            "INSERT INTO audit_log (actor_kind, actor_id, actor_name, action, "
            "target_kind, target_id, summary, client_ip) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                actor_kind,
                actor_id,
                (actor_name or "")[:200] or None,
                action[:100],
                target_kind,
                None if target_id is None else str(target_id)[:200],
                summary,
                (client_ip or "")[:64] or None,
            ),
        )

    async def list_audit_log(
        self,
        *,
        limit: int = 200,
        before_id: int | None = None,
        since: datetime | None = None,
        target_ids: list[str] | None = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Entries newest first. *filters* are ``_FILTERS`` names; None or
        blank ones are ignored. *target_ids* limits to those targets."""
        where: list[str] = []
        params: list[Any] = []
        for name, value in filters.items():
            if name not in _FILTERS:
                raise ValueError(f"unknown audit filter {name!r}")
            if value in (None, ""):
                continue
            where.append(_FILTERS[name])
            params.append(f"{value}%" if name == "action_prefix" else value)
        if before_id is not None:
            where.append("id < ?")
            params.append(before_id)
        if since is not None:
            where.append("at >= ?")
            params.append(self._bind_time(since))
        if target_ids is not None:
            if not target_ids:
                return []
            where.append("target_id IN (" + ", ".join("?" * len(target_ids)) + ")")
            params.extend(target_ids)
        sql = f"SELECT {_COLUMNS} FROM audit_log"  # noqa: S608 - fixed columns; filters are bound
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, min(limit, 1000)))
        rows = await self._select_all(sql, tuple(params))
        return [
            {
                "id": int(r["id"]),
                "at": _utc_datetime(r["at"]),
                "actor_kind": r["actor_kind"],
                "actor_id": None if r["actor_id"] is None else int(r["actor_id"]),
                "actor_name": r["actor_name"],
                "action": r["action"],
                "target_kind": r["target_kind"],
                "target_id": r["target_id"],
                "summary": r["summary"],
                "client_ip": r["client_ip"],
            }
            for r in rows
        ]

    async def delete_audit_entries_before(self, before: datetime) -> int:
        return await self._execute_rowcount(
            "DELETE FROM audit_log WHERE at < ?", (self._bind_time(before),)
        )
