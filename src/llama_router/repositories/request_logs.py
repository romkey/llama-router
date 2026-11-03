"""The request log, spend and usage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

from ..config import settings
from ..models import RequestLog
from .base import Repository, _utc_datetime


class RequestLogRepository(Repository):
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

    async def save_request_logs(self, entries: Sequence[RequestLog]) -> None:
        """Insert *entries* in one transaction (one executemany)."""
        if not entries:
            return
        rows = [
            {
                "provider_id": e.provider_id,
                "provider_name": e.provider_name,
                "protocol": e.protocol,
                "endpoint": e.endpoint,
                "source_ip": e.source_ip,
                "model": e.model,
                "request_size": e.request_size,
                "response_size": e.response_size,
                "request_meta": e.request_meta,
                "duration_ms": e.duration_ms,
                "ttfb_ms": e.ttfb_ms,
                "router_ms": e.router_ms,
                "status": e.status,
                "streamed": int(e.streamed),
                "error_detail": e.error_detail,
                "locality": e.locality,
                "api_key_id": e.api_key_id,
                "prompt_tokens": e.prompt_tokens,
                "completion_tokens": e.completion_tokens,
                "cost_usd": e.cost_usd,
                "routed_model": e.routed_model,
            }
            for e in entries
        ]
        stmt = text(
            "INSERT INTO request_log "
            "(provider_id, provider_name, protocol, endpoint, source_ip, model, "
            "request_size, response_size, request_meta, duration_ms, ttfb_ms, "
            "router_ms, status, streamed, error_detail, locality, api_key_id, "
            "prompt_tokens, completion_tokens, cost_usd, routed_model) "
            "VALUES (:provider_id, :provider_name, :protocol, :endpoint, "
            ":source_ip, :model, :request_size, :response_size, :request_meta, "
            ":duration_ms, :ttfb_ms, :router_ms, :status, :streamed, "
            ":error_detail, :locality, :api_key_id, :prompt_tokens, "
            ":completion_tokens, :cost_usd, :routed_model)"
        )
        sf = self._require_session_factory()
        async with sf() as session, session.begin():
            await session.execute(stmt, rows)

    def _log_time(self, when: datetime) -> Any:
        """*when* bound for a comparison with ``request_log.created_at``: UTC
        without a zone (asyncpg refuses an aware one), as text on SQLite."""
        if when.tzinfo is not None:
            when = when.astimezone(UTC).replace(tzinfo=None)
        if self._dialect_name == "sqlite":
            return when.strftime("%Y-%m-%d %H:%M:%S")
        return when

    def _retained_since(self, since: datetime | None) -> datetime | None:
        """The later of *since* and the start of the retention window (UTC,
        naive): bounding an aggregate by ``created_at`` lets it read only the
        rows still kept, through the index, even before a purge has run."""
        days = settings.request_log_retention_days
        if days <= 0:
            return since
        floor = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
        if since is None:
            return floor
        if since.tzinfo is not None:
            since = since.astimezone(UTC).replace(tzinfo=None)
        return max(since, floor)

    async def delete_request_logs_before(self, before: datetime) -> int:
        """Delete request log rows created before *before* (UTC); the count."""
        return await self._execute_rowcount(
            "DELETE FROM request_log WHERE created_at < ?", (self._log_time(before),)
        )

    async def spend_since(self, key_id: int, since: datetime) -> float:
        """USD a key has spent on requests logged since *since* (UTC)."""
        when = self._log_time(since)
        row = await self._select_one(
            "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM request_log "
            "WHERE api_key_id = ? AND created_at >= ?",
            (key_id, when),
        )
        return float(row["total"] or 0) if row else 0.0

    async def spend_by_key_since(self, since: datetime) -> dict[int, float]:
        when = self._log_time(since)
        rows = await self._select_all(
            "SELECT api_key_id, COALESCE(SUM(cost_usd), 0) AS total FROM request_log "
            "WHERE api_key_id IS NOT NULL AND created_at >= ? GROUP BY api_key_id",
            (when,),
        )
        return {int(r["api_key_id"]): float(r["total"] or 0) for r in rows}

    async def usage_by_key_and_provider(
        self, since: datetime | None = None, key_id: int | None = None
    ) -> list[dict[str, Any]]:
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
        since = self._retained_since(since)
        if since is not None:
            where.append("r.created_at >= ?")
            params.append(self._log_time(since))
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

    async def get_request_logs(
        self, limit: int = 200, offset: int = 0, *, owner_user_id: int | None = None
    ) -> list[RequestLog]:
        """Return all logs, or only requests made with this user's existing keys."""
        # id breaks ties: MySQL TIMESTAMP has whole-second precision.
        if owner_user_id is None:
            rows = await self._select_all(
                "SELECT * FROM request_log ORDER BY created_at DESC, id DESC "
                "LIMIT ? OFFSET ?",
                (limit, offset),
            )
        else:
            rows = await self._select_all(
                "SELECT * FROM request_log WHERE api_key_id IN "
                "(SELECT id FROM api_keys WHERE owner_user_id = ?) "
                "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (owner_user_id, limit, offset),
            )
        return [_row_to_request_log(r) for r in rows]

    async def get_model_request_counts(self) -> dict[str, int]:
        since = self._retained_since(None)
        if since is None:
            rows = await self._select_all(
                "SELECT model, COUNT(*) AS cnt FROM request_log "
                "WHERE model IS NOT NULL GROUP BY model"
            )
        else:
            rows = await self._select_all(
                "SELECT model, COUNT(*) AS cnt FROM request_log "
                "WHERE model IS NOT NULL AND created_at >= ? GROUP BY model",
                (self._log_time(since),),
            )
        return {r["model"]: r["cnt"] for r in rows}

    async def count_request_logs(self, *, owner_user_id: int | None = None) -> int:
        """Count the same user-scoped rows as get_request_logs."""
        if owner_user_id is None:
            row = await self._select_one("SELECT COUNT(*) AS cnt FROM request_log")
        else:
            row = await self._select_one(
                "SELECT COUNT(*) AS cnt FROM request_log WHERE api_key_id IN "
                "(SELECT id FROM api_keys WHERE owner_user_id = ?)",
                (owner_user_id,),
            )
        return row["cnt"] if row else 0


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
