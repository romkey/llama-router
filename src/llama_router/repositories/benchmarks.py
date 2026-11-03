"""Benchmark results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..db_events import CHANGE_BENCHMARKS, _notifies
from ..models import BenchmarkResult
from .base import Repository


class BenchmarkRepository(Repository):
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

    async def get_all_benchmarks(self) -> list[dict[str, Any]]:
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
