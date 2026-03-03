"""Manages provider lifecycle: discovery, health checking, status tracking."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from .config import settings
from .database import Database
from .models import (
    BenchmarkResult,
    Provider,
    ProviderInfo,
    ProviderModel,
    ProviderStatus,
)
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class ProviderManager:
    def __init__(self, db: Database):
        self._db = db
        self._clients: dict[int, OllamaClient] = {}
        self._active_requests: dict[int, int] = defaultdict(int)
        self._health_task: asyncio.Task | None = None

    async def start(self) -> None:
        providers = await self._db.list_providers()
        for p in providers:
            assert p.id is not None
            self._clients[p.id] = OllamaClient(p.url)
        self._health_task = asyncio.create_task(self._health_check_loop())

    async def stop(self) -> None:
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        for client in self._clients.values():
            await client.close()

    async def add_provider(self, name: str, url: str) -> Provider:
        provider = await self._db.add_provider(name, url)
        assert provider.id is not None
        client = OllamaClient(url)
        self._clients[provider.id] = client

        try:
            await self._discover_provider(provider.id, client)
            await self._db.update_provider_status(provider.id, ProviderStatus.IDLE)
            provider.status = ProviderStatus.IDLE
        except Exception:
            logger.exception("Failed to discover provider %s", name)
            await self._db.update_provider_status(provider.id, ProviderStatus.OFFLINE)
            provider.status = ProviderStatus.OFFLINE

        return provider

    async def remove_provider(self, provider_id: int) -> None:
        if provider_id in self._clients:
            await self._clients[provider_id].close()
            del self._clients[provider_id]
        self._active_requests.pop(provider_id, None)
        await self._db.remove_provider(provider_id)

    async def get_provider_info(self, provider_id: int) -> ProviderInfo | None:
        provider = await self._db.get_provider(provider_id)
        if not provider:
            return None
        models = await self._db.get_provider_models(provider_id)
        benchmarks = await self._db.get_benchmarks_for_provider(provider_id)
        return ProviderInfo(
            provider=provider,
            models=models,
            benchmarks=benchmarks,
            active_requests=self._active_requests.get(provider_id, 0),
        )

    async def list_provider_infos(self) -> list[ProviderInfo]:
        providers = await self._db.list_providers()
        infos = []
        for p in providers:
            assert p.id is not None
            models = await self._db.get_provider_models(p.id)
            benchmarks = await self._db.get_benchmarks_for_provider(p.id)
            infos.append(
                ProviderInfo(
                    provider=p,
                    models=models,
                    benchmarks=benchmarks,
                    active_requests=self._active_requests.get(p.id, 0),
                )
            )
        return infos

    def get_client(self, provider_id: int) -> OllamaClient:
        return self._clients[provider_id]

    def acquire(self, provider_id: int) -> None:
        self._active_requests[provider_id] += 1

    def release(self, provider_id: int) -> None:
        self._active_requests[provider_id] = max(
            0, self._active_requests[provider_id] - 1
        )

    def active_requests(self, provider_id: int) -> int:
        return self._active_requests.get(provider_id, 0)

    async def refresh_provider(self, provider_id: int) -> None:
        client = self._clients.get(provider_id)
        if not client:
            return
        try:
            await self._discover_provider(provider_id, client)
            await self._db.update_provider_status(provider_id, ProviderStatus.IDLE)
        except Exception:
            logger.exception("Failed to refresh provider %d", provider_id)
            await self._db.update_provider_status(provider_id, ProviderStatus.OFFLINE)

    async def benchmark_provider(
        self, provider_id: int, model_name: str
    ) -> BenchmarkResult | None:
        client = self._clients.get(provider_id)
        if not client:
            return None
        try:
            metrics = await client.benchmark_chat(model_name, settings.benchmark_prompt)
            result = BenchmarkResult(
                provider_id=provider_id,
                model_name=model_name,
                startup_time_ms=metrics["startup_time_ms"],
                tokens_per_second=metrics["tokens_per_second"],
            )
            await self._db.save_benchmark(result)
            return result
        except Exception:
            logger.exception(
                "Benchmark failed for provider %d model %s", provider_id, model_name
            )
            return None

    async def _discover_provider(self, provider_id: int, client: OllamaClient) -> None:
        tags = await client.get_tags()
        models = [
            ProviderModel(
                provider_id=provider_id,
                name=m.name,
                size=m.size,
                digest=m.digest,
                modified_at=m.modified_at,
                details=m.details,
            )
            for m in tags
        ]
        await self._db.set_provider_models(provider_id, models)
        logger.info("Discovered %d models on provider %d", len(models), provider_id)

    async def _health_check_loop(self) -> None:
        while True:
            await asyncio.sleep(settings.health_check_interval_seconds)
            try:
                await self._run_health_checks()
            except Exception:
                logger.exception("Health check cycle failed")

    async def _run_health_checks(self) -> None:
        providers = await self._db.list_providers()
        for p in providers:
            assert p.id is not None
            client = self._clients.get(p.id)
            if not client:
                continue
            reachable = await client.is_reachable()
            if reachable:
                if p.status == ProviderStatus.OFFLINE:
                    logger.info("Provider %s is back online, re-discovering", p.name)
                    await self._discover_provider(p.id, client)
                if self._active_requests.get(p.id, 0) > 0:
                    await self._db.update_provider_status(p.id, ProviderStatus.BUSY)
                else:
                    await self._db.update_provider_status(p.id, ProviderStatus.IDLE)
            else:
                if p.status != ProviderStatus.OFFLINE:
                    logger.warning("Provider %s went offline", p.name)
                await self._db.update_provider_status(p.id, ProviderStatus.OFFLINE)
