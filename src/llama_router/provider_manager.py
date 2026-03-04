"""Manages provider lifecycle: discovery, health checking, status tracking."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from .config import settings
from .database import Database
from .llamacpp_client import LlamaCppClient
from .models import (
    BenchmarkResult,
    Provider,
    ProviderInfo,
    ProviderModel,
    ProviderStatus,
    ProviderType,
)
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class ProviderManager:
    def __init__(self, db: Database):
        self._db = db
        self._ollama_clients: dict[int, OllamaClient] = {}
        self._llamacpp_clients: dict[int, LlamaCppClient] = {}
        self._active_requests: dict[int, int] = defaultdict(int)
        self._health_task: asyncio.Task | None = None

    async def start(self) -> None:
        providers = await self._db.list_providers()
        for p in providers:
            assert p.id is not None
            self._create_clients(p)
        self._health_task = asyncio.create_task(self._health_check_loop())

    async def stop(self) -> None:
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        for client in self._ollama_clients.values():
            await client.close()
        for client in self._llamacpp_clients.values():
            await client.close()

    def _create_clients(self, provider: Provider) -> None:
        assert provider.id is not None
        if provider.supports_ollama:
            self._ollama_clients[provider.id] = OllamaClient(provider.url)
        if provider.supports_llamacpp:
            url = provider.llamacpp_url or provider.url
            self._llamacpp_clients[provider.id] = LlamaCppClient(url)

    async def _close_clients(self, provider_id: int) -> None:
        if provider_id in self._ollama_clients:
            await self._ollama_clients[provider_id].close()
            del self._ollama_clients[provider_id]
        if provider_id in self._llamacpp_clients:
            await self._llamacpp_clients[provider_id].close()
            del self._llamacpp_clients[provider_id]

    async def add_provider(
        self,
        name: str,
        url: str,
        provider_type: ProviderType = ProviderType.OLLAMA,
        llamacpp_url: str | None = None,
    ) -> Provider:
        provider = await self._db.add_provider(name, url, provider_type, llamacpp_url)
        assert provider.id is not None
        self._create_clients(provider)

        try:
            await self._discover_provider(provider)
            await self._db.update_provider_status(provider.id, ProviderStatus.IDLE)
            provider.status = ProviderStatus.IDLE
        except Exception:
            logger.exception("Failed to discover provider %s", name)
            await self._db.update_provider_status(provider.id, ProviderStatus.OFFLINE)
            provider.status = ProviderStatus.OFFLINE

        return provider

    async def update_provider(
        self,
        provider_id: int,
        name: str,
        url: str,
        provider_type: ProviderType | None = None,
        llamacpp_url: str | None = None,
    ) -> None:
        await self._db.update_provider(
            provider_id, name, url, provider_type, llamacpp_url
        )
        await self._close_clients(provider_id)
        provider = await self._db.get_provider(provider_id)
        if not provider:
            return
        self._create_clients(provider)
        try:
            await self._discover_provider(provider)
            await self._db.update_provider_status(provider_id, ProviderStatus.IDLE)
        except Exception:
            logger.exception("Failed to discover updated provider %d", provider_id)
            await self._db.update_provider_status(provider_id, ProviderStatus.OFFLINE)

    async def delete_remote_model(self, provider_id: int, model_name: str) -> None:
        ollama = self._ollama_clients.get(provider_id)
        if ollama:
            await ollama.delete_model(model_name)
        provider = await self._db.get_provider(provider_id)
        if provider:
            await self._discover_provider(provider)

    async def remove_provider(self, provider_id: int) -> None:
        await self._close_clients(provider_id)
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

    def get_ollama_client(self, provider_id: int) -> OllamaClient:
        return self._ollama_clients[provider_id]

    def get_llamacpp_client(self, provider_id: int) -> LlamaCppClient:
        return self._llamacpp_clients[provider_id]

    def get_client(self, provider_id: int) -> OllamaClient:
        """Backward compat: return Ollama client."""
        return self._ollama_clients[provider_id]

    def acquire(self, provider_id: int) -> None:
        self._active_requests[provider_id] += 1

    def release(self, provider_id: int) -> None:
        self._active_requests[provider_id] = max(
            0, self._active_requests[provider_id] - 1
        )

    def active_requests(self, provider_id: int) -> int:
        return self._active_requests.get(provider_id, 0)

    async def refresh_provider(self, provider_id: int) -> None:
        provider = await self._db.get_provider(provider_id)
        if not provider:
            return
        try:
            await self._discover_provider(provider)
            await self._db.update_provider_status(provider_id, ProviderStatus.IDLE)
        except Exception:
            logger.exception("Failed to refresh provider %d", provider_id)
            await self._db.update_provider_status(provider_id, ProviderStatus.OFFLINE)

    async def benchmark_provider(
        self, provider_id: int, model_name: str
    ) -> BenchmarkResult | None:
        provider = await self._db.get_provider(provider_id)
        if not provider:
            return None
        try:
            if provider.supports_ollama and provider_id in self._ollama_clients:
                metrics = await self._ollama_clients[provider_id].benchmark_chat(
                    model_name, settings.benchmark_prompt
                )
            elif provider.supports_llamacpp and provider_id in self._llamacpp_clients:
                metrics = await self._llamacpp_clients[provider_id].benchmark_chat(
                    model_name, settings.benchmark_prompt
                )
            else:
                return None
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

    async def _discover_provider(self, provider: Provider) -> None:
        assert provider.id is not None
        all_models: list[ProviderModel] = []
        seen_names: set[str] = set()

        if provider.supports_ollama and provider.id in self._ollama_clients:
            tags = await self._ollama_clients[provider.id].get_tags()
            for m in tags:
                if m.name not in seen_names:
                    seen_names.add(m.name)
                    all_models.append(
                        ProviderModel(
                            provider_id=provider.id,
                            name=m.name,
                            size=m.size,
                            digest=m.digest,
                            modified_at=m.modified_at,
                            details=m.details,
                        )
                    )

        if provider.supports_llamacpp and provider.id in self._llamacpp_clients:
            lcpp_models = await self._llamacpp_clients[provider.id].get_models()
            for m in lcpp_models:
                if m.name not in seen_names:
                    seen_names.add(m.name)
                    all_models.append(
                        ProviderModel(
                            provider_id=provider.id,
                            name=m.name,
                            size=m.size,
                            details=m.details,
                        )
                    )

        await self._db.set_provider_models(provider.id, all_models)
        logger.info(
            "Discovered %d models on provider %s", len(all_models), provider.name
        )

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
            reachable = False

            ollama = self._ollama_clients.get(p.id)
            lcpp = self._llamacpp_clients.get(p.id)

            if ollama and await ollama.is_reachable():
                reachable = True
            if lcpp and await lcpp.is_reachable():
                reachable = True

            if reachable:
                if p.status == ProviderStatus.OFFLINE:
                    logger.info("Provider %s is back online, re-discovering", p.name)
                    await self._discover_provider(p)
                if self._active_requests.get(p.id, 0) > 0:
                    await self._db.update_provider_status(p.id, ProviderStatus.BUSY)
                else:
                    await self._db.update_provider_status(p.id, ProviderStatus.IDLE)
            else:
                if p.status != ProviderStatus.OFFLINE:
                    logger.warning("Provider %s went offline", p.name)
                await self._db.update_provider_status(p.id, ProviderStatus.OFFLINE)
