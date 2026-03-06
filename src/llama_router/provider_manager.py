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
    ProviderAddress,
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
        self._model_raw_names: dict[int, dict[str, str]] = {}

    async def start(self) -> None:
        providers = await self._db.list_providers()
        for p in providers:
            assert p.id is not None
            await self._rebuild_clients(p)
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

    # --- Address helpers ---

    def _best_address(self, addresses: list[ProviderAddress]) -> ProviderAddress | None:
        """Pick the best address: prefer live+preferred > live > preferred > any."""
        if not addresses:
            return None
        live_preferred = [a for a in addresses if a.is_live and a.is_preferred]
        if live_preferred:
            return live_preferred[0]
        live = [a for a in addresses if a.is_live]
        if live:
            return live[0]
        preferred = [a for a in addresses if a.is_preferred]
        if preferred:
            return preferred[0]
        return addresses[0]

    async def _rebuild_clients(self, provider: Provider) -> None:
        """Close existing clients and create new ones from the best live address."""
        assert provider.id is not None
        await self._close_clients(provider.id)
        addresses = await self._db.get_addresses(provider.id)
        addr = self._best_address(addresses)
        if not addr:
            return
        if provider.supports_ollama:
            self._ollama_clients[provider.id] = OllamaClient(addr.url)
        if provider.supports_llamacpp:
            url = addr.llamacpp_url or addr.url
            self._llamacpp_clients[provider.id] = LlamaCppClient(url)

    async def _close_clients(self, provider_id: int) -> None:
        if provider_id in self._ollama_clients:
            await self._ollama_clients[provider_id].close()
            del self._ollama_clients[provider_id]
        if provider_id in self._llamacpp_clients:
            await self._llamacpp_clients[provider_id].close()
            del self._llamacpp_clients[provider_id]

    # --- Provider CRUD ---

    async def add_provider(
        self,
        name: str,
        url: str,
        provider_type: ProviderType = ProviderType.OLLAMA,
        llamacpp_url: str | None = None,
        machine_type: str | None = None,
        gpu_type: str | None = None,
        gpu_ram: str | None = None,
    ) -> Provider:
        provider = await self._db.add_provider(
            name,
            url,
            provider_type,
            llamacpp_url,
            machine_type=machine_type,
            gpu_type=gpu_type,
            gpu_ram=gpu_ram,
        )
        assert provider.id is not None
        await self._db.add_address(provider.id, url, llamacpp_url, is_preferred=True)
        await self._rebuild_clients(provider)

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
        machine_type: str | None = None,
        gpu_type: str | None = None,
        gpu_ram: str | None = None,
    ) -> None:
        await self._db.update_provider(
            provider_id,
            name,
            url,
            provider_type,
            llamacpp_url,
            machine_type=machine_type,
            gpu_type=gpu_type,
            gpu_ram=gpu_ram,
        )
        provider = await self._db.get_provider(provider_id)
        if not provider:
            return
        await self._rebuild_clients(provider)
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

    # --- Address CRUD ---

    async def add_address(
        self,
        provider_id: int,
        url: str,
        llamacpp_url: str | None = None,
        is_preferred: bool = False,
    ) -> ProviderAddress:
        addr = await self._db.add_address(provider_id, url, llamacpp_url, is_preferred)
        provider = await self._db.get_provider(provider_id)
        if provider:
            await self._rebuild_clients(provider)
        return addr

    async def update_address(
        self,
        address_id: int,
        url: str,
        llamacpp_url: str | None = None,
        is_preferred: bool | None = None,
    ) -> None:
        addr = await self._db.get_address(address_id)
        if not addr:
            return
        await self._db.update_address(address_id, url, llamacpp_url, is_preferred)
        provider = await self._db.get_provider(addr.provider_id)
        if provider:
            await self._rebuild_clients(provider)

    async def remove_address(self, address_id: int) -> None:
        addr = await self._db.get_address(address_id)
        if not addr:
            return
        await self._db.remove_address(address_id)
        provider = await self._db.get_provider(addr.provider_id)
        if provider:
            await self._rebuild_clients(provider)

    async def toggle_address_preferred(self, address_id: int) -> None:
        addr = await self._db.get_address(address_id)
        if not addr:
            return
        await self._db.set_address_preferred(address_id, not addr.is_preferred)
        provider = await self._db.get_provider(addr.provider_id)
        if provider:
            await self._rebuild_clients(provider)

    # --- Info ---

    async def get_provider_info(self, provider_id: int) -> ProviderInfo | None:
        provider = await self._db.get_provider(provider_id)
        if not provider:
            return None
        models = await self._db.get_provider_models(provider_id)
        benchmarks = await self._db.get_benchmarks_for_provider(provider_id)
        addresses = await self._db.get_addresses(provider_id)
        return ProviderInfo(
            provider=provider,
            models=models,
            benchmarks=benchmarks,
            addresses=addresses,
            active_requests=self._active_requests.get(provider_id, 0),
        )

    async def list_provider_infos(self) -> list[ProviderInfo]:
        providers = await self._db.list_providers()
        infos = []
        for p in providers:
            assert p.id is not None
            models = await self._db.get_provider_models(p.id)
            benchmarks = await self._db.get_benchmarks_for_provider(p.id)
            addresses = await self._db.get_addresses(p.id)
            infos.append(
                ProviderInfo(
                    provider=p,
                    models=models,
                    benchmarks=benchmarks,
                    addresses=addresses,
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
            protocol: str | None = None
            metrics: dict[str, float] | None = None

            if provider.supports_ollama and provider_id in self._ollama_clients:
                client = self._ollama_clients[provider_id]
                protocol = "ollama"
                try:
                    metrics = await client.benchmark_chat(
                        model_name, settings.benchmark_prompt
                    )
                except Exception:
                    logger.info(
                        "Chat benchmark failed for %s on provider %d, trying embed",
                        model_name,
                        provider_id,
                    )
                    metrics = await client.benchmark_embed(
                        model_name, settings.benchmark_prompt
                    )
            elif provider.supports_llamacpp and provider_id in self._llamacpp_clients:
                client = self._llamacpp_clients[provider_id]
                protocol = "llamacpp"
                try:
                    metrics = await client.benchmark_chat(
                        model_name, settings.benchmark_prompt
                    )
                except Exception:
                    logger.info(
                        "Chat benchmark failed for %s on provider %d, trying embed",
                        model_name,
                        provider_id,
                    )
                    metrics = await client.benchmark_embed(
                        model_name, settings.benchmark_prompt
                    )
            else:
                return None

            result = BenchmarkResult(
                provider_id=provider_id,
                model_name=model_name,
                protocol=protocol,
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

    @staticmethod
    def _strip_cache_prefix(name: str) -> str:
        """Remove the cache registry prefix that Ollama adds to model names.

        Models pulled through the cache get stored as e.g.
        ``host:9200/library/llama3.2:latest``.  Strip the ``host:port/library/``
        (or ``host:port/``) prefix so we display just ``llama3.2:latest``.
        """
        if "/" not in name:
            return name
        host = settings.cache_external_host
        port = str(settings.cache_port)
        prefixes = []
        if host:
            prefixes.append(f"{host}:{port}/library/")
            prefixes.append(f"{host}:{port}/")
        prefixes.append(f"127.0.0.1:{port}/library/")
        prefixes.append(f"127.0.0.1:{port}/")
        for pfx in prefixes:
            if name.startswith(pfx):
                return name[len(pfx) :]
        return name

    def resolve_backend_model_name(self, provider_id: int, clean_name: str) -> str:
        """Return the raw model name the backend knows for a given clean name.

        If the model was pulled through the cache, Ollama stores it with a
        prefix (e.g. ``host:9200/library/qwen3.5:9B``).  The router stores
        and displays the stripped name (``qwen3.5:9B``).  When forwarding a
        request to the backend we must use the raw name the backend recognises.
        """
        raw_map = self._model_raw_names.get(provider_id, {})
        return raw_map.get(clean_name, clean_name)

    async def _discover_provider(self, provider: Provider) -> None:
        assert provider.id is not None
        all_models: list[ProviderModel] = []
        seen_names: set[str] = set()
        raw_names: dict[str, str] = {}

        if provider.supports_ollama and provider.id in self._ollama_clients:
            tags = await self._ollama_clients[provider.id].get_tags()
            for m in tags:
                clean_name = self._strip_cache_prefix(m.name)
                if clean_name not in seen_names:
                    seen_names.add(clean_name)
                    if clean_name != m.name:
                        raw_names[clean_name] = m.name
                    all_models.append(
                        ProviderModel(
                            provider_id=provider.id,
                            name=clean_name,
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

        self._model_raw_names[provider.id] = raw_names
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
            addresses = await self._db.get_addresses(p.id)
            any_live = False

            for addr in addresses:
                assert addr.id is not None
                reachable = await self._probe_address(p, addr)
                await self._db.set_address_live(addr.id, reachable)
                if reachable:
                    any_live = True

            if any_live:
                if p.status == ProviderStatus.OFFLINE:
                    logger.info("Provider %s is back online, re-discovering", p.name)
                    await self._rebuild_clients(p)
                    await self._discover_provider(p)
                if self._active_requests.get(p.id, 0) > 0:
                    await self._db.update_provider_status(p.id, ProviderStatus.BUSY)
                else:
                    await self._db.update_provider_status(p.id, ProviderStatus.IDLE)
            else:
                if p.status != ProviderStatus.OFFLINE:
                    logger.warning("Provider %s went offline", p.name)
                await self._db.update_provider_status(p.id, ProviderStatus.OFFLINE)

    async def _probe_address(self, provider: Provider, addr: ProviderAddress) -> bool:
        """Check if a single address is reachable via the provider's protocol(s)."""
        if provider.supports_ollama:
            tmp = OllamaClient(addr.url)
            try:
                if await tmp.is_reachable():
                    return True
            finally:
                await tmp.close()

        if provider.supports_llamacpp:
            url = addr.llamacpp_url or addr.url
            tmp = LlamaCppClient(url)
            try:
                if await tmp.is_reachable():
                    return True
            finally:
                await tmp.close()

        return False
