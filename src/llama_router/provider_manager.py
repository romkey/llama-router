"""Manages provider lifecycle: discovery, health checking, status tracking."""

from __future__ import annotations

import asyncio
import logging
import time
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
        self._hot_models: dict[int, list[dict]] = {}
        self._health_task: asyncio.Task | None = None

    async def start(self) -> None:
        providers = await self._db.list_providers()
        for p in providers:
            assert p.id is not None
            await self._rebuild_clients(p)
            try:
                await self._discover_provider(p)
            except Exception:
                logger.exception("Initial discovery failed for provider %s", p.name)
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
        """Delete a model from the remote provider and re-discover.

        If the backend returns 404 (model already gone), the local model
        list is still refreshed and an ``httpx.HTTPStatusError`` is raised
        so callers can display a notice.
        """
        import httpx

        backend_name = await self._db.get_backend_model_name(provider_id, model_name)
        ollama = self._ollama_clients.get(provider_id)
        not_found = False
        if ollama:
            try:
                await ollama.delete_model(backend_name)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    not_found = True
                else:
                    raise
        provider = await self._db.get_provider(provider_id)
        if provider:
            await self._discover_provider(provider)
        if not_found:
            raise httpx.HTTPStatusError(
                "model not found on backend",
                request=httpx.Request("DELETE", "/api/delete"),
                response=httpx.Response(404),
            )

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
            hot_models=self._hot_models.get(provider_id, []),
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
                    hot_models=self._hot_models.get(p.id, []),
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

    def get_hot_models(self, provider_id: int) -> list[dict]:
        return self._hot_models.get(provider_id, [])

    async def _refresh_hot_models(self, provider: Provider) -> None:
        """Fetch running models via /api/ps for Ollama providers."""
        assert provider.id is not None
        if not provider.supports_ollama or provider.id not in self._ollama_clients:
            self._hot_models[provider.id] = []
            return
        try:
            raw = await self._ollama_clients[provider.id].get_ps()
            hot: list[dict] = []
            for m in raw:
                name = self._strip_cache_prefix(m.get("name", ""))
                entry: dict = {"name": name}
                if m.get("size"):
                    entry["size"] = m["size"]
                if m.get("size_vram"):
                    entry["size_vram"] = m["size_vram"]
                if m.get("expires_at"):
                    entry["expires_at"] = m["expires_at"]
                hot.append(entry)
            self._hot_models[provider.id] = hot
        except Exception:
            logger.debug("Failed to fetch /api/ps for provider %s", provider.name)

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
    ) -> BenchmarkResult:
        """Run a benchmark. Raises ``RuntimeError`` with a human-readable
        message on failure so the caller can surface it to the user."""
        from .request_logger import log_request

        provider = await self._db.get_provider(provider_id)
        if not provider:
            raise RuntimeError(f"Provider {provider_id} not found")

        backend_name = await self._db.get_backend_model_name(provider_id, model_name)

        start = time.monotonic()
        protocol: str | None = None

        try:
            metrics: dict[str, float] | None = None

            if provider.supports_ollama and provider_id in self._ollama_clients:
                client = self._ollama_clients[provider_id]
                protocol = "ollama"
                metrics = await self._try_benchmark(
                    client, backend_name, model_name, provider_id
                )
            elif provider.supports_llamacpp and provider_id in self._llamacpp_clients:
                client = self._llamacpp_clients[provider_id]
                protocol = "llamacpp"
                metrics = await self._try_benchmark(
                    client, backend_name, model_name, provider_id
                )
            else:
                raise RuntimeError(f"No client available for provider {provider.name}")

            result = BenchmarkResult(
                provider_id=provider_id,
                model_name=model_name,
                protocol=protocol,
                startup_time_ms=metrics["startup_time_ms"],
                tokens_per_second=metrics["tokens_per_second"],
            )
            await self._db.save_benchmark(result)

            duration = (time.monotonic() - start) * 1000
            await log_request(
                self._db,
                provider=provider,
                protocol=protocol or "unknown",
                endpoint="benchmark",
                model=model_name,
                duration_ms=duration,
                source_ip="internal",
            )

            return result
        except RuntimeError:
            duration = (time.monotonic() - start) * 1000
            await log_request(
                self._db,
                provider=provider,
                protocol=protocol or "unknown",
                endpoint="benchmark",
                model=model_name,
                duration_ms=duration,
                source_ip="internal",
                status="error",
                error_detail=f"Benchmark failed for {model_name}",
            )
            raise
        except Exception as exc:
            duration = (time.monotonic() - start) * 1000
            detail = self._format_benchmark_error(exc, model_name, provider)
            logger.error("Benchmark failed: %s", detail)
            await log_request(
                self._db,
                provider=provider,
                protocol=protocol or "unknown",
                endpoint="benchmark",
                model=model_name,
                duration_ms=duration,
                source_ip="internal",
                status="error",
                error_detail=detail,
            )
            raise RuntimeError(detail) from exc

    async def _try_benchmark(
        self, client, backend_name: str, display_name: str, provider_id: int
    ) -> dict[str, float]:
        """Try chat benchmark first, fall back to embed."""
        chat_err = None
        try:
            return await client.benchmark_chat(backend_name, settings.benchmark_prompt)
        except Exception as exc:
            chat_err = exc
            logger.info(
                "Chat benchmark failed for %s on provider %d, trying embed",
                display_name,
                provider_id,
            )

        try:
            return await client.benchmark_embed(backend_name, settings.benchmark_prompt)
        except Exception as embed_err:
            raise embed_err from chat_err

    @staticmethod
    def _format_benchmark_error(
        exc: Exception, model_name: str, provider: Provider
    ) -> str:
        import httpx

        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            try:
                body = exc.response.json()
                msg = body.get("error", exc.response.text[:200])
            except Exception:
                msg = exc.response.text[:200] if exc.response.text else ""
            return (
                f"Backend {provider.name} returned HTTP {status} "
                f"for {model_name}: {msg}"
            )
        return f"Benchmark failed for {model_name} on {provider.name}: {exc}"

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

    async def _discover_provider(self, provider: Provider) -> None:
        assert provider.id is not None
        all_models: list[ProviderModel] = []
        seen_names: set[str] = set()
        cache_prefixed = 0

        if provider.supports_ollama and provider.id in self._ollama_clients:
            tags = await self._ollama_clients[provider.id].get_tags()
            for m in tags:
                clean_name = self._strip_cache_prefix(m.name)
                if clean_name not in seen_names:
                    seen_names.add(clean_name)
                    raw_name = m.name if clean_name != m.name else None
                    if raw_name:
                        cache_prefixed += 1
                    all_models.append(
                        ProviderModel(
                            provider_id=provider.id,
                            name=clean_name,
                            raw_name=raw_name,
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
        if cache_prefixed:
            logger.info(
                "Discovered %d models on provider %s (%d with cache prefix)",
                len(all_models),
                provider.name,
                cache_prefixed,
            )
        else:
            logger.info(
                "Discovered %d models on provider %s",
                len(all_models),
                provider.name,
            )

        await self._refresh_hot_models(provider)

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
                    live_urls = [a.url for a in addresses if a.is_live]
                    logger.info(
                        "Provider %s is back online (%s), re-discovering",
                        p.name,
                        ", ".join(live_urls),
                    )
                    await self._rebuild_clients(p)
                    await self._discover_provider(p)
                else:
                    await self._refresh_hot_models(p)
                if self._active_requests.get(p.id, 0) > 0:
                    await self._db.update_provider_status(p.id, ProviderStatus.BUSY)
                else:
                    await self._db.update_provider_status(p.id, ProviderStatus.IDLE)
            else:
                if p.status != ProviderStatus.OFFLINE:
                    addr_urls = [a.url for a in addresses]
                    logger.warning(
                        "Provider %s went offline (%s)",
                        p.name,
                        ", ".join(addr_urls),
                    )
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
