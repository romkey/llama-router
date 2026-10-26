"""Manages provider lifecycle: discovery, health checking, status tracking."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
import time
from typing import TYPE_CHECKING

import httpx

from .config import settings
from .database import Database
from .httpx_errors import describe_httpx_error
from .llamacpp_client import LlamaCppClient
from .models import (
    BenchmarkResult,
    HotModel,
    Provider,
    ProviderAddress,
    ProviderInfo,
    ProviderModel,
    ProviderStatus,
    ProviderType,
    RequestLog,
    WORKER_URL_SCHEME,
)
from .ollama_client import OllamaClient
from .worker_link.hub import WorkerConnection, WorkerHub

if TYPE_CHECKING:
    from .router import Router

logger = logging.getLogger(__name__)

# Reachability probes must be quick: one black-holed address must not stall the
# health cycle for every other provider.
_PROBE_TIMEOUT = httpx.Timeout(5.0)
# Providers checked at once during startup and each health cycle.
_CHECK_CONCURRENCY = 8


def _failure_reason(exc: BaseException) -> str:
    """Short, safe description of why discovery failed, for status and the Log."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in (401, 403):
            return (
                f"authentication required (HTTP {code} from "
                f"{exc.request.url.path}); set the provider's API key"
            )
        return f"HTTP {code} from {exc.request.url.path}"
    if isinstance(exc, httpx.HTTPError):
        return describe_httpx_error(exc)
    return str(exc) or type(exc).__name__


def _transient_chat_benchmark_failure(exc: BaseException) -> bool:
    """Return True if chat benchmark likely failed transiently — do not try embed."""
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
            httpx.LocalProtocolError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (408, 425, 429, 500, 502, 503, 504)
    return False


class ProviderManager:
    _cached_prefixes: list[str] | None = None

    def __init__(self, db: Database, hub: WorkerHub | None = None):
        self._db = db
        self.worker_hub = hub if hub is not None else WorkerHub()
        self._worker_locks: dict[int, asyncio.Lock] = {}
        self.worker_hub.set_callbacks(
            self._on_worker_connected, self._on_worker_disconnected
        )
        self._ollama_clients: dict[int, OllamaClient] = {}
        self._llamacpp_clients: dict[int, LlamaCppClient] = {}
        self._active_requests: dict[int, int] = defaultdict(int)
        self._hot_models: dict[int, list[HotModel]] = {}
        self._active_urls: dict[int, tuple[str, str]] = {}
        self._router: Router | None = None
        self._health_task: asyncio.Task | None = None
        # Why a provider with a reachable address is still not usable, shown in
        # the dashboard. Cleared once discovery succeeds.
        self._status_detail: dict[int, str] = {}
        # Last offline reason written to the Log, so automatic checks log a
        # failure once rather than on every cycle.
        self._logged_detail: dict[int, str] = {}

    def attach_router(self, router: Router) -> None:
        self._router = router

    def _notify_model_routing_cache(self, provider_id: int) -> None:
        rt = self._router
        if rt is None:
            return
        rt.invalidate_providers_for_model_cache()
        rt.invalidate_benchmark_cache_for_provider(provider_id)

    @classmethod
    def _get_cache_prefixes(cls) -> list[str]:
        if cls._cached_prefixes is None:
            host = settings.cache_external_host
            port = str(settings.cache_port)
            raw = [
                f"{host}:{port}/library/" if host else None,
                f"{host}:{port}/" if host else None,
                f"127.0.0.1:{port}/library/",
                f"127.0.0.1:{port}/",
            ]
            cls._cached_prefixes = [p for p in raw if p]
        return cls._cached_prefixes

    @staticmethod
    def _address_url_sig(provider: Provider, addr: ProviderAddress) -> tuple[str, str]:
        o = addr.url.rstrip("/") if provider.supports_ollama else ""
        lcpp = ""
        if provider.supports_llamacpp:
            lcpp = (addr.llamacpp_url or addr.url).rstrip("/")
        return (o, lcpp)

    async def start(self) -> None:
        providers = await self._db.list_providers()
        http_providers: list[Provider] = []
        for p in providers:
            assert p.id is not None
            await self._rebuild_clients(p)
            if p.is_worker:
                # Workers come online when they dial in, not at startup.
                if not self.worker_hub.is_connected(p.id):
                    await self._mark_worker_offline(p.id)
                continue
            http_providers.append(p)
        # Probe and discover now so the dashboard shows real state immediately
        # instead of whatever the database held from the last run.
        await self._check_many(http_providers, rediscover=True, source="startup")
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
            self._active_urls.pop(provider.id, None)
            return
        if provider.is_worker:
            hub = self.worker_hub
            if provider.supports_ollama:
                self._ollama_clients[provider.id] = OllamaClient(
                    provider.url, transport=hub.transport(provider.id, "ollama")
                )
            if provider.supports_llamacpp:
                self._llamacpp_clients[provider.id] = LlamaCppClient(
                    provider.url, transport=hub.transport(provider.id, "llamacpp")
                )
            self._active_urls[provider.id] = (provider.url, provider.url)
            return
        if provider.supports_ollama:
            self._ollama_clients[provider.id] = OllamaClient(addr.url)
        if provider.supports_llamacpp:
            url = addr.llamacpp_url or addr.url
            self._llamacpp_clients[provider.id] = LlamaCppClient(url)
        self._active_urls[provider.id] = self._address_url_sig(provider, addr)

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
        await self.check_provider(provider, rediscover=True, source="add")
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
        existing = await self._db.get_provider(provider_id)
        if existing is not None and existing.is_worker:
            # A worker's address is its socket; the name and URL stay tied to the
            # worker record so the hub can find it again.
            name = existing.name
            url = existing.url
            llamacpp_url = None
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
        await self.check_provider(provider, rediscover=True, source="edit")

    async def delete_remote_model(self, provider_id: int, model_name: str) -> None:
        """Delete a model from the remote provider and re-discover.

        If the backend returns 404 (model already gone), the local model
        list is still refreshed and an ``httpx.HTTPStatusError`` is raised
        so callers can display a notice.
        """
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
        worker = await self._db.get_worker_by_provider(provider_id)
        if worker is not None:
            from .worker_link.protocol import CLOSE_REVOKED

            await self.worker_hub.disconnect(
                provider_id, CLOSE_REVOKED, "worker removed"
            )
            await self._db.delete_worker(int(worker["id"]))
        await self._close_clients(provider_id)
        self._active_requests.pop(provider_id, None)
        self._active_urls.pop(provider_id, None)
        self._hot_models.pop(provider_id, None)
        self._status_detail.pop(provider_id, None)
        self._logged_detail.pop(provider_id, None)
        await self._db.remove_provider(provider_id)
        self._notify_model_routing_cache(provider_id)

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
            worker=self.worker_summary(provider),
            status_detail=self._status_detail.get(provider_id),
        )

    def worker_summary(self, provider: Provider) -> dict | None:
        """Live socket details for a worker provider; None for HTTP providers."""
        if not provider.is_worker or provider.id is None:
            return None
        conn = self.worker_hub.get(provider.id)
        if conn is None:
            return {"connected": False}
        return conn.summary()

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
                    worker=self.worker_summary(p),
                    status_detail=self._status_detail.get(p.id),
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
        current = self._active_requests[provider_id]
        if current <= 0:
            logger.warning(
                "release() called on provider %d but active_requests=%d — possible double-release",
                provider_id,
                current,
            )
            return
        self._active_requests[provider_id] = current - 1

    @asynccontextmanager
    async def acquire_provider(self, provider_id: int):
        self.acquire(provider_id)
        try:
            yield
        finally:
            self.release(provider_id)

    def active_requests(self, provider_id: int) -> int:
        return self._active_requests.get(provider_id, 0)

    def get_hot_models(self, provider_id: int) -> list[HotModel]:
        return self._hot_models.get(provider_id, [])

    async def _refresh_hot_models(self, provider: Provider) -> None:
        """Fetch running models via /api/ps for Ollama providers."""
        assert provider.id is not None
        if not provider.supports_ollama or provider.id not in self._ollama_clients:
            self._hot_models[provider.id] = []
            return
        try:
            raw = await self._ollama_clients[provider.id].get_ps()
            hot: list[HotModel] = []
            for m in raw:
                name = self._strip_cache_prefix(m.get("name", ""))
                entry: HotModel = {"name": name}
                if m.get("size"):
                    entry["size"] = m["size"]
                if m.get("size_vram"):
                    entry["size_vram"] = m["size_vram"]
                if m.get("expires_at"):
                    entry["expires_at"] = m["expires_at"]
                hot.append(entry)
            self._hot_models[provider.id] = hot
        except Exception as exc:
            if isinstance(exc, httpx.HTTPError):
                logger.warning(
                    "Failed to refresh running models (/api/ps) for provider %r: %s",
                    provider.name,
                    describe_httpx_error(exc),
                )
            else:
                logger.debug(
                    "Failed to fetch /api/ps for provider %s: %s",
                    provider.name,
                    exc,
                )

    async def refresh_provider(self, provider_id: int) -> ProviderStatus | None:
        """Re-probe every address, rebuild clients and re-discover models.

        Writes the outcome to the dashboard Log so a manual refresh is visible.
        """
        provider = await self._db.get_provider(provider_id)
        if not provider:
            return None
        return await self.check_provider(provider, rediscover=True, source="refresh")

    def status_detail(self, provider_id: int) -> str | None:
        """Why a provider is not usable although an address answers, if known."""
        return self._status_detail.get(provider_id)

    async def _log_event(
        self,
        provider: Provider,
        endpoint: str,
        *,
        ok: bool,
        detail: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Record a provider lifecycle event in the dashboard Log."""
        try:
            await self._db.save_request_log(
                RequestLog(
                    provider_id=provider.id,
                    provider_name=provider.name,
                    protocol="system",
                    endpoint=endpoint,
                    source_ip="internal",
                    request_meta=None if not ok else detail,
                    duration_ms=duration_ms,
                    status="ok" if ok else "error",
                    error_detail=None if ok else detail,
                )
            )
        except Exception:
            logger.exception("Could not write %s event for %s", endpoint, provider.name)

    async def _check_many(
        self, providers: list[Provider], *, rediscover: bool, source: str
    ) -> None:
        """Check providers concurrently; one failure never stops the others."""
        sem = asyncio.Semaphore(_CHECK_CONCURRENCY)

        async def one(p: Provider) -> None:
            async with sem:
                try:
                    await self.check_provider(p, rediscover=rediscover, source=source)
                except Exception:
                    logger.exception("Health check failed for provider %s", p.name)

        await asyncio.gather(*(one(p) for p in providers))

    async def check_provider(
        self, provider: Provider, *, rediscover: bool = False, source: str = "health"
    ) -> ProviderStatus:
        """Probe all addresses, update liveness, clients, models and status.

        *source* is ``startup``, ``refresh`` or ``health``. Manual refreshes
        always write a Log entry; automatic checks write one only when the
        status or the failure reason changes, so the Log is not flooded.
        """
        assert provider.id is not None
        if provider.is_worker:
            # Probe and discover without the worker lock, which the connect and
            # disconnect handlers need promptly; write results only if the same
            # socket is still current when we are done.
            conn = self.worker_hub.get(provider.id)
            return await self._check_provider_impl(
                provider,
                rediscover,
                source,
                commit_lock=self._worker_lock(provider.id),
                still_current=lambda: self.worker_hub.get(provider.id) is conn,
            )
        return await self._check_provider_impl(provider, rediscover, source)

    async def _check_provider_impl(
        self,
        provider: Provider,
        rediscover: bool,
        source: str,
        *,
        commit_lock: asyncio.Lock | None = None,
        still_current: Callable[[], bool] | None = None,
    ) -> ProviderStatus:
        pid = provider.id
        assert pid is not None
        started = time.monotonic()
        prev_status = provider.status

        addresses = await self._db.get_addresses(pid)
        results = await asyncio.gather(
            *(self._probe_address(provider, a) for a in addresses)
        )
        changed = [
            (addr, reachable)
            for addr, reachable in zip(addresses, results)
            if addr.is_live != reachable
        ]
        for addr, reachable in zip(addresses, results):
            addr.is_live = reachable
        live = [a for a in addresses if a.is_live]

        detail: str | None = None
        if not live:
            status = ProviderStatus.OFFLINE
            urls = ", ".join(a.url for a in addresses) or "no addresses"
            detail = f"no address responded ({urls})"
            self._hot_models[pid] = []
        else:
            await self._maybe_rebuild_clients(provider, addresses)
            needs_discovery = rediscover or prev_status in (
                ProviderStatus.OFFLINE,
                ProviderStatus.UNKNOWN,
            )
            try:
                if needs_discovery:
                    await self._discover_provider(provider)
                else:
                    await self._refresh_hot_models(provider)
            except Exception as exc:
                # Reachable but unusable (for example a key is required to list
                # models). Keep it out of routing and say why.
                status = ProviderStatus.OFFLINE
                detail = _failure_reason(exc)
            else:
                busy = self._active_requests.get(pid, 0) > 0
                status = ProviderStatus.BUSY if busy else ProviderStatus.IDLE

        if commit_lock is None:
            return await self._commit_check(
                provider, prev_status, status, detail, live, changed, source, started
            )
        async with commit_lock:
            if still_current is not None and not still_current():
                # A connect or disconnect handler took over while we probed.
                return provider.status
            return await self._commit_check(
                provider, prev_status, status, detail, live, changed, source, started
            )

    async def _commit_check(
        self,
        provider: Provider,
        prev_status: ProviderStatus,
        status: ProviderStatus,
        detail: str | None,
        live: list[ProviderAddress],
        changed: list[tuple[ProviderAddress, bool]],
        source: str,
        started: float,
    ) -> ProviderStatus:
        pid = provider.id
        assert pid is not None
        prev_detail = self._logged_detail.get(pid)
        for addr, reachable in changed:
            assert addr.id is not None
            await self._db.set_address_live(addr.id, reachable)

        if detail and status == ProviderStatus.OFFLINE and live:
            self._status_detail[pid] = detail
        else:
            self._status_detail.pop(pid, None)

        await self._db.update_provider_status(pid, status)
        provider.status = status

        online = status != ProviderStatus.OFFLINE
        if online:
            self._logged_detail.pop(pid, None)
        elif detail:
            self._logged_detail[pid] = detail
        was_online = prev_status not in (ProviderStatus.OFFLINE, ProviderStatus.UNKNOWN)
        duration_ms = (time.monotonic() - started) * 1000
        live_urls = ", ".join(a.url for a in live)
        if source in ("refresh", "add", "edit"):
            summary = f"online via {live_urls}" if online else (detail or "offline")
            await self._log_event(
                provider, source, ok=online, detail=summary, duration_ms=duration_ms
            )
        elif online and not was_online:
            logger.info("Provider %s is online (%s)", provider.name, live_urls)
            await self._log_event(
                provider,
                source,
                ok=True,
                detail=f"online via {live_urls}",
                duration_ms=duration_ms,
            )
        elif not online and (was_online or detail != prev_detail):
            logger.warning("Provider %s is offline: %s", provider.name, detail)
            await self._log_event(
                provider, source, ok=False, detail=detail, duration_ms=duration_ms
            )
        return status

    async def _maybe_rebuild_clients(
        self, provider: Provider, addresses: list[ProviderAddress]
    ) -> None:
        """Point clients at the best live address unless requests are in flight."""
        pid = provider.id
        assert pid is not None
        best = self._best_address(addresses)
        if best is None:
            return
        has_clients = pid in self._ollama_clients or pid in self._llamacpp_clients
        new_sig = self._address_url_sig(provider, best)
        if has_clients and (
            provider.is_worker or new_sig == self._active_urls.get(pid)
        ):
            return
        in_flight = self._active_requests.get(pid, 0)
        if has_clients and in_flight > 0:
            logger.info(
                "Skipping client rebuild for provider %d — %d requests in flight",
                pid,
                in_flight,
            )
            return
        await self._rebuild_clients(provider)

    async def benchmark_provider(
        self,
        provider_id: int,
        model_name: str,
        benchmark_api: str | None = None,
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

            api_choice = (benchmark_api or "auto").strip().lower()

            if api_choice == "ollama":
                if not (
                    provider.supports_ollama and provider_id in self._ollama_clients
                ):
                    raise RuntimeError(
                        f"Provider {provider.name} does not support ollama benchmarks"
                    )
                client = self._ollama_clients[provider_id]
                protocol = "ollama"
                metrics = await self._try_benchmark(
                    client, backend_name, model_name, provider_id
                )
            elif api_choice == "llamacpp":
                if not (
                    provider.supports_llamacpp and provider_id in self._llamacpp_clients
                ):
                    raise RuntimeError(
                        f"Provider {provider.name} does not support llama.cpp benchmarks"
                    )
                client = self._llamacpp_clients[provider_id]
                protocol = "llamacpp"
                metrics = await self._try_benchmark(
                    client, backend_name, model_name, provider_id
                )
            elif provider.supports_ollama and provider_id in self._ollama_clients:
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
            if self._router:
                self._router.invalidate_benchmark_cache_for_provider(provider_id)

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
        """Try chat benchmark first, fall back to embed only for non-transient chat errors.

        LLMs that do not implement embeddings often return HTTP 501 on embed. If chat
        failed for a timeout or 5xx while the model was loading, falling back to embed
        surfaces that misleading 501 instead of the real chat error — so we skip embed
        in those cases. If embed returns 501 after a prior chat error, we report both.
        """
        chat_err: BaseException | None = None
        try:
            return await client.benchmark_chat(backend_name, settings.benchmark_prompt)
        except Exception as exc:
            chat_err = exc
            if _transient_chat_benchmark_failure(exc):
                raise exc
            detail = (
                describe_httpx_error(exc)
                if isinstance(exc, httpx.HTTPError)
                else str(exc)
            )
            logger.info(
                "Chat benchmark failed for %s on provider %d (%s), trying embed",
                display_name,
                provider_id,
                detail,
            )

        try:
            return await client.benchmark_embed(backend_name, settings.benchmark_prompt)
        except Exception as embed_err:
            if chat_err is not None and isinstance(embed_err, httpx.HTTPStatusError):
                if embed_err.response.status_code == 501:
                    ce = (
                        describe_httpx_error(chat_err)
                        if isinstance(chat_err, httpx.HTTPError)
                        else str(chat_err)
                    )
                    raise RuntimeError(
                        f"Chat benchmark failed ({ce}); this model does not support "
                        f"embeddings, so there is no fallback. If chat failed while the "
                        f"model was loading or the GPU was busy, retry the benchmark."
                    ) from embed_err
            raise embed_err from chat_err

    async def _format_provider_urls(self, provider_id: int) -> str:
        addrs = await self._db.get_addresses(provider_id)
        if not addrs:
            return "(no addresses)"
        return ", ".join(a.url for a in addrs)

    @staticmethod
    def _format_benchmark_error(
        exc: Exception, model_name: str, provider: Provider
    ) -> str:
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
        if isinstance(exc, httpx.RequestError):
            return (
                f"Could not reach backend {provider.name} for {model_name}: "
                f"{describe_httpx_error(exc)}"
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
        for pfx in ProviderManager._get_cache_prefixes():
            if name.startswith(pfx):
                return name[len(pfx) :]
        return name

    async def _discover_provider(self, provider: Provider) -> None:
        assert provider.id is not None
        models_by_name: dict[str, ProviderModel] = {}
        cache_prefixed = 0

        def _upsert_model(
            *,
            source: str,
            clean_name: str,
            raw_name: str | None,
            size: int | None = None,
            digest: str | None = None,
            modified_at: str | None = None,
            details: dict | None = None,
        ) -> None:
            existing = models_by_name.get(clean_name)
            if existing is None:
                merged_details = dict(details or {})
                merged_details["_in_ollama"] = source == "ollama"
                merged_details["_in_llamacpp"] = source == "llamacpp"
                models_by_name[clean_name] = ProviderModel(
                    provider_id=provider.id,
                    name=clean_name,
                    raw_name=raw_name,
                    size=size,
                    digest=digest,
                    modified_at=modified_at,
                    details=merged_details,
                )
                return

            merged_details = dict(existing.details or {})
            if details:
                merged_details.update(details)
            merged_details["_in_ollama"] = bool(
                merged_details.get("_in_ollama", False) or source == "ollama"
            )
            merged_details["_in_llamacpp"] = bool(
                merged_details.get("_in_llamacpp", False) or source == "llamacpp"
            )

            # Prefer the first non-null raw_name and metadata values.
            if existing.raw_name is None and raw_name is not None:
                existing.raw_name = raw_name
            if existing.size is None and size is not None:
                existing.size = size
            if existing.digest is None and digest is not None:
                existing.digest = digest
            if existing.modified_at is None and modified_at is not None:
                existing.modified_at = modified_at
            existing.details = merged_details

        if provider.supports_ollama and provider.id in self._ollama_clients:
            tags = await self._ollama_clients[provider.id].get_tags()
            for m in tags:
                clean_name = self._strip_cache_prefix(m.name)
                raw_name = m.name if clean_name != m.name else None
                if raw_name:
                    cache_prefixed += 1
                _upsert_model(
                    source="ollama",
                    clean_name=clean_name,
                    raw_name=raw_name,
                    size=m.size,
                    digest=m.digest,
                    modified_at=m.modified_at,
                    details=m.details,
                )

        if provider.supports_llamacpp and provider.id in self._llamacpp_clients:
            lcpp_models = await self._llamacpp_clients[provider.id].get_models()
            for m in lcpp_models:
                clean_name = self._strip_cache_prefix(m.name)
                raw_name = m.name if clean_name != m.name else None
                if raw_name:
                    cache_prefixed += 1
                _upsert_model(
                    source="llamacpp",
                    clean_name=clean_name,
                    raw_name=raw_name,
                    size=m.size,
                    details=m.details,
                )

            # Also probe each configured llama.cpp instance for its currently loaded
            # model via /props; some instances expose active model there even when
            # /v1/models is sparse.
            if provider.is_worker:
                props = await self._llamacpp_clients[provider.id].get_props()
                current = self._extract_llamacpp_current_model(props)
                if current:
                    clean_name = self._strip_cache_prefix(current)
                    raw_name = current if clean_name != current else None
                    _upsert_model(
                        source="llamacpp",
                        clean_name=clean_name,
                        raw_name=raw_name,
                        details={"_from_props": True},
                    )
            addresses = (
                [] if provider.is_worker else await self._db.get_addresses(provider.id)
            )
            timeout = httpx.Timeout(10.0)
            async with httpx.AsyncClient(timeout=timeout) as probe_client:
                for addr in addresses:
                    probe_url = (addr.llamacpp_url or addr.url).rstrip("/")
                    try:
                        resp = await probe_client.get(f"{probe_url}/props")
                        props = resp.json() if resp.status_code == 200 else None
                    except httpx.HTTPError:
                        props = None
                    current = self._extract_llamacpp_current_model(props)
                    if not current:
                        continue
                    clean_name = self._strip_cache_prefix(current)
                    raw_name = current if clean_name != current else None
                    if raw_name:
                        cache_prefixed += 1
                    _upsert_model(
                        source="llamacpp",
                        clean_name=clean_name,
                        raw_name=raw_name,
                        details={
                            "_from_props": True,
                            "_llamacpp_source_url": probe_url,
                        },
                    )

        all_models = list(models_by_name.values())
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
        self._notify_model_routing_cache(provider.id)

    @staticmethod
    def _extract_llamacpp_current_model(props: dict | None) -> str | None:
        if not props:
            return None
        for key in ("model_alias", "model", "model_name"):
            val = props.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        model_path = props.get("model_path")
        if isinstance(model_path, str) and model_path.strip():
            name = Path(model_path).name
            if name.endswith(".gguf"):
                name = name[: -len(".gguf")]
            return name
        return None

    async def _health_check_loop(self) -> None:
        while True:
            await asyncio.sleep(settings.health_check_interval_seconds)
            try:
                await self._run_health_checks()
            except Exception as exc:
                detail = (
                    describe_httpx_error(exc)
                    if isinstance(exc, httpx.HTTPError)
                    else str(exc)
                )
                logger.error("Health check cycle failed: %s", detail)

    async def _run_health_checks(self) -> None:
        providers = await self._db.list_providers()
        await self._check_many(providers, rediscover=False, source="health")

    async def _probe_address(self, provider: Provider, addr: ProviderAddress) -> bool:
        """Check if a single address is reachable via the provider's protocol(s)."""
        if provider.is_worker:
            return await self._probe_worker(provider)
        if provider.supports_ollama:
            tmp = OllamaClient(addr.url, timeout=_PROBE_TIMEOUT)
            try:
                if await tmp.is_reachable():
                    return True
            finally:
                await tmp.close()

        if provider.supports_llamacpp:
            url = addr.llamacpp_url or addr.url
            tmp = LlamaCppClient(url, timeout=_PROBE_TIMEOUT)
            try:
                if await tmp.is_reachable():
                    return True
            finally:
                await tmp.close()

        return False

    # --- Workers ---

    async def _probe_worker(self, provider: Provider) -> bool:
        """A worker is live when its socket is up and its upstream answers."""
        assert provider.id is not None
        if not self.worker_hub.is_connected(provider.id):
            return False
        if provider.id not in self._ollama_clients and (
            provider.id not in self._llamacpp_clients
        ):
            await self._rebuild_clients(provider)
        ollama = self._ollama_clients.get(provider.id)
        if ollama is not None and await ollama.is_reachable(timeout=_PROBE_TIMEOUT):
            return True
        lcpp = self._llamacpp_clients.get(provider.id)
        if lcpp is not None and await lcpp.is_reachable(timeout=_PROBE_TIMEOUT):
            return True
        return False

    async def add_worker_provider(self, name: str) -> Provider:
        """Create the provider record a worker attaches to when it dials in."""
        url = f"{WORKER_URL_SCHEME}{name}"
        provider = await self._db.add_provider(name, url, ProviderType.OLLAMA, None)
        assert provider.id is not None
        await self._db.add_address(provider.id, url, None, is_preferred=True)
        await self._db.update_provider_status(provider.id, ProviderStatus.OFFLINE)
        provider.status = ProviderStatus.OFFLINE
        return provider

    def _worker_lock(self, provider_id: int) -> asyncio.Lock:
        lock = self._worker_locks.get(provider_id)
        if lock is None:
            lock = self._worker_locks[provider_id] = asyncio.Lock()
        return lock

    def _is_current_worker(self, conn: WorkerConnection) -> bool:
        return self.worker_hub.get(conn.provider_id) is conn and not conn.closed

    async def _on_worker_connected(self, conn: WorkerConnection) -> None:
        """Bring a freshly connected worker online.

        Connect and disconnect handlers for one provider run one at a time, and
        each re-checks that its socket is still current before writing status,
        so a handler from an older session cannot undo a newer one.
        """
        provider_id = conn.provider_id
        async with self._worker_lock(provider_id):
            if not self._is_current_worker(conn):
                return
            provider = await self._db.get_provider(provider_id)
            if provider is None or not provider.is_worker:
                return
            protocols = conn.protocols
            if "ollama" in protocols and "llamacpp" in protocols:
                ptype = ProviderType.BOTH
            elif "llamacpp" in protocols:
                ptype = ProviderType.LLAMACPP
            else:
                ptype = ProviderType.OLLAMA
            if ptype != provider.provider_type:
                await self._db.update_provider(
                    provider_id,
                    provider.name,
                    provider.url,
                    ptype,
                    None,
                    machine_type=provider.machine_type,
                    gpu_type=provider.gpu_type,
                    gpu_ram=provider.gpu_ram,
                )
                provider.provider_type = ptype
            await self._rebuild_clients(provider)
            detail: str | None = None
            try:
                await self._discover_provider(provider)
                ok = True
            except Exception as exc:
                ok = False
                detail = _failure_reason(exc)
                if self._is_current_worker(conn):
                    logger.warning(
                        "Worker %s connected but model discovery failed: %s",
                        provider.name,
                        detail,
                    )
            # Liveness means the same as in check_provider: the socket is up and
            # the worker's upstream answers, whether or not discovery worked.
            reachable = ok or await self._probe_worker(provider)
            if not self._is_current_worker(conn):
                return
            if detail is not None and reachable:
                self._status_detail[provider_id] = detail
            else:
                self._status_detail.pop(provider_id, None)
                if ok:
                    self._logged_detail.pop(provider_id, None)
            await self._set_worker_addresses_live(provider_id, reachable)
            await self._db.update_provider_status(
                provider_id, ProviderStatus.IDLE if ok else ProviderStatus.OFFLINE
            )

    async def _on_worker_disconnected(self, conn: WorkerConnection) -> None:
        provider_id = conn.provider_id
        async with self._worker_lock(provider_id):
            if self.worker_hub.is_connected(provider_id):
                # A newer session already took over; its handler owns status.
                return
            await self._mark_worker_offline(provider_id)

    async def _set_worker_addresses_live(self, provider_id: int, live: bool) -> None:
        for addr in await self._db.get_addresses(provider_id):
            if addr.id is not None:
                await self._db.set_address_live(addr.id, live)

    async def _mark_worker_offline(self, provider_id: int) -> None:
        provider = await self._db.get_provider(provider_id)
        if provider is None:
            return
        await self._set_worker_addresses_live(provider_id, False)
        await self._db.update_provider_status(provider_id, ProviderStatus.OFFLINE)
        self._hot_models[provider_id] = []
        # No socket means nothing answers; any "reachable but unusable" reason
        # from the last session no longer applies.
        self._status_detail.pop(provider_id, None)
