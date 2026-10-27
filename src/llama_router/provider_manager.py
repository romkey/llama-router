"""Manages provider lifecycle: discovery, health checking, status tracking."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx

from .config import settings
from .database import Database
from .httpx_errors import describe_httpx_error
from .llamacpp_client import LlamaCppClient
from .models import (
    BenchmarkResult,
    HotModel,
    measured_cold_start,
    Provider,
    ProviderAddress,
    ProviderInfo,
    ProviderModel,
    ProviderStatus,
    ProviderProtocol,
    RequestLog,
    WORKER_URL_SCHEME,
)
from .ollama_client import OllamaClient
from .probe import merge_results, probe_url
from .upstream_url import guess_locality, split_api_base
from .routing_stats import ProviderStats
from .provider_secrets import (
    DEFAULT_HEADER,
    ApiKeyUnavailable,
    auth_headers,
    decrypt_api_key,
    encrypt_api_key,
    normalize_header,
)
from .worker_link.hub import WorkerConnection, WorkerHub

if TYPE_CHECKING:
    from .router import Router

logger = logging.getLogger(__name__)

# Reachability probes must be quick: one black-holed address must not stall the
# health cycle for every other provider.
_PROBE_TIMEOUT = httpx.Timeout(5.0)
# Providers checked at once during startup and each health cycle.
_CHECK_CONCURRENCY = 8
# Upper bound on one worker health check, which runs under the worker's lock.
_WORKER_CHECK_TIMEOUT = 20.0
# Checks that rediscover models (manual refresh, add, edit, or a worker
# coming back from offline) can take longer.
_WORKER_MANUAL_TIMEOUT = 60.0


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


def worker_protocol(targets: list[str]) -> ProviderProtocol:
    """The one protocol a worker provider is reached by.

    A worker that relays its own router offers both targets, and the router's
    Ollama port also serves every OpenAI-style route, so the Ollama target
    reaches everything. Use the OpenAI target only when it is all there is.
    """
    if "ollama" in targets or "llamacpp" not in targets:
        return ProviderProtocol.OLLAMA
    return ProviderProtocol.OPENAI


def preset_settings(preset) -> dict[str, Any]:
    """Provider settings a cloud preset starts with."""
    return {
        "preset": preset.id,
        "extra_headers": dict(preset.extra_headers) or None,
        "manual_models": list(preset.models) or None,
        "locality": "cloud",
    }


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
        # The model each llama.cpp upstream reports as loaded (/props).
        self._llamacpp_loaded: dict[int, set[str]] = {}
        self._active_urls: dict[int, str] = {}
        self._router: Router | None = None
        # Live round-trip and request samples that feed the routing table.
        self.stats = ProviderStats()
        # When each provider last started or finished a proxied request, for
        # idle detection; providers with none count from process start.
        self._started_at = time.monotonic()
        self._last_activity: dict[int, float] = {}
        self._activity_listeners: list[Callable[[int], None]] = []
        self._health_task: asyncio.Task | None = None
        # Why a provider with a reachable address is still not usable, shown in
        # the dashboard. Cleared once discovery succeeds.
        self._status_detail: dict[int, str] = {}
        # Last offline reason written to the Log, so automatic checks log a
        # failure once rather than on every cycle.
        self._logged_detail: dict[int, str] = {}
        # Endpoint probes running in the background, one per provider.
        self._probing: dict[int, asyncio.Task] = {}

    def attach_router(self, router: Router) -> None:
        self._router = router

    @property
    def router(self) -> Router | None:
        return self._router

    def _routing_changed(self, reason: str) -> None:
        """In-memory routing input changed (database writes notify on their own)."""
        if self._router is not None:
            self._router.mark_dirty(reason, reload=False)

    def schedule_probe(self, provider_id: int, reason: str = "auto") -> None:
        """Probe a provider's endpoints in the background, once at a time.

        Probes never run inference, so they run on add, on every edit and
        address change, when a provider comes back, and when the last one is
        older than LLAMA_ROUTER_PROBE_MAX_AGE_HOURS.
        """
        if not settings.auto_probe_enabled:
            return
        running = self._probing.get(provider_id)
        if running is not None and not running.done():
            return

        async def _run() -> None:
            try:
                await self.probe_provider(provider_id)
            except ValueError as exc:
                logger.debug("Probe of provider %d skipped: %s", provider_id, exc)
            except Exception:
                logger.exception(
                    "Background probe failed (provider_id=%s, %s)", provider_id, reason
                )
            finally:
                self._probing.pop(provider_id, None)

        self._probing[provider_id] = asyncio.create_task(
            _run(), name=f"probe-{provider_id}"
        )

    def _probe_is_stale(self, provider: Provider) -> bool:
        max_age = settings.probe_max_age_hours
        if max_age <= 0:
            return False
        at = provider.probed_at
        if at is None:
            return True
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - at).total_seconds()
        return age >= max_age * 3600

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
    def _address_url_sig(addr: ProviderAddress) -> str:
        return addr.url.rstrip("/")

    async def provision_from_env(self) -> list[Provider]:
        """Create or update the providers named by LLAMA_ROUTER_PROVIDER_*_API_KEY.

        A docker-compose file can attach cloud services without the
        dashboard: each preset whose key is set gets a provider (the first one
        created from that preset, else a new one named after it), and its key
        is replaced when the variable holds a different one.
        """
        from .cloud_presets import env_provisioned

        out: list[Provider] = []
        existing = await self._db.list_providers()
        for preset, key, params in env_provisioned():
            try:
                url = preset.url(**params)
            except ValueError as exc:
                logger.warning("%s set but %s; skipped", preset.env_var, exc)
                continue
            match = next((p for p in existing if p.preset == preset.id), None)
            if match is None:
                name = preset.name
                taken = {p.name for p in existing}
                n = 2
                while name in taken:
                    name = f"{preset.name} {n}"
                    n += 1
                provider = await self._db.add_provider(
                    name, ProviderProtocol(preset.protocol)
                )
                assert provider.id is not None
                await self._db.update_provider_settings(
                    provider.id, **preset_settings(preset)
                )
                await self._db.add_address(provider.id, url, is_preferred=True)
                provider = await self._db.get_provider(provider.id) or provider
                await self._store_api_key(
                    provider, key, normalize_header(preset.api_key_header)
                )
                existing.append(provider)
                logger.info("Added provider %s from %s", name, preset.env_var)
            else:
                provider = match
                try:
                    current = (
                        decrypt_api_key(provider.api_key_enc)
                        if provider.api_key_enc
                        else None
                    )
                except ApiKeyUnavailable:
                    current = None
                if current != key:
                    await self._store_api_key(
                        provider,
                        key,
                        normalize_header(
                            provider.api_key_header or preset.api_key_header
                        ),
                    )
                    logger.info(
                        "Updated the key of provider %s from %s",
                        provider.name,
                        preset.env_var,
                    )
            out.append(provider)
        return out

    async def start(self) -> None:
        await self.provision_from_env()
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
        for task in list(self._probing.values()):
            task.cancel()
        for task in list(self._probing.values()):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._probing.clear()
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

    @staticmethod
    def _upstream_headers(provider: Provider) -> dict[str, str] | None:
        """Auth headers for a direct provider's upstream, or None.

        Worker providers never get them: the link strips Authorization and
        API-KEY, and the worker adds its own upstream key.
        """
        if provider.is_worker:
            return None
        headers = dict(provider.extra_headers or {})
        if provider.api_key_enc:
            try:
                key = decrypt_api_key(provider.api_key_enc)
            except ApiKeyUnavailable as exc:
                logger.warning("Provider %s: %s", provider.name, exc)
            else:
                headers.update(auth_headers(key, provider.api_key_header))
        return headers or None

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
            transport = self.worker_hub.transport(provider.id, provider.worker_target)
            if provider.is_ollama:
                self._ollama_clients[provider.id] = OllamaClient(
                    addr.url, transport=transport
                )
            else:
                self._llamacpp_clients[provider.id] = LlamaCppClient(
                    addr.url, transport=transport
                )
            self._active_urls[provider.id] = self._address_url_sig(addr)
            return
        headers = self._upstream_headers(provider)
        if provider.is_ollama:
            self._ollama_clients[provider.id] = OllamaClient(addr.url, headers=headers)
        else:
            self._llamacpp_clients[provider.id] = LlamaCppClient(
                addr.url, headers=headers
            )
        self._active_urls[provider.id] = self._address_url_sig(addr)

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
        protocol: ProviderProtocol = ProviderProtocol.OLLAMA,
        *,
        machine_type: str | None = None,
        gpu_type: str | None = None,
        gpu_ram: str | None = None,
        api_key: str | None = None,
        api_key_header: str | None = None,
        provider_settings: dict[str, Any] | None = None,
    ) -> Provider:
        """Create a provider with *url* as its first address.

        *provider_settings* are optional columns (preset, extra_headers,
        manual_models, ...) written before the first check.
        """
        header = normalize_header(api_key_header) if api_key else None
        provider_settings = dict(provider_settings or {})
        if not provider_settings.get("locality"):
            provider_settings["locality"] = guess_locality(url)
        provider = await self._db.add_provider(
            name,
            protocol,
            machine_type=machine_type,
            gpu_type=gpu_type,
            gpu_ram=gpu_ram,
        )
        assert provider.id is not None
        if provider_settings:
            await self._db.update_provider_settings(provider.id, **provider_settings)
            provider = await self._db.get_provider(provider.id) or provider
        if api_key:
            await self._store_api_key(provider, api_key, header)
        await self._db.add_address(provider.id, url, is_preferred=True)
        await self._rebuild_clients(provider)
        self.schedule_probe(provider.id, "add")
        await self.check_provider(provider, rediscover=True, source="add")
        return provider

    async def update_provider(
        self,
        provider_id: int,
        name: str,
        protocol: ProviderProtocol | None = None,
        *,
        machine_type: str | None = None,
        gpu_type: str | None = None,
        gpu_ram: str | None = None,
        api_key: str | None = None,
        api_key_header: str | None = None,
        clear_api_key: bool = False,
        provider_settings: dict[str, Any] | None = None,
    ) -> None:
        """Update provider details. Addresses are edited separately.

        *api_key* None keeps the stored key; *clear_api_key* removes it.
        *provider_settings* writes optional columns; ones left out are kept.
        """
        existing = await self._db.get_provider(provider_id)
        if existing is None:
            return
        # Validate before writing anything, so a bad header name cannot leave
        # the edit half-applied.
        header = (
            normalize_header(api_key_header) if api_key_header is not None else None
        )
        if existing.is_worker:
            # A worker's name ties it to its token on the hub, and its protocol
            # follows what the worker offers when it connects.
            name = existing.name
            protocol = None
        await self._db.update_provider(
            provider_id,
            name,
            protocol,
            machine_type=machine_type,
            gpu_type=gpu_type,
            gpu_ram=gpu_ram,
        )
        if provider_settings:
            await self._db.update_provider_settings(provider_id, **provider_settings)
        if not existing.is_worker:
            if clear_api_key:
                await self._db.set_provider_api_key(provider_id, None, None)
            elif api_key:
                await self._store_api_key(existing, api_key, header)
            elif existing.api_key_enc and header is not None:
                await self._db.set_provider_api_key(
                    provider_id,
                    existing.api_key_enc,
                    None if header == DEFAULT_HEADER else header,
                )
        provider = await self._db.get_provider(provider_id)
        if not provider:
            return
        await self._rebuild_clients(provider)
        self.schedule_probe(provider_id, "edit")
        await self.check_provider(provider, rediscover=True, source="edit")

    async def _store_api_key(
        self, provider: Provider, api_key: str, header: str | None
    ) -> None:
        assert provider.id is not None
        if provider.is_worker:
            raise ValueError(
                "worker providers use the worker's own upstream key "
                "(LLAMA_ROUTER_WORKER_UPSTREAM_API_KEY)"
            )
        enc = encrypt_api_key(api_key.strip())
        stored_header = None if header in (None, DEFAULT_HEADER) else header
        await self._db.set_provider_api_key(provider.id, enc, stored_header)
        provider.api_key_enc = enc
        provider.api_key_header = stored_header

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
        self._llamacpp_loaded.pop(provider_id, None)
        self._status_detail.pop(provider_id, None)
        self._logged_detail.pop(provider_id, None)
        self.stats.forget_provider(provider_id)
        await self._db.remove_provider(provider_id)

    # --- Address CRUD ---

    async def add_address(
        self,
        provider_id: int,
        url: str,
        is_preferred: bool = False,
    ) -> ProviderAddress:
        addr = await self._db.add_address(provider_id, url, is_preferred)
        provider = await self._db.get_provider(provider_id)
        if provider:
            await self._rebuild_clients(provider)
            self.schedule_probe(provider_id, "address")
        return addr

    async def update_address(
        self,
        address_id: int,
        url: str,
        is_preferred: bool | None = None,
    ) -> None:
        addr = await self._db.get_address(address_id)
        if not addr:
            return
        await self._db.update_address(address_id, url, is_preferred)
        provider = await self._db.get_provider(addr.provider_id)
        if provider:
            await self._rebuild_clients(provider)
            if url.rstrip("/") != addr.url.rstrip("/"):
                self.schedule_probe(addr.provider_id, "address")

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

    def add_activity_listener(self, fn: Callable[[int], None]) -> None:
        """Call *fn(provider_id)* whenever a proxied request starts."""
        self._activity_listeners.append(fn)

    def idle_seconds(self, provider_id: int, now: float | None = None) -> float:
        """Seconds since the provider last served a request (0 while busy)."""
        if self._active_requests.get(provider_id, 0) > 0:
            return 0.0
        t = time.monotonic() if now is None else now
        return max(0.0, t - self._last_activity.get(provider_id, self._started_at))

    def acquire(self, provider_id: int) -> None:
        self._active_requests[provider_id] += 1
        self._last_activity[provider_id] = time.monotonic()
        for fn in self._activity_listeners:
            try:
                fn(provider_id)
            except Exception:
                logger.exception("Activity listener failed")

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
        self._last_activity[provider_id] = time.monotonic()

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

    def hot_model_names(
        self, provider_id: int, upstream: str | None = None
    ) -> set[str]:
        """Models loaded right now: reported by Ollama's /api/ps (*upstream*
        "ollama"), by llama.cpp's /props ("openai"), or either when None."""
        ollama = {
            hm["name"] for hm in self._hot_models.get(provider_id, []) if hm.get("name")
        }
        if upstream == "ollama":
            return ollama
        llamacpp = self._llamacpp_loaded.get(provider_id, set())
        if upstream == "openai":
            return set(llamacpp)
        return ollama | llamacpp

    def _set_llamacpp_loaded(self, provider_id: int, loaded: set[str]) -> None:
        if self._llamacpp_loaded.get(provider_id, set()) != loaded:
            self._llamacpp_loaded[provider_id] = loaded
            self._routing_changed("hot models")

    def _clear_loaded(self, provider_id: int) -> None:
        """Nothing is loaded on a provider that does not answer."""
        self._set_hot_models(provider_id, [])
        self._set_llamacpp_loaded(provider_id, set())

    def _set_hot_models(self, provider_id: int, hot: list[HotModel]) -> None:
        before = self.hot_model_names(provider_id)
        self._hot_models[provider_id] = hot
        if self.hot_model_names(provider_id) != before:
            self._routing_changed("hot models")

    async def _refresh_hot_models(self, provider: Provider) -> None:
        """Refresh which models are loaded: /api/ps on Ollama, /props on llama.cpp."""
        await self._refresh_llamacpp_loaded(provider)
        await self._refresh_ollama_hot(provider)

    async def _refresh_llamacpp_loaded(self, provider: Provider) -> None:
        assert provider.id is not None
        client = self._llamacpp_clients.get(provider.id)
        if not provider.is_openai or client is None or provider.lacks("/props"):
            # Only llama.cpp-style servers report their loaded model in /props.
            self._set_llamacpp_loaded(provider.id, set())
            return
        try:
            current = self._extract_llamacpp_current_model(await client.get_props())
        except Exception as exc:
            logger.debug(
                "Failed to read /props for provider %s: %s", provider.name, exc
            )
            current = None
        self._set_llamacpp_loaded(
            provider.id, {self._strip_cache_prefix(current)} if current else set()
        )

    async def _refresh_ollama_hot(self, provider: Provider) -> None:
        assert provider.id is not None
        if not provider.is_ollama or provider.id not in self._ollama_clients:
            self._set_hot_models(provider.id, [])
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
            self._set_hot_models(provider.id, hot)
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
            return await self._check_worker(provider, rediscover, source)
        return await self._check_provider_impl(provider, rediscover, source)

    async def _check_worker(
        self, provider: Provider, rediscover: bool, source: str
    ) -> ProviderStatus:
        """Check a worker provider under its lock, bounded in time.

        Every state change (clients, models, hot models, liveness, status)
        happens under the per-worker lock the connect and disconnect handlers
        use, so a check can never interleave with them. An automatic check
        steps aside while a handler holds the lock, and the whole check is
        capped so a hung upstream cannot keep the lock for long.
        """
        pid = provider.id
        assert pid is not None
        lock = self._worker_lock(pid)
        if source == "health" and lock.locked():
            return provider.status
        async with lock:
            # Re-read under the lock: a connect or disconnect handler may have
            # changed (and logged) the status since this cycle listed providers.
            fresh = await self._db.get_provider(pid)
            if fresh is None:
                return provider.status
            self._refresh_from(provider, fresh)
            conn = self.worker_hub.get(pid)
            prev_status = provider.status
            started = time.monotonic()
            # A check that will rediscover models gets the longer cap, so one
            # slow model listing cannot keep a connected worker offline.
            rediscovering = rediscover or prev_status in (
                ProviderStatus.OFFLINE,
                ProviderStatus.UNKNOWN,
            )
            cap = (
                _WORKER_MANUAL_TIMEOUT
                if source != "health" or rediscovering
                else _WORKER_CHECK_TIMEOUT
            )
            try:
                return await asyncio.wait_for(
                    self._check_provider_impl(
                        provider,
                        rediscover,
                        source,
                        still_current=lambda: self.worker_hub.get(pid) is conn,
                    ),
                    timeout=cap,
                )
            except asyncio.TimeoutError:
                # Commit a consistent result rather than leaving liveness
                # written and status, reason and Log untouched.
                logger.warning(
                    "Check of worker %s timed out after %.0fs", provider.name, cap
                )
                if self.worker_hub.get(pid) is not conn:
                    # A new socket arrived; its connect handler, waiting on the
                    # lock, sets status. Do not mark it offline first.
                    return await self._superseded(provider, source)
                self._clear_loaded(pid)
                live = [a for a in await self._db.get_addresses(pid) if a.is_live]
                return await self._commit_check(
                    provider,
                    prev_status,
                    ProviderStatus.OFFLINE,
                    f"check timed out after {cap:.0f}s",
                    live,
                    source,
                    started,
                )

    async def _check_provider_impl(
        self,
        provider: Provider,
        rediscover: bool,
        source: str,
        *,
        still_current: Callable[[], bool] | None = None,
    ) -> ProviderStatus:
        pid = provider.id
        assert pid is not None
        started = time.monotonic()
        if not provider.is_worker:
            # The cycle's provider list can be seconds old by now.
            fresh = await self._db.get_provider(pid)
            if fresh is not None:
                self._refresh_from(provider, fresh)
        prev_status = provider.status

        addresses = await self._db.get_addresses(pid)
        results = await asyncio.gather(
            *(self._probe_address(provider, a) for a in addresses)
        )
        # Write liveness before rebuilding clients: the rebuild re-reads
        # addresses and ranks them by is_live.
        for addr, reachable in zip(addresses, results):
            assert addr.id is not None
            if addr.is_live != reachable:
                await self._db.set_address_live(addr.id, reachable)
            addr.is_live = reachable
        live = [a for a in addresses if a.is_live]
        if addresses:
            self.stats.record_check(pid, ok=bool(live))

        detail: str | None = None
        if not live:
            status = ProviderStatus.OFFLINE
            if provider.is_worker and not self.worker_hub.is_connected(pid):
                detail = "worker is not connected"
            else:
                urls = ", ".join(a.url for a in addresses) or "no addresses"
                detail = f"no address responded ({urls})"
            self._clear_loaded(pid)
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

        if still_current is not None and not still_current():
            # A new socket arrived mid-check; its connect handler (waiting on
            # the lock) sets status, so do not overwrite it with ours.
            return await self._superseded(provider, source)
        return await self._commit_check(
            provider, prev_status, status, detail, live, source, started
        )

    async def _superseded(self, provider: Provider, source: str) -> ProviderStatus:
        """A reconnect overtook this check: keep status, but log manual checks."""
        if source in ("refresh", "add", "edit"):
            await self._log_event(
                provider,
                source,
                ok=True,
                detail="worker reconnected during the check; status follows "
                "the new connection",
            )
        return provider.status

    @staticmethod
    def _refresh_from(provider: Provider, fresh: Provider) -> None:
        """Update *provider* in place from a freshly read row.

        Status, type, API key and header can all change while a cycle runs
        (for example an edit saving a key), so copy every field, not just
        the status.
        """
        for name in type(fresh).model_fields:
            setattr(provider, name, getattr(fresh, name))

    async def _commit_check(
        self,
        provider: Provider,
        prev_status: ProviderStatus,
        status: ProviderStatus,
        detail: str | None,
        live: list[ProviderAddress],
        source: str,
        started: float,
    ) -> ProviderStatus:
        pid = provider.id
        assert pid is not None
        prev_detail = self._logged_detail.get(pid)

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
            if prev_status == ProviderStatus.OFFLINE:
                # It may have come back as a different server or version.
                self.schedule_probe(pid, "recovered")
            await self._log_event(
                provider,
                source,
                ok=True,
                detail=f"online via {live_urls}",
                duration_ms=duration_ms,
            )
        # A provider already offline before this process started has no
        # remembered reason: remember it without logging, so a restart does not
        # re-log every provider that was down anyway.
        elif not online and (
            was_online or (prev_detail is not None and detail != prev_detail)
        ):
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
        new_sig = self._address_url_sig(best)
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
        *,
        source: str = "manual",
    ) -> BenchmarkResult:
        """Run a benchmark. Raises ``RuntimeError`` with a human-readable
        message on failure so the caller can surface it to the user.

        *source* is ``manual`` or ``auto`` (the idle auto-benchmarker). When
        the model was already loaded, the run cannot measure a cold start, so
        its startup time is stored as unknown rather than as near zero.
        """
        from .request_logger import log_request

        provider = await self._db.get_provider(provider_id)
        if not provider:
            raise RuntimeError(f"Provider {provider_id} not found")

        backend_name = await self._db.get_backend_model_name(provider_id, model_name)
        log_endpoint = "auto-benchmark" if source == "auto" else "benchmark"

        start = time.monotonic()
        protocol: str | None = None

        try:
            metrics: dict[str, float] | None = None

            api_choice = (benchmark_api or "auto").strip().lower()
            if api_choice != "auto":
                wanted = ProviderProtocol.parse(api_choice)
                if wanted is not provider.protocol:
                    raise RuntimeError(
                        f"Provider {provider.name} speaks the "
                        f"{provider.protocol.value} API, not {wanted.value}"
                    )
            client: OllamaClient | LlamaCppClient | None
            if provider.is_ollama:
                client = self._ollama_clients.get(provider_id)
            else:
                client = self._llamacpp_clients.get(provider_id)
            if client is None:
                raise RuntimeError(f"No client available for provider {provider.name}")
            protocol = provider.protocol.value
            metrics = await self._try_benchmark(
                client, backend_name, model_name, provider_id
            )

            result = BenchmarkResult(
                provider_id=provider_id,
                model_name=model_name,
                protocol=protocol,
                # The backend's own load time says whether this was a cold
                # start; the router's view of what is loaded can be stale.
                startup_time_ms=measured_cold_start(metrics["startup_time_ms"]),
                tokens_per_second=metrics["tokens_per_second"],
                source=source,
            )
            await self._db.save_benchmark(result)

            duration = (time.monotonic() - start) * 1000
            await log_request(
                self._db,
                provider=provider,
                protocol=protocol or "unknown",
                endpoint=log_endpoint,
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
                endpoint=log_endpoint,
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
                endpoint=log_endpoint,
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
            clean_name: str,
            raw_name: str | None,
            size: int | None = None,
            digest: str | None = None,
            modified_at: str | None = None,
            details: dict | None = None,
        ) -> None:
            existing = models_by_name.get(clean_name)
            if existing is None:
                models_by_name[clean_name] = ProviderModel(
                    provider_id=provider.id,
                    name=clean_name,
                    raw_name=raw_name,
                    size=size,
                    digest=digest,
                    modified_at=modified_at,
                    details=dict(details) if details else None,
                )
                return

            merged_details = dict(existing.details or {})
            if details:
                merged_details.update(details)

            # Prefer the first non-null raw_name and metadata values.
            if existing.raw_name is None and raw_name is not None:
                existing.raw_name = raw_name
            if existing.size is None and size is not None:
                existing.size = size
            if existing.digest is None and digest is not None:
                existing.digest = digest
            if existing.modified_at is None and modified_at is not None:
                existing.modified_at = modified_at
            existing.details = merged_details or None

        manual = [m for m in (provider.manual_models or []) if m]

        async def _listed(fetch):
            """The provider's own list; with hand-added models, a failed list
            only costs the listed ones."""
            try:
                return await fetch()
            except (httpx.HTTPError, ValueError) as exc:
                if not manual:
                    raise
                logger.warning(
                    "Provider %s: model list failed (%s); using its %d added models",
                    provider.name,
                    (
                        describe_httpx_error(exc)
                        if isinstance(exc, httpx.HTTPError)
                        else exc
                    ),
                    len(manual),
                )
                return []

        if provider.is_ollama and provider.id in self._ollama_clients:
            tags = await _listed(self._ollama_clients[provider.id].get_tags)
            for m in tags:
                clean_name = self._strip_cache_prefix(m.name)
                raw_name = m.name if clean_name != m.name else None
                if raw_name:
                    cache_prefixed += 1
                _upsert_model(
                    clean_name=clean_name,
                    raw_name=raw_name,
                    size=m.size,
                    digest=m.digest,
                    modified_at=m.modified_at,
                    details=m.details,
                )

        if provider.is_openai and provider.id in self._llamacpp_clients:
            lcpp_models = await _listed(self._llamacpp_clients[provider.id].get_models)
            for m in lcpp_models:
                clean_name = self._strip_cache_prefix(m.name)
                raw_name = m.name if clean_name != m.name else None
                if raw_name:
                    cache_prefixed += 1
                _upsert_model(
                    clean_name=clean_name,
                    raw_name=raw_name,
                    size=m.size,
                    details=m.details,
                )

            # Also probe each configured llama.cpp instance for its currently loaded
            # model via /props; some instances expose active model there even when
            # /v1/models is sparse. Skip servers whose probe found no /props.
            if provider.lacks("/props"):
                pass
            elif provider.is_worker:
                props = await self._llamacpp_clients[provider.id].get_props()
                current = self._extract_llamacpp_current_model(props)
                if current:
                    clean_name = self._strip_cache_prefix(current)
                    raw_name = current if clean_name != current else None
                    _upsert_model(
                        clean_name=clean_name,
                        raw_name=raw_name,
                        details={"_from_props": True},
                    )
            addresses = (
                []
                if provider.is_worker or provider.lacks("/props")
                else await self._db.get_addresses(provider.id)
            )
            timeout = httpx.Timeout(10.0)
            async with httpx.AsyncClient(
                timeout=timeout, headers=self._upstream_headers(provider)
            ) as probe_client:
                for addr in addresses:
                    base = split_api_base(addr.url)[0]
                    try:
                        resp = await probe_client.get(f"{base}/props")
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
                        clean_name=clean_name,
                        raw_name=raw_name,
                        details={
                            "_from_props": True,
                            "_llamacpp_source_url": base,
                        },
                    )

        for name in manual:
            if name not in models_by_name:
                _upsert_model(clean_name=name, raw_name=None, details={"_manual": True})
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
        for p in providers:
            if (
                p.id is not None
                and p.status != ProviderStatus.OFFLINE
                and self._probe_is_stale(p)
            ):
                self.schedule_probe(p.id, "stale")

    async def _probe_address(self, provider: Provider, addr: ProviderAddress) -> bool:
        """Check one address and record how long a successful check took.

        The round trip includes connecting, so it reflects what a request to
        that address pays before the backend starts work.
        """
        started = time.monotonic()
        ok = False
        try:
            ok = await self._probe_address_once(provider, addr)
            return ok
        finally:
            # Errors and timeouts (including a cancelled check) count as
            # failed checks in the address's tally.
            if provider.id is not None and addr.id is not None:
                self.stats.record_probe(
                    provider.id,
                    addr.id,
                    ok=ok,
                    ms=(time.monotonic() - started) * 1000,
                )

    async def _probe_address_once(
        self, provider: Provider, addr: ProviderAddress
    ) -> bool:
        """Check if a single address is reachable via the provider's protocol."""
        if provider.is_worker:
            return await self._probe_worker(provider)
        liveness = provider.liveness_path
        if liveness:
            # The probe found this GET answers cheaply on this server.
            try:
                async with httpx.AsyncClient(
                    timeout=_PROBE_TIMEOUT, headers=self._upstream_headers(provider)
                ) as http:
                    resp = await http.get(addr.url.rstrip("/") + liveness)
                return 200 <= resp.status_code < 300
            except httpx.HTTPError:
                return False
        client_cls = OllamaClient if provider.is_ollama else LlamaCppClient
        tmp = client_cls(
            addr.url,
            timeout=_PROBE_TIMEOUT,
            headers=self._upstream_headers(provider),
        )
        try:
            return await tmp.is_reachable()
        finally:
            await tmp.close()

    # --- Endpoint probing ---

    async def probe_provider(self, provider_id: int) -> dict | None:
        """Probe the provider's best address and store what its upstream supports.

        Workers are probed through their socket.
        """
        provider = await self._db.get_provider(provider_id)
        if provider is None:
            return None
        addresses = await self._db.get_addresses(provider_id)
        best = self._best_address(addresses)
        if best is None:
            raise ValueError("provider has no address to probe")
        results = []
        if provider.is_worker:
            if not self.worker_hub.is_connected(provider_id):
                raise ValueError("worker is not connected")
            r = await probe_url(
                "http://worker",
                transport=self.worker_hub.transport(
                    provider_id, provider.worker_target
                ),
            )
            r.base_url = provider.worker_address or r.base_url
            results.append(r)
        else:
            results.append(
                await probe_url(
                    best.url,
                    headers=self._upstream_headers(provider),
                    key_sent=provider.has_api_key,
                )
            )
        caps = merge_results(results)
        await self._db.set_provider_capabilities(provider_id, caps)
        await self._log_event(
            provider,
            "probe",
            ok=caps["reachable"],
            detail=(
                (
                    f"{len(caps['supported'])} endpoints; servers: "
                    + (", ".join(caps["servers"]) or "unrecognised")
                )
                if caps["reachable"]
                else "no response from any endpoint"
            ),
        )
        return caps

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
        provider = await self._db.add_provider(
            name, ProviderProtocol.OLLAMA, worker_name=name
        )
        assert provider.id is not None
        url = f"{WORKER_URL_SCHEME}{name}"
        await self._db.add_address(provider.id, url, is_preferred=True)
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
            protocol = worker_protocol(conn.protocols)
            if protocol is not provider.protocol:
                await self._db.update_provider(
                    provider_id,
                    provider.name,
                    protocol,
                    machine_type=provider.machine_type,
                    gpu_type=provider.gpu_type,
                    gpu_ram=provider.gpu_ram,
                )
                provider.protocol = protocol
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
            provider = await self._db.get_provider(provider_id)
            was_online = provider is not None and provider.status not in (
                ProviderStatus.OFFLINE,
                ProviderStatus.UNKNOWN,
            )
            await self._mark_worker_offline(provider_id)
            if provider is not None:
                # The disconnect is the transition, so log it here; the
                # remembered reason stops the next health check logging it again.
                detail = "worker is not connected"
                self._logged_detail[provider_id] = detail
                if was_online:
                    logger.warning("Worker %s disconnected", provider.name)
                    await self._log_event(
                        provider, "disconnect", ok=False, detail=detail
                    )

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
        self._clear_loaded(provider_id)
        # No socket means nothing answers; any "reachable but unusable" reason
        # from the last session no longer applies.
        self._status_detail.pop(provider_id, None)
