"""Manages provider lifecycle: discovery, health checking, status tracking."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from . import router_identity
from .alerts import alerts
from .capacity import (
    DEFAULT_LEVEL,
    CapacityLimits,
    ProviderAtCapacity,
    QueueTimeout,
    SlotGate,
)
from .client_pool import ClientPool, upstream_headers
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
    ProviderProtocol,
    ProviderStatus,
)
from .ollama_client import OllamaClient
from .probe import (
    BlockedAddressError,
    check_not_metadata,
    normalize_base_url,
)
from .provider_benchmark import Benchmarking
from .provider_health import HealthChecks
from .provider_secrets import (
    DEFAULT_HEADER,
    ApiKeyUnavailableError,
    decrypt_api_key,
    encrypt_api_key,
    normalize_header,
    reencrypt_legacy,
)
from .routing_stats import ProviderStats
from .upstream_url import guess_locality, split_api_base
from .worker_link.hub import WorkerHub
from .worker_sync import WorkerSync

if TYPE_CHECKING:
    from .cloud_presets import CloudPreset
    from .router import Router
    from .translate import OllamaOverOpenAI

logger = logging.getLogger(__name__)


async def _check_provider_url(url: str) -> None:
    """Refuse a provider address that is not plain http(s) (another scheme, or
    credentials in it) or that reaches a cloud metadata service."""
    try:
        normalize_base_url(url)
    except ValueError as exc:
        raise BlockedAddressError(str(exc)) from exc
    await check_not_metadata(url)


def preset_settings(preset: CloudPreset) -> dict[str, Any]:
    """Provider settings a cloud preset starts with."""
    return {
        "preset": preset.id,
        "extra_headers": dict(preset.extra_headers) or None,
        "manual_models": list(preset.models) or None,
        "locality": "cloud",
        "model_filter": preset.model_filter or None,
    }


class ProviderManager(HealthChecks, WorkerSync, Benchmarking):
    """Providers' lifecycle: settings, clients, discovery, request counts.

    Health checks, worker links and benchmarks live in their own modules
    (provider_health, worker_sync, provider_benchmark).
    """

    _cached_prefixes: list[str] | None = None

    def __init__(self, db: Database, hub: WorkerHub | None = None):
        self._db = db
        self.worker_hub = hub if hub is not None else WorkerHub()
        self._worker_locks: dict[int, asyncio.Lock] = {}
        self.worker_hub.set_callbacks(
            self._on_worker_connected,
            self._on_worker_disconnected,
            self._on_worker_catalog,
        )
        # Live round-trip and request samples that feed the routing table.
        self.stats = ProviderStats()
        self.clients = ClientPool(self.worker_hub, self.stats)
        self._active_requests: dict[int, int] = defaultdict(int)
        # Gates of the providers with a max_concurrent limit; configured from
        # the provider rows each time the routing table reloads.
        self.capacity = CapacityLimits()
        self._hot_models: dict[int, list[HotModel]] = {}
        # The model each llama.cpp upstream reports as loaded (/props).
        self._llamacpp_loaded: dict[int, set[str]] = {}
        self._router: Router | None = None
        # When each provider last started or finished a proxied request, for
        # idle detection; providers with none count from process start.
        self._started_at = time.monotonic()
        self._last_activity: dict[int, float] = {}
        self._activity_listeners: list[Callable[[int], None]] = []
        self._health_task: asyncio.Task[None] | None = None
        # When request log retention last ran (monotonic); None: not yet.
        self._last_log_purge: float | None = None
        # Why a provider with a reachable address is still not usable, shown in
        # the dashboard. Cleared once discovery succeeds.
        self._status_detail: dict[int, str] = {}
        # Last offline reason written to the Log, so automatic checks log a
        # failure once rather than on every cycle.
        self._logged_detail: dict[int, str] = {}
        # Endpoint probes running in the background, one per provider.
        self._probing: dict[int, asyncio.Task[None]] = {}
        # Software version each worker last connected with, so a worker that
        # comes back upgraded (with new routes to relay) is probed again.
        self._worker_versions: dict[int, str | None] = {}
        # Why a provider a worker shared individually is offline while its
        # worker is still connected, so health checks keep saying so.
        self._child_reasons: dict[int, str] = {}
        # One catalog sync at a time per worker, so an older one cannot undo
        # a newer one.
        self._catalog_locks: dict[int, asyncio.Lock] = {}
        # When each provider a worker shared individually stopped being
        # shared (monotonic), and the timers that delete them once
        # worker_unshared_purge_hours have passed, one per worker.
        self._unshared_since: dict[int, float] = {}
        self._purge_timers: dict[int, asyncio.Task[None]] = {}
        # What those use to tell and wait for the time; replaced in tests.
        self.clock: Callable[[], float] = time.monotonic
        self.sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

    def attach_router(self, router: Router) -> None:
        self._router = router

    @property
    def router(self) -> Router | None:
        return self._router

    def _routing_changed(self, reason: str) -> None:
        """In-memory routing input changed (database writes notify on their own)."""
        if self._router is not None:
            self._router.mark_dirty(reason, reload=False)

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
                provider_id = provider.require_id()
                await self._db.update_provider_settings(
                    provider_id, **preset_settings(preset)
                )
                await self._db.add_address(provider_id, url, is_preferred=True)
                provider = await self._db.get_provider(provider_id) or provider
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
                except ApiKeyUnavailableError:
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

    async def _reencrypt_legacy_keys(self, providers: list[Provider]) -> None:
        """Move keys stored under the old fixed development key to the current one."""
        for prov in providers:
            if not prov.api_key_enc:
                continue
            fresh = reencrypt_legacy(prov.api_key_enc)
            if fresh is None:
                continue
            await self._db.set_provider_api_key(
                prov.require_id(), fresh, prov.api_key_header
            )
            prov.api_key_enc = fresh
            logger.info("Provider %s: re-encrypted its stored API key", prov.name)

    async def start(self) -> None:
        await self.provision_from_env()
        providers = await self._db.list_providers()
        await self._reencrypt_legacy_keys(providers)
        http_providers: list[Provider] = []
        for p in providers:
            p_id = p.require_id()
            if p.is_worker_child:
                self._alias_child(p)
            await self._rebuild_clients(p)
            if p.is_worker:
                # Workers come online when they dial in, not at startup.
                if not self.worker_hub.is_connected(p_id):
                    # Not an outage: its state is unknown until it dials in.
                    alerts.forget_provider(p_id)
                    await self._mark_worker_offline(p_id)
                continue
            http_providers.append(p)
        # Probe and discover now so the dashboard shows real state immediately
        # instead of whatever the database held from the last run.
        await self._check_many(http_providers, rediscover=True, source="startup")
        self._health_task = asyncio.create_task(self._health_check_loop())

    async def stop(self) -> None:
        if self._health_task:
            self._health_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._health_task
        for task in list(self._probing.values()):
            task.cancel()
        for task in list(self._probing.values()):
            with suppress(asyncio.CancelledError, Exception):
                await task
        self._probing.clear()
        for timer in list(self._purge_timers.values()):
            timer.cancel()
        self._purge_timers.clear()
        await self.clients.close_all()

    async def _rebuild_clients(self, provider: Provider) -> None:
        """Point the provider's clients at its best address (see ClientPool)."""
        addresses = await self._db.get_addresses(provider.require_id())
        await self.clients.rebuild(provider, addresses)

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
        await _check_provider_url(url)
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
        provider_id = provider.require_id()
        if provider_settings:
            await self._db.update_provider_settings(provider_id, **provider_settings)
            provider = await self._db.get_provider(provider_id) or provider
        if api_key:
            await self._store_api_key(provider, api_key, header)
        await self._db.add_address(provider_id, url, is_preferred=True)
        await self._rebuild_clients(provider)
        self.schedule_probe(provider_id, "add")
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
        provider_id = provider.require_id()
        if provider.is_worker:
            raise ValueError(
                "worker providers use the worker's own upstream key "
                "(LLAMA_ROUTER_WORKER_UPSTREAM_API_KEY)"
            )
        enc = encrypt_api_key(api_key.strip())
        stored_header = None if header in (None, DEFAULT_HEADER) else header
        await self._db.set_provider_api_key(provider_id, enc, stored_header)
        provider.api_key_enc = enc
        provider.api_key_header = stored_header

    async def delete_remote_model(self, provider_id: int, model_name: str) -> None:
        """Delete a model from the remote provider and re-discover.

        If the backend returns 404 (model already gone), the local model
        list is still refreshed and an ``httpx.HTTPStatusError`` is raised
        so callers can display a notice.
        """
        backend_name = await self._db.get_backend_model_name(provider_id, model_name)
        ollama = self.clients.ollama.get(provider_id)
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
        self.worker_hub.remove_child(provider_id)
        for child in await self._db.list_worker_children(provider_id):
            # A worker's individually shared providers go with it.
            await self.remove_provider(child.require_id())
        worker = await self._db.get_worker_by_provider(provider_id)
        if worker is not None:
            from .worker_link.protocol import CLOSE_REVOKED

            await self.worker_hub.disconnect(
                provider_id, CLOSE_REVOKED, "worker removed"
            )
            await self._db.delete_worker(int(worker["id"]))
        await self.clients.close(provider_id)
        self._active_requests.pop(provider_id, None)
        self.clients.active_urls.pop(provider_id, None)
        self._hot_models.pop(provider_id, None)
        self._llamacpp_loaded.pop(provider_id, None)
        self._status_detail.pop(provider_id, None)
        self._logged_detail.pop(provider_id, None)
        self._child_reasons.pop(provider_id, None)
        self._catalog_locks.pop(provider_id, None)
        self._unshared_since.pop(provider_id, None)
        timer = self._purge_timers.pop(provider_id, None)
        if timer is not None and timer is not asyncio.current_task():
            timer.cancel()
        self.stats.forget_provider(provider_id)
        await self._db.remove_provider(provider_id)

    # --- Address CRUD ---

    async def add_address(
        self,
        provider_id: int,
        url: str,
        is_preferred: bool = False,
    ) -> ProviderAddress:
        await _check_provider_url(url)
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
        await _check_provider_url(url)
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
            queued_requests=self._queued(provider_id),
            hot_models=self._hot_models.get(provider_id, []),
            worker=self.worker_summary(provider),
            status_detail=self._reported_detail(provider),
        )

    def hub_loopback_ids(self) -> set[int]:
        """Remotes dialing in from the hub this router serves as a worker.

        Sharing them back would send the hub's own jobs to itself, so the hub
        never sees them, whatever ``hub_shared`` says.
        """
        hub = router_identity.hub_id()
        if hub is None:
            return set()
        parents = {
            c.provider_id
            for c in self.worker_hub.connections()
            if not c.closed and c.router_id == hub
        }
        return parents | {
            child for parent in parents for child in self.worker_hub.children_of(parent)
        }

    def shares_with_hub(self, provider: Provider) -> bool:
        return provider.hub_shared and provider.id not in self.hub_loopback_ids()

    def hub_offers(self, providers: list[Provider]) -> list[Provider]:
        """Which of *providers* to list for a hub that sees them one by one.

        A remote that shares its own providers one by one is replaced by
        them. Those rows are offered only while that remote serves them: one
        that went back to sharing as one provider, or is not connected,
        leaves rows behind that nothing can reach.
        """
        has_children = {p.worker_parent_id for p in providers if p.is_worker_child}
        out: list[Provider] = []
        for prov in providers:
            if prov.id is None or not self.shares_with_hub(prov):
                continue
            if prov.is_worker_child:
                if not self.worker_hub.is_connected(prov.id):
                    continue
            elif prov.is_worker:
                if self.shares_individually(prov):
                    continue
                if prov.id in has_children and not self.worker_hub.is_connected(
                    prov.id
                ):
                    # Unknown until it reconnects whether it will share as
                    # one provider or as these; it serves neither meanwhile.
                    continue
            out.append(prov)
        return out

    def worker_summary(self, provider: Provider) -> dict[str, Any] | None:
        """Live socket details for a worker provider; None for HTTP providers."""
        if not provider.is_worker or provider.id is None:
            return None
        conn = self.worker_hub.get(provider.id)
        if conn is None:
            return {"connected": False}
        return conn.summary()

    async def list_provider_infos(self) -> list[ProviderInfo]:
        # Four queries however many providers there are, grouped here; each
        # bulk query keeps the order the per-provider one used.
        providers = await self._db.list_providers()
        models: dict[int, list[ProviderModel]] = {}
        for m in await self._db.get_all_provider_models():
            models.setdefault(m.provider_id, []).append(m)
        addresses: dict[int, list[ProviderAddress]] = {}
        for a in await self._db.get_all_addresses():
            addresses.setdefault(a.provider_id, []).append(a)
        benchmarks: dict[int, list[BenchmarkResult]] = {}
        for b in await self._db.get_all_benchmark_results():
            benchmarks.setdefault(b.provider_id, []).append(b)
        infos = []
        for p in providers:
            p_id = p.require_id()
            infos.append(
                ProviderInfo(
                    provider=p,
                    models=models.get(p_id, []),
                    benchmarks=benchmarks.get(p_id, []),
                    addresses=addresses.get(p_id, []),
                    active_requests=self._active_requests.get(p_id, 0),
                    queued_requests=self._queued(p_id),
                    hot_models=self._hot_models.get(p_id, []),
                    worker=self.worker_summary(p),
                    status_detail=self._reported_detail(p),
                )
            )
        return infos

    def get_ollama_client(self, provider_id: int) -> OllamaClient:
        return self.clients.ollama[provider_id]

    def get_llamacpp_client(self, provider_id: int) -> LlamaCppClient:
        return self.clients.llamacpp[provider_id]

    def find_ollama_client(self, provider_id: int) -> OllamaClient | None:
        return self.clients.ollama.get(provider_id)

    def find_llamacpp_client(self, provider_id: int) -> LlamaCppClient | None:
        return self.clients.llamacpp.get(provider_id)

    def ollama_api_client(self, provider_id: int) -> OllamaClient | OllamaOverOpenAI:
        """What the Ollama API handlers call: the provider's Ollama client, or
        for an OpenAI-style provider, one that translates (see translate).

        Raises NoReachableAddressError when the provider has neither.
        """
        client = self.clients.ollama.get(provider_id)
        if client is not None:
            return client
        lcpp = self.clients.llamacpp.get(provider_id)
        if lcpp is None:
            from .v1_client import NoReachableAddressError

            raise NoReachableAddressError()
        from .translate import OllamaOverOpenAI

        return OllamaOverOpenAI(lcpp)

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
                "release() called on provider %d but active_requests=%d "
                "— possible double-release",
                provider_id,
                current,
            )
            return
        self._active_requests[provider_id] = current - 1
        self._last_activity[provider_id] = time.monotonic()

    async def take_slot(
        self,
        provider_id: int,
        deadline: float | None = None,
        level: int = DEFAULT_LEVEL,
    ) -> SlotGate | None:
        """Take one of the provider's ``max_concurrent`` slots, waiting in
        line at *level* (see :func:`~.capacity.queue_level`) when there is
        none (:class:`ProviderAtCapacity` when the wait runs past the queue
        timeout or *deadline*).

        Returns the gate to release, or None for a provider without a limit.
        Never suspends when a slot is free.
        """
        gate = self.capacity.gate(provider_id)
        if gate is None:
            return None
        try:
            waited = await gate.acquire(deadline, level)
        except QueueTimeout:
            logger.info(
                "Provider %d at capacity (%d): request at queue level %d "
                "timed out waiting",
                provider_id,
                gate.limit,
                level,
            )
            raise ProviderAtCapacity(provider_id, gate.limit) from None
        if waited > 0:
            logger.debug(
                "Request waited %.3fs at queue level %d for provider %d",
                waited,
                level,
                provider_id,
            )
        return gate

    @asynccontextmanager
    async def acquire_provider(
        self,
        provider_id: int,
        *,
        deadline: float | None = None,
        slot: SlotGate | None = None,
        slot_taken: bool = False,
        level: int = DEFAULT_LEVEL,
    ) -> AsyncIterator[None]:
        """Take a slot on the provider (see :meth:`take_slot`), then count
        the request against it for its duration.

        Pass *slot_taken* with the gate :meth:`take_slot` returned (*slot*)
        when the slot was taken beforehand; this block then releases it.

        The provider's clients are kept open until it ends even if an edit
        replaces them, so fetch the client inside this block, not before.
        """
        if not slot_taken:
            slot = await self.take_slot(provider_id, deadline, level)
        try:
            pinned = self.clients.pin(provider_id)
            self.acquire(provider_id)
        except BaseException:
            if slot is not None:
                slot.release()
            raise
        try:
            yield
        finally:
            self.release(provider_id)
            if slot is not None:
                slot.release()
            await self.clients.unpin(pinned)

    def _queued(self, provider_id: int) -> int:
        gate = self.capacity.gate(provider_id)
        return gate.queued if gate is not None else 0

    def has_room(self, provider_id: int, level: int = DEFAULT_LEVEL) -> bool:
        """A request at *level* sent to the provider now would not wait."""
        return self.capacity.has_room(provider_id, level)

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
        provider_id = provider.require_id()
        client = self.clients.llamacpp.get(provider_id)
        if not provider.is_openai or client is None or provider.lacks("/props"):
            # Only llama.cpp-style servers report their loaded model in /props.
            self._set_llamacpp_loaded(provider_id, set())
            return
        try:
            current = self._extract_llamacpp_current_model(await client.get_props())
        except Exception as exc:
            logger.debug(
                "Failed to read /props for provider %s: %s", provider.name, exc
            )
            current = None
        self._set_llamacpp_loaded(
            provider_id, {self._strip_cache_prefix(current)} if current else set()
        )

    async def _refresh_ollama_hot(self, provider: Provider) -> None:
        provider_id = provider.require_id()
        if not provider.is_ollama or provider_id not in self.clients.ollama:
            self._set_hot_models(provider_id, [])
            return
        try:
            raw = await self.clients.ollama[provider_id].get_ps()
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
            self._set_hot_models(provider_id, hot)
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

    def _reported_detail(self, provider: Provider) -> str | None:
        """The reason shown with a provider; an offline one always has one.

        The fallback names no address: it is shown to non-admins too.
        """
        pid = provider.require_id()
        detail = self._status_detail.get(pid)
        if detail is None and provider.status == ProviderStatus.OFFLINE:
            if provider.is_worker and not self.worker_hub.is_connected(pid):
                return self._child_reasons.get(pid, "worker is not connected")
            return "no address responded"
        return detail

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
        provider_id = provider.require_id()
        if self.shares_individually(provider):
            # Its models belong to its providers; the worker row has none.
            await self._db.set_provider_models(provider_id, [])
            self._hot_models.pop(provider_id, None)
            return
        models_by_name: dict[str, ProviderModel] = {}
        cache_prefixed = 0

        def _upsert_model(
            *,
            clean_name: str,
            raw_name: str | None,
            size: int | None = None,
            digest: str | None = None,
            modified_at: str | None = None,
            details: dict[str, Any] | None = None,
        ) -> None:
            existing = models_by_name.get(clean_name)
            if existing is None:
                models_by_name[clean_name] = ProviderModel(
                    provider_id=provider_id,
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

        async def _listed(
            fetch: Callable[[], Awaitable[list[ProviderModel]]],
        ) -> list[ProviderModel]:
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

        if provider.is_ollama and provider.id in self.clients.ollama:
            tags = await _listed(self.clients.ollama[provider.id].get_tags)
            tags = [m for m in tags if provider.keeps_model(m.name)]
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

        if provider.is_openai and provider.id in self.clients.llamacpp:
            lcpp_models = await _listed(self.clients.llamacpp[provider.id].get_models)
            lcpp_models = [m for m in lcpp_models if provider.keeps_model(m.name)]
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
                props = await self.clients.llamacpp[provider.id].get_props()
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
                timeout=timeout, headers=upstream_headers(provider)
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
        await self._db.set_provider_models(provider_id, all_models)
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
    def _extract_llamacpp_current_model(props: dict[str, Any] | None) -> str | None:
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
