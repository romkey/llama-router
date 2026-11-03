"""Workers on the hub: connect, disconnect, and the providers they share."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from .config import settings
from .models import WORKER_URL_SCHEME, Provider, ProviderProtocol, ProviderStatus
from .provider_health import _PROBE_TIMEOUT, _failure_reason
from .worker_link.hub import WorkerConnection
from .worker_link.protocol import EXPOSE_INDIVIDUAL

if TYPE_CHECKING:
    from .client_pool import ClientPool
    from .database import Database
    from .worker_link.hub import WorkerHub

logger = logging.getLogger(__name__)

# A failed sync from the purge timer is tried again after this long.
_PURGE_RETRY_S = 60.0


def worker_protocol(targets: list[str]) -> ProviderProtocol:
    """The one protocol a worker provider is reached by.

    A worker that relays its own router offers both targets, and the router's
    Ollama port also serves every OpenAI-style route, so the Ollama target
    reaches everything. Use the OpenAI target only when it is all there is.
    """
    if "ollama" in targets or "llamacpp" not in targets:
        return ProviderProtocol.OLLAMA
    return ProviderProtocol.OPENAI


class WorkerSync:
    """ProviderManager's handling of worker links and their catalogs."""

    if TYPE_CHECKING:
        # Set up and provided by ProviderManager.
        _db: Database
        clients: ClientPool
        worker_hub: WorkerHub
        _worker_locks: dict[int, asyncio.Lock]
        _status_detail: dict[int, str]
        _logged_detail: dict[int, str]
        _child_reasons: dict[int, str]
        _worker_versions: dict[int, str | None]
        _catalog_locks: dict[int, asyncio.Lock]
        _unshared_since: dict[int, float]
        _purge_timers: dict[int, asyncio.Task[None]]
        clock: Callable[[], float]
        sleep: Callable[[float], Awaitable[None]]

        def schedule_probe(self, provider_id: int, reason: str = "auto") -> None: ...
        def _clear_loaded(self, provider_id: int) -> None: ...
        async def _rebuild_clients(self, provider: Provider) -> None: ...
        async def _discover_provider(self, provider: Provider) -> None: ...
        async def remove_provider(self, provider_id: int) -> None: ...
        async def _log_event(
            self,
            provider: Provider,
            endpoint: str,
            *,
            ok: bool,
            detail: str | None = None,
            duration_ms: float = 0.0,
        ) -> None: ...

    # --- Workers ---

    async def _probe_worker(self, provider: Provider) -> bool:
        """A worker is live when its socket is up and its upstream answers."""
        provider_id = provider.require_id()
        if not self.worker_hub.is_connected(provider_id):
            return False
        if provider_id not in self.clients.ollama and (
            provider_id not in self.clients.llamacpp
        ):
            await self._rebuild_clients(provider)
        ollama = self.clients.ollama.get(provider_id)
        if ollama is not None and await ollama.is_reachable(timeout=_PROBE_TIMEOUT):
            return True
        lcpp = self.clients.llamacpp.get(provider_id)
        return lcpp is not None and await lcpp.is_reachable(timeout=_PROBE_TIMEOUT)

    async def add_worker_provider(self, name: str) -> Provider:
        """Create the provider record a worker attaches to when it dials in."""
        provider = await self._db.add_provider(
            name, ProviderProtocol.OLLAMA, worker_name=name
        )
        provider_id = provider.require_id()
        url = f"{WORKER_URL_SCHEME}{name}"
        await self._db.add_address(provider_id, url, is_preferred=True)
        await self._db.update_provider_status(provider_id, ProviderStatus.OFFLINE)
        provider.status = ProviderStatus.OFFLINE
        return provider

    def _worker_lock(self, provider_id: int) -> asyncio.Lock:
        lock = self._worker_locks.get(provider_id)
        if lock is None:
            lock = self._worker_locks[provider_id] = asyncio.Lock()
        return lock

    def _is_current_worker(self, conn: WorkerConnection) -> bool:
        return self.worker_hub.get(conn.provider_id) is conn and not conn.closed

    def shares_individually(self, provider: Provider) -> bool:
        """A worker's own row while it shares its providers one by one."""
        if not provider.is_worker or provider.is_worker_child or provider.id is None:
            return False
        conn = self.worker_hub.get(provider.id)
        return conn is not None and conn.expose == EXPOSE_INDIVIDUAL

    def _alias_child(self, child: Provider) -> None:
        if child.worker_parent_id is not None and child.worker_remote_id is not None:
            self.worker_hub.add_child(
                child.require_id(), child.worker_parent_id, child.worker_remote_id
            )

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
            await self._bring_worker_online(conn, provider)
        if conn.expose != EXPOSE_INDIVIDUAL:
            # Shared as one provider now: its individual ones are unreachable.
            for child in await self._db.list_worker_children(provider_id):
                await self._take_child_offline(child, "worker shares as one provider")

    async def _bring_worker_online(
        self, conn: WorkerConnection, provider: Provider
    ) -> None:
        """Discover *provider* (the worker, or one it shares) through *conn*.

        The caller holds the provider's worker lock.
        """
        provider_id = provider.require_id()
        self._child_reasons.pop(provider_id, None)
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
        if self.shares_individually(provider):
            # Nothing to probe: jobs go to the providers it shares.
            return
        known = provider_id in self._worker_versions
        previous = self._worker_versions.get(provider_id)
        self._worker_versions[provider_id] = conn.software_version
        if reachable and (
            not known or previous != conn.software_version or provider.probed_at is None
        ):
            # What the worker relays depends on its version and upstream.
            self.schedule_probe(provider_id, "worker connected")

    async def _on_worker_catalog(self, conn: WorkerConnection) -> None:
        """Match the providers on the hub to the ones the worker shares.

        Syncs for one worker run one at a time, each from the latest catalog,
        so a slow older sync can never undo what a newer one did.
        """
        parent_id = conn.provider_id
        lock = self._catalog_locks.setdefault(parent_id, asyncio.Lock())
        async with lock:
            await self._sync_catalog(conn)

    async def _sync_catalog(self, conn: WorkerConnection) -> None:
        parent_id = conn.provider_id
        async with self._worker_lock(parent_id):
            # A newer frame replaces conn.catalog; this sync works from the
            # list it started with throughout (the next sync applies the new).
            catalog = conn.catalog
            if not self._is_current_worker(conn) or catalog is None:
                return
            parent = await self._db.get_provider(parent_id)
            if parent is None or not parent.is_worker or parent.worker_name is None:
                return
            existing = {
                c.worker_remote_id: c
                for c in await self._db.list_worker_children(parent_id)
            }
            # Providers the worker no longer shares give up their names first,
            # so one shared in their place can have <worker>/<provider>.
            shared = {int(e["id"]) for e in catalog}
            for rid, old in existing.items():
                if rid in shared or old.name.endswith(f"#{rid}"):
                    continue
                aside = f"{old.name}#{rid}"
                if await self._name_free(aside, old.id):
                    await self._rename(old, aside)
            wanted: list[Provider] = []
            # Children that could not take their name yet (another provider
            # had it, maybe a sibling renamed later in this same catalog).
            retry: list[tuple[Provider, str]] = []
            for entry in catalog:
                rid = int(entry["id"])
                name = f"{parent.name}/{entry['name']}"
                try:
                    child = await self._sync_child(
                        parent, existing.pop(rid, None), entry, name
                    )
                except Exception as exc:
                    logger.warning(
                        "Worker %s: could not add or update %s: %s",
                        parent.name,
                        name,
                        exc,
                    )
                    continue
                if child is None:
                    continue
                self._unshared_since.pop(child.require_id(), None)
                if child.name != name:
                    retry.append((child, name))
                self._alias_child(child)
                if child.status in (ProviderStatus.OFFLINE, ProviderStatus.UNKNOWN):
                    wanted.append(child)
            for child, name in retry:
                if await self._name_free(name, child.id):
                    await self._rename(child, name)
            gone, purge = self._purge_unshared(list(existing.values()), len(catalog))
        for child in purge:
            logger.info(
                "Worker %s: deleting %s, no longer shared by the worker",
                parent.name,
                child.name,
            )
            await self.remove_provider(child.require_id())
        for child in gone:
            self.worker_hub.remove_child(child.require_id())
            await self._take_child_offline(child, "no longer shared by the worker")
        grace = settings.worker_unshared_purge_hours * 3600.0
        if gone and grace > 0:
            # Due when the longest unshared one reaches the grace period.
            oldest = min(self._unshared_since[c.require_id()] for c in gone)
            self._schedule_purge(conn, oldest + grace - self.clock())
        for child in wanted:
            async with self._worker_lock(child.require_id()):
                if self.worker_hub.get(child.require_id()) is conn:
                    await self._bring_worker_online(conn, child)

    def _purge_unshared(
        self, unshared: list[Provider], shared: int
    ) -> tuple[list[Provider], list[Provider]]:
        """Split a worker's unshared providers into those to keep (offline)
        and those to delete: unshared for longer than
        worker_unshared_purge_hours, and the longest unshared beyond
        worker_max_shared_providers, so a worker cannot pile up rows."""
        now = self.clock()
        for child in unshared:
            self._unshared_since.setdefault(child.require_id(), now)
        unshared.sort(key=lambda c: self._unshared_since[c.require_id()])
        grace = settings.worker_unshared_purge_hours * 3600.0
        excess = max(0, shared + len(unshared) - settings.worker_max_shared_providers)
        keep: list[Provider] = []
        purge: list[Provider] = []
        for i, child in enumerate(unshared):
            since = self._unshared_since[child.require_id()]
            if i < excess or (grace > 0 and now - since >= grace):
                purge.append(child)
            else:
                keep.append(child)
        return keep, purge

    def _schedule_purge(self, conn: WorkerConnection, delay: float) -> None:
        """Sync the worker's catalog again in *delay* seconds, when its
        longest unshared provider is due for deletion. That sync deletes it
        and schedules the next. One timer per worker (a running one is due no
        later, as providers unshared since are due after it); it uses
        whichever socket is current when it fires, so a reconnect in between
        does not lose it."""
        parent_id = conn.provider_id
        timer = self._purge_timers.get(parent_id)
        if timer is not None and not timer.done():
            return

        async def later() -> None:
            await self.sleep(max(0.0, delay) + 1.0)
            if self._purge_timers.get(parent_id) is asyncio.current_task():
                del self._purge_timers[parent_id]
            current = self.worker_hub.get(parent_id)
            if current is None or not self._is_current_worker(current):
                return
            try:
                await self._on_worker_catalog(current)
            except Exception:
                # The worker may not send its catalog again; try later.
                logger.exception(
                    "Could not delete unshared providers of worker %d", parent_id
                )
                self._schedule_purge(current, _PURGE_RETRY_S)

        self._purge_timers[parent_id] = asyncio.create_task(
            later(), name=f"purge-unshared-{parent_id}"
        )

    async def _name_free(self, name: str, own_id: int | None) -> bool:
        holder = await self._db.get_provider_by_name(name)
        return holder is None or holder.id == own_id

    async def _child_name(self, name: str, rid: int, own_id: int | None) -> str | None:
        """*name*, or ``<name>#<id on the worker>`` when another provider has it."""
        for candidate in (name, f"{name}#{rid}"):
            if await self._name_free(candidate, own_id):
                return candidate
        return None

    async def _rename(self, child: Provider, name: str) -> None:
        await self._db.update_provider(
            child.require_id(),
            name,
            machine_type=child.machine_type,
            gpu_type=child.gpu_type,
            gpu_ram=child.gpu_ram,
        )
        child.name = name

    async def _sync_child(
        self,
        parent: Provider,
        child: Provider | None,
        entry: dict[str, Any],
        name: str,
    ) -> Provider | None:
        """Create or update the hub's row for one catalog entry."""
        rid = int(entry["id"])
        parent_id = parent.require_id()
        free = await self._child_name(name, rid, child.id if child else None)
        if free is None:
            logger.warning(
                "Worker %s shares %s, but providers named %s and %s#%d already "
                "exist here; rename one of them",
                parent.name,
                entry["name"],
                name,
                name,
                rid,
            )
        if child is None:
            if free is None:
                return None
            child = await self._db.add_provider(
                free,
                parent.protocol,
                worker_name=f"{parent.worker_name}/{rid}",
                worker_parent_id=parent_id,
                worker_remote_id=rid,
            )
            await self._db.add_address(
                child.require_id(),
                f"{WORKER_URL_SCHEME}{parent.worker_name}/{rid}",
                is_preferred=True,
            )
            logger.info("Worker %s now shares %s", parent.name, free)
        elif free is not None and child.name != free:
            await self._rename(child, free)
        if child.locality != entry["locality"]:
            await self._db.update_provider_settings(
                child.require_id(), locality=entry["locality"]
            )
            child.locality = entry["locality"]
            await self._rebuild_clients(child)
        return child

    async def _take_child_offline(self, child: Provider, detail: str) -> None:
        child_id = child.require_id()
        async with self._worker_lock(child_id):
            if self.worker_hub.is_connected(child_id):
                return
            await self._mark_worker_offline(child_id)
            self._status_detail[child_id] = detail
            self._logged_detail[child_id] = detail
            self._child_reasons[child_id] = detail

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
        for child in await self._db.list_worker_children(provider_id):
            child_id = child.require_id()
            async with self._worker_lock(child_id):
                if not self.worker_hub.is_connected(child_id):
                    await self._mark_worker_offline(child_id)
                    self._child_reasons.pop(child_id, None)
                    self._logged_detail[child_id] = "worker is not connected"

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
