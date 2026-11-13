"""Health checks: the periodic loop, address liveness, endpoint probes."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from .client_pool import address_url_sig, best_address, upstream_headers
from .config import settings
from .httpx_errors import describe_httpx_error
from .llamacpp_client import LlamaCppClient
from .models import (
    Provider,
    ProviderAddress,
    ProviderStatus,
    RequestLog,
)
from .ollama_client import OllamaClient
from .probe import merge_results, probe_url

if TYPE_CHECKING:
    from .client_pool import ClientPool
    from .database import Database
    from .routing_stats import ProviderStats
    from .worker_link.hub import WorkerHub

logger = logging.getLogger(__name__)

# Request log retention runs at most this often.
LOG_PURGE_INTERVAL_S = 3600.0
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
    if isinstance(exc, httpx.RequestError):
        # The path only: the full URL can carry credentials and internal
        # hostnames, and this text reaches non-admins on the dashboard.
        # TLS and transport errors often repeat the host, so drop it too.
        try:
            url = exc.request.url
        except RuntimeError:
            return type(exc).__name__
        text = str(exc)
        for secret in (str(url), url.netloc.decode("ascii", "replace"), url.host):
            if secret:
                text = text.replace(secret, "(provider)")
        return f"{type(exc).__name__} for {url.path}: {text}".rstrip(": ")
    if isinstance(exc, httpx.HTTPError):
        return type(exc).__name__
    return str(exc) or type(exc).__name__


class HealthChecks:
    """ProviderManager's health checks and endpoint probes."""

    if TYPE_CHECKING:
        # Set up and provided by ProviderManager.
        _db: Database
        clients: ClientPool
        worker_hub: WorkerHub
        stats: ProviderStats
        _active_requests: dict[int, int]
        _status_detail: dict[int, str]
        _logged_detail: dict[int, str]
        _child_reasons: dict[int, str]
        _probing: dict[int, asyncio.Task[None]]
        _last_log_purge: float | None

        def _worker_lock(self, provider_id: int) -> asyncio.Lock: ...
        def _clear_loaded(self, provider_id: int) -> None: ...
        async def _rebuild_clients(self, provider: Provider) -> None: ...
        async def _discover_provider(self, provider: Provider) -> None: ...
        async def _refresh_hot_models(self, provider: Provider) -> None: ...
        async def _probe_worker(self, provider: Provider) -> bool: ...

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
            at = at.replace(tzinfo=UTC)
        age = (datetime.now(UTC) - at).total_seconds()
        return age >= max_age * 3600

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
                    provider_id=provider.require_id(),
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
        provider.require_id()  # a provider that was never saved cannot be checked
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
        pid = provider.require_id()
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
            except TimeoutError:
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
        pid = provider.require_id()
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
        for addr, reachable in zip(addresses, results, strict=False):
            addr_id = addr.require_id()
            if addr.is_live != reachable:
                await self._db.set_address_live(addr_id, reachable)
            addr.is_live = reachable
        live = [a for a in addresses if a.is_live]
        if addresses:
            self.stats.record_check(pid, ok=bool(live))

        detail: str | None = None
        if not live:
            status = ProviderStatus.OFFLINE
            if provider.is_worker and not self.worker_hub.is_connected(pid):
                detail = self._child_reasons.get(pid, "worker is not connected")
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
        pid = provider.require_id()
        prev_detail = self._logged_detail.get(pid)

        # A provider its worker stopped sharing keeps saying why, though it
        # has no live address.
        if (
            detail
            and status == ProviderStatus.OFFLINE
            and (live or pid in self._child_reasons)
        ):
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
        pid = provider.require_id()
        best = best_address(addresses)
        if best is None:
            return
        has_clients = pid in self.clients.ollama or pid in self.clients.llamacpp
        new_sig = address_url_sig(best)
        if has_clients and (
            provider.is_worker or new_sig == self.clients.active_urls.get(pid)
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
            try:
                await self.purge_request_logs()
            except Exception:
                logger.exception("Request log retention failed")

    async def purge_request_logs(self, *, force: bool = False) -> int:
        """Delete request log rows older than LLAMA_ROUTER_REQUEST_LOG_RETENTION_DAYS.

        Runs from the health-check loop at most once per hour (every call
        with *force*). Returns the number of rows deleted.
        """
        days = settings.request_log_retention_days
        audit_days = settings.audit_retention_days
        if days <= 0 and audit_days <= 0:
            return 0
        now = time.monotonic()
        last = self._last_log_purge
        if not force and last is not None and now - last < LOG_PURGE_INTERVAL_S:
            return 0
        self._last_log_purge = now
        if audit_days > 0:
            gone = await self._db.delete_audit_entries_before(
                datetime.now(UTC) - timedelta(days=audit_days)
            )
            if gone:
                logger.info(
                    "Deleted %d audit entries older than %d days", gone, audit_days
                )
        if days <= 0:
            return 0
        cutoff = datetime.now(UTC) - timedelta(days=days)
        deleted = await self._db.delete_request_logs_before(cutoff)
        if deleted:
            logger.info("Deleted %d request log rows older than %g days", deleted, days)
        return deleted

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
                    timeout=_PROBE_TIMEOUT, headers=upstream_headers(provider)
                ) as http:
                    resp = await http.get(addr.url.rstrip("/") + liveness)
                return 200 <= resp.status_code < 300
            except httpx.HTTPError:
                return False
        client_cls = OllamaClient if provider.is_ollama else LlamaCppClient
        tmp = client_cls(
            addr.url,
            timeout=_PROBE_TIMEOUT,
            headers=upstream_headers(provider),
        )
        try:
            return await tmp.is_reachable()
        finally:
            await tmp.close()

    # --- Endpoint probing ---

    async def probe_provider(self, provider_id: int) -> dict[str, Any] | None:
        """Probe the provider's best address and store what its upstream supports.

        Workers are probed through their socket.
        """
        provider = await self._db.get_provider(provider_id)
        if provider is None:
            return None
        addresses = await self._db.get_addresses(provider_id)
        best = best_address(addresses)
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
                    headers=upstream_headers(provider),
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
