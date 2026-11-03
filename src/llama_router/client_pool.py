"""The HTTP clients each provider is reached through.

``ClientPool`` holds one client per provider (Ollama or OpenAI-style), picks
the address it points at, adds the upstream's auth headers, and closes a
replaced client only once the last request using it has finished.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

import httpx

from . import router_identity
from .llamacpp_client import LlamaCppClient
from .models import Provider, ProviderAddress
from .ollama_client import OllamaClient
from .provider_secrets import ApiKeyUnavailableError, auth_headers, decrypt_api_key
from .routing_stats import ProviderStats, retry_after_seconds
from .worker_link.hub import WorkerHub

logger = logging.getLogger(__name__)

type BackendClient = OllamaClient | LlamaCppClient


def best_address(addresses: list[ProviderAddress]) -> ProviderAddress | None:
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


def address_url_sig(addr: ProviderAddress) -> str:
    return addr.url.rstrip("/")


def upstream_headers(provider: Provider) -> dict[str, str] | None:
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
        except ApiKeyUnavailableError as exc:
            logger.warning("Provider %s: %s", provider.name, exc)
        else:
            headers.update(auth_headers(key, provider.api_key_header))
    return headers or None


class ClientPool:
    """Each provider's current client, and the replaced ones still in use."""

    def __init__(self, hub: WorkerHub, stats: ProviderStats):
        self._hub = hub
        self._stats = stats
        self.ollama: dict[int, OllamaClient] = {}
        self.llamacpp: dict[int, LlamaCppClient] = {}
        # The address URL each provider's clients point at.
        self.active_urls: dict[int, str] = {}
        # Requests using each client right now (see acquire_provider), and the
        # clients a rebuild replaced while still in use: those close when
        # their last request finishes, not mid-request.
        self._users: dict[BackendClient, int] = {}
        self._retired: set[BackendClient] = set()

    async def rebuild(
        self, provider: Provider, addresses: list[ProviderAddress]
    ) -> None:
        """Replace the provider's clients with ones for the best live address.

        Requests already running keep the old client until they finish.
        """
        provider_id = provider.require_id()
        addr = best_address(addresses)
        if not addr:
            await self.close(provider_id)
            self.active_urls.pop(provider_id, None)
            return
        kw: dict[str, Any] = {"on_response": self._response_hook(provider_id)}
        if not provider.is_cloud:
            # Any local provider may be another llama-router; a cloud
            # service never is, so it is not told this router's ID.
            kw["on_request"] = router_identity.add_path_header
        if provider.read_timeout_s:
            kw["timeout"] = httpx.Timeout(10.0, read=provider.read_timeout_s)
        if provider.is_worker:
            kw["transport"] = self._hub.transport(provider_id, provider.worker_target)
        else:
            kw["headers"] = upstream_headers(provider)
        # A worker's address only names it; the transport picks the socket,
        # and a path in it (worker://<name>/<id>) must not reach the worker.
        base = "http://worker" if provider.is_worker else addr.url
        new: BackendClient
        if provider.is_ollama:
            new = OllamaClient(base, **kw)
        else:
            new = LlamaCppClient(base, **kw)
        await self.close(provider_id)
        if isinstance(new, OllamaClient):
            self.ollama[provider_id] = new
        else:
            self.llamacpp[provider_id] = new
        self.active_urls[provider_id] = address_url_sig(addr)

    def _response_hook(
        self, provider_id: int
    ) -> Callable[[httpx.Response], Awaitable[None]]:
        """Every upstream response passes here: a 429 marks the provider busy."""

        async def on_response(resp: httpx.Response) -> None:
            if resp.status_code == 429:
                seconds = retry_after_seconds(resp.headers.get("retry-after"))
                self._stats.mark_busy(provider_id, seconds)
                logger.info(
                    "Provider %d rate-limited (HTTP 429); routing around it for %.0fs",
                    provider_id,
                    min(seconds, 300.0),
                )

        return on_response

    async def close(self, provider_id: int) -> None:
        """Drop the provider's clients; each closes once no request uses it."""
        ollama = self.ollama.pop(provider_id, None)
        if ollama is not None:
            await self._retire(ollama)
        lcpp = self.llamacpp.pop(provider_id, None)
        if lcpp is not None:
            await self._retire(lcpp)

    async def close_all(self) -> None:
        for client in [
            *self.ollama.values(),
            *self.llamacpp.values(),
            *self._retired,
        ]:
            await client.close()
        self._retired.clear()

    async def _retire(self, client: BackendClient) -> None:
        if self._users.get(client, 0) > 0:
            self._retired.add(client)
        else:
            await client.close()

    def pin(self, provider_id: int) -> list[BackendClient]:
        pinned: list[BackendClient] = []
        ollama = self.ollama.get(provider_id)
        if ollama is not None:
            pinned.append(ollama)
        lcpp = self.llamacpp.get(provider_id)
        if lcpp is not None:
            pinned.append(lcpp)
        for client in pinned:
            self._users[client] = self._users.get(client, 0) + 1
        return pinned

    async def unpin(self, pinned: list[BackendClient]) -> None:
        for client in pinned:
            left = self._users.get(client, 0) - 1
            if left > 0:
                self._users[client] = left
                continue
            self._users.pop(client, None)
            if client in self._retired:
                self._retired.discard(client)
                with suppress(Exception):
                    await client.close()
