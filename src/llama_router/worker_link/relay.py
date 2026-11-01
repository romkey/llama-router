"""Telling jobs relayed by this router's own worker link from client requests.

A worker relays the hub's jobs to its upstream, usually this router's own API
on 127.0.0.1, marked with ``HUB_HEADER`` (and ``PROVIDER_HEADER``). Anyone
who can reach the API could send those headers too, so they count only when
they come with ``RELAY_HEADER`` holding a secret made fresh in each process,
which only the worker link in this same process knows.

:class:`HubRelayMiddleware` checks that, records the result in the ASGI scope
under ``SCOPE_KEY`` and strips all three headers from every request, so the
routes never see them and the secret is never logged.
"""

from __future__ import annotations

import hmac
import ipaddress
import secrets
from urllib.parse import urlsplit

from starlette.types import ASGIApp, Receive, Scope, Send

from .protocol import HUB_HEADER, HUB_HEADER_VALUE, PROVIDER_HEADER

RELAY_HEADER = "X-Llama-Router-Relay"
# Set in the scope of a request this process's worker link relayed: a dict
# with "provider", the one provider the job is for, or None.
SCOPE_KEY = "llama_router.hub_relay"

_SECRET = secrets.token_urlsafe(32)
_STRIP = frozenset(
    h.lower().encode("latin-1") for h in (HUB_HEADER, PROVIDER_HEADER, RELAY_HEADER)
)


def relay_secret() -> str:
    """The secret the worker link sends in ``RELAY_HEADER``."""
    return _SECRET


def is_loopback_url(url: str) -> bool:
    """*url* points at this machine, where the secret may be sent."""
    host = (urlsplit(url).hostname or "").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class HubRelayMiddleware:
    """Mark requests relayed by this process's worker link; strip the headers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        via = provider = relay = ""
        kept: list[tuple[bytes, bytes]] = []
        for k, v in scope.get("headers") or ():
            name = k.lower()
            if name not in _STRIP:
                kept.append((k, v))
                continue
            value = v.decode("latin-1").strip()
            if name == HUB_HEADER.lower().encode():
                via = value
            elif name == PROVIDER_HEADER.lower().encode():
                provider = value
            else:
                relay = value
        scope = dict(scope)
        scope["headers"] = kept
        scope.pop(SCOPE_KEY, None)
        if (
            relay
            and hmac.compare_digest(relay.encode(), _SECRET.encode())
            and via.lower() == HUB_HEADER_VALUE
        ):
            scope[SCOPE_KEY] = {"provider": _provider_id(provider)}
        await self.app(scope, receive, send)


def _provider_id(raw: str) -> int | None:
    if raw.isascii() and raw.isdigit() and len(raw) <= 18:
        return int(raw)
    return None
