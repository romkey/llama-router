"""Proof that an API request was relayed by this process's own worker link.

A worker replays hub jobs to its router's API ports on 127.0.0.1, without the
hub client's credentials, marked with ``HUB_HEADER`` (and ``PROVIDER_HEADER``
for a provider shared individually). Anyone who can reach the API could send
those headers too, and with anonymous access off (the default for new
installs) the jobs would be refused for want of a key. So the worker adds
``RELAY_HEADER`` with a token made fresh in each process, sent only to this
router's own default upstreams.

:class:`HubRelayMiddleware` accepts the token only from a loopback peer,
records the result in the ASGI scope under ``SCOPE_KEY`` and strips all three
headers from every request, so the routes never see them and the token is
never logged. :func:`is_own_relay` and ``auth.from_hub`` read the scope.
"""

from __future__ import annotations

import hmac
import ipaddress
import secrets

from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from .protocol import HUB_HEADER, HUB_HEADER_VALUE, PROVIDER_HEADER

RELAY_HEADER = "X-Llama-Router-Relay"
# Set in the scope of a request this process's worker link relayed: a dict
# with "hub" (it carried HUB_HEADER, so routing is limited to the providers
# shared with the hub) and "provider" (the one provider it is for, or None).
SCOPE_KEY = "llama_router.hub_relay"

_TOKEN = secrets.token_urlsafe(32)
_STRIP = frozenset(
    h.lower().encode("latin-1") for h in (HUB_HEADER, PROVIDER_HEADER, RELAY_HEADER)
)


def relay_token() -> str:
    """The token the worker link sends in ``RELAY_HEADER``."""
    return _TOKEN


def _loopback(host: object) -> bool:
    """*host*, the TCP peer, is this machine. A dual-stack socket reports
    IPv4 peers as ``::ffff:127.0.0.1``, which counts too."""
    try:
        addr = ipaddress.ip_address(str(host))
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr.is_loopback


def is_own_relay(request: Request) -> bool:
    """True for a request this process's worker relayed to itself."""
    return isinstance(request.scope.get(SCOPE_KEY), dict)


class HubRelayMiddleware:
    """Mark requests relayed by this process's worker link; strip the headers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        via = provider = token = ""
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
                token = value
        scope = dict(scope)
        scope["headers"] = kept
        scope.pop(SCOPE_KEY, None)
        client = scope.get("client")
        if (
            token
            and client
            and _loopback(client[0])
            and hmac.compare_digest(token.encode(), _TOKEN.encode())
        ):
            hub = via.lower() == HUB_HEADER_VALUE
            scope[SCOPE_KEY] = {
                "hub": hub,
                "provider": _provider_id(provider) if hub else None,
            }
        await self.app(scope, receive, send)


def _provider_id(raw: str) -> int | None:
    if raw.isascii() and raw.isdigit() and len(raw) <= 18:
        return int(raw)
    return None
