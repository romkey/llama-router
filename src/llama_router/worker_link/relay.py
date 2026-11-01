"""Proof that an API request was relayed by this process's own worker link.

A worker replays hub jobs to its router's API ports on 127.0.0.1 without the
hub client's credentials. With anonymous access off (the default for new
installs) those requests would be refused, so the worker adds a token that
only this process knows. It is random per start, is only sent to this
process over loopback, and is accepted only from a loopback address.
"""

from __future__ import annotations

import secrets

from starlette.requests import Request

RELAY_HEADER = "X-Llama-Router-Relay"

_TOKEN = secrets.token_urlsafe(32)
_LOOPBACK = frozenset({"127.0.0.1", "::1"})


def relay_token() -> str:
    return _TOKEN


def is_own_relay(request: Request) -> bool:
    """True for a request this process's worker relayed to itself."""
    client = request.client
    if client is None or client.host not in _LOOPBACK:
        return False
    got = request.headers.get(RELAY_HEADER) or ""
    return bool(got) and secrets.compare_digest(got, _TOKEN)
