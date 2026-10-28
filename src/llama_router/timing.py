"""Stamp each request with the moment it reached the app.

A pure ASGI middleware, so it adds one dict write per request and never
buffers or wraps streamed responses. Handlers and the request logger read
the stamp to measure how long the router itself took before dispatching.
"""

from __future__ import annotations

import time
from collections.abc import MutableMapping
from typing import Any

RECEIVED_AT = "received_at"


class RequestTimingMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            scope.setdefault("state", {})[RECEIVED_AT] = time.monotonic()
        await self.app(scope, receive, send)


def received_at(scope: MutableMapping[str, Any] | None) -> float | None:
    if not scope:
        return None
    state = scope.get("state")
    if isinstance(state, dict):
        value = state.get(RECEIVED_AT)
        if isinstance(value, float):
            return value
    return None
