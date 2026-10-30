"""Who this router is, and the guard that stops requests going round in circles.

Every router has a random ID, made once and kept in the database. Routers
swap IDs when a worker links to a hub, and every request a router relays to
another router carries the IDs it has passed through in ``PATH_HEADER``. A
router that finds its own ID there, or a path that is too long, answers 508
instead of routing the request again.
"""

from __future__ import annotations

import contextvars
import json
import logging
import uuid
from collections.abc import Callable
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

DB_KEY = "router_id"
PATH_HEADER = "X-Llama-Router-Path"
# A path this long is a loop through routers that do not add their IDs, or a
# setup nobody means to build.
MAX_HOPS = 8
LOOP_STATUS = 508


class _Settings(Protocol):
    async def get_app_setting(self, key: str) -> str | None: ...

    async def set_app_setting(self, key: str, value: str) -> None: ...


_router_id: str | None = None
# Router ID of the hub this router serves as a worker, while linked.
_hub_id: str | None = None
_hub_listeners: list[Callable[[], None]] = []
# Routers the request being handled has already passed through.
_incoming: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "llama_router_path", default=()
)


def parse_id(value: Any) -> str | None:
    """A router ID in canonical form, or None when *value* is not one."""
    try:
        return str(uuid.UUID(str(value or "").strip()))
    except ValueError:
        return None


async def load(db: _Settings) -> str:
    """Read this router's ID, making one on first start."""
    global _router_id
    rid = parse_id(await db.get_app_setting(DB_KEY))
    if rid is None:
        rid = str(uuid.uuid4())
        await db.set_app_setting(DB_KEY, rid)
        logger.info("Router ID %s created", rid)
    _router_id = rid
    return rid


async def regenerate(db: _Settings) -> str:
    """Give this router a new ID, for a copy made from another's data volume."""
    global _router_id
    rid = str(uuid.uuid4())
    await db.set_app_setting(DB_KEY, rid)
    _router_id = rid
    logger.info("Router ID changed to %s", rid)
    return rid


def router_id() -> str | None:
    return _router_id


def hub_id() -> str | None:
    """Router ID of the hub this router serves, or None when not linked."""
    return _hub_id


def set_hub_id(value: str | None) -> None:
    global _hub_id
    value = parse_id(value)
    if value == _hub_id:
        return
    _hub_id = value
    for listener in list(_hub_listeners):
        try:
            listener()
        except Exception:
            logger.exception("Hub ID listener failed")


def on_hub_change(listener: Callable[[], None]) -> None:
    _hub_listeners.append(listener)


def remove_hub_listener(listener: Callable[[], None]) -> None:
    if listener in _hub_listeners:
        _hub_listeners.remove(listener)


def parse_path(value: str | None) -> tuple[str, ...]:
    return tuple(part.strip() for part in (value or "").split(",") if part.strip())


def loop_reason(path: tuple[str, ...]) -> str | None:
    """Why a request with *path* must not be routed here, or None."""
    if _router_id is not None and _router_id in path:
        return "request has already passed through this router"
    if len(path) >= MAX_HOPS:
        return f"request has passed through {len(path)} routers"
    return None


def outgoing_path() -> str | None:
    """The path header value for a request this router sends on."""
    if _router_id is None:
        return None
    return ",".join((*_incoming.get(), _router_id))


async def add_path_header(request: httpx.Request) -> None:
    """``httpx`` request hook for clients that may reach another router."""
    value = outgoing_path()
    if value is not None:
        request.headers[PATH_HEADER] = value


class LoopGuardMiddleware:
    """Refuses requests that have been here before; remembers the path of the rest."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        raw = b",".join(
            v
            for k, v in scope.get("headers") or ()
            if k == PATH_HEADER.lower().encode()
        )
        path = parse_path(raw.decode("latin-1"))
        reason = loop_reason(path)
        if reason is not None:
            logger.warning(
                "Refused %s %s: routing loop (%s; path %s)",
                scope.get("method"),
                scope.get("path"),
                reason,
                ",".join(path),
            )
            body = json.dumps({"error": f"routing loop: {reason}"}).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": LOOP_STATUS,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        token = _incoming.set(path)
        try:
            await self.app(scope, receive, send)
        finally:
            _incoming.reset(token)
