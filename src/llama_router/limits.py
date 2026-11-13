"""Limits on what one client can make the router do.

- ``BodySizeLimitMiddleware`` refuses request bodies over
  ``max_body_bytes`` (``max_media_body_bytes`` for image and audio
  endpoints) with 413, by ``Content-Length`` or while a chunked body arrives.
- ``clamp_generation_params`` caps ``n`` and the output-token fields of chat
  and completion requests.
- ``upstream_slot`` holds a global and a per-key slot for each upstream call.
  When every slot is taken the request waits, in the order its routing mode
  sets (latency first, throughput and cheapest last), up to
  ``upstream_queue_timeout_seconds`` (and never past its deadline), then gets
  429. Streams also count against ``max_streams_per_key``, which refuses at
  once with 429.
- Every proxied request, streamed or not, must finish within
  ``request_deadline_seconds`` of reaching the router, or it gets 504 (a
  stream already under way is ended instead).
"""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from collections.abc import AsyncIterator, MutableMapping
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypeGuard

from fastapi import HTTPException, Request

from .capacity import DEFAULT_LEVEL, QueueTimeout, SlotGate, queue_level
from .config import settings
from .timing import received_at

if TYPE_CHECKING:
    from .router import RoutingPreferences

logger = logging.getLogger(__name__)

MEDIA_PATH_PREFIXES = ("/v1/audio/", "/v1/images/")

# Output-token fields across the OpenAI chat, completions and responses APIs.
_TOKEN_FIELDS = ("max_tokens", "max_completion_tokens", "max_output_tokens")
_CHOICE_FIELDS = ("n", "best_of")


class RequestBodyTooLarge(HTTPException):
    def __init__(self, limit: int) -> None:
        super().__init__(
            status_code=413, detail=f"request body larger than {limit} bytes"
        )


class DeadlineExceeded(HTTPException):
    def __init__(self) -> None:
        super().__init__(
            status_code=504,
            detail=(
                f"request deadline of {settings.request_deadline_seconds:g}s exceeded"
            ),
        )


def body_limit_for(path: str) -> int:
    if path.startswith(MEDIA_PATH_PREFIXES):
        return settings.max_media_body_bytes
    return settings.max_body_bytes


class BodySizeLimitMiddleware:
    """Pure ASGI: refuse oversized bodies before a handler reads them.

    A declared ``Content-Length`` over the limit is refused without reading
    anything. Otherwise the body is counted as it arrives and reading stops
    with 413 once it passes the limit.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(
        self, scope: MutableMapping[str, Any], receive: Any, send: Any
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = body_limit_for(scope.get("path", ""))
        if limit <= 0:
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers") or ():
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = -1
                if declared < 0 or declared > limit:
                    await _reject(send, limit)
                    return

        received = 0
        started = False

        async def limited_receive() -> Any:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise RequestBodyTooLarge(limit)
            return message

        async def tracking_send(message: Any) -> None:
            nonlocal started
            if message.get("type") == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except RequestBodyTooLarge:
            # Normally the app's exception handling answers 413; this covers
            # a body read outside it.
            if not started:
                await _reject(send, limit)


async def _reject(send: Any, limit: int) -> None:
    body = f'{{"detail":"request body larger than {limit} bytes"}}'.encode()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def clamp_generation_params(body: Any) -> None:
    """Cap ``n``/``best_of`` at ``max_n`` and output tokens at
    ``max_tokens_cap``, in place. Ollama's ``options.num_predict`` counts as
    output tokens; a negative one (unlimited) is capped too. Fields that are
    absent or not numbers are left alone; a float over a cap (``1e9``) is
    capped too."""
    if not isinstance(body, dict):
        return
    if settings.max_n > 0:
        for field in _CHOICE_FIELDS:
            value = body.get(field)
            if _is_number(value) and value > settings.max_n:
                body[field] = settings.max_n
    cap = settings.max_tokens_cap
    if cap <= 0:
        return
    for field in _TOKEN_FIELDS:
        value = body.get(field)
        if _is_number(value) and value > cap:
            body[field] = cap
    options = body.get("options")
    if isinstance(options, dict):
        value = options.get("num_predict")
        if _is_number(value) and (value > cap or value < 0):
            options["num_predict"] = cap


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def request_deadline(request: Request) -> float | None:
    """When the request must be finished, on the ``time.monotonic`` clock."""
    seconds = settings.request_deadline_seconds
    if seconds <= 0:
        return None
    start = received_at(request.scope)
    return (start if start is not None else time.monotonic()) + seconds


class UpstreamLimiter:
    """Global and per-key slot gates, and per-key stream counts.

    The gates are :class:`~.capacity.SlotGate`, so waiters are served in
    queue-level order (see :func:`~.capacity.queue_level`), first come first
    served within a level. Sized from settings when first used; ``reset``
    drops them so new settings take effect (tests). Requests without an API
    key share only the global limit.
    """

    def __init__(self) -> None:
        self._global: SlotGate | None = None
        self._global_size = 0
        self._per_key: dict[int, SlotGate] = {}
        self._streams: dict[int, int] = {}

    def reset(self) -> None:
        self._global = None
        self._global_size = 0
        self._per_key.clear()
        self._streams.clear()

    def active_streams(self, key_id: int) -> int:
        return self._streams.get(key_id, 0)

    def global_gate(self) -> SlotGate | None:
        size = settings.upstream_max_concurrency
        if size <= 0:
            return None
        if self._global is None or self._global_size != size:
            self._global = SlotGate(size)
            self._global_size = size
        return self._global

    def _key_gate(self, key_id: int | None) -> SlotGate | None:
        size = settings.upstream_max_concurrency_per_key
        if key_id is None or size <= 0:
            return None
        gate = self._per_key.get(key_id)
        if gate is None:
            gate = self._per_key[key_id] = SlotGate(size)
        return gate

    async def acquire(
        self,
        key_id: int | None,
        *,
        stream: bool,
        deadline: float | None,
        level: int = DEFAULT_LEVEL,
    ) -> list[SlotGate]:
        """Take the slots for one upstream call, or raise 429."""
        held: list[SlotGate] = []
        if stream and key_id is not None:
            cap = settings.max_streams_per_key
            if cap > 0 and self._streams.get(key_id, 0) >= cap:
                raise HTTPException(
                    status_code=429,
                    detail=f"too many concurrent streams for this API key (max {cap})",
                    headers={"Retry-After": "1"},
                )
            self._streams[key_id] = self._streams.get(key_id, 0) + 1
        try:
            for gate in (self._key_gate(key_id), self.global_gate()):
                if gate is None:
                    continue
                try:
                    waited = await gate.acquire(deadline, level)
                except QueueTimeout:
                    logger.info(
                        "Upstream slots full: request at queue level %d timed out",
                        level,
                    )
                    raise HTTPException(
                        status_code=429,
                        detail="too many concurrent requests; try again shortly",
                        headers={"Retry-After": "1"},
                    ) from None
                if waited > 0:
                    logger.debug(
                        "Request waited %.3fs at queue level %d for an upstream slot",
                        waited,
                        level,
                    )
                held.append(gate)
        except BaseException:
            self.release(key_id, stream=stream, held=held)
            raise
        return held

    def release(
        self, key_id: int | None, *, stream: bool, held: list[SlotGate]
    ) -> None:
        for gate in held:
            gate.release()
        held.clear()
        if stream and key_id is not None:
            left = self._streams.get(key_id, 0) - 1
            if left > 0:
                self._streams[key_id] = left
            else:
                self._streams.pop(key_id, None)


limiter = UpstreamLimiter()


class UpstreamSlot:
    """Slots held for one upstream call, released exactly once.

    ``async with slot`` releases them on exit. For a non-streamed call it
    also enforces the request deadline over the block (504). A stream gets
    its deadline from ``slot.stream(...)`` instead, since a timeout must not
    span a ``yield``. A slot whose stream is never started is released when
    it is garbage-collected.
    """

    def __init__(
        self,
        key_id: int | None,
        held: list[SlotGate],
        *,
        stream: bool,
        deadline: float | None,
    ) -> None:
        self.stream_mode = stream
        self.deadline = deadline
        # The queue level the request waits at (see capacity.queue_level).
        self.level = DEFAULT_LEVEL
        self._release = weakref.finalize(
            self, limiter.release, key_id, stream=stream, held=held
        )
        self._timeout: asyncio.Timeout | None = None

    def release(self) -> None:
        self._release()

    async def __aenter__(self) -> UpstreamSlot:
        if not self.stream_mode and self.deadline is not None:
            self._timeout = asyncio.timeout(_remaining(self.deadline))
            await self._timeout.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            if self._timeout is not None:
                try:
                    await self._timeout.__aexit__(exc_type, exc, tb)
                except TimeoutError as err:
                    if self._timeout.expired():
                        raise DeadlineExceeded() from err
                    raise
        finally:
            self.release()

    async def stream(self, inner: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """*inner*, ended with ``DeadlineExceeded`` at the request deadline."""
        deadline = self.deadline
        if deadline is None:
            async for chunk in inner:
                yield chunk
            return
        # One timer for the whole stream, not one per chunk: when it fires
        # while a chunk is awaited it cancels that wait (as asyncio.timeout
        # does); otherwise the next wait sees it has fired.
        it = aiter(inner)
        expired = False
        waiting: asyncio.Task[Any] | None = None
        cancelled = False

        def expire() -> None:
            nonlocal expired, cancelled
            expired = True
            if waiting is not None:
                cancelled = True
                waiting.cancel()

        handle = asyncio.get_running_loop().call_later(_remaining(deadline), expire)
        try:
            while True:
                if expired:
                    raise DeadlineExceeded()
                waiting = asyncio.current_task()
                try:
                    chunk = await anext(it)
                except StopAsyncIteration:
                    return
                except asyncio.CancelledError as err:
                    task = asyncio.current_task()
                    if cancelled and task is not None and task.uncancel() == 0:
                        raise DeadlineExceeded() from err
                    raise
                finally:
                    waiting = None
                yield chunk
        finally:
            handle.cancel()
            aclose = getattr(it, "aclose", None)
            if aclose is not None:
                await aclose()


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


async def upstream_slot(
    request: Request, prefs: RoutingPreferences | None, *, stream: bool = False
) -> UpstreamSlot:
    """Take the slots for one upstream call (429 when none frees up in time).

    Call it before the call's ``try`` block so a 429 is not logged as the
    provider's failure, then ``async with`` the result around the call.
    """
    key_id = prefs.key_id if prefs is not None else None
    deadline = request_deadline(request)
    level = queue_level(prefs.mode if prefs is not None else None)
    held = await limiter.acquire(key_id, stream=stream, deadline=deadline, level=level)
    slot = UpstreamSlot(key_id, held, stream=stream, deadline=deadline)
    slot.level = level
    return slot
