"""Fire-and-forget tasks that run after a response has been sent.

Keeps a reference to each task so it is not garbage collected mid-flight,
and logs failures instead of leaving them unretrieved.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)

_tasks: set[asyncio.Task[Any]] = set()
# Called by drain() to empty queues that are not tasks (the request log).
_flushers: list[Callable[[], Awaitable[None]]] = []


def on_drain(flush: Callable[[], Awaitable[None]]) -> None:
    """Have drain() also await *flush*."""
    _flushers.append(flush)


def spawn(
    coro: Coroutine[Any, Any, Any], *, name: str | None = None
) -> asyncio.Task[Any]:
    task = asyncio.get_running_loop().create_task(coro, name=name)
    _tasks.add(task)
    task.add_done_callback(_done)
    return task


def _done(task: asyncio.Task[Any]) -> None:
    _tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "Background task %s failed: %s", task.get_name(), exc, exc_info=exc
        )


async def drain() -> None:
    """Wait for every pending background task and queued write (tests and
    shutdown)."""
    while True:
        for flush in _flushers:
            await flush()
        if not _tasks:
            return
        await asyncio.gather(*list(_tasks), return_exceptions=True)
