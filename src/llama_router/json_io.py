"""JSON parsing and serialising that keeps large bodies off the event loop.

One router process serves every request on one event loop (see
docs/configuration.md, "One router process"). Parsing or serialising a JSON
body of many megabytes (inline images, long documents) holds that loop for
tens of milliseconds, and every other request waits. Bodies over
:data:`LARGE_JSON_BYTES` are handled in a worker thread instead; smaller
ones stay inline, where a thread hop would cost more than it saves.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

# Bodies at least this large are parsed and serialised in a worker thread.
LARGE_JSON_BYTES = 1_000_000


def _str_bytes(s: str) -> int:
    """At least the bytes *s* takes as UTF-8 JSON, without encoding it:
    ASCII is one byte a character, anything else up to four."""
    return len(s) + 2 if s.isascii() else 4 * len(s) + 2


def approx_json_size(value: Any, limit: int = LARGE_JSON_BYTES) -> int:
    """A cheap estimate of *value*'s UTF-8 JSON size, never below it for
    strings (non-ASCII text counts four bytes a character) and with numbers
    at their printed length. Stops once it reaches *limit*: large bodies are
    large because of a few long strings or long arrays, so it stays cheap
    for the bodies it has to say are small."""
    total = 0
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, str):
            total += _str_bytes(v)
        elif isinstance(v, dict):
            total += 2
            for k, item in v.items():
                total += (_str_bytes(k) if isinstance(k, str) else 8) + 2
                stack.append(item)
        elif isinstance(v, (list, tuple)):
            total += 2
            stack.extend(v)
        elif isinstance(v, bool) or v is None:
            total += 6
        elif isinstance(v, (int, float)):
            total += len(repr(v)) + 1
        else:
            total += 8
        if total >= limit:
            return total
    return total


def _dumps(value: Any) -> bytes:
    # As httpx encodes json=: compact, UTF-8, no NaN.
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


async def loads(raw: bytes) -> Any:
    """Parse *raw* JSON, in a worker thread when it is large. Raises what
    ``json.loads`` raises."""
    if len(raw) >= LARGE_JSON_BYTES:
        return await asyncio.to_thread(json.loads, raw)
    return json.loads(raw)


async def json_kwargs(body: Any) -> dict[str, Any]:
    """httpx request arguments that send *body* as JSON: ``json=body`` for a
    small body, else the bytes serialised in a worker thread."""
    if approx_json_size(body) < LARGE_JSON_BYTES:
        return {"json": body}
    content = await asyncio.to_thread(_dumps, body)
    return {"content": content, "headers": {"Content-Type": "application/json"}}


async def json_size(value: Any) -> int:
    """Bytes *value* takes as JSON (0 if it cannot be serialised), measured
    in a worker thread when it is large."""
    if approx_json_size(value) >= LARGE_JSON_BYTES:
        return await asyncio.to_thread(_size, value)
    return _size(value)


def _size(value: Any) -> int:
    try:
        return len(json.dumps(value).encode())
    except Exception:
        return 0
