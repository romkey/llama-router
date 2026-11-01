"""Spend caps: reservations, estimates for unknown usage, stream usage."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import cast

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from llama_router.auth import route_within_cap
from llama_router.database import Database
from llama_router.models import Provider
from llama_router.pricing import Price, estimate_tokens
from llama_router.request_logger import (
    StreamLogger,
    _UsageChunkFilter,
    ask_stream_usage,
    attach_pricer,
    log_request,
    log_request_background,
)
from llama_router.router import Router, RouteResult, RoutingPreferences
from llama_router.spend import SpendTracker, tracker_for

CHAT = "/v1/chat/completions"
IMAGES = "/v1/images/generations"
PRICE = Price(1.0, 10.0)  # USD per million tokens in, out

CLOUD = Provider(id=1, name="cloud", locality="cloud")
LOCAL = Provider(id=2, name="lan", locality="local")


def _request(path: str = CHAT) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("router", 8080),
        }
    )


class _FakeRouter:
    """Routes to the cloud provider unless limited to local ones."""

    def __init__(self, price: Price | None, local: bool = True) -> None:
        self.price = price
        self.local = local
        self.asked: list[str | None] = []

    async def route(
        self,
        model: str,
        endpoint: str,
        prefs: RoutingPreferences | None = None,
        *,
        session: str | None = None,
    ) -> RouteResult | None:
        locality = prefs.locality if prefs else None
        self.asked.append(locality)
        if locality == "local":
            return RouteResult(LOCAL, model) if self.local else None
        return RouteResult(CLOUD, model)

    def price_for(self, provider_id: int, model: str) -> Price | None:
        return self.price if provider_id == CLOUD.id else None


def _capped(locality: str | None = None) -> RoutingPreferences:
    return RoutingPreferences(key_id=5, daily_spend_cap_usd=1.0, locality=locality)


def test_estimate_tokens() -> None:
    body = {"messages": [{"role": "user", "content": "x" * 300}], "max_tokens": 50}
    prompt, completion = estimate_tokens(body, 4096)
    assert prompt >= 100  # 300 characters at 3 per token, plus the keys
    assert completion == 50
    assert estimate_tokens({"prompt": "hi"}, 4096)[1] == 4096
    assert estimate_tokens({"options": {"num_predict": 7}}, 4096)[1] == 7
    assert estimate_tokens({"max_completion_tokens": 10, "n": 3}, 4096)[1] == 30
    # An inline image counts as a fixed number of tokens, not its base64 size.
    image = {"image_url": "data:image/png;base64," + "A" * 1_000_000}
    assert estimate_tokens(image, 0)[0] < 5000
    assert estimate_tokens(None, 9) == (0, 9)


async def test_reservations_hold_against_the_cap(db: Database) -> None:
    t = SpendTracker(db)
    first = await t.reserve(1, 1.0, 0.6)
    assert first is not None
    assert await t.committed_today(1) == pytest.approx(0.6)
    assert await t.reserve(1, 1.0, 0.6) is None
    t.settle(first, 0.3)
    assert await t.spent_today(1) == pytest.approx(0.3)
    assert await t.committed_today(1) == pytest.approx(0.3)
    second = await t.reserve(1, 1.0, 0.6)
    assert second is not None
    t.settle(second, None)  # failed: released, nothing charged
    t.settle(second, 5.0)  # settling twice does nothing
    assert await t.committed_today(1) == pytest.approx(0.3)


async def test_concurrent_reservations_cannot_jointly_exceed_the_cap(
    db: Database,
) -> None:
    t = SpendTracker(db)
    held = await asyncio.gather(*(t.reserve(1, 1.0, 0.3) for _ in range(10)))
    assert sum(r is not None for r in held) == 3


async def test_route_within_cap_reserves_the_estimate(db: Database) -> None:
    rt = _FakeRouter(PRICE)
    req = _request()
    body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 1000}
    result = await route_within_cap(cast(Router, rt), db, req, "m", _capped(), body)
    assert result is not None
    assert result.provider is CLOUD
    held = req.state.spend_reservation
    assert held.amount == pytest.approx(PRICE.cost(*estimate_tokens(body, 0)))
    assert await tracker_for(db).committed_today(5) == pytest.approx(held.amount)


async def test_route_within_cap_goes_local_when_the_estimate_does_not_fit(
    db: Database,
) -> None:
    rt = _FakeRouter(PRICE)
    # 200k output tokens at $10 per million: $2, over the $1 cap.
    body = {"prompt": "hi", "max_tokens": 200_000}
    result = await route_within_cap(
        cast(Router, rt), db, _request(), "m", _capped(), body
    )
    assert result is not None
    assert result.provider is LOCAL
    assert rt.asked == [None, "local"]
    with pytest.raises(HTTPException) as exc:
        await route_within_cap(
            cast(Router, rt), db, _request(), "m", _capped("cloud"), body
        )
    assert exc.value.status_code == 429


async def test_unpriced_media_is_refused_on_capped_keys(db: Database) -> None:
    rt = _FakeRouter(None, local=False)
    with pytest.raises(HTTPException) as exc:
        await route_within_cap(
            cast(Router, rt), db, _request(IMAGES), "m", _capped(), {"prompt": "a"}
        )
    assert exc.value.status_code == 403
    # Local providers may still serve it.
    rt.local = True
    result = await route_within_cap(
        cast(Router, rt), db, _request(IMAGES), "m", _capped(), {"prompt": "a"}
    )
    assert result is not None
    assert result.provider is LOCAL
    # Chat on an unpriced model is not refused (its cost is unknown, as before);
    # nor is anything on a key without a cap.
    chat = await route_within_cap(
        cast(Router, rt), db, _request(), "m", _capped(), {"prompt": "a"}
    )
    assert chat is not None
    assert chat.provider is CLOUD
    uncapped = await route_within_cap(
        cast(Router, rt),
        db,
        _request(IMAGES),
        "m",
        RoutingPreferences(key_id=5),
        {"prompt": "a"},
    )
    assert uncapped is not None
    assert uncapped.provider is CLOUD


async def test_unknown_usage_is_charged_the_estimate(db: Database) -> None:
    attach_pricer(lambda pid, model: PRICE if pid == CLOUD.id else None)
    try:
        body = {"prompt": "hi", "max_tokens": 100}
        await log_request(
            db,
            provider=CLOUD,
            protocol="v1",
            endpoint=CHAT,
            model="m",
            request_body=body,
            duration_ms=1,
            api_key_id=5,
        )
        # A failed request is not charged.
        await log_request(
            db,
            provider=CLOUD,
            protocol="v1",
            endpoint=CHAT,
            model="m",
            request_body=body,
            duration_ms=1,
            status="error",
            api_key_id=5,
        )
    finally:
        attach_pricer(None)
    failed, ok = await db.get_request_logs(10, 0)
    assert ok.cost_usd == pytest.approx(PRICE.cost(*estimate_tokens(body, 0)))
    assert (ok.cost_usd or 0) > 0
    assert failed.cost_usd is None


async def test_the_log_settles_the_reservation(db: Database) -> None:
    t = tracker_for(db)
    held = await t.reserve(5, 1.0, 0.5)
    assert held is not None
    req = _request()
    req.state.api_key_id = 5
    req.state.spend_reservation = held
    attach_pricer(lambda pid, model: PRICE)
    try:
        log_request_background(
            db,
            provider=CLOUD,
            protocol="v1",
            endpoint=CHAT,
            request=req,
            model="m",
            duration_ms=1,
            response_body={"usage": {"prompt_tokens": 1000, "completion_tokens": 0}},
        )
        assert req.state.spend_reservation is None
        for _ in range(50):
            if held.done:
                break
            await asyncio.sleep(0.01)
    finally:
        attach_pricer(None)
    assert held.done
    assert await t.committed_today(5) == pytest.approx(0.001)


def test_ask_stream_usage() -> None:
    body: dict[str, object] = {"stream": True}
    assert ask_stream_usage(body, CLOUD)
    assert body["stream_options"] == {"include_usage": True}
    asked: dict[str, object] = {"stream_options": {"include_usage": True}}
    assert not ask_stream_usage(asked, CLOUD)
    local: dict[str, object] = {}
    assert not ask_stream_usage(local, LOCAL)
    assert "stream_options" not in local


def _sse(obj: object) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


CONTENT = _sse({"choices": [{"delta": {"content": "hi"}}], "usage": None})
USAGE = _sse({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}})
DONE = b"data: [DONE]\n\n"


def test_usage_chunk_filter_drops_only_the_usage_event() -> None:
    raw = CONTENT + USAGE + DONE
    f = _UsageChunkFilter()
    # Chunks that do not line up with events.
    out = b"".join(f.feed(raw[i : i + 7]) for i in range(0, len(raw), 7))
    out += f.flush()
    assert out == CONTENT + DONE
    crlf = _UsageChunkFilter()
    assert crlf.feed(USAGE.replace(b"\n", b"\r\n")) == b""


async def test_stream_logger_hides_usage_but_reads_it(db: Database) -> None:
    async def upstream() -> AsyncIterator[bytes]:
        for chunk in (CONTENT, USAGE[:10], USAGE[10:] + DONE):
            yield chunk

    logged = StreamLogger(
        upstream(),
        db=db,
        provider=CLOUD,
        protocol="v1",
        endpoint=CHAT,
        request=_request(),
        model="m",
        request_body={},
        start_time=0.0,
        hide_usage=True,
    )
    out = b"".join([c async for c in logged])
    assert out == CONTENT + DONE
    for _ in range(50):
        logs = await db.get_request_logs(10, 0)
        if logs:
            break
        await asyncio.sleep(0.01)
    (entry,) = logs
    assert (entry.prompt_tokens, entry.completion_tokens) == (3, 1)
