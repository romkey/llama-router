"""One router process: loop lag, large JSON off the loop, the startup guard."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from llama_router import json_io, metrics
from llama_router.database import Database
from llama_router.request_body import json_object
from llama_router.request_logger import _measure
from llama_router.routing_stats import ProviderStats
from llama_router.single_process import (
    LEASE_KEY,
    LEASE_STALE_S,
    PER_PROCESS_STATE,
    LoopLag,
    ProcessLease,
)

BIG = "x" * (json_io.LARGE_JSON_BYTES + 10)


# --- Event loop lag ---


def test_lag_metric_is_exported() -> None:
    text = metrics.render(metrics.collect([], ProviderStats()))
    lines = text.splitlines()
    assert "# TYPE llama_router_event_loop_lag_seconds histogram" in lines
    assert any(
        line.startswith('llama_router_event_loop_lag_seconds_bucket{le="+Inf"} ')
        for line in lines
    )
    assert any(
        line.startswith("llama_router_event_loop_lag_seconds_sum ") for line in lines
    )
    assert any(
        line.startswith("llama_router_event_loop_lag_seconds_count ") for line in lines
    )


async def test_lag_is_the_overshoot_of_a_fixed_sleep() -> None:
    now = [0.0]

    async def late_sleep(seconds: float) -> None:
        now[0] += seconds + 0.03  # woken 30 ms late

    lag = LoopLag(0.5, (0.01, 0.05, 0.1), clock=lambda: now[0], sleep=late_sleep)
    assert await lag.tick() == pytest.approx(0.03)
    lag.observe(0.2)
    assert lag.cumulative() == [(0.01, 0), (0.05, 1), (0.1, 1), (float("inf"), 2)]
    assert lag.count == 2
    assert lag.sum == pytest.approx(0.23)
    family = metrics.Histogram("x", "help", lag.cumulative(), lag.sum, lag.count)
    assert 'llama_router_x_bucket{le="0.05"} 1' in list(family.lines())


async def test_a_blocked_loop_shows_as_lag() -> None:
    lag = LoopLag(0.01)
    task = asyncio.create_task(lag.tick())
    await asyncio.sleep(0)  # let it start its sleep
    time.sleep(0.06)  # hold the loop, as a large parse would
    assert await task >= 0.04


# --- Large JSON bodies ---


def _request(raw: bytes) -> Request:
    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": raw, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
            "scheme": "http",
            "server": ("router", 8080),
        },
        receive,
    )


@pytest.fixture
def threads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Names of the functions json_io sent to a worker thread."""
    calls: list[str] = []
    real = asyncio.to_thread

    async def to_thread(fn: Any, *args: Any) -> Any:
        calls.append(fn.__name__)
        return await real(fn, *args)

    monkeypatch.setattr("llama_router.json_io.asyncio.to_thread", to_thread)
    return calls


async def test_large_body_parses_off_the_loop(threads: list[str]) -> None:
    body = {"model": "m", "messages": [{"role": "user", "content": BIG}]}
    parsed = await json_object(_request(json.dumps(body).encode()))
    assert parsed == body
    assert threads == ["loads"]

    threads.clear()
    assert await json_object(_request(b'{"model": "m"}')) == {"model": "m"}
    assert threads == []


@pytest.mark.parametrize(
    ("raw", "detail"),
    [
        (b'{"model": "m", "prompt": "' + BIG.encode() + b'"', "JSON body required"),
        (b'["' + BIG.encode() + b'"]', "JSON body must be an object"),
        (b'{"model": 5, "prompt": "' + BIG.encode() + b'"}', "model must be a string"),
        (b"\xff" * (json_io.LARGE_JSON_BYTES + 1), "JSON body required"),
    ],
)
async def test_large_bad_bodies_get_the_same_400s(
    threads: list[str], raw: bytes, detail: str
) -> None:
    with pytest.raises(HTTPException) as exc:
        await json_object(_request(raw))
    assert exc.value.status_code == 400
    assert exc.value.detail == detail
    assert threads == ["loads"]


async def test_large_upstream_bodies_are_serialised_off_the_loop(
    threads: list[str],
) -> None:
    small = {"model": "m", "prompt": "hi"}
    assert await json_io.json_kwargs(small) == {"json": small}
    big = {"model": "m", "prompt": BIG, "ünï": "çödé"}
    kwargs = await json_io.json_kwargs(big)
    assert threads == ["_dumps"]
    assert kwargs["headers"] == {"Content-Type": "application/json"}
    assert json.loads(kwargs["content"]) == big
    # Encoded as httpx encodes json=: compact and UTF-8.
    assert b'"\xc3\xbcn\xc3\xaf":"\xc3\xa7\xc3\xb6d\xc3\xa9"' in kwargs["content"]


async def test_large_logged_bodies_are_measured_off_the_loop(
    threads: list[str],
) -> None:
    big = {"choices": [{"message": {"content": BIG}}]}
    row: dict[str, Any] = {"response_body": big, "request_body": {"p": BIG}}
    await _measure(row)
    assert row["response_size"] == len(json.dumps(big).encode())
    assert row["request_size"] == len(json.dumps({"p": BIG}).encode())
    assert threads == ["_size", "_size"]
    small: dict[str, Any] = {"response_body": {"a": 1}}
    threads.clear()
    await _measure(small)
    assert small["response_size"] == len(b'{"a": 1}')
    assert threads == []


def test_size_estimate_stops_at_the_limit() -> None:
    assert json_io.approx_json_size({"a": [1, "bc"]}) < 50
    assert json_io.approx_json_size({"a": BIG}) >= json_io.LARGE_JSON_BYTES
    many = [{"k": "v"}] * 1_000_000
    assert json_io.approx_json_size(many, limit=1000) >= 1000


# --- Startup guard ---


async def test_second_process_logs_the_guard_error(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    now = [1_000_000.0]
    first = ProcessLease(db, clock=lambda: now[0], host="box-a", pid=1)
    second = ProcessLease(db, clock=lambda: now[0], host="box-b", pid=1)
    with caplog.at_level(logging.ERROR, logger="llama_router.single_process"):
        await first.beat()
        assert not caplog.records
        await second.beat()
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    message = errors[0].getMessage()
    assert "box-a" in message
    for item in PER_PROCESS_STATE:
        assert item in message

    # The first process notices too, once; neither repeats itself.
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="llama_router.single_process"):
        await first.beat()
        await second.beat()
        await first.beat()
    assert len(caplog.records) == 1
    assert "box-b" in caplog.records[0].getMessage()


async def test_stale_or_own_previous_lease_is_not_a_second_process(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    now = [1_000_000.0]
    gone = ProcessLease(db, clock=lambda: now[0], host="box-a", pid=7)
    await gone.beat()
    now[0] += LEASE_STALE_S + 1
    with caplog.at_level(logging.ERROR, logger="llama_router.single_process"):
        await ProcessLease(db, clock=lambda: now[0], host="box-b", pid=7).beat()
        # A restarted container: same host and PID, new process.
        await ProcessLease(db, clock=lambda: now[0], host="box-b", pid=7).beat()
    assert not caplog.records


async def test_stopping_gives_the_lease_up(db: Database) -> None:
    lease = ProcessLease(db, heartbeat_s=3600)
    await lease.start()
    stored = json.loads(await db.get_app_setting(LEASE_KEY) or "{}")
    assert stored["id"] == lease.id
    await lease.stop()
    assert not await db.get_app_setting(LEASE_KEY)
