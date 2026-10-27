from __future__ import annotations

import time

import httpx
import pytest

from llama_router.api import deps as api_deps
from llama_router.api.app import app as api_app
from llama_router.background import drain
from llama_router.database import Database
from llama_router.models import (
    ProviderModel,
    ProviderStatus,
    ProviderProtocol,
    RequestLog,
)
from llama_router.provider_manager import ProviderManager
from llama_router.request_logger import StreamLogger, log_request_background
from llama_router.router import Router
from llama_router.timing import RECEIVED_AT, RequestTimingMiddleware, received_at
from starlette.requests import Request


async def test_request_log_round_trips_latency_fields(db: Database):
    await db.save_request_log(
        RequestLog(
            protocol="ollama",
            endpoint="/api/chat",
            duration_ms=120.0,
            ttfb_ms=35.5,
            router_ms=0.4,
        )
    )
    entry = (await db.get_request_logs(1, 0))[0]
    assert entry.ttfb_ms == pytest.approx(35.5)
    assert entry.router_ms == pytest.approx(0.4)


async def test_middleware_stamps_arrival_time():
    seen = {}

    async def app(scope, receive, send):
        seen["at"] = received_at(scope)

    before = time.monotonic()
    await RequestTimingMiddleware(app)({"type": "http"}, None, None)
    assert seen["at"] is not None and seen["at"] >= before
    # Non-HTTP scopes pass through untouched.
    scope = {"type": "lifespan"}
    await RequestTimingMiddleware(app)(scope, None, None)
    assert "state" not in scope


def _stamped_request(arrived: float) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/chat",
            "headers": [],
            "state": {RECEIVED_AT: arrived},
        }
    )


async def test_router_overhead_from_duration(db: Database):
    now = time.monotonic()
    # Arrived 50 ms ago; the backend took 30 ms, so routing took about 20 ms.
    log_request_background(
        db,
        provider=None,
        protocol="ollama",
        endpoint="/api/chat",
        request=_stamped_request(now - 0.050),
        model="m",
        duration_ms=30.0,
    )
    await drain()
    entry = (await db.get_request_logs(1, 0))[0]
    assert entry.router_ms == pytest.approx(20.0, abs=10.0)
    assert entry.ttfb_ms is None


async def test_stream_logs_first_byte_and_overhead(db: Database):
    dispatched = time.monotonic()
    arrived = dispatched - 0.010

    async def upstream():
        yield b'{"done":false}\n'
        yield b'{"done":true}\n'

    logged = StreamLogger(
        upstream(),
        db=db,
        provider=None,
        protocol="ollama",
        endpoint="/api/chat",
        request=_stamped_request(arrived),
        model="m",
        request_body={},
        start_time=dispatched,
    )
    async for _ in logged:
        pass
    await drain()
    entry = (await db.get_request_logs(1, 0))[0]
    assert entry.streamed
    assert entry.router_ms == pytest.approx(10.0, abs=0.5)
    assert entry.ttfb_ms is not None and entry.ttfb_ms >= 0


async def test_proxied_request_records_latency_end_to_end(db: Database, httpx_mock):
    p = await db.add_provider("srv", ProviderProtocol.OLLAMA)
    await db.add_address(p.id, "http://backend.test:11434", is_preferred=True)
    await db.update_provider_status(p.id, ProviderStatus.IDLE)
    await db.set_provider_models(p.id, [ProviderModel(provider_id=p.id, name="m")])
    pm = ProviderManager(db)
    await pm._rebuild_clients(await db.get_provider(p.id))
    rt = Router(db, pm)
    api_deps.init(db, pm, rt)
    httpx_mock.add_response(
        url="http://backend.test:11434/api/chat",
        method="POST",
        stream=httpx.ByteStream(
            b'{"message":{"content":"hi"},"done":false}\n{"done":true}\n'
        ),
    )
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://router") as c:
        resp = await c.post(
            "/api/chat", json={"model": "m", "messages": [], "stream": True}
        )
    assert resp.status_code == 200
    await drain()
    entry = next(
        e for e in await db.get_request_logs(10, 0) if e.endpoint == "/api/chat"
    )
    assert entry.provider_id == p.id
    assert entry.ttfb_ms is not None and entry.ttfb_ms >= 0
    assert entry.router_ms is not None and 0 <= entry.router_ms < 1000
    await pm.stop()
