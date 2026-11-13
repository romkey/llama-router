"""Guards for the warm request path.

A warm request (its API key cached, a slot free) must not touch the
database before the upstream call, must route and take its slots without
yielding to the event loop, and ``route_with`` must stay fast. Every feature
that adds per-request work extends the first test's configuration.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from llama_router import proxy
from llama_router.auth import generate_api_key, key_prefix
from llama_router.database import Database
from llama_router.limits import limiter
from llama_router.llamacpp_api import deps as v1_deps
from llama_router.llamacpp_api.app import app as v1_app
from llama_router.models import (
    BenchmarkResult,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.provider_manager import ProviderManager
from llama_router.provider_secrets import key_hash
from llama_router.router import Router, RoutingPreferences
from tests.conftest import asgi_client

CHAT = "/v1/chat/completions"
BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
BASELINE = Path(__file__).with_name("route_with_baseline.json")


@pytest.fixture
def pm(db: Database) -> ProviderManager:
    return ProviderManager(db)


@pytest.fixture
async def rt(db: Database, pm: ProviderManager) -> Router:
    return Router(db, pm)


@pytest.fixture
async def v1(
    db: Database, pm: ProviderManager, rt: Router
) -> AsyncIterator[httpx.AsyncClient]:
    v1_deps.init(db, pm, rt)
    async with asgi_client(v1_app) as client:
        yield client
    v1_deps._db, v1_deps._pm, v1_deps._router = None, None, None


async def _provider(
    db: Database,
    name: str,
    *,
    startup: float,
    limit: int | None = None,
    locality: str = "local",
) -> int:
    p = await db.add_provider(name, ProviderProtocol.OPENAI)
    pid = p.require_id()
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.set_provider_models(pid, [ProviderModel(provider_id=pid, name="m")])
    await db.update_provider_settings(pid, max_concurrent=limit, locality=locality)
    await db.save_benchmark(
        BenchmarkResult(
            provider_id=pid,
            model_name="m",
            protocol="openai",
            startup_time_ms=startup,
            tokens_per_second=50.0,
        )
    )
    return pid


class DbSpy:
    """Counts every database round trip (all go through the session
    factory)."""

    def __init__(self, db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls = 0
        real: Callable[[], Any] = db._require_session_factory

        def counted() -> Any:
            self.calls += 1
            return real()

        monkeypatch.setattr(db, "_require_session_factory", counted)


async def _configure_everything(db: Database) -> tuple[str, int]:
    """A key and providers with every per-request feature switched on.

    Later features add theirs here.
    """
    a = await _provider(db, "a", startup=100.0, limit=8)
    await _provider(db, "b", startup=300.0, limit=8)
    await _provider(db, "cloud", startup=50.0, limit=8, locality="cloud")
    plaintext = generate_api_key()
    key_id = await db.create_api_key(
        key_prefix(plaintext),
        key_hash(plaintext),
        "balanced",
        True,
        session_affinity=True,
        daily_spend_cap_usd=5.0,
        strict_pins=False,
    )
    await db.set_api_key_model_pin(key_id, "other", a)
    return plaintext, key_id


async def test_warm_request_makes_no_database_calls_before_upstream(
    db: Database,
    v1: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key, _ = await _configure_everything(db)
    calls_at_upstream: list[int] = []
    spy: DbSpy | None = None

    def client(pm: ProviderManager, provider_id: int) -> Any:
        class Client:
            async def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
                calls_at_upstream.append(spy.calls if spy is not None else -1)
                return {"choices": [{"message": {"content": "ok"}}]}

        return Client()

    monkeypatch.setattr(proxy, "get_v1_client", client)
    monkeypatch.setattr(proxy, "log_request_background", lambda *a, **k: None)
    headers = {"Authorization": f"Bearer {key}"}
    # Cold: resolves the key, builds the table, seeds per-key totals.
    assert (await v1.post(CHAT, json=BODY, headers=headers)).status_code == 200
    spy = DbSpy(db, monkeypatch)
    resp = await v1.post(CHAT, json=BODY, headers=headers)
    assert resp.status_code == 200, resp.text
    assert calls_at_upstream[-1] == 0


async def test_route_and_slots_complete_without_yielding(
    db: Database, pm: ProviderManager, rt: Router
) -> None:
    await _configure_everything(db)
    table = await rt.current_table()
    prefs = RoutingPreferences(key_id=1, session_affinity=True)
    limiter.reset()

    async def warm() -> None:
        result = rt.route_with(table, "m", CHAT, prefs, session="s")
        assert result is not None
        pid = result.provider.require_id()
        held = await limiter.acquire(1, stream=False, deadline=None)
        gate = await pm.take_slot(pid)
        async with pm.acquire_provider(pid, slot=gate, slot_taken=True):
            pass
        limiter.release(1, stream=False, held=held)

    coro = warm()
    # One send runs it to the end: nothing awaited suspended.
    with pytest.raises(StopIteration):
        coro.send(None)


def _calibrate() -> float:
    """Fixed pure-Python work, to express route_with's time in units that
    do not depend on the machine."""
    data = {i: (i * 7) % 13 for i in range(64)}
    start = time.perf_counter()
    for _ in range(200):
        best = None
        for k, v in data.items():
            if best is None or v < best[1]:
                best = (k, v)
        _ = sorted(data.values())
    return time.perf_counter() - start


async def test_route_with_benchmark_20_candidates(
    db: Database, pm: ProviderManager, rt: Router
) -> None:
    for n in range(20):
        await _provider(db, f"p{n}", startup=100.0 + n * 10, limit=4)
    table = await rt.current_table()
    prefs = RoutingPreferences(key_id=1, session_affinity=True)
    for n in range(20):
        # Some busy, so the queue cost is weighed.
        for _ in range(n % 3):
            pm.acquire(n + 1)

    def run() -> float:
        start = time.perf_counter()
        for i in range(500):
            rt.route_with(table, "m", CHAT, prefs, session=f"s{i % 50}")
        return time.perf_counter() - start

    run()
    ratio = min(run() / _calibrate() for _ in range(7))
    recorded = json.loads(BASELINE.read_text())
    if recorded.get("ratio") is None:
        pytest.fail(f"no baseline recorded; measured {ratio:.3f}")
    assert ratio <= recorded["ratio"] * recorded.get("tolerance", 1.2), (
        f"route_with with 20 candidates is {ratio / recorded['ratio']:.2f}x "
        "its recorded baseline"
    )
