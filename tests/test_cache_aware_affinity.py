"""Cache-aware session affinity: stay while the prompt cache is worth the wait."""

from __future__ import annotations

import math

import pytest

from llama_router import metrics
from llama_router.affinity import SessionAffinity
from llama_router.database import Database
from llama_router.models import (
    Provider,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.provider_manager import ProviderManager
from llama_router.request_logger import _prompt_tps_from_body, _prompt_tps_from_tail
from llama_router.router import Router
from llama_router.routing_stats import UNKNOWN_PROMPT_TPS, ProviderStats
from tests.conftest import FakeClock

MODEL = "m"
CHAT = "/v1/chat/completions"
SESSION = "0:m:abc"
# A long conversation: re-reading it at the default speed takes 100 s.
LARGE = 50_000


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def pm(db: Database, clock: FakeClock) -> ProviderManager:
    manager = ProviderManager(db)
    manager.stats = ProviderStats(clock=clock)
    return manager


async def _provider(db: Database, name: str) -> Provider:
    p = await db.add_provider(name, ProviderProtocol.OLLAMA)
    pid = p.require_id()
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.set_provider_models(pid, [ProviderModel(provider_id=pid, name=MODEL)])
    return p


async def _setup(
    db: Database, pm: ProviderManager, clock: FakeClock, *, cache_aware: bool = True
) -> tuple[Router, int, int, float]:
    """Two equal providers and a session bound to one of them. Returns the
    router, the bound and other provider ids, and the balanced queue weight."""
    a = (await _provider(db, "a")).require_id()
    b = (await _provider(db, "b")).require_id()
    rt = Router(
        db,
        pm,
        affinity=SessionAffinity(1800, 1000, clock),
        session_affinity=True,
        cache_aware=cache_aware,
    )
    table = await rt.current_table()
    weight = table.ranked_for(MODEL, CHAT, "balanced", None, False)[0].queue_weight[
        "balanced"
    ]
    return rt, a, b, weight


async def _bind(rt: Router, prompt_tokens: int) -> int:
    first = await rt.route(MODEL, CHAT, session=SESSION, prompt_tokens=prompt_tokens)
    assert first is not None
    return first.provider.require_id()


def _other(bound: int, a: int, b: int) -> int:
    return b if bound == a else a


async def test_large_session_stays_through_a_small_queue_and_moves_at_a_large_one(
    db: Database, pm: ProviderManager, clock: FakeClock
) -> None:
    rt, a, b, weight = await _setup(db, pm, clock)
    bound = await _bind(rt, LARGE)
    other = _other(bound, a, b)
    cache_ms = LARGE / UNKNOWN_PROMPT_TPS * 1000
    stay = math.floor(cache_ms / weight)
    assert stay >= 1  # the test needs a queue the cache is worth waiting out

    pm._active_requests[bound] = stay
    r = await rt.route(MODEL, CHAT, session=SESSION, prompt_tokens=LARGE)
    assert r is not None
    assert r.provider.id == bound
    assert r.affinity is not None
    assert not r.affinity.moved
    assert rt.affinity.overflows == 0

    pm._active_requests[bound] = stay + 1
    r = await rt.route(MODEL, CHAT, session=SESSION, prompt_tokens=LARGE)
    assert r is not None
    assert r.provider.id == other
    assert r.affinity is not None
    assert r.affinity.moved
    assert r.affinity.extra_wait_ms == pytest.approx((stay + 1) * weight)
    assert r.affinity.cache_value_ms == pytest.approx(cache_ms)
    stats = rt.affinity.stats()
    assert (stats["overflows"], stats["rebinds"]) == (1, 0)
    # The session is rebound where it went.
    binding = rt.affinity.peek(SESSION)
    assert binding is not None
    assert binding.provider_id == other


async def test_small_session_moves_at_the_first_extra_request(
    db: Database, pm: ProviderManager, clock: FakeClock
) -> None:
    rt, a, b, _ = await _setup(db, pm, clock)
    bound = await _bind(rt, 20)
    pm._active_requests[bound] = 1
    r = await rt.route(MODEL, CHAT, session=SESSION, prompt_tokens=20)
    assert r is not None
    assert r.provider.id == _other(bound, a, b)
    assert rt.affinity.overflows == 1


async def test_prompt_speed_samples_shift_the_threshold(
    db: Database, pm: ProviderManager, clock: FakeClock
) -> None:
    rt, a, b, weight = await _setup(db, pm, clock)
    bound = await _bind(rt, LARGE)
    other = _other(bound, a, b)
    stay = math.floor(LARGE / UNKNOWN_PROMPT_TPS * 1000 / weight)
    pm._active_requests[bound] = stay + 1

    # The other provider reads prompts ten times slower than assumed: the
    # cache is worth ten times as much, so the session stays.
    pm.stats.record_prompt_speed(other, MODEL, UNKNOWN_PROMPT_TPS / 10)
    r = await rt.route(MODEL, CHAT, session=SESSION, prompt_tokens=LARGE)
    assert r is not None
    assert r.provider.id == bound

    # Much faster than assumed: the cache is worth little, so it moves.
    fast = ProviderStats(clock=clock)
    fast.record_prompt_speed(other, MODEL, UNKNOWN_PROMPT_TPS * 100)
    pm.stats = fast
    pm._active_requests[bound] = 1
    r = await rt.route(MODEL, CHAT, session=SESSION, prompt_tokens=LARGE)
    assert r is not None
    assert r.provider.id == other


async def test_setting_off_never_moves(
    db: Database, pm: ProviderManager, clock: FakeClock
) -> None:
    rt, _a, _b, _ = await _setup(db, pm, clock, cache_aware=False)
    bound = await _bind(rt, 1)
    pm._active_requests[bound] = 50
    for _ in range(3):
        r = await rt.route(MODEL, CHAT, session=SESSION, prompt_tokens=1)
        assert r is not None
        assert r.provider.id == bound
        assert r.affinity is None
    assert rt.affinity.overflows == 0
    assert rt.affinity.hits == 3


async def test_pins_still_override_affinity(
    db: Database, pm: ProviderManager, clock: FakeClock
) -> None:
    from llama_router.router import RoutingPreferences

    rt, _a, _b, _ = await _setup(db, pm, clock)
    bound = await _bind(rt, LARGE)
    prefs = RoutingPreferences(pinned_providers={MODEL: bound})
    pm._active_requests[bound] = 1000
    r = await rt.route(MODEL, CHAT, prefs, session=SESSION, prompt_tokens=1)
    assert r is not None
    assert r.provider.id == bound


async def test_explain_agrees_when_a_session_stays_and_moves(
    db: Database, pm: ProviderManager, clock: FakeClock
) -> None:
    rt, a, b, weight = await _setup(db, pm, clock)
    bound = await _bind(rt, LARGE)
    other = _other(bound, a, b)
    stay = math.floor(LARGE / UNKNOWN_PROMPT_TPS * 1000 / weight)

    pm._active_requests[bound] = stay
    out = await rt.explain(MODEL, CHAT, session=SESSION)
    assert out["chosen"]["provider_id"] == bound
    assert out["session"]["honored"] is True
    assert out["session"]["weighing"]["moved"] is False

    pm._active_requests[bound] = stay + 1
    out = await rt.explain(MODEL, CHAT, session=SESSION)
    assert out["chosen"]["provider_id"] == other
    session = out["session"]
    assert session["honored"] is False
    assert session["weighing"]["moved"] is True
    assert session["weighing"]["prompt_tokens"] == LARGE
    assert "ms" in session["reason"]
    # Explaining changes nothing.
    assert rt.affinity.overflows == 0
    binding = rt.affinity.peek(SESSION)
    assert binding is not None
    assert binding.provider_id == bound
    routed = await rt.route(MODEL, CHAT, session=SESSION, prompt_tokens=LARGE)
    assert routed is not None
    assert routed.provider.id == other


async def test_overflows_are_exported(
    db: Database, pm: ProviderManager, clock: FakeClock
) -> None:
    rt, _a, _b, _ = await _setup(db, pm, clock)
    bound = await _bind(rt, 1)
    pm._active_requests[bound] = 1
    await rt.route(MODEL, CHAT, session=SESSION, prompt_tokens=1)
    text = metrics.render(
        metrics.collect(await pm.list_provider_infos(), pm.stats, router=rt)
    )
    assert "llama_router_session_affinity_overflows_total 1" in text.splitlines()


def test_prompt_speed_is_read_from_responses() -> None:
    ollama = {"prompt_eval_count": 1000, "prompt_eval_duration": 500_000_000}
    assert _prompt_tps_from_body(ollama) == pytest.approx(2000.0)
    llamacpp = {"timings": {"prompt_n": 300, "prompt_ms": 150.0}}
    assert _prompt_tps_from_body(llamacpp) == pytest.approx(2000.0)
    rate = {"timings": {"prompt_per_second": 812.5}}
    assert _prompt_tps_from_body(rate) == pytest.approx(812.5)
    assert _prompt_tps_from_body({"usage": {"prompt_tokens": 9}}) is None
    assert _prompt_tps_from_body(None) is None

    tail = b'{"done":true,"prompt_eval_count":400,"prompt_eval_duration":200000000}'
    assert _prompt_tps_from_tail(tail) == pytest.approx(2000.0)
    tail = b'data: {"timings":{"prompt_n":10,"prompt_per_second":1234.5}}'
    assert _prompt_tps_from_tail(tail) == pytest.approx(1234.5)
    assert _prompt_tps_from_tail(b'{"done":true}') is None


def test_prompt_speed_is_a_running_average() -> None:
    stats = ProviderStats()
    assert stats.prompt_tps(1, MODEL) == UNKNOWN_PROMPT_TPS
    stats.record_prompt_speed(1, MODEL, 1000.0)
    stats.record_prompt_speed(1, MODEL, 0.0)  # ignored
    assert stats.prompt_tps(1, MODEL) == pytest.approx(1000.0)
    stats.record_prompt_speed(1, MODEL, 2000.0)
    assert 1000.0 < stats.prompt_tps(1, MODEL) < 2000.0
    stats.forget_provider(1)
    assert stats.prompt_tps(1, MODEL) == UNKNOWN_PROMPT_TPS


async def test_handlers_bind_the_estimated_prompt(
    db: Database, pm: ProviderManager, clock: FakeClock
) -> None:
    from starlette.requests import Request

    from llama_router.auth import route_within_cap
    from llama_router.pricing import estimate_tokens

    rt, _a, _b, _ = await _setup(db, pm, clock)
    body = {"model": MODEL, "messages": [{"role": "user", "content": "x" * 3000}]}
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": CHAT,
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("router", 8080),
        }
    )
    await route_within_cap(rt, db, request, MODEL, None, body, session=SESSION)
    binding = rt.affinity.peek(SESSION)
    assert binding is not None
    assert binding.prompt_tokens == estimate_tokens(body, 0)[0] > 500
