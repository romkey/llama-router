"""Running cloud providers: rate limits, big catalogs, benchmarks, timeouts."""

from __future__ import annotations

import time

import httpx
import pytest

from llama_router.auto_benchmark import AutoBenchmarker
from llama_router.models import (
    BenchmarkResult,
    Provider,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.provider_manager import ProviderManager
from llama_router.request_logger import _is_provider_failure
from llama_router.router import Router
from llama_router.routing_stats import ProviderStats, retry_after_seconds
from tests.test_dashboard_auth import dash_client  # noqa: F401  (fixture)

pytestmark = pytest.mark.httpx_mock(
    assert_all_responses_were_requested=False,
    can_send_already_matched_responses=True,
)

CHAT = "/v1/chat/completions"


async def _provider(
    db, name, locality="local", models=("m",), startup=100.0, bench=True, **kw
):
    p = await db.add_provider(name, ProviderProtocol.OPENAI)
    await db.update_provider_settings(p.id, locality=locality, **kw)
    await db.update_provider_status(p.id, ProviderStatus.IDLE)
    await db.set_provider_models(
        p.id, [ProviderModel(provider_id=p.id, name=m) for m in models]
    )
    for m in models if bench else ():
        await db.save_benchmark(
            BenchmarkResult(
                provider_id=p.id,
                model_name=m,
                protocol="openai",
                startup_time_ms=startup,
                tokens_per_second=50.0,
            )
        )
    return await db.get_provider(p.id)


def test_retry_after() -> None:
    assert retry_after_seconds("7") == 7.0
    assert retry_after_seconds(None) == 10.0
    assert retry_after_seconds("soon") == 10.0
    future = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(time.time() + 60))
    assert 55 <= retry_after_seconds(future) <= 61


def test_busy_cooldown_expires() -> None:
    now = [100.0]
    stats = ProviderStats(clock=lambda: now[0])
    stats.mark_busy(1, 5)
    assert stats.busy_for(1) == 5
    stats.mark_busy(1, 1)  # a shorter one does not cut it short
    assert stats.busy_for(1) == 5
    now[0] += 6
    assert stats.busy_for(1) == 0
    stats.mark_busy(2, 10_000)
    assert stats.busy_for(2) == 300


def test_429_is_not_a_provider_failure() -> None:
    assert not _is_provider_failure("HTTP 429: slow down")
    assert _is_provider_failure("HTTP 503: down")


async def test_router_skips_a_rate_limited_provider(db) -> None:
    pm = ProviderManager(db)
    fast = await _provider(db, "fast", startup=60)
    slow = await _provider(db, "slow", startup=5000)
    rt = Router(db, pm)
    assert (await rt.route("m", CHAT)).provider.id == fast.id
    pm.stats.mark_busy(fast.id, 30)
    assert (await rt.route("m", CHAT)).provider.id == slow.id
    out = await rt.explain("m", CHAT)
    row = next(c for c in out["steps"][0]["candidates"] if c["provider"] == "fast")
    assert row["excluded"].startswith("rate limited (HTTP 429)")
    # When every provider is cooling down, the best is still used.
    pm.stats.mark_busy(slow.id, 30)
    assert (await rt.route("m", CHAT)).provider.id == fast.id


async def test_client_429_marks_provider_busy(db, httpx_mock) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(429, headers={"Retry-After": "42"}, json={})
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    httpx_mock.add_callback(handler, is_reusable=True, is_optional=True)
    pm = ProviderManager(db)
    p = await pm.add_provider(
        "cloud", "https://api.example.com/v1", ProviderProtocol.OPENAI
    )
    client = pm.get_llamacpp_client(p.id)
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat_completions({"model": "m"})
    assert 40 <= pm.stats.busy_for(p.id) <= 42
    await pm.stop()


async def test_model_filter_limits_discovery(db, httpx_mock) -> None:
    listed = ["anthropic/claude-x", "openai/gpt-y", "someone/tiny-z"]
    httpx_mock.add_callback(
        lambda r: httpx.Response(200, json={"data": [{"id": m} for m in listed]}),
        is_reusable=True,
        is_optional=True,
    )
    pm = ProviderManager(db)
    p = await pm.add_provider(
        "or",
        "https://openrouter.ai/api/v1",
        ProviderProtocol.OPENAI,
        provider_settings={
            "model_filter": "anthropic/*\nopenai/*",
            "manual_models": ["someone/kept-by-hand"],
        },
    )
    names = {m.name for m in await db.get_provider_models(p.id)}
    assert names == {"anthropic/claude-x", "openai/gpt-y", "someone/kept-by-hand"}
    await pm.stop()


def test_keeps_model() -> None:
    p = Provider(name="x", model_filter="a/*, b")
    assert p.keeps_model("a/1") and p.keeps_model("b") and not p.keeps_model("c")
    assert Provider(name="y").keeps_model("anything")


async def test_cloud_skipped_by_auto_benchmark_unless_opted_in(db) -> None:
    pm = ProviderManager(db)
    await _provider(db, "lan", bench=False)
    await _provider(db, "cloud", locality="cloud", bench=False)
    await _provider(db, "cloud-on", locality="cloud", bench=False, auto_benchmark=True)
    await _provider(db, "lan-off", bench=False, auto_benchmark=False)
    ab = AutoBenchmarker(db, pm)
    assert {j.provider_name for j in await ab.plan()} == {"lan", "cloud-on"}
    await pm.stop()


async def test_cloud_models_count_as_loaded(db) -> None:
    pm = ProviderManager(db)
    await _provider(db, "cloud", locality="cloud")
    await _provider(db, "lan")
    rt = Router(db, pm)
    table = await rt.rebuild()
    hot = {c.provider.name: c.is_hot for c in table.all_candidates[("m", CHAT)]}
    assert hot == {"cloud": True, "lan": False}
    await pm.stop()


async def test_read_timeout_reaches_the_client(db) -> None:
    pm = ProviderManager(db)
    p = await _provider(db, "slow", read_timeout_s=600.0)
    await db.add_address(p.id, "http://slow:8080", True)
    await pm._rebuild_clients(await db.get_provider(p.id))
    assert pm.get_llamacpp_client(p.id)._http.timeout.read == 600.0
    await pm.stop()


async def test_dashboard_edit_cloud_settings(
    dash_client, httpx_mock
) -> None:  # noqa: F811
    httpx_mock.add_exception(httpx.ConnectError("offline"), is_reusable=True)
    client, db = dash_client
    p = await db.add_provider("or", ProviderProtocol.OPENAI)
    await db.add_address(p.id, "https://openrouter.ai/api/v1", True)
    form = {
        "name": "or",
        "protocol": "openai",
        "settings_form": "1",
        "model_filter": "anthropic/*, openai/*",
        "read_timeout_s": "300",
        "auto_benchmark": "on",
    }
    assert (await client.post(f"/providers/{p.id}/edit", data=form)).status_code == 303
    p = await db.get_provider(p.id)
    assert p.model_filter == "anthropic/*\nopenai/*"
    assert (p.read_timeout_s, p.auto_benchmark) == (300.0, True)
    bad = {**form, "read_timeout_s": "0"}
    assert (await client.post(f"/providers/{p.id}/edit", data=bad)).status_code == 400
    page = await client.get(f"/providers/{p.id}")
    assert "anthropic/*" in page.text and "PROVIDER_IS_CLOUD" in page.text
