"""Prometheus metrics: text format, collected values and access control."""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio

from llama_router import metrics
from llama_router.config import settings
from llama_router.dashboard.auth_core import hash_password
from llama_router.dashboard.middleware import invalidate_dashboard_user_count_cache
from llama_router.models import (
    BenchmarkResult,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.routing_stats import ProviderStats
from tests.test_dashboard_auth import _session_cookie_header, dash_client  # noqa: F401

CHAT = "/v1/chat/completions"


def _lines(text: str) -> dict[str, str]:
    """Sample lines keyed by ``name{labels}``."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        key, _, value = line.rpartition(" ")
        out[key] = value
    return out


def test_family_renders_prometheus_text() -> None:
    fam = metrics.Family("thing", "gauge", 'Has "quotes" and\nnewlines.')
    fam.add(1.5, provider='a"b', url="http://x/\\")
    fam.add(True)
    fam.add(None)
    text = metrics.render([fam])
    assert text.splitlines() == [
        '# HELP llama_router_thing Has \\"quotes\\" and\\nnewlines.',
        "# TYPE llama_router_thing gauge",
        'llama_router_thing{provider="a\\"b",url="http://x/\\\\"} 1.5',
        "llama_router_thing 1",
        "llama_router_thing NaN",
    ]
    assert text.endswith("\n")


def test_fmt() -> None:
    assert metrics._fmt(3) == "3"
    assert metrics._fmt(float("inf")) == "+Inf"
    assert metrics._fmt(0.1) == "0.1"
    assert metrics._fmt(False) == "0"


@pytest_asyncio.fixture
async def seeded(dash_client: Any) -> tuple[Any, Any, int, int]:
    client, db = dash_client
    p = await db.add_provider("gpu", ProviderProtocol.OPENAI)
    await db.update_provider_status(p.id, ProviderStatus.IDLE)
    addr = await db.add_address(p.id, "http://10.0.0.2:8080", True)
    await db.set_address_live(addr.id, True)
    await db.set_provider_models(p.id, [ProviderModel(provider_id=p.id, name="m")])
    await db.save_benchmark(
        BenchmarkResult(
            provider_id=p.id,
            model_name="m",
            protocol="openai",
            startup_time_ms=1200.0,
            tokens_per_second=42.5,
        )
    )
    cloud = await db.add_provider("api", ProviderProtocol.OPENAI)
    await db.update_provider_settings(cloud.id, locality="cloud")
    await db.update_provider_status(cloud.id, ProviderStatus.OFFLINE)
    return client, db, p.id, cloud.id


async def test_collect_reports_providers_benchmarks_and_requests(
    seeded: tuple[Any, Any, int, int],
) -> None:
    client, db, pid, cloud_id = seeded
    from llama_router.dashboard import deps

    pm = deps.get_pm()
    stats: ProviderStats = pm.stats
    stats.record_request(pid, "m", ok=True, tokens_per_second=30.0, ttft_ms=90.0)
    stats.record_request(pid, "m", ok=False)
    pm.acquire(pid)
    try:
        text = metrics.render(metrics.collect(await pm.list_provider_infos(), stats))
    finally:
        pm.release(pid)
    got = _lines(text)
    gpu = f'provider="gpu",provider_id="{pid}",protocol="openai",locality="local"'
    assert got['llama_router_providers{status="idle"}'] == "1"
    assert got['llama_router_providers{status="offline"}'] == "1"
    assert got["llama_router_models"] == "1"
    assert got[f"llama_router_provider_up{{{gpu}}}"] == "1"
    assert got[f"llama_router_provider_active_requests{{{gpu}}}"] == "1"
    assert got[f'llama_router_address_up{{{gpu},url="http://10.0.0.2:8080"}}'] == "1"
    bench = f'{gpu},model="m",benchmark_protocol="openai"'
    assert got[f"llama_router_benchmark_tokens_per_second{{{bench}}}"] == "42.5"
    assert got[f"llama_router_benchmark_startup_milliseconds{{{bench}}}"] == "1200.0"
    req = f'{gpu},model="m",endpoint_class="generate"'
    assert got[f'llama_router_requests_total{{{req},outcome="ok"}}'] == "1"
    assert got[f'llama_router_requests_total{{{req},outcome="error"}}'] == "1"
    assert got[f"llama_router_model_tokens_per_second{{{req}}}"] == "30.0"
    assert (
        got[f"llama_router_model_time_to_first_token_milliseconds{{{req}}}"] == "90.0"
    )
    cloud = (
        f'provider="api",provider_id="{cloud_id}",protocol="openai",locality="cloud"'
    )
    assert got[f"llama_router_provider_up{{{cloud}}}"] == "0"
    assert "llama_router_routing_table_version" not in text  # no router attached


async def test_metrics_endpoint_open_before_first_user(
    seeded: tuple[Any, Any, int, int],
) -> None:
    client, db, pid, _cloud = seeded
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert "llama_router_info{version=" in resp.text
    assert "llama_router_worker_link_connected" in resp.text


async def test_metrics_endpoint_needs_token_or_session(
    seeded: tuple[Any, Any, int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, db, pid, _cloud = seeded
    uid = await db.create_dashboard_user("admin", hash_password("pw"), True)
    invalidate_dashboard_user_count_cache()

    assert (await client.get("/metrics")).status_code == 302  # to /login

    monkeypatch.setattr(settings, "metrics_token", "scrape-me")
    assert (await client.get("/metrics")).status_code == 302
    bad = {"Authorization": "Bearer nope"}
    assert (await client.get("/metrics", headers=bad)).status_code == 302
    good = {"Authorization": "Bearer scrape-me"}
    assert (await client.get("/metrics", headers=good)).status_code == 200

    monkeypatch.setattr(settings, "metrics_token", "")
    resp = await client.get("/metrics", headers=_session_cookie_header(uid))
    assert resp.status_code == 200


async def test_metrics_endpoint_can_be_disabled(
    seeded: tuple[Any, Any, int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, db, pid, _cloud = seeded
    monkeypatch.setattr(settings, "metrics_enabled", False)
    assert (await client.get("/metrics")).status_code == 404
