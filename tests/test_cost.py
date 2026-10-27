"""Cost: prices, usage, the cheapest routing mode and daily spend caps."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from llama_router.auth import (
    generate_api_key,
    key_hash,
    key_prefix,
    routing_preferences_from_request,
)
from llama_router.models import (
    BenchmarkResult,
    Provider,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
    RequestLog,
)
from llama_router.pricing import FREE, PriceBook
from llama_router.provider_manager import ProviderManager
from llama_router.request_logger import (
    attach_pricer,
    log_request,
    usage_from_body,
    usage_from_tail,
)
from llama_router.router import Router, RoutingPreferences
from llama_router.spend import tracker_for
from tests.test_dashboard_auth import dash_client  # noqa: F401  (fixture)

CHAT = "/v1/chat/completions"


async def _provider(db, name, locality="local", models=("m",), details=None):
    p = await db.add_provider(name, ProviderProtocol.OPENAI)
    await db.update_provider_settings(p.id, locality=locality)
    await db.update_provider_status(p.id, ProviderStatus.IDLE)
    await db.set_provider_models(
        p.id,
        [ProviderModel(provider_id=p.id, name=m, details=details) for m in models],
    )
    for m in models:
        await db.save_benchmark(
            BenchmarkResult(
                provider_id=p.id,
                model_name=m,
                protocol="openai",
                startup_time_ms=100,
                tokens_per_second=50.0,
            )
        )
    return await db.get_provider(p.id)


def test_price_book_resolution() -> None:
    providers = [
        Provider(id=1, name="cloud", locality="cloud"),
        Provider(id=2, name="lan"),
    ]
    models = {
        1: [
            ProviderModel(
                provider_id=1,
                name="listed",
                details={"pricing": {"prompt": "0.000003", "completion": "0.000015"}},
            ),
            ProviderModel(
                provider_id=1,
                name="varies",
                details={"pricing": {"prompt": "-1", "completion": "-1"}},
            ),
        ]
    }
    rows = [
        {
            "provider_id": 1,
            "model_pattern": "*",
            "input_per_mtok": 9,
            "output_per_mtok": 9,
        },
        {
            "provider_id": 1,
            "model_pattern": "gpt-*",
            "input_per_mtok": 2,
            "output_per_mtok": 8,
        },
        {
            "provider_id": 1,
            "model_pattern": "gpt-mini",
            "input_per_mtok": 0.1,
            "output_per_mtok": 0.4,
        },
    ]
    book = PriceBook(providers, models, rows)
    assert book.price(1, "gpt-mini").input_per_mtok == 0.1
    assert book.price(1, "gpt-big").input_per_mtok == 2
    # A hand-set "*" wins over the provider's own list.
    assert book.price(1, "listed").input_per_mtok == 9
    no_star = PriceBook(providers, models, rows[1:])
    listed = no_star.price(1, "listed")
    assert (listed.input_per_mtok, listed.output_per_mtok, listed.source) == (
        pytest.approx(3.0),
        pytest.approx(15.0),
        "listed",
    )
    assert no_star.price(1, "varies") is None
    assert no_star.price(2, "anything") is FREE
    assert book.cost(1, "gpt-mini", 1_000_000, 1_000_000) == pytest.approx(0.5)
    assert book.cost(1, "gpt-mini", None, None) is None


def test_usage_parsing() -> None:
    assert usage_from_body(
        {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    ) == (10, 5)
    assert usage_from_body({"usage": {"input_tokens": 7, "output_tokens": 3}}) == (7, 3)
    assert usage_from_body({"prompt_eval_count": 4, "eval_count": 9}) == (4, 9)
    assert usage_from_body({"choices": []}) == (None, None)
    tail = (
        b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":34}}\n\n'
        b"data: [DONE]\n\n"
    )
    assert usage_from_tail(tail) == (12, 34)
    assert usage_from_tail(b'{"done":true,"prompt_eval_count":8,"eval_count":21}') == (
        8,
        21,
    )
    assert usage_from_tail(b"data: [DONE]") == (None, None)


async def test_cheapest_mode(db) -> None:
    pm = ProviderManager(db)
    await _provider(db, "pricey", "cloud")
    cheap = await _provider(db, "cheap", "cloud")
    await _provider(db, "unknown", "cloud")
    pricey = await db.get_provider_by_name("pricey")
    await db.set_model_price(pricey.id, "*", 10, 30)
    await db.set_model_price(cheap.id, "m", 0.5, 1.5)
    rt = Router(db, pm)
    prefs = RoutingPreferences(mode="cheapest")
    assert (await rt.route("m", CHAT, prefs)).provider.id == cheap.id
    lan = await _provider(db, "lan")
    await rt.rebuild()
    assert (await rt.route("m", CHAT, prefs)).provider.id == lan.id
    out = await rt.explain("m", CHAT, prefs)
    ranked = [
        c["provider"]
        for c in sorted(out["steps"][0]["candidates"], key=lambda c: c["rank"])
    ]
    assert ranked == ["lan", "cheap", "pricey", "unknown"]


async def test_logged_cost_and_key(db) -> None:
    pm = ProviderManager(db)
    p = await _provider(db, "cloud", "cloud")
    await db.set_model_price(p.id, "m", 1.0, 2.0)
    rt = Router(db, pm)
    await rt.rebuild()
    attach_pricer(rt.price_for)
    try:
        req = Request({"type": "http", "headers": [], "method": "POST", "path": "/"})
        req.state.api_key_id = 7
        await log_request(
            db,
            provider=p,
            protocol="v1",
            endpoint=CHAT,
            request=req,
            model="m",
            duration_ms=5,
            response_body={"usage": {"prompt_tokens": 1000, "completion_tokens": 500}},
        )
    finally:
        attach_pricer(None)
    (entry,) = await db.get_request_logs(10, 0)
    assert (entry.api_key_id, entry.prompt_tokens, entry.completion_tokens) == (
        7,
        1000,
        500,
    )
    assert entry.cost_usd == pytest.approx(0.002)


def _request(key: str) -> Request:
    return Request(
        {
            "type": "http",
            "headers": [(b"api-key", key.encode())],
            "method": "POST",
            "path": "/",
        }
    )


async def test_spend_cap_limits_a_key_to_local(db) -> None:
    async def key(locality=None):
        plain = generate_api_key()
        kid = await db.create_api_key(
            key_prefix(plain),
            key_hash(plain),
            "balanced",
            True,
            locality=locality,
            daily_spend_cap_usd=1.0,
        )
        return plain, kid

    plain, kid = await key()
    prefs = await routing_preferences_from_request(db, _request(plain))
    assert prefs.locality is None and not prefs.spend_capped
    await db.save_request_log(
        RequestLog(protocol="v1", endpoint=CHAT, api_key_id=kid, cost_usd=0.6)
    )
    tracker_for(db).add(kid, 0.6)  # what log_request does
    req = _request(plain)
    await db.save_request_log(
        RequestLog(protocol="v1", endpoint=CHAT, api_key_id=kid, cost_usd=0.6)
    )
    tracker_for(db).add(kid, 0.6)
    prefs = await routing_preferences_from_request(db, req)
    assert prefs.locality == "local" and prefs.spend_capped
    assert req.state.api_key_id == kid

    cloud_plain, cloud_id = await key("cloud")
    await db.save_request_log(
        RequestLog(protocol="v1", endpoint=CHAT, api_key_id=cloud_id, cost_usd=5.0)
    )
    with pytest.raises(HTTPException) as exc:
        await routing_preferences_from_request(db, _request(cloud_plain))
    assert exc.value.status_code == 429


async def test_dashboard_prices_and_caps(dash_client) -> None:  # noqa: F811
    client, db = dash_client
    p = await db.add_provider("cloud", ProviderProtocol.OPENAI)
    r = await client.post(
        f"/api/providers/{p.id}/prices",
        json={"model_pattern": "gpt-*", "input_per_mtok": "2.5", "output_per_mtok": 10},
    )
    assert r.status_code == 200
    data = (await client.get(f"/api/providers/{p.id}/prices")).json()
    (row,) = data["prices"]
    assert (row["model_pattern"], row["input_per_mtok"]) == ("gpt-*", 2.5)
    bad = await client.post(
        f"/api/providers/{p.id}/prices",
        json={"model_pattern": "x", "input_per_mtok": -1, "output_per_mtok": 1},
    )
    assert bad.status_code == 400
    assert (
        await client.delete(f"/api/providers/{p.id}/prices/{row['id']}")
    ).status_code == 200

    r = await client.post(
        "/api/keys/generate",
        json={"routing_mode": "cheapest", "daily_spend_cap_usd": "5"},
    )
    assert r.status_code == 200 and r.json()["daily_spend_cap_usd"] == 5.0
    (k,) = await db.list_api_keys()
    assert (k["routing_mode"], k["daily_spend_cap_usd"]) == ("cheapest", 5.0)
    r = await client.post(
        f"/api/keys/{k['id']}/spend-cap", json={"daily_spend_cap_usd": None}
    )
    assert r.status_code == 200
    assert (await db.list_api_keys())[0]["daily_spend_cap_usd"] is None
    page = await client.get("/")
    assert "setKeySpendCap(" in page.text and "cheapest" in page.text
    detail = await client.get(f"/providers/{p.id}")
    assert 'id="prices-card"' in detail.text
