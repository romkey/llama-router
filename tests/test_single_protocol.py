"""One protocol per provider: parsing, worker mapping, forms and benchmarks."""

from __future__ import annotations

import httpx
import pytest

from llama_router.models import ProviderModel, ProviderProtocol
from llama_router.provider_manager import ProviderManager, worker_protocol
from tests.test_dashboard_auth import dash_client  # noqa: F401  (fixture)


def test_parse_accepts_old_name_and_rejects_both() -> None:
    assert ProviderProtocol.parse("openai") is ProviderProtocol.OPENAI
    assert ProviderProtocol.parse("llamacpp") is ProviderProtocol.OPENAI
    assert ProviderProtocol.parse(" Ollama ") is ProviderProtocol.OLLAMA
    assert ProviderProtocol.parse(None) is ProviderProtocol.OLLAMA
    with pytest.raises(ValueError):
        ProviderProtocol.parse("both")


def test_worker_protocol_prefers_ollama_target() -> None:
    # The Ollama port of a relayed router serves the OpenAI routes too.
    assert worker_protocol(["ollama", "llamacpp"]) is ProviderProtocol.OLLAMA
    assert worker_protocol(["ollama"]) is ProviderProtocol.OLLAMA
    assert worker_protocol(["llamacpp"]) is ProviderProtocol.OPENAI


def test_worker_target_keeps_wire_names() -> None:
    from llama_router.models import Provider
    from llama_router.worker_link.protocol import TARGETS

    for proto in ProviderProtocol:
        target = Provider(name="w", protocol=proto).worker_target
        assert target in TARGETS


@pytest.mark.asyncio
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_add_provider_form_takes_protocol(
    dash_client, httpx_mock  # noqa: F811
) -> None:
    httpx_mock.add_exception(httpx.ConnectError("refused"), is_reusable=True)
    client, db = dash_client

    r = await client.post(
        "/providers/add",
        data={"name": "x", "url": "http://x:8080", "protocol": "both"},
    )
    assert r.status_code == 400
    assert await db.list_providers() == []

    await client.post(
        "/providers/add",
        data={"name": "lcpp", "url": "http://lcpp:8080", "protocol": "llamacpp"},
    )
    (p,) = await db.list_providers()
    assert p.protocol is ProviderProtocol.OPENAI
    (addr,) = await db.get_addresses(p.id)
    assert addr.url == "http://lcpp:8080"


@pytest.mark.asyncio
async def test_benchmark_rejects_the_other_protocol(db) -> None:
    p = await db.add_provider("lcpp", ProviderProtocol.OPENAI)
    await db.set_provider_models(p.id, [ProviderModel(provider_id=p.id, name="m")])
    pm = ProviderManager(db)
    try:
        with pytest.raises(RuntimeError, match="speaks the openai API, not ollama"):
            await pm.benchmark_provider(p.id, "m", "ollama")
    finally:
        await pm.stop()
