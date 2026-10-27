"""Cloud presets: the table, the Add dialog, extra headers, added models and
provisioning from the environment."""

from __future__ import annotations

import httpx
import pytest

from llama_router.cloud_presets import (
    PRESETS,
    env_provisioned,
    get_preset,
    preset_for_url,
)
from llama_router.models import ProviderProtocol, ProviderStatus
from llama_router.provider_manager import ProviderManager
from llama_router.provider_secrets import decrypt_api_key, parse_extra_headers
from llama_router.upstream_url import split_api_base
from tests.test_dashboard_auth import dash_client  # noqa: F401  (fixture)

pytestmark = pytest.mark.httpx_mock(
    assert_all_responses_were_requested=False,
    can_send_already_matched_responses=True,
)


def test_table_is_sane() -> None:
    ids = [p.id for p in PRESETS]
    assert len(ids) == len(set(ids))
    for asked in ("openai", "openrouter", "anthropic", "perplexity"):
        assert get_preset(asked) is not None
    for p in PRESETS:
        assert p.base_url.startswith("https://")
        assert p.protocol in ("openai", "ollama")
        assert p.hosts and p.docs_url and p.key_url
        if p.protocol == "openai" and not p.url_params:
            # Every OpenAI preset URL is an API base, not a server root.
            assert split_api_base(p.base_url)[1] != "/v1" or p.base_url.endswith(
                "/v1"
            ), p.id


def test_preset_for_url_by_hostname() -> None:
    assert preset_for_url("https://api.openai.com/v1").id == "openai"
    assert preset_for_url("https://my-res.openai.azure.com/openai/v1").id == (
        "azure-openai"
    )
    assert preset_for_url("http://192.168.1.5:8080") is None


def test_url_params() -> None:
    azure = get_preset("azure-openai")
    assert azure.url(resource="my-res") == "https://my-res.openai.azure.com/openai/v1"
    with pytest.raises(ValueError):
        azure.url(resource="bad/host")
    with pytest.raises(ValueError):
        azure.url()


def test_env_provisioned(monkeypatch) -> None:
    for p in PRESETS:
        monkeypatch.delenv(p.env_var, raising=False)
    monkeypatch.setenv("LLAMA_ROUTER_PROVIDER_OPENROUTER_API_KEY", "sk-or")
    monkeypatch.setenv("LLAMA_ROUTER_PROVIDER_AZURE_OPENAI_API_KEY", "az")
    monkeypatch.setenv("LLAMA_ROUTER_PROVIDER_AZURE_OPENAI_RESOURCE", "res")
    got = {p.id: (key, params) for p, key, params in env_provisioned()}
    assert got == {
        "openrouter": ("sk-or", {}),
        "azure-openai": ("az", {"resource": "res"}),
    }


def test_parse_extra_headers() -> None:
    assert parse_extra_headers("anthropic-version: 2023-06-01\n\nX-Title: me") == {
        "anthropic-version": "2023-06-01",
        "X-Title": "me",
    }
    for bad in ("no colon", "Authorization: Bearer x", "x-api-key: k", "bad name: v"):
        with pytest.raises(ValueError):
            parse_extra_headers(bad)


async def test_provision_from_env_creates_then_updates(db, monkeypatch) -> None:
    for p in PRESETS:
        monkeypatch.delenv(p.env_var, raising=False)
    monkeypatch.setenv("LLAMA_ROUTER_PROVIDER_ANTHROPIC_API_KEY", "sk-ant-1")
    pm = ProviderManager(db)
    (made,) = await pm.provision_from_env()
    p = await db.get_provider(made.id)
    assert (p.name, p.preset, p.protocol) == (
        "Anthropic",
        "anthropic",
        ProviderProtocol.OPENAI,
    )
    assert p.extra_headers == {"anthropic-version": "2023-06-01"}
    assert p.api_key_header == "Authorization, x-api-key"
    assert decrypt_api_key(p.api_key_enc) == "sk-ant-1"
    (addr,) = await db.get_addresses(p.id)
    assert addr.url == "https://api.anthropic.com/v1"

    monkeypatch.setenv("LLAMA_ROUTER_PROVIDER_ANTHROPIC_API_KEY", "sk-ant-2")
    await pm.provision_from_env()
    providers = await db.list_providers()
    assert len(providers) == 1
    assert decrypt_api_key(providers[0].api_key_enc) == "sk-ant-2"
    await pm.stop()


async def test_extra_headers_and_key_go_upstream(db, httpx_mock) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "claude-x"}]})
        return httpx.Response(404)

    httpx_mock.add_callback(handler, is_reusable=True, is_optional=True)
    pm = ProviderManager(db)
    p = await pm.add_provider(
        "ant",
        "https://api.anthropic.com/v1",
        ProviderProtocol.OPENAI,
        api_key="sk",
        api_key_header="Authorization, x-api-key",
        provider_settings={"extra_headers": {"anthropic-version": "2023-06-01"}},
    )
    await pm.stop()
    listing = next(r for r in seen if r.url.path == "/v1/models")
    assert listing.headers["anthropic-version"] == "2023-06-01"
    assert listing.headers["x-api-key"] == "sk"
    assert listing.headers["authorization"] == "Bearer sk"
    assert (await db.get_provider(p.id)).status == ProviderStatus.IDLE


async def test_added_models_survive_a_failed_listing(db, httpx_mock) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "agent-x"}]})
        return httpx.Response(404)

    httpx_mock.add_callback(handler, is_reusable=True, is_optional=True)
    pm = ProviderManager(db)
    p = await pm.add_provider(
        "pplx",
        "https://api.perplexity.ai/v1",
        ProviderProtocol.OPENAI,
        provider_settings={"manual_models": ["sonar", "sonar-pro"]},
    )
    names = {m.name: m.details for m in await db.get_provider_models(p.id)}
    assert names == {
        "agent-x": None,
        "sonar": {"_manual": True},
        "sonar-pro": {"_manual": True},
    }

    httpx_mock.reset()
    httpx_mock.add_callback(
        lambda r: (
            httpx.Response(500) if r.url.path == "/v1/models" else httpx.Response(404)
        ),
        is_reusable=True,
        is_optional=True,
    )
    await pm._discover_provider(await db.get_provider(p.id))
    assert {m.name for m in await db.get_provider_models(p.id)} == {
        "sonar",
        "sonar-pro",
    }
    await pm.stop()


async def test_dashboard_add_from_preset(dash_client, httpx_mock) -> None:  # noqa: F811
    httpx_mock.add_exception(httpx.ConnectError("offline"), is_reusable=True)
    client, db = dash_client
    r = await client.post(
        "/providers/add",
        data={"name": "Perplexity", "url": "", "preset": "perplexity", "api_key": "k"},
    )
    assert r.status_code == 303
    (p,) = await db.list_providers()
    assert p.preset == "perplexity" and p.protocol == ProviderProtocol.OPENAI
    assert p.manual_models == [
        "sonar",
        "sonar-pro",
        "sonar-reasoning-pro",
        "sonar-deep-research",
    ]
    (addr,) = await db.get_addresses(p.id)
    assert addr.url == "https://api.perplexity.ai/v1"

    r = await client.post(
        "/providers/add",
        data={"name": "Az", "url": "x", "preset": "azure-openai", "api_key": "k"},
    )
    assert r.status_code == 400  # needs a resource name
    r = await client.post(
        "/providers/add",
        data={
            "name": "Az",
            "url": "",
            "preset": "azure-openai",
            "preset_resource": "my-res",
            "api_key": "k",
        },
    )
    assert r.status_code == 303
    az = await db.get_provider_by_name("Az")
    assert az.api_key_header == "api-key"
    (addr,) = await db.get_addresses(az.id)
    assert addr.url == "https://my-res.openai.azure.com/openai/v1"

    page = await client.get("/")
    assert 'data-preset="openrouter"' in page.text


async def test_dashboard_edit_extra_headers_and_models(
    dash_client, httpx_mock  # noqa: F811
) -> None:
    httpx_mock.add_exception(httpx.ConnectError("offline"), is_reusable=True)
    client, db = dash_client
    p = await db.add_provider("box", ProviderProtocol.OPENAI)
    await db.add_address(p.id, "http://box:8080", True)
    form = {
        "name": "box",
        "protocol": "openai",
        "extra_headers": "X-Team: ml",
        "manual_models": "a, b\nc",
        "settings_form": "1",
    }
    r = await client.post(f"/providers/{p.id}/edit", data=form)
    assert r.status_code == 303
    p = await db.get_provider(p.id)
    assert p.extra_headers == {"X-Team": "ml"}
    assert p.manual_models == ["a", "b", "c"]
    page = await client.get(f"/providers/{p.id}")
    assert "X-Team: ml" in page.text

    r = await client.post(
        f"/providers/{p.id}/edit", data={**form, "extra_headers": "Authorization: x"}
    )
    assert r.status_code == 400
    await client.post(
        f"/providers/{p.id}/edit",
        data={**form, "extra_headers": "", "manual_models": ""},
    )
    p = await db.get_provider(p.id)
    assert p.extra_headers is None and p.manual_models is None
