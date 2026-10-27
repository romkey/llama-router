"""The dashboard port can also serve the LLM APIs and worker links."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from starlette.datastructures import URL
from starlette.requests import Request

from llama_router.api import deps as api_deps
from llama_router.api.app import app as api_app
from llama_router.config import settings
from llama_router.dashboard import deps as dash_deps
from llama_router.dashboard.app import app as dashboard_app
from llama_router.dashboard.auth_core import hash_password
from llama_router.dashboard.middleware import invalidate_dashboard_user_count_cache
from llama_router.dashboard.routes import _hub_url_hint
from llama_router.database import Database
from llama_router.llamacpp_api import deps as v1_deps
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router
from llama_router.shared_port import SharedPortApp, api_routes, routes_to_api

_ROUTES = api_routes(api_app)


def _scope(method: str, path: str, kind: str = "http") -> dict:
    scope = {"type": kind, "path": path, "root_path": "", "headers": []}
    if kind == "http":
        scope["method"] = method
    return scope


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/api/chat"),
        ("POST", "/api/generate"),
        ("POST", "/api/embed"),
        ("POST", "/api/embeddings"),
        ("GET", "/api/tags"),
        ("GET", "/api/version"),
        ("GET", "/api/ps"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/chat/completions"),
        ("POST", "/v1/completions"),
        ("POST", "/v1/embeddings"),
        ("POST", "/v1/responses"),
        ("GET", "/v1/models"),
        ("GET", "/v1/models/llama3.2:latest"),
        ("POST", "/v1/audio/speech"),
        ("POST", "/v1/images/generations"),
        ("POST", "/v1/systemone"),
        ("POST", "/systemone"),
        ("POST", "/v1/decisions"),
        ("POST", "/decisions"),
    ],
)
def test_llm_routes_go_to_api(method: str, path: str) -> None:
    assert routes_to_api(_scope(method, path), _ROUTES)


def test_worker_socket_goes_to_api() -> None:
    assert routes_to_api(_scope("", "/worker/ws", kind="websocket"), _ROUTES)


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/"),
        ("GET", "/health"),
        ("GET", "/login"),
        ("GET", "/decisions"),
        ("GET", "/api/status"),
        ("POST", "/api/keys/generate"),
        ("POST", "/api/pull"),
        ("GET", "/api/workers"),
        ("GET", "/docs"),
        ("GET", "/openapi.json"),
        ("GET", "/api/chat"),
    ],
)
def test_dashboard_routes_stay_on_dashboard(method: str, path: str) -> None:
    assert not routes_to_api(_scope(method, path), _ROUTES)


def test_lifespan_goes_to_dashboard() -> None:
    assert not routes_to_api({"type": "lifespan"}, _ROUTES)


@pytest_asyncio.fixture
async def shared_client(tmp_path, monkeypatch):
    """The combined app with a dashboard user, so the dashboard demands a login."""
    monkeypatch.setattr(
        settings, "session_secret", "test-session-secret-key-min-16chars"
    )
    db = Database(str(tmp_path / "shared.db"))
    await db.connect()
    pm = ProviderManager(db)
    rt = Router(db, pm)
    pm.attach_router(rt)
    api_deps.init(db, pm, rt)
    v1_deps.init(db, pm, rt)
    dash_deps.invalidate_dashboard_session_secret_cache()
    invalidate_dashboard_user_count_cache()
    dash_deps.init(db, pm)
    await db.create_dashboard_user("admin", hash_password("pw-123456"), True)
    await db.set_allow_unauthenticated(False)

    transport = ASGITransport(app=SharedPortApp(dashboard_app, api_app))
    async with AsyncClient(
        transport=transport, base_url="http://testserver", follow_redirects=False
    ) as client:
        yield client

    await db.close()
    dash_deps._db = None
    dash_deps._pm = None
    dash_deps.invalidate_dashboard_session_secret_cache()
    invalidate_dashboard_user_count_cache()


async def test_api_request_skips_dashboard_login(shared_client) -> None:
    r = await shared_client.get("/api/version")
    assert r.status_code == 200
    assert "version" in r.json()


async def test_dashboard_still_requires_login(shared_client) -> None:
    r = await shared_client.get("/api/status")
    assert r.status_code == 401
    r = await shared_client.get("/")
    assert r.status_code == 302
    assert r.headers["location"].startswith("/login")


async def test_api_key_auth_still_applies(shared_client) -> None:
    r = await shared_client.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401
    assert "API-KEY" in r.json()["detail"]


def _request(url: str, headers: dict[str, str] | None = None) -> Request:
    u = URL(url)
    raw = [(b"host", u.netloc.encode())]
    raw += [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request(
        {
            "type": "http",
            "method": "GET",
            "scheme": u.scheme,
            "server": (u.hostname, u.port or (443 if u.scheme == "https" else 80)),
            "path": u.path or "/",
            "root_path": "",
            "query_string": b"",
            "headers": raw,
        }
    )


def test_hub_hint_uses_api_port_by_default(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_on_dashboard_port", False)
    hint = _hub_url_hint(_request("https://llm.example.org/"))
    assert hint == f"https://llm.example.org:{settings.api_port}"


def test_hub_hint_uses_page_address_when_shared(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_on_dashboard_port", True)
    assert _hub_url_hint(_request("https://llm.example.org/")) == (
        "https://llm.example.org"
    )
    assert _hub_url_hint(_request("http://10.0.0.5:8099/")) == ("http://10.0.0.5:8099")


def test_hub_hint_follows_proxy_headers_when_shared(monkeypatch) -> None:
    """A TLS-terminating proxy talks plain http to the router."""
    monkeypatch.setattr(settings, "api_on_dashboard_port", True)
    req = _request(
        "http://10.0.0.5/",
        {"X-Forwarded-Proto": "https", "X-Forwarded-Host": "llm.example.org"},
    )
    assert _hub_url_hint(req) == "https://llm.example.org"
