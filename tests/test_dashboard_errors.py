"""Dashboard error handling: HTML error pages, form failures that go back to
the form, JSON bodies, and the open-access toggle."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

import httpx
import pytest

from llama_router.dashboard.routes import providers
from llama_router.database import Database
from llama_router.models import ProviderProtocol

DashClient = tuple[httpx.AsyncClient, Database]
TEMPLATES = Path(__file__).parent.parent / "src" / "llama_router" / "templates"


async def test_browser_errors_render_an_html_page(dash_client: DashClient) -> None:
    c, _ = dash_client
    resp = await c.get("/providers/9999", headers={"Accept": "text/html"})
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("text/html")
    assert "404 Not Found" in resp.text
    assert 'href="/"' in resp.text


async def test_api_errors_stay_json(dash_client: DashClient) -> None:
    c, _ = dash_client
    resp = await c.get("/api/providers/9999/prices", headers={"Accept": "text/html"})
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Provider not found"}


@pytest.mark.parametrize("accept", ["application/json", "*/*", ""])
async def test_script_clients_get_json_off_the_api(
    dash_client: DashClient, accept: str
) -> None:
    """fetch() sends */*; WebAuthn and other script routes need JSON."""
    c, _ = dash_client
    resp = await c.get("/providers/9999", headers={"Accept": accept})
    assert resp.status_code == 404
    assert "detail" in resp.json()
    resp = await c.post(
        "/account/security/register/options", json={}, headers={"Accept": accept}
    )
    assert resp.status_code >= 400
    assert "detail" in resp.json()


async def test_failed_add_provider_reopens_the_dialog_filled_in(
    dash_client: DashClient,
) -> None:
    c, db = dash_client
    resp = await c.post(
        "/providers/add",
        data={
            "name": "gpu-1",
            "url": "http://x:8080",
            "protocol": "both",
            "api_key": "sk-secret",
            "gpu_type": "RTX 4090",
        },
    )
    assert resp.status_code == 303
    location = resp.headers["location"]
    assert location.startswith("/?add_provider=1&error=")
    assert providers._ADD_FORM_COOKIE in resp.cookies
    assert await db.list_providers() == []

    page = await c.get(location)  # the client sends the cookie back
    assert page.status_code == 200
    assert '"gpu-1"' in page.text
    assert '"RTX 4090"' in page.text
    # The key is never echoed back.
    assert "sk-secret" not in page.text
    assert "addProviderModal')).show()" in page.text
    # Shown once: the page clears the cookie, with the attributes it was set
    # with so browsers honour the deletion.
    cleared = page.headers.get("set-cookie", "").lower()
    assert providers._ADD_FORM_COOKIE in cleared
    assert "max-age=0" in cleared
    assert "samesite=strict" in cleared
    assert "httponly" in cleared


async def test_dashboard_ignores_a_bad_prefill_cookie(dash_client: DashClient) -> None:
    c, _ = dash_client
    c.cookies.set(providers._ADD_FORM_COOKIE, "not base64 json")
    page = await c.get("/?add_provider=1&error=x")
    assert page.status_code == 200
    assert "addProviderModal')).show()" in page.text


async def test_failed_address_edit_goes_back_to_the_provider(
    dash_client: DashClient,
) -> None:
    c, db = dash_client
    pid = (await db.add_provider("p", ProviderProtocol.OLLAMA)).require_id()
    resp = await c.post(
        f"/providers/{pid}/addresses/save", data={"url": "x", "address_id": "nope"}
    )
    assert resp.status_code == 303
    assert (
        unquote(resp.headers["location"]) == f"/providers/{pid}?error=Invalid address"
    )


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/auth/allow-unauthenticated", b"{not json"),
        ("/api/auth/allow-unauthenticated", b"[true]"),
        ("/api/fallbacks", b"{not json"),
        ("/api/model-groups", b"{not json"),
        ("/api/benchmark", b"{}"),
        ("/api/benchmark", b'{"provider_id": "x", "model": "m"}'),
        ("/api/pull", b'{"model": "m", "provider_id": "x"}'),
    ],
)
async def test_bad_json_bodies_are_400(
    dash_client: DashClient, path: str, body: bytes
) -> None:
    c, _ = dash_client
    resp = await c.post(
        path, content=body, headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 400
    assert "detail" in resp.json()


async def test_allow_unauthenticated_round_trip(dash_client: DashClient) -> None:
    c, _ = dash_client
    resp = await c.post("/api/auth/allow-unauthenticated", json={"allow": True})
    assert resp.json() == {"allow_unauthenticated": True}
    page = await c.get("/")
    assert 'id="open-access-badge"' in page.text
    assert 'id="open-access-badge" class="badge text-bg-warning fs-6' in page.text
    assert "d-none" not in page.text.split('id="open-access-badge"')[1].split(">")[0]


def test_dashboard_template_guards_open_access_and_polling() -> None:
    html = (TEMPLATES / "dashboard.html").read_text()
    toggle = html.split("function setAllowUnauthenticated", 1)[1].split("\n}\n", 1)[0]
    assert "confirm(" in toggle
    assert "input.checked = !allow" in toggle  # reverted on failure
    assert "showToast(" in toggle
    assert 'id="security-card"' in html
    poll = html.split("function refreshStatus", 1)[1].split("\n}\n", 1)[0]
    assert "document.hidden" in poll
    assert "r.status === 401" in poll
    assert "stopStatusPolling()" in poll
    assert 'id="session-expired"' in html
