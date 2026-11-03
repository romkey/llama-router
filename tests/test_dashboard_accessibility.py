"""Dashboard accessibility and structure: labels, headings, status, dates."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import httpx
import pytest

from llama_router.config import settings
from llama_router.dashboard import routes
from llama_router.dashboard.auth_core import AUTH_COOKIE, hash_password, sign_session
from llama_router.dashboard.middleware import invalidate_dashboard_user_count_cache
from llama_router.database import Database
from llama_router.models import ProviderProtocol, ProviderStatus
from llama_router.provider_manager import ProviderManager

DashClient = tuple[httpx.AsyncClient, Database]
TEMPLATES = Path(routes.__file__).resolve().parent.parent / "templates"


def _as(user_id: int) -> dict[str, str]:
    token = sign_session(user_id, settings.session_secret.strip())
    return {"Cookie": f"{AUTH_COOKIE}={token}"}


async def _admin(db: Database) -> int:
    user_id = await db.create_dashboard_user("adm", hash_password("pw-123456789"), True)
    invalidate_dashboard_user_count_cache()
    return user_id


class _Page(HTMLParser):
    """Headings and buttons, with each button's text and attributes."""

    def __init__(self, html: str) -> None:
        super().__init__()
        self.headings: list[str] = []
        self.buttons: list[tuple[dict[str, str | None], str]] = []
        self._button: tuple[dict[str, str | None], list[str]] | None = None
        self.feed(html)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if re.fullmatch(r"h[1-6]", tag):
            self.headings.append(tag)
        if tag == "button":
            self._button = (dict(attrs), [])

    def handle_data(self, data: str) -> None:
        if self._button is not None:
            self._button[1].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "button" and self._button is not None:
            attrs, text = self._button
            self.buttons.append((attrs, "".join(text).strip()))
            self._button = None


def _unlabelled(page: _Page) -> list[dict[str, str | None]]:
    return [a for a, text in page.buttons if not text and not a.get("aria-label")]


# --- Templates as written ---


@pytest.mark.parametrize("name", sorted(p.name for p in TEMPLATES.glob("*.html")))
def test_icon_only_buttons_in_templates_have_labels(name: str) -> None:
    html = (TEMPLATES / name).read_text()
    for m in re.finditer(r"<button\b([^>]*)>(.*?)</button>", html, re.S):
        attrs, inner = m.group(1), m.group(2)
        text = re.sub(r"\{\{.*?\}\}|\{%.*?%\}", "x", re.sub(r"<[^>]+>", "", inner))
        if not text.strip():
            assert "aria-label" in attrs, f"{name}: {m.group(0)[:120]}"


def test_status_badges_pair_colour_with_an_icon() -> None:
    macros = (TEMPLATES / "status_macros.html").read_text()
    for icon in ("check-circle-fill", "hourglass-split", "x-circle-fill"):
        assert icon in macros
    assert "visually-hidden" in macros
    for name in ("dashboard.html", "provider_detail.html"):
        assert "bi-circle-fill" not in (TEMPLATES / name).read_text(), name


# --- Rendered pages ---


async def test_dashboard_structure(dash_client: DashClient) -> None:
    client, db = dash_client
    admin = await _admin(db)
    await db.add_provider("box", ProviderProtocol.OLLAMA)
    html = (await client.get("/", headers=_as(admin))).text
    page = _Page(html)

    assert page.headings.count("h1") == 1
    assert _unlabelled(page) == []
    # The stat cards belong to the Providers tab and are not headings.
    providers_pane = html.index('id="providers-pane"')
    assert providers_pane < html.index('id="stat-providers"') < html.index(
        'id="benchmarks-pane"'
    )
    # Routing has its own tab; API Keys keeps keys and their usage.
    keys = html[html.index('id="keys-pane"') : html.index('id="routing-pane"')]
    routing = html[html.index('id="routing-pane"') : html.index('id="workers-pane"')]
    assert "Issued Keys" in keys and "key-usage" in keys
    assert "Routing Preview" in routing
    # Table and explorer show once the routing table has been built.
    for card in ("Routing Preview", "Routing Table", "Route Explorer"):
        assert card not in keys
    assert "Connect to a hub (run as a worker)" in html
    assert "modal-dialog-scrollable" in html
    assert re.search(r"onclick='return confirm\(\"Remove provider box\?\"\)'", html)


async def test_every_page_has_one_h1_and_labelled_buttons(
    dash_client: DashClient,
) -> None:
    client, db = dash_client
    admin = await _admin(db)
    provider = await db.add_provider("box", ProviderProtocol.OLLAMA)
    for path in ("/users", "/my-keys", "/chat", "/decisions", f"/providers/{provider.id}"):
        page = _Page((await client.get(path, headers=_as(admin))).text)
        assert page.headings.count("h1") == 1, path
        assert _unlabelled(page) == [], path


async def test_log_badge_shows_on_every_page(
    dash_client: DashClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, db = dash_client
    admin = await _admin(db)

    async def count(_db: Database, _owner: int | None = None) -> int:
        return 42

    monkeypatch.setattr(routes, "_log_count", count)
    for path in ("/chat", "/my-keys", "/users"):
        html = (await client.get(path, headers=_as(admin))).text
        assert re.search(r'href="/#log-pane">.*?<span class="badge[^"]*">42</span>', html), path


async def test_users_actions_sit_in_a_named_menu(dash_client: DashClient) -> None:
    client, db = dash_client
    admin = await _admin(db)
    await db.create_dashboard_user("o'brien", hash_password("pw-123456789"), False)
    html = (await client.get("/users", headers=_as(admin))).text
    assert 'aria-label="Actions for o&#39;brien"' in html
    # The name reaches confirm() through a data attribute, never JS source.
    assert 'data-confirm="Delete user o&#39;brien?"' in html
    assert "confirm('Delete user" not in html


# --- Offline providers always say why ---


async def test_offline_provider_always_has_a_reason(database_url: str) -> None:
    db = Database(database_url)
    await db.connect()
    try:
        pm = ProviderManager(db)
        provider = await db.add_provider("box", ProviderProtocol.OLLAMA)
        assert provider.id is not None
        info = await pm.get_provider_info(provider.id)
        assert info is not None and info.status_detail is None

        await db.update_provider_status(provider.id, ProviderStatus.OFFLINE)
        info = await pm.get_provider_info(provider.id)
        assert info is not None and info.status_detail == "no address responded"
        [listed] = await pm.list_provider_infos()
        assert listed.status_detail == "no address responded"

        # A reason the check recorded wins over the fallback.
        pm._status_detail[provider.id] = "HTTP 401 listing models"
        info = await pm.get_provider_info(provider.id)
        assert info is not None and info.status_detail == "HTTP 401 listing models"
    finally:
        await db.close()


# --- One date format ---


def test_time_tag_filter() -> None:
    tag = str(routes._time_tag("2026-01-02 03:04:05"))
    assert 'datetime="2026-01-02T03:04:05+00:00"' in tag
    assert "data-local" in tag and "data-relative" not in tag
    assert re.search(r">\d{4}-\d\d-\d\d \d\d:\d\d</time>$", tag)
    assert 'title="2026-01-0' in tag

    aware = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert "data-relative" in str(routes._time_tag(aware, relative=True))
    assert routes._time_tag(None) == "—"
    assert routes._time_tag("") == "—"
    # Text it cannot read is returned as plain text, which Jinja escapes.
    assert routes._time_tag("not a date") == "not a date"
    assert not hasattr(routes._time_tag("<b>"), "__html__")


# --- API key labels ---


async def test_admin_key_label(dash_client: DashClient) -> None:
    client, db = dash_client
    admin = await _admin(db)
    resp = await client.post(
        "/api/keys/generate", json={"label": "  home assistant  "}, headers=_as(admin)
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["label"] == "home assistant"
    [key] = await db.list_api_keys()
    assert key["label"] == "home assistant"
    html = (await client.get("/", headers=_as(admin))).text
    assert '<div class="small text-muted">home assistant</div>' in html

    resp = await client.post("/api/keys/generate", json={"label": ""}, headers=_as(admin))
    assert resp.json()["label"] is None


@pytest.mark.parametrize("label", ["x" * 101, 5, ["a"]])
async def test_bad_key_labels_are_refused(dash_client: DashClient, label: object) -> None:
    client, db = dash_client
    admin = await _admin(db)
    for path in ("/api/keys/generate", "/api/my-keys"):
        resp = await client.post(path, json={"label": label}, headers=_as(admin))
        assert resp.status_code == 400, path
    assert await db.list_api_keys() == []


async def test_self_issued_key_label(dash_client: DashClient) -> None:
    client, db = dash_client
    ann = await db.create_dashboard_user("ann", hash_password("pw-123456789"), False)
    invalidate_dashboard_user_count_cache()
    resp = await client.post("/api/my-keys", json={"label": "laptop"}, headers=_as(ann))
    assert resp.status_code == 200, resp.text
    mine = (await client.get("/api/my-keys", headers=_as(ann))).json()
    assert mine["keys"][0]["label"] == "laptop"
