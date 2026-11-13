"""The dashboard menu keeps its order and visibility across page changes."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

import httpx
import pytest

from llama_router.config import settings
from llama_router.dashboard import deps
from llama_router.dashboard.auth_core import AUTH_COOKIE, hash_password, sign_session
from llama_router.database import Database
from llama_router.models import ProviderProtocol
from llama_router.provider_secrets import session_signing_key
from llama_router.registry_cache.cache import BlobCache


class _Navigation(HTMLParser):
    """Links in the main menu, including those in its Access and Tools menus."""

    def __init__(self, html: str) -> None:
        super().__init__()
        # Depth of <ul> inside #main-nav-tabs; 0 outside it.
        self.depth = 0
        self.items: list[dict[str, str | None]] = []
        self.menus: list[dict[str, str | None]] = []
        self.feed(html)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "ul" and (self.depth or attributes.get("id") == "main-nav-tabs"):
            self.depth += 1
        classes = (attributes.get("class") or "").split()
        if not self.depth:
            return
        if "dropdown-toggle" in classes:
            self.menus.append(attributes)
        elif "nav-link" in classes or "dropdown-item" in classes:
            self.items.append(attributes)

    def handle_endtag(self, tag: str) -> None:
        if tag == "ul" and self.depth:
            self.depth -= 1


@pytest.mark.parametrize("role", ["admin", "viewer"])
@pytest.mark.parametrize("cache_state", ["absent", "enabled", "disabled"])
async def test_navigation_stays_consistent_across_pages(
    dash_client: tuple[httpx.AsyncClient, Database],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    role: str,
    cache_state: str,
) -> None:
    client, db = dash_client
    monkeypatch.setattr(settings, "cache_enabled", cache_state == "enabled")
    cache = None if cache_state == "absent" else BlobCache(str(tmp_path / "cache"))
    monkeypatch.setattr(deps, "_blob_cache", cache)
    user_id = await db.create_dashboard_user(
        "test-user", hash_password("test-password"), role == "admin"
    )
    client.cookies.set(AUTH_COOKIE, sign_session(user_id, session_signing_key()))
    provider = await db.add_provider("test-provider", ProviderProtocol.OLLAMA)
    pages = ["/", "/chat", "/my-keys", "/decisions", f"/providers/{provider.id}"]
    # Operate tabs, then the Access menu (admins only), then Tools.
    expected = ["/#providers-pane", "/#benchmarks-pane", "/#models-pane"]
    if role != "viewer":
        pages.extend(["/users", "/audit"])
        expected.extend(["/#workers-pane", "/#log-pane"])
        if cache is not None:
            expected.append("/#cache-pane")
        expected.extend(["/#keys-pane", "/#routing-pane", "/users", "/audit"])
    else:
        expected.append("/#log-pane")
    expected.extend(["/decisions", "/chat"])

    for path in pages:
        response = await client.get(path)
        assert response.status_code == 200, path
        items = _Navigation(response.text).items
        # Provider details append their page-specific marker.
        shared = [
            item for item in items if item.get("href") or item.get("data-bs-target")
        ]
        destinations = [
            item.get("href") or "/" + str(item["data-bs-target"]) for item in shared
        ]
        assert destinations == expected, path
        active = [
            item for item in shared if "active" in (item.get("class") or "").split()
        ]
        # The menu holding the current page is marked too.
        open_menus = [
            m for m in _Navigation(response.text).menus if "active" in str(m["class"])
        ]
        assert len(open_menus) == (
            path in ("/chat", "/users", "/audit", "/decisions")
        ), path
        assert response.text.count('href="/decisions"') == 1, path
        if path in ("/", "/chat", "/users", "/decisions"):
            assert len(active) == 1, path
            assert active[0].get("href") == path or (
                path == "/" and active[0].get("data-bs-target") == "#providers-pane"
            )
        if path == "/":
            for item in shared:
                target = item.get("data-bs-target")
                if target:
                    assert f'id="{target.removeprefix("#")}"' in response.text
