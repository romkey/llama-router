"""The favicon and touch icons are public and cacheable."""

from __future__ import annotations

import httpx
import pytest

from llama_router.dashboard.auth_core import hash_password
from llama_router.dashboard.middleware import invalidate_dashboard_user_count_cache
from llama_router.database import Database


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "media_type"),
    [
        ("/favicon.ico", "image/x-icon"),
        ("/favicon-32x32.png", "image/png"),
        ("/apple-touch-icon.png", "image/png"),
        ("/apple-touch-icon-precomposed.png", "image/png"),
        ("/apple-touch-icon-120x120.png", "image/png"),
        ("/android-chrome-512x512.png", "image/png"),
        ("/site.webmanifest", "application/manifest+json"),
    ],
)
async def test_icons_need_no_login(
    dash_client: tuple[httpx.AsyncClient, Database], path: str, media_type: str
) -> None:
    client, db = dash_client
    await db.create_dashboard_user("adm", hash_password("password-12345"), True)
    invalidate_dashboard_user_count_cache()

    r = await client.get(path, follow_redirects=False)

    assert r.status_code == 200
    assert r.headers["content-type"].startswith(media_type)
    assert "max-age" in r.headers["cache-control"]
    assert r.content


@pytest.mark.asyncio
async def test_pages_link_the_icons(
    dash_client: tuple[httpx.AsyncClient, Database],
) -> None:
    client, db = dash_client
    await db.create_dashboard_user("adm", hash_password("password-12345"), True)
    invalidate_dashboard_user_count_cache()

    r = await client.get("/login")

    assert 'rel="apple-touch-icon"' in r.text
    assert 'href="/favicon.ico"' in r.text
