"""Dashboard login, sessions, and RBAC (admin vs viewer) access control."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from llama_router.config import settings
from llama_router.dashboard import deps as dash_deps
from llama_router.dashboard.app import app as dashboard_app
from llama_router.dashboard.auth_core import (
    AUTH_COOKIE,
    hash_password,
    non_admin_forbidden_get_path,
    normalize_username,
    sign_session,
    verify_password,
)
from llama_router.dashboard.middleware import invalidate_dashboard_user_count_cache
from llama_router.database import Database
from llama_router.provider_manager import ProviderManager


def _session_cookie_header(user_id: int) -> dict[str, str]:
    secret = settings.session_secret.strip()
    token = sign_session(user_id, secret)
    return {"Cookie": f"{AUTH_COOKIE}={token}"}


@pytest_asyncio.fixture
async def dash_client(tmp_path, monkeypatch):
    """ASGI client with isolated DB and fixed session secret."""
    monkeypatch.setattr(
        settings, "session_secret", "test-session-secret-key-min-16chars"
    )
    monkeypatch.setattr(settings, "dashboard_cookie_secure", False)

    db = Database(str(tmp_path / "dash_auth.db"))
    await db.connect()
    pm = ProviderManager(db)
    dash_deps.invalidate_dashboard_session_secret_cache()
    invalidate_dashboard_user_count_cache()
    dash_deps.init(db, pm)

    transport = ASGITransport(app=dashboard_app)
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
        follow_redirects=False,
    ) as client:
        yield client, db

    await db.close()
    dash_deps._db = None
    dash_deps._pm = None
    dash_deps.invalidate_dashboard_session_secret_cache()
    invalidate_dashboard_user_count_cache()


def test_normalize_username_accepts_valid() -> None:
    assert normalize_username("Alice_01") == "alice_01"


@pytest.mark.parametrize(
    "raw",
    [
        "a",
        "x" * 65,
        "bad name",
        "bad!char",
    ],
)
def test_normalize_username_rejects_invalid(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_username(raw)


def test_password_hash_roundtrip() -> None:
    stored = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", stored)
    assert not verify_password("wrong", stored)


@pytest.mark.parametrize(
    "path",
    [
        "/users",
        "/users/1/delete",
        "/api/keys",
        "/api/keys/9/pins",
        "/api/workers",
        "/api/worker-link",
        "/api/cache/status",
        "/api/fallbacks",
        "/api/pulls",
        "/api/pulls/abc",
        "/api/auth/allow-unauthenticated",
        "/api/benchmarks/fill-missing",
        "/api/benchmarks/fill-missing/job1",
    ],
)
def test_non_admin_forbidden_get_path_blocks_sensitive_routes(path: str) -> None:
    assert non_admin_forbidden_get_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/providers/1",
        "/api/status",
        "/api/benchmarks/42",
        "/login",
        "/health",
    ],
)
def test_non_admin_forbidden_get_path_allows_public_dashboard_paths(path: str) -> None:
    assert not non_admin_forbidden_get_path(path)


@pytest.mark.asyncio
async def test_health_always_public_without_db_users(dash_client) -> None:
    client, _db = dash_client
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


@pytest.mark.asyncio
async def test_healthz_public_after_users_exist(dash_client) -> None:
    client, db = dash_client
    await db.create_dashboard_user("adm", hash_password("password-12345"), True)
    r = await client.get("/healthz", follow_redirects=False)
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_bootstrap_no_users_dashboard_and_api_status_open(dash_client) -> None:
    client, _db = dash_client
    r = await client.get("/")
    assert r.status_code == 200
    assert "Dashboard" in r.text or "llama-router" in r.text

    r2 = await client.get("/api/status")
    assert r2.status_code == 200
    body = r2.json()
    assert "workers" in body
    assert "cache" in body


@pytest.mark.asyncio
async def test_bootstrap_login_page_redirects_home(dash_client) -> None:
    client, _db = dash_client
    r = await client.get("/login", follow_redirects=False)
    assert r.status_code == 302
    loc = r.headers.get("location") or ""
    assert loc in ("/", "http://testserver/")


@pytest.mark.asyncio
async def test_bootstrap_can_create_first_user_via_form(dash_client) -> None:
    client, db = dash_client
    r = await client.post(
        "/users/add",
        data={
            "username": "firstadmin",
            "password": "first-pass-8chars",
        },
    )
    assert r.status_code == 303
    assert await db.count_dashboard_users() == 1
    row = await db.get_dashboard_user_by_username("firstadmin")
    assert row is not None
    assert row["is_admin"] is True


@pytest.mark.asyncio
async def test_after_users_exist_unauthenticated_redirects_to_login(
    dash_client,
) -> None:
    client, db = dash_client
    await db.create_dashboard_user("adm", hash_password("password-12345"), True)
    invalidate_dashboard_user_count_cache()

    r = await client.get("/", follow_redirects=False)
    assert r.status_code == 302
    loc = r.headers.get("location", "")
    assert "/login" in loc


@pytest.mark.asyncio
async def test_after_users_exist_api_returns_401_json(dash_client) -> None:
    client, db = dash_client
    await db.create_dashboard_user("adm", hash_password("password-12345"), True)
    invalidate_dashboard_user_count_cache()

    r = await client.get("/api/status")
    assert r.status_code == 401
    assert r.json().get("detail") == "Not authenticated"


@pytest.mark.asyncio
async def test_worker_socket_path_is_not_on_dashboard(dash_client) -> None:
    """The hub socket lives on the API port; the dashboard must not expose it."""
    client, db = dash_client
    await db.create_dashboard_user("adm", hash_password("password-12345"), True)
    invalidate_dashboard_user_count_cache()
    r = await client.get("/worker/ws")
    assert r.status_code in (302, 401, 404)


@pytest.mark.asyncio
async def test_login_success_allows_dashboard_html(dash_client) -> None:
    client, db = dash_client
    await db.create_dashboard_user("alice", hash_password("password-12345"), True)
    invalidate_dashboard_user_count_cache()

    r = await client.post(
        "/login",
        data={"username": "alice", "password": "password-12345", "next": "/"},
    )
    assert r.status_code == 303
    assert AUTH_COOKIE in r.cookies

    home = await client.get("/")
    assert home.status_code == 200


@pytest.mark.asyncio
async def test_login_rejects_bad_password(dash_client) -> None:
    client, db = dash_client
    await db.create_dashboard_user("bob", hash_password("right-pass-12345"), True)
    invalidate_dashboard_user_count_cache()

    r = await client.post(
        "/login",
        data={"username": "bob", "password": "wrong-pass-12345", "next": "/"},
    )
    assert r.status_code == 303
    assert "error=invalid" in (r.headers.get("location") or "")

    blocked = await client.get("/")
    assert blocked.status_code == 302
    assert "/login" in (blocked.headers.get("location") or "")


@pytest.mark.asyncio
async def test_admin_api_status_includes_sensitive_sections(dash_client) -> None:
    client, db = dash_client
    uid = await db.create_dashboard_user("admin", hash_password("password-12345"), True)
    invalidate_dashboard_user_count_cache()

    r = await client.get("/api/status", headers=_session_cookie_header(uid))
    assert r.status_code == 200
    body = r.json()
    assert "workers" in body
    assert "cache" in body
    assert "active_pulls" in body
    assert "active_benchmarks" in body


@pytest.mark.asyncio
async def test_viewer_api_status_hides_sensitive_sections(dash_client) -> None:
    client, db = dash_client
    uid = await db.create_dashboard_user(
        "viewer", hash_password("password-12345"), False
    )
    invalidate_dashboard_user_count_cache()

    r = await client.get("/api/status", headers=_session_cookie_header(uid))
    assert r.status_code == 200
    body = r.json()
    assert "workers" not in body
    assert "worker_link" not in body
    assert "cache" not in body
    assert body.get("active_pulls") == {}
    assert body.get("active_benchmarks") == {}
    assert body.get("log_total") == 0
    assert "providers" in body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/users",
        "/api/keys/1/pins",
        "/api/fallbacks",
        "/api/pulls",
        "/api/cache/status",
        "/api/workers",
        "/api/worker-link",
        "/api/benchmarks/fill-missing/job-x",
    ],
)
async def test_viewer_get_forbidden_paths_blocked(dash_client, path: str) -> None:
    client, db = dash_client
    uid = await db.create_dashboard_user("v", hash_password("password-12345"), False)
    invalidate_dashboard_user_count_cache()

    r = await client.get(
        path, headers=_session_cookie_header(uid), follow_redirects=False
    )
    if path == "/users":
        assert r.status_code == 302
        assert "error=forbidden" in (r.headers.get("location") or "")
    else:
        assert r.status_code == 403
        assert r.json().get("detail") == "Forbidden"


@pytest.mark.asyncio
async def test_viewer_post_to_api_keys_generate_forbidden_json(dash_client) -> None:
    client, db = dash_client
    uid = await db.create_dashboard_user("v", hash_password("password-12345"), False)
    invalidate_dashboard_user_count_cache()

    r = await client.post(
        "/api/keys/generate",
        headers=_session_cookie_header(uid),
        json={"routing_mode": "latency", "allow_fallback": True},
    )
    assert r.status_code == 403
    assert r.json().get("detail") == "Forbidden"


@pytest.mark.asyncio
async def test_viewer_post_providers_add_redirects_forbidden(dash_client) -> None:
    client, db = dash_client
    uid = await db.create_dashboard_user("v", hash_password("password-12345"), False)
    invalidate_dashboard_user_count_cache()

    r = await client.post(
        "/providers/add",
        headers=_session_cookie_header(uid),
        data={
            "name": "evil",
            "url": "http://127.0.0.1:11434",
            "protocol": "ollama",
        },
        follow_redirects=False,
    )
    assert r.status_code == 302
    assert "error=forbidden" in (r.headers.get("location") or "")


@pytest.mark.asyncio
async def test_viewer_get_dashboard_and_provider_detail_allowed(dash_client) -> None:
    client, db = dash_client
    uid = await db.create_dashboard_user("v", hash_password("password-12345"), False)
    p = await db.add_provider("p1")
    invalidate_dashboard_user_count_cache()
    hdr = _session_cookie_header(uid)

    r = await client.get("/", headers=hdr)
    assert r.status_code == 200

    r2 = await client.get(f"/providers/{p.id}", headers=hdr)
    assert r2.status_code == 200


@pytest.mark.asyncio
async def test_admin_save_provider_address_add_and_edit(dash_client) -> None:
    """Address modals must POST to /addresses/save, not to GET-only /providers/{id}."""
    client, db = dash_client
    uid = await db.create_dashboard_user("admin", hash_password("password-12345"), True)
    p = await db.add_provider("p1")
    invalidate_dashboard_user_count_cache()
    hdr = _session_cookie_header(uid)

    bad = await client.post(
        f"/providers/{p.id}",
        headers=hdr,
        data={"url": "http://oops:11434"},
        follow_redirects=False,
    )
    assert bad.status_code == 405

    add = await client.post(
        f"/providers/{p.id}/addresses/save",
        headers=hdr,
        data={"url": "http://10.0.0.2:11434"},
        follow_redirects=False,
    )
    assert add.status_code == 303
    assert add.headers.get("location", "").endswith(f"/providers/{p.id}")
    rows = await db.get_addresses(p.id)
    assert len(rows) == 1
    aid = rows[0].id

    edit = await client.post(
        f"/providers/{p.id}/addresses/save",
        headers=hdr,
        data={
            "address_id": str(aid),
            "url": "http://10.0.0.3:11434",
            "is_preferred": "1",
        },
        follow_redirects=False,
    )
    assert edit.status_code == 303
    updated = await db.get_address(aid)
    assert updated is not None
    assert updated.url == "http://10.0.0.3:11434"
    assert updated.is_preferred is True


@pytest.mark.asyncio
async def test_admin_can_open_users_page(dash_client) -> None:
    client, db = dash_client
    uid = await db.create_dashboard_user("admin", hash_password("password-12345"), True)
    invalidate_dashboard_user_count_cache()

    r = await client.get("/users", headers=_session_cookie_header(uid))
    assert r.status_code == 200
    assert "Dashboard users" in r.text or "users" in r.text.lower()


@pytest.mark.asyncio
async def test_viewer_cannot_post_users_add(dash_client) -> None:
    client, db = dash_client
    await db.create_dashboard_user("admin", hash_password("password-12345"), True)
    vid = await db.create_dashboard_user("v", hash_password("password-12345"), False)
    invalidate_dashboard_user_count_cache()

    r = await client.post(
        "/users/add",
        headers=_session_cookie_header(vid),
        data={
            "username": "hacker",
            "password": "password-12345",
            "is_admin": "1",
        },
        follow_redirects=False,
    )
    assert r.status_code == 302
    assert "error=forbidden" in (r.headers.get("location") or "")
    assert await db.get_dashboard_user_by_username("hacker") is None


@pytest.mark.asyncio
async def test_invalid_session_cookie_treated_as_unauthenticated(dash_client) -> None:
    client, db = dash_client
    await db.create_dashboard_user("a", hash_password("password-12345"), True)
    invalidate_dashboard_user_count_cache()

    r = await client.get(
        "/api/status",
        headers={"Cookie": f"{AUTH_COOKIE}=not-a-valid-token"},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_session_for_deleted_user_rejected_when_accounts_remain(
    dash_client,
) -> None:
    client, db = dash_client
    uid = await db.create_dashboard_user("gone", hash_password("password-12345"), True)
    await db.create_dashboard_user("other", hash_password("password-12345"), True)
    invalidate_dashboard_user_count_cache()
    token = sign_session(uid, settings.session_secret.strip())
    await db.delete_dashboard_user(uid)
    invalidate_dashboard_user_count_cache()

    r = await client.get("/api/status", headers={"Cookie": f"{AUTH_COOKIE}={token}"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_deleting_last_dashboard_user_reopens_bootstrap(dash_client) -> None:
    """With zero users the dashboard is intentionally open (initial setup)."""
    client, db = dash_client
    uid = await db.create_dashboard_user("only", hash_password("password-12345"), True)
    invalidate_dashboard_user_count_cache()
    token = sign_session(uid, settings.session_secret.strip())
    await db.delete_dashboard_user(uid)
    invalidate_dashboard_user_count_cache()

    r = await client.get("/api/status", headers={"Cookie": f"{AUTH_COOKIE}={token}"})
    assert r.status_code == 200
    assert "workers" in r.json()


@pytest.mark.asyncio
async def test_logout_prevents_further_access(dash_client) -> None:
    client, db = dash_client
    uid = await db.create_dashboard_user("u", hash_password("password-12345"), True)
    invalidate_dashboard_user_count_cache()
    hdr = _session_cookie_header(uid)

    ok = await client.get("/", headers=hdr)
    assert ok.status_code == 200

    out = await client.post("/logout", headers=hdr)
    assert out.status_code == 303

    again = await client.get("/")
    assert again.status_code == 302
    assert "/login" in (again.headers.get("location") or "")


@pytest.mark.asyncio
async def test_admin_can_create_rotate_disable_and_delete_worker(dash_client) -> None:
    from llama_router.worker_link.hub import worker_token_hash

    client, db = dash_client
    uid = await db.create_dashboard_user("adm", hash_password("password-12345"), True)
    invalidate_dashboard_user_count_cache()
    hdr = _session_cookie_header(uid)

    r = await client.post("/api/workers", headers=hdr, json={"name": "Alice-GPU"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "alice-gpu"
    token = body["token"]
    assert token.startswith("lrw_")
    assert "LLAMA_ROUTER_WORKER_TOKEN=" + token in body["env_example"]
    row = await db.get_worker_by_token_hash(worker_token_hash(token))
    assert row is not None and row["provider_id"] == body["provider_id"]
    prov = await db.get_provider(body["provider_id"])
    assert prov.worker_name == "alice-gpu" and prov.is_worker
    assert [a.url for a in await db.get_addresses(prov.id)] == ["worker://alice-gpu"]
    assert prov.status.value == "offline"

    dup = await client.post("/api/workers", headers=hdr, json={"name": "alice-gpu"})
    assert dup.status_code == 409
    bad = await client.post("/api/workers", headers=hdr, json={"name": "no spaces"})
    assert bad.status_code == 400

    listed = (await client.get("/api/workers", headers=hdr)).json()["workers"]
    assert [w["name"] for w in listed] == ["alice-gpu"]
    assert listed[0]["connected"] is False

    page = await client.get("/", headers=hdr)
    assert page.status_code == 200
    assert "workers-pane" in page.text
    assert "alice-gpu" in page.text
    assert "wireguard" not in page.text.lower()

    rot = await client.post(f"/api/workers/{body['id']}/rotate", headers=hdr)
    new_token = rot.json()["token"]
    assert new_token != token
    assert await db.get_worker_by_token_hash(worker_token_hash(token)) is None
    assert await db.get_worker_by_token_hash(worker_token_hash(new_token)) is not None

    dis = await client.post(
        f"/api/workers/{body['id']}/enabled", headers=hdr, json={"enabled": False}
    )
    assert dis.status_code == 200
    assert (await db.get_worker(body["id"]))["enabled"] is False

    rm = await client.delete(f"/api/workers/{body['id']}", headers=hdr)
    assert rm.status_code == 200
    assert await db.get_worker(body["id"]) is None
    assert await db.get_provider(body["provider_id"]) is None


@pytest.mark.asyncio
async def test_viewer_cannot_create_worker(dash_client) -> None:
    client, db = dash_client
    uid = await db.create_dashboard_user("v", hash_password("password-12345"), False)
    invalidate_dashboard_user_count_cache()
    r = await client.post(
        "/api/workers", headers=_session_cookie_header(uid), json={"name": "x1"}
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_worker_link_save_and_clear(dash_client, monkeypatch) -> None:
    monkeypatch.setattr(settings, "worker_hub_url", "")
    monkeypatch.setattr(settings, "worker_token", "")
    client, db = dash_client
    uid = await db.create_dashboard_user("adm", hash_password("password-12345"), True)
    invalidate_dashboard_user_count_cache()
    hdr = _session_cookie_header(uid)

    r = await client.get("/api/worker-link", headers=hdr)
    assert r.json()["state"] == "disabled"
    assert r.json()["configured"] is False

    # Port 9 (discard) refuses connections, so the link stays in backoff.
    r = await client.post(
        "/api/worker-link",
        headers=hdr,
        json={"hub_url": "http://127.0.0.1:9", "token": "lrw_abc"},
    )
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["configured"] is True and st["source"] == "dashboard"
    assert st["hub_url"] == "http://127.0.0.1:9"
    assert st["state"] in ("connecting", "backoff")
    assert await db.get_app_setting("worker_token") == "lrw_abc"

    r = await client.post(
        "/api/worker-link", headers=hdr, json={"hub_url": "", "token": ""}
    )
    st = r.json()
    assert st["configured"] is False and st["state"] == "disabled"
    await dash_deps.get_worker_link().stop()


@pytest.mark.asyncio
async def test_worker_link_locked_when_set_by_env(dash_client, monkeypatch) -> None:
    monkeypatch.setattr(settings, "worker_hub_url", "https://hub.example")
    monkeypatch.setattr(settings, "worker_token", "lrw_env")
    client, db = dash_client
    uid = await db.create_dashboard_user("adm", hash_password("password-12345"), True)
    invalidate_dashboard_user_count_cache()
    r = await client.post(
        "/api/worker-link",
        headers=_session_cookie_header(uid),
        json={"hub_url": "https://other", "token": "x"},
    )
    assert r.status_code == 409
