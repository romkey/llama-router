"""Usage per API key and provider, from the request log."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx

from llama_router.config import settings
from llama_router.dashboard.auth_core import AUTH_COOKIE, hash_password, sign_session
from llama_router.dashboard.middleware import invalidate_dashboard_user_count_cache
from llama_router.database import Database
from llama_router.models import RequestLog

CHAT = "/v1/chat/completions"


async def _key(db: Database, prefix: str) -> int:
    return await db.create_api_key(prefix, f"hash-{prefix}", "balanced", True)


async def _log(
    db: Database,
    key_id: int | None,
    provider_id: int | None,
    provider_name: str | None,
    *,
    status: str = "ok",
    prompt: int | None = None,
    completion: int | None = None,
    cost: float | None = None,
) -> None:
    await db.save_request_log(
        RequestLog(
            provider_id=provider_id,
            provider_name=provider_name,
            protocol="v1",
            endpoint=CHAT,
            api_key_id=key_id,
            status=status,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cost_usd=cost,
        )
    )


async def _seed(db: Database) -> dict[str, int]:
    alice = await _key(db, "lr_alice")
    bob = await _key(db, "lr_bob")
    gone_key = await _key(db, "lr_gone")
    gpu = (await db.add_provider("gpu")).require_id()
    cloud = (await db.add_provider("cloud")).require_id()
    old = (await db.add_provider("old")).require_id()

    await _log(db, alice, gpu, "gpu", prompt=10, completion=5)
    await _log(db, alice, gpu, "gpu", prompt=20, completion=7, status="error")
    await _log(db, alice, cloud, "cloud", prompt=100, completion=50, cost=0.25)
    await _log(db, bob, gpu, "gpu", prompt=1, completion=1)
    await _log(db, None, gpu, "gpu")
    await _log(db, gone_key, old, "old-name")
    # Refused before routing: no provider, not usage of one.
    await _log(db, alice, None, None, status="error")
    await db.delete_api_key(gone_key)
    await db.remove_provider(old)
    return {"alice": alice, "bob": bob, "gone": gone_key, "gpu": gpu, "cloud": cloud}


async def test_usage_groups_by_key_and_provider(db: Database) -> None:
    ids = await _seed(db)
    rows = await db.usage_by_key_and_provider()
    by = {(r["api_key_id"], r["provider"]): r for r in rows}
    assert set(by) == {
        (ids["alice"], "gpu"),
        (ids["alice"], "cloud"),
        (ids["bob"], "gpu"),
        (None, "gpu"),
        (ids["gone"], "old-name"),
    }
    a_gpu = by[(ids["alice"], "gpu")]
    assert rows[0] is a_gpu  # busiest first
    assert a_gpu["key_prefix"] == "lr_alice"
    assert (a_gpu["requests"], a_gpu["errors"]) == (2, 1)
    assert (a_gpu["prompt_tokens"], a_gpu["completion_tokens"]) == (30, 12)
    assert a_gpu["cost_usd"] == 0
    assert a_gpu["provider_exists"] is True
    # Stored in UTC; returned so, whichever database stores it.
    assert a_gpu["last_request_at"].tzinfo is not None
    assert abs(datetime.now(UTC) - a_gpu["last_request_at"]) < timedelta(minutes=5)
    assert by[(ids["alice"], "cloud")]["cost_usd"] == 0.25
    gone = by[(ids["gone"], "old-name")]
    assert gone["key_prefix"] is None
    assert gone["provider_exists"] is False

    only_bob = await db.usage_by_key_and_provider(key_id=ids["bob"])
    assert [(r["api_key_id"], r["requests"]) for r in only_bob] == [(ids["bob"], 1)]


async def test_usage_names_follow_renames(db: Database) -> None:
    ids = await _seed(db)
    await db.update_provider(ids["gpu"], "big-gpu")
    names = {r["provider"] for r in await db.usage_by_key_and_provider()}
    assert "big-gpu" in names
    assert "gpu" not in names


def test_utc_datetime() -> None:
    from llama_router.database import _utc_datetime

    assert _utc_datetime(None) is None
    assert _utc_datetime("junk") is None
    assert _utc_datetime(42) is None
    got = _utc_datetime("2026-10-30 12:34:56")
    assert got == datetime(2026, 10, 30, 12, 34, 56, tzinfo=UTC)
    aware = datetime(2026, 1, 1, tzinfo=UTC)
    assert _utc_datetime(aware) is aware
    assert _utc_datetime(aware.replace(tzinfo=None)) == aware


async def test_usage_period(db: Database) -> None:
    ids = await _seed(db)
    await db._execute(
        "UPDATE request_log SET created_at = ? WHERE api_key_id = ?",
        ("2000-01-01 00:00:00", ids["bob"]),
    )
    since = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=1)
    recent = await db.usage_by_key_and_provider(since)
    assert ids["bob"] not in {r["api_key_id"] for r in recent}
    everything = await db.usage_by_key_and_provider()
    assert ids["bob"] in {r["api_key_id"] for r in everything}


def _cookie(user_id: int) -> dict[str, str]:
    token = sign_session(user_id, settings.session_secret.strip())
    return {"Cookie": f"{AUTH_COOKIE}={token}"}


async def test_usage_api(dash_client: tuple[httpx.AsyncClient, Database]) -> None:
    client, db = dash_client
    ids = await _seed(db)
    admin = await db.create_dashboard_user("adm", hash_password("password-12345"), True)
    viewer = await db.create_dashboard_user("v", hash_password("password-12345"), False)
    invalidate_dashboard_user_count_cache()

    r = await client.get("/api/keys/usage", headers=_cookie(admin))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["period"] == "7d"
    assert body["since"] is not None
    assert len(body["rows"]) == 5
    assert all(row["last_request_at"].endswith("+00:00") for row in body["rows"])

    r = await client.get(
        f"/api/keys/usage?period=all&key_id={ids['alice']}", headers=_cookie(admin)
    )
    body = r.json()
    assert body["since"] is None
    assert {row["provider"] for row in body["rows"]} == {"gpu", "cloud"}

    r = await client.get("/api/keys/usage?period=1y", headers=_cookie(admin))
    assert r.status_code == 400
    # Key details are for administrators only.
    r = await client.get("/api/keys/usage", headers=_cookie(viewer))
    assert r.status_code == 403

    page = await client.get("/", headers=_cookie(admin))
    assert "key-usage-tbody" in page.text


async def test_usage_api_with_a_management_key(
    dash_client: tuple[httpx.AsyncClient, Database],
) -> None:
    client, db = dash_client
    await _seed(db)
    admin = await db.create_dashboard_user("adm", hash_password("password-12345"), True)
    invalidate_dashboard_user_count_cache()

    async def issue(privileges: list[str]) -> dict[str, str]:
        r = await client.post(
            "/api/management-keys",
            json={"name": "bot", "privileges": privileges},
            headers=_cookie(admin),
        )
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['management_key']}"}

    r = await client.get("/api/keys/usage", headers=await issue(["keys:read"]))
    assert r.status_code == 200
    assert len(r.json()["rows"]) == 5
    r = await client.get("/api/keys/usage", headers=await issue(["models:read"]))
    assert r.status_code == 403
