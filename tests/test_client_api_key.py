"""Clients send their key as API-KEY or as Authorization: Bearer."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from starlette.requests import Request

from llama_router import auth
from llama_router.api import deps as api_deps
from llama_router.api.app import app as ollama_app
from llama_router.auth import (
    client_api_key,
    generate_api_key,
    key_hash,
    key_prefix,
    routing_preferences_from_request,
)
from llama_router.database import Database
from llama_router.llamacpp_api import deps as v1_deps
from llama_router.llamacpp_api.app import app as v1_app
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router


def _request(headers: dict[str, str]) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "headers": raw, "method": "POST", "path": "/"})


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"API-KEY": "lr_a"}, "lr_a"),
        ({"Authorization": "Bearer lr_b"}, "lr_b"),
        ({"Authorization": "bearer   lr_c  "}, "lr_c"),
        ({"API-KEY": "lr_a", "Authorization": "Bearer lr_b"}, "lr_a"),
        ({"API-KEY": "  ", "Authorization": "Bearer lr_b"}, "lr_b"),
        ({"Authorization": "Basic dXNlcjpwYXNz"}, ""),
        ({"Authorization": "Bearer"}, ""),
        ({}, ""),
    ],
)
def test_client_api_key(headers: dict[str, str], expected: str) -> None:
    assert client_api_key(_request(headers)) == expected


async def test_bearer_key_resolves_like_api_key(db: Database) -> None:
    plain = generate_api_key()
    await db.create_api_key(
        key_prefix(plain), key_hash(plain), "balanced", True, locality="local"
    )
    prefs = await routing_preferences_from_request(
        db, _request({"Authorization": f"Bearer {plain}"})
    )
    assert prefs is not None
    assert prefs.locality == "local"


async def test_missing_and_invalid_bearer_key_refused(db: Database) -> None:
    await db.set_allow_unauthenticated(False)
    with pytest.raises(HTTPException) as missing:
        await routing_preferences_from_request(db, _request({}))
    assert missing.value.status_code == 401
    with pytest.raises(HTTPException) as invalid:
        await routing_preferences_from_request(
            db, _request({"Authorization": "Bearer lr_nope"})
        )
    assert invalid.value.status_code == 401


async def test_malformed_key_refused_without_hashing(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _count_hashes(monkeypatch)
    monkeypatch.setattr(db, "lookup_api_key", _fail_lookup)
    for bad in ("lr_nope", "lrk_short", "x" * 47, "lrk_" + "a" * 42 + "!"):
        with pytest.raises(HTTPException) as exc:
            await routing_preferences_from_request(db, _request({"API-KEY": bad}))
        assert exc.value.status_code == 401
    assert calls == []


async def _fail_lookup(_hash: str) -> None:
    raise AssertionError("malformed keys must not reach the database")


def _count_hashes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    real = auth.key_hash

    def counting(key: str) -> str:
        calls.append(key)
        return real(key)

    monkeypatch.setattr(auth, "key_hash", counting)
    return calls


async def test_unknown_key_hashed_once_then_refused_from_cache(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _count_hashes(monkeypatch)
    unknown = generate_api_key()
    for _ in range(3):
        with pytest.raises(HTTPException) as exc:
            await routing_preferences_from_request(db, _request({"API-KEY": unknown}))
        assert exc.value.status_code == 401
    assert calls == [unknown]


async def test_unknown_key_cache_expires(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _count_hashes(monkeypatch)
    unknown = generate_api_key()
    invalid = auth._cache_for(db).invalid
    for _ in range(2):
        with pytest.raises(HTTPException):
            await routing_preferences_from_request(db, _request({"API-KEY": unknown}))
        # Pretend the entry's 30 seconds are up.
        for fingerprint in invalid:
            invalid[fingerprint] -= auth._INVALID_TTL_S
    assert calls == [unknown, unknown]


async def test_creating_a_key_clears_unknown_key_cache(db: Database) -> None:
    plain = generate_api_key()
    with pytest.raises(HTTPException):
        await routing_preferences_from_request(db, _request({"API-KEY": plain}))
    await db.create_api_key(key_prefix(plain), key_hash(plain), "balanced", True)
    prefs = await routing_preferences_from_request(db, _request({"API-KEY": plain}))
    assert prefs is not None


async def test_unknown_key_cache_is_bounded(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(auth, "_MAX_INVALID_KEYS", 3)
    for _ in range(5):
        with pytest.raises(HTTPException):
            await routing_preferences_from_request(
                db, _request({"API-KEY": generate_api_key()})
            )
    assert len(auth._cache_for(db).invalid) <= 3


async def test_allow_unauthenticated_cache_expires(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another replica may change the setting; this one must notice."""
    await db.set_allow_unauthenticated(True)
    assert await routing_preferences_from_request(db, _request({})) is None

    async def changed_elsewhere() -> bool:
        return False

    monkeypatch.setattr(db, "get_allow_unauthenticated", changed_elsewhere)
    assert await routing_preferences_from_request(db, _request({})) is None
    cache = auth._cache_for(db)
    assert cache.allow_unauthenticated is not None
    fetched, value = cache.allow_unauthenticated
    cache.allow_unauthenticated = (fetched - auth._PREFS_TTL_S, value)
    with pytest.raises(HTTPException) as exc:
        await routing_preferences_from_request(db, _request({}))
    assert exc.value.status_code == 401


@pytest.fixture
async def listing_app(db: Database) -> str:
    await db.set_allow_unauthenticated(False)
    pm = ProviderManager(db)
    rt = Router(db, pm)
    await rt.rebuild()
    api_deps.init(db, pm, rt)
    v1_deps.init(db, pm, rt)
    plain = generate_api_key()
    await db.create_api_key(key_prefix(plain), key_hash(plain), "balanced", True)
    return plain


@pytest.mark.parametrize(
    ("app", "path"),
    [
        (ollama_app, "/api/tags"),
        (ollama_app, "/api/version"),
        (ollama_app, "/api/ps"),
        (v1_app, "/v1/models"),
    ],
)
async def test_listing_endpoints_need_a_key(
    listing_app: str, app: FastAPI, path: str
) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://r") as c:
        assert (await c.get(path)).status_code == 401
        ok = await c.get(path, headers={"Authorization": f"Bearer {listing_app}"})
        assert ok.status_code == 200


async def test_single_model_needs_a_key(listing_app: str) -> None:
    transport = httpx.ASGITransport(app=v1_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://r") as c:
        assert (await c.get("/v1/models/nope")).status_code == 401
        missing = await c.get("/v1/models/nope", headers={"API-KEY": listing_app})
        assert missing.status_code == 404


async def test_health_stays_public(listing_app: str) -> None:
    transport = httpx.ASGITransport(app=v1_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://r") as c:
        assert (await c.get("/health")).status_code == 200
