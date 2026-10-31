"""Clients send their key as API-KEY or as Authorization: Bearer."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from llama_router.auth import (
    client_api_key,
    generate_api_key,
    key_hash,
    key_prefix,
    routing_preferences_from_request,
)


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


async def test_bearer_key_resolves_like_api_key(db) -> None:
    plain = generate_api_key()
    await db.create_api_key(
        key_prefix(plain), key_hash(plain), "balanced", True, locality="local"
    )
    prefs = await routing_preferences_from_request(
        db, _request({"Authorization": f"Bearer {plain}"})
    )
    assert prefs.locality == "local"


async def test_missing_and_invalid_bearer_key_refused(db) -> None:
    await db.set_allow_unauthenticated(False)
    with pytest.raises(HTTPException) as missing:
        await routing_preferences_from_request(db, _request({}))
    assert missing.value.status_code == 401
    with pytest.raises(HTTPException) as invalid:
        await routing_preferences_from_request(
            db, _request({"Authorization": "Bearer lr_nope"})
        )
    assert invalid.value.status_code == 401
