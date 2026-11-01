"""Router IDs and the loop guard on relayed requests."""

from __future__ import annotations

import uuid
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request

from llama_router import router_identity as ri
from llama_router.database import Database

ME = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ri, "_router_id", None)
    monkeypatch.setattr(ri, "_hub_id", None)
    monkeypatch.setattr(ri, "_hub_listeners", [])


async def test_id_is_made_once_and_kept(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "r.db"))
    await db.connect()
    try:
        first = await ri.load(db)
        assert uuid.UUID(first)
        assert await ri.load(db) == first
        assert ri.router_id() == first
        new = await ri.regenerate(db)
        assert new != first
        assert await ri.load(db) == new
    finally:
        await db.close()


async def test_a_bad_stored_id_is_replaced(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "r.db"))
    await db.connect()
    try:
        await db.set_app_setting(ri.DB_KEY, "not-a-uuid")
        rid = await ri.load(db)
        assert uuid.UUID(rid)
        assert await db.get_app_setting(ri.DB_KEY) == rid
    finally:
        await db.close()


def test_parse_path() -> None:
    assert ri.parse_path(None) == ()
    assert ri.parse_path(f" {ME} , ,{OTHER.upper()},") == (ME, OTHER)
    # Anything that is not a router ID is dropped, not passed on.
    assert ri.parse_path(f"a,{ME},<script>,b") == (ME,)


async def test_relayed_path_keeps_only_router_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ri, "_router_id", ME)
    sent: list[httpx.Request] = []
    resp = await _get(_guarded_app(sent), {ri.PATH_HEADER: f"junk,{OTHER},x y"})
    assert resp.status_code == 200
    assert sent[-1].headers[ri.PATH_HEADER] == f"{OTHER},{ME}"


async def test_an_oversized_path_header_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ri, "_router_id", ME)
    sent: list[httpx.Request] = []
    huge = "x" * (ri.MAX_PATH_HEADER_CHARS + 1)
    resp = await _get(_guarded_app(sent), {ri.PATH_HEADER: huge})
    assert resp.status_code == ri.LOOP_STATUS
    assert "too long" in resp.json()["error"]
    assert not sent
    # A full path of real IDs still fits.
    full = ",".join(str(uuid.uuid4()) for _ in range(ri.MAX_HOPS - 1))
    assert len(full) <= ri.MAX_PATH_HEADER_CHARS
    assert (await _get(_guarded_app(sent), {ri.PATH_HEADER: full})).status_code == 200


def test_loop_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ri, "_router_id", ME)
    assert ri.loop_reason(()) is None
    assert ri.loop_reason((OTHER,)) is None
    assert "this router" in (ri.loop_reason((OTHER, ME)) or "")
    long = tuple(str(uuid.uuid4()) for _ in range(ri.MAX_HOPS))
    assert "8 routers" in (ri.loop_reason(long) or "")


def test_hub_id_listeners_fire_on_change_only() -> None:
    calls: list[str | None] = []

    def listener() -> None:
        calls.append(ri.hub_id())

    def broken() -> None:
        raise RuntimeError("listener failed")

    ri.on_hub_change(broken)
    ri.on_hub_change(listener)
    ri.set_hub_id(OTHER)
    ri.set_hub_id(OTHER)
    ri.set_hub_id("junk")
    ri.remove_hub_listener(listener)
    ri.remove_hub_listener(listener)
    ri.set_hub_id(ME)
    assert calls == [OTHER, None]


def _guarded_app(sent: list[httpx.Request]) -> FastAPI:
    """An app that relays each request to a recorded upstream, like the router."""

    async def record(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"ok": True})

    app = FastAPI()
    app.add_middleware(ri.LoopGuardMiddleware)

    @app.get("/relay")
    async def relay(request: Request) -> dict[str, bool]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(record),
            event_hooks={"request": [ri.add_path_header]},
        ) as client:
            await client.get("http://upstream/api/tags")
        return {"ok": True}

    return app


async def _get(app: FastAPI, headers: dict[str, str] | None = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://r") as c:
        return await c.get("/relay", headers=headers)


async def test_relayed_requests_carry_the_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ri, "_router_id", ME)
    sent: list[httpx.Request] = []
    app = _guarded_app(sent)

    assert (await _get(app)).status_code == 200
    assert sent[-1].headers[ri.PATH_HEADER] == ME

    assert (await _get(app, {ri.PATH_HEADER: OTHER})).status_code == 200
    assert sent[-1].headers[ri.PATH_HEADER] == f"{OTHER},{ME}"
    # The path belongs to that request only.
    assert ri.outgoing_path() == ME


async def test_a_request_that_was_here_before_gets_508(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ri, "_router_id", ME)
    sent: list[httpx.Request] = []
    resp = await _get(_guarded_app(sent), {ri.PATH_HEADER: f"{ME},{OTHER}"})
    assert resp.status_code == ri.LOOP_STATUS
    assert "routing loop" in resp.json()["error"]
    assert not sent


async def test_no_header_before_the_id_is_loaded() -> None:
    sent: list[httpx.Request] = []
    assert (await _get(_guarded_app(sent))).status_code == 200
    assert ri.PATH_HEADER not in sent[-1].headers
