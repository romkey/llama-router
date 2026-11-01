from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
import sqlalchemy as sa
from fastapi import FastAPI, Request, Response
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import RequestResponseEndpoint

from llama_router import request_logger
from llama_router.config import Settings, settings
from llama_router.dashboard import deps as dash_deps
from llama_router.dashboard.app import app as dashboard_app
from llama_router.dashboard.auth_core import DashboardUser
from llama_router.dashboard.decisions import router as dash_decisions_router
from llama_router.dashboard.middleware import (
    DashboardAuthMiddleware,
    invalidate_dashboard_user_count_cache,
)
from llama_router.database import Database
from llama_router.llamacpp_api import audio, deps
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router
from llama_router.routing_stats import ProviderStats

# Point the ``db`` fixture at a real server instead of a temporary SQLite file,
# e.g. LLAMA_ROUTER_TEST_DATABASE_URL=mysql+asyncmy://root:pw@127.0.0.1:3306/lr
# The database is emptied before each test, so never use one you care about.
_TEST_DATABASE_URL = os.environ.get("LLAMA_ROUTER_TEST_DATABASE_URL", "").strip()


def _empty_database(async_url: str) -> None:
    sync_url = Settings().sync_database_url_for_alembic(async_url)
    engine = sa.create_engine(sync_url)
    try:
        meta = sa.MetaData()
        meta.reflect(bind=engine)
        meta.drop_all(bind=engine)
    finally:
        engine.dispose()


@pytest.fixture(autouse=True)
def _no_background_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Background endpoint probes would hit every fake upstream; tests that
    exercise them turn them back on."""
    monkeypatch.setattr(settings, "auto_probe_enabled", False)


_TEST_ADMIN = DashboardUser(id=0, username="test-admin", is_admin=True)


@pytest.fixture(autouse=True)
def _dashboard_admin_without_accounts(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no dashboard accounts, act as a signed-in administrator.

    Most dashboard tests are about something other than sign-in. Tests of
    the real no-accounts behaviour (a locked setup page) are marked
    ``no_accounts_locked``.
    """
    if request.node.get_closest_marker("no_accounts_locked"):
        return

    async def as_admin(
        _self: DashboardAuthMiddleware,
        req: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        req.state.dashboard_user = _TEST_ADMIN
        return await call_next(req)

    monkeypatch.setattr(DashboardAuthMiddleware, "no_accounts", as_admin)


@pytest.fixture(autouse=True)
def _fresh_voices_cache() -> None:
    """Each test fetches /v1/audio/voices afresh."""
    audio._voices_cache.clear()


@pytest.fixture(autouse=True)
def _isolated_secret_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the generated secret (used without a session secret) per test."""
    monkeypatch.setattr(settings, "secret_key_path", str(tmp_path / "secret.key"))


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    if _TEST_DATABASE_URL:
        _empty_database(_TEST_DATABASE_URL)
        database = Database(_TEST_DATABASE_URL)
    else:
        database = Database(str(tmp_path / "test.db"))
    await database.connect()
    yield database
    await database.close()


@pytest_asyncio.fixture
async def dash_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[httpx.AsyncClient, Database]]:
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


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
async def pm(db: Database, clock: FakeClock) -> AsyncIterator[ProviderManager]:
    manager = ProviderManager(db)
    manager.stats = ProviderStats(clock=clock)
    request_logger.attach_stats(manager.stats)
    yield manager
    request_logger.attach_stats(None)
    await manager.stop()


@pytest.fixture
async def router(db: Database, pm: ProviderManager) -> Router:
    await db.set_allow_unauthenticated(True)
    rt = Router(db, pm)
    deps.init(db, pm, rt)
    return rt


@pytest.fixture
async def dash(db: Database, pm: ProviderManager) -> AsyncIterator[httpx.AsyncClient]:
    dash_deps.init(db, pm)
    app = FastAPI()
    app.include_router(dash_decisions_router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://dash") as c:
        yield c
