from __future__ import annotations

import contextlib
import os
import sqlite3
from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
import sqlalchemy as sa
from alembic.script import ScriptDirectory
from fastapi import FastAPI, Request, Response
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import RequestResponseEndpoint

import llama_router
from llama_router import request_logger
from llama_router.api import deps as api_deps
from llama_router.config import Settings, settings
from llama_router.dashboard import deps as dash_deps
from llama_router.dashboard import management_keys
from llama_router.dashboard.app import app as dashboard_app
from llama_router.dashboard.auth_core import DashboardUser
from llama_router.dashboard.decisions import router as dash_decisions_router
from llama_router.dashboard.login_throttle import throttle as login_throttle
from llama_router.dashboard.middleware import (
    DashboardAuthMiddleware,
    invalidate_dashboard_user_count_cache,
)
from llama_router.database import Database
from llama_router.db_migrate import run_upgrade_sync
from llama_router.limits import limiter
from llama_router.llamacpp_api import audio, deps
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router
from llama_router.routing_stats import ProviderStats

# Point the ``db`` fixture at a real server instead of a temporary SQLite file,
# e.g. LLAMA_ROUTER_TEST_DATABASE_URL=mysql+asyncmy://root:pw@127.0.0.1:3306/lr
# The database is emptied before each test, so never use one you care about.
_TEST_DATABASE_URL = os.environ.get("LLAMA_ROUTER_TEST_DATABASE_URL", "").strip()


class _ServerDatabase:
    """Empties the LLAMA_ROUTER_TEST_DATABASE_URL server between tests.

    Migrating a server database takes seconds, so it is migrated once; each
    test then empties every table and puts back the rows the migrations
    seeded. A test that leaves the schema anywhere but head gets a full
    drop and migration instead.
    """

    def __init__(self, async_url: str) -> None:
        self.sync_url = Settings().sync_database_url_for_alembic(async_url)
        self.engine = sa.create_engine(self.sync_url)
        self.meta: sa.MetaData | None = None
        self.seed: dict[str, list[dict[str, Any]]] = {}

    def _at_head(self) -> bool:
        if self.meta is None:
            return False
        names = set(sa.inspect(self.engine).get_table_names())
        if names != set(self.meta.tables):
            return False
        with self.engine.connect() as conn:
            version = conn.execute(
                sa.text("SELECT version_num FROM alembic_version")
            ).scalar()
        return version == _HEAD

    def _migrate(self) -> None:
        meta = sa.MetaData()
        meta.reflect(bind=self.engine)
        meta.drop_all(bind=self.engine)
        run_upgrade_sync(self.sync_url)
        self.meta = sa.MetaData()
        self.meta.reflect(bind=self.engine)
        with self.engine.connect() as conn:
            self.seed = {
                table.name: [dict(r) for r in conn.execute(table.select()).mappings()]
                for table in self.meta.sorted_tables
                if table.name != "alembic_version"
            }

    def reset(self) -> None:
        if not self._at_head():
            self._migrate()
            return
        assert self.meta is not None
        tables = [t for t in self.meta.sorted_tables if t.name != "alembic_version"]
        quote = self.engine.dialect.identifier_preparer.quote
        with self.engine.begin() as conn:
            if self.engine.dialect.name == "postgresql":
                names = ", ".join(quote(t.name) for t in tables)
                conn.execute(sa.text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))
            else:
                conn.execute(sa.text("SET FOREIGN_KEY_CHECKS = 0"))
                for t in tables:
                    conn.execute(sa.text(f"TRUNCATE TABLE {quote(t.name)}"))
                conn.execute(sa.text("SET FOREIGN_KEY_CHECKS = 1"))
            for t in tables:
                if self.seed.get(t.name):
                    conn.execute(t.insert(), self.seed[t.name])


_HEAD = ScriptDirectory(
    str(Path(llama_router.__file__).resolve().parent / "alembic")
).get_current_head()
_server_database = _ServerDatabase(_TEST_DATABASE_URL) if _TEST_DATABASE_URL else None


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
def _fresh_upstream_limiter() -> Iterator[None]:
    """Each test gets new upstream semaphores (they bind to one event loop)."""
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture(autouse=True)
def _fresh_voices_cache() -> None:
    """Each test fetches /v1/audio/voices afresh."""
    audio._voices_cache.clear()


@pytest.fixture(autouse=True)
def _fresh_management_key_touches() -> None:
    """Key ids repeat across tests' databases; forget when each was used."""
    management_keys._touched.clear()


@pytest.fixture(autouse=True)
def _isolated_secret_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the generated secret (used without a session secret) per test."""
    monkeypatch.setattr(settings, "secret_key_path", str(tmp_path / "secret.key"))


@pytest.fixture(scope="session")
def _migrated_sqlite(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A SQLite file migrated to head once per session (per xdist worker).

    Migrating takes far longer than copying, so tests start from a copy.
    """
    path = tmp_path_factory.mktemp("template") / "migrated.db"
    run_upgrade_sync(f"sqlite:///{path}")
    return path


@pytest.fixture
def database_url(tmp_path: Path, _migrated_sqlite: Path) -> str:
    """An empty database for one test: LLAMA_ROUTER_TEST_DATABASE_URL, emptied,
    or a fresh copy of the migrated SQLite template."""
    if _server_database is not None:
        _server_database.reset()
        return _TEST_DATABASE_URL
    path = tmp_path / "test.db"
    with contextlib.closing(sqlite3.connect(_migrated_sqlite)) as con:
        con.execute("VACUUM INTO ?", (str(path),))
    return str(path)


@pytest_asyncio.fixture
async def db(database_url: str) -> AsyncIterator[Database]:
    database = Database(database_url)
    await database.connect()
    yield database
    await database.close()


@pytest_asyncio.fixture
async def dash_client(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[httpx.AsyncClient, Database]]:
    """ASGI client with isolated DB and fixed session secret."""
    monkeypatch.setattr(
        settings, "session_secret", "test-session-secret-key-min-16chars"
    )
    monkeypatch.setattr(settings, "dashboard_cookie_secure", False)

    db = Database(database_url)
    await db.connect()
    pm = ProviderManager(db)
    dash_deps.invalidate_dashboard_session_secret_cache()
    invalidate_dashboard_user_count_cache()
    dash_deps.init(db, pm)
    login_throttle.clear()

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


def asgi_client(app: FastAPI) -> httpx.AsyncClient:
    """An async client for *app* on the test's own event loop (the sync
    TestClient runs the app on another, which server databases refuse)."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )


def db_timestamp(db: Database, when: datetime) -> Any:
    """*when* as the database wants it bound: SQLite stores text."""
    if db._dialect_name == "sqlite":
        return when.strftime("%Y-%m-%d %H:%M:%S")
    return when.replace(tzinfo=None)


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def _reset_deps() -> None:
    """Forget the database and managers the route modules were given, so a
    later test cannot reach a closed database through them."""
    deps._db, deps._pm, deps._router = None, None, None
    api_deps._db, api_deps._pm, api_deps._router = None, None, None
    dash_deps._db = None
    dash_deps._pm = None


@pytest.fixture
async def pm(db: Database, clock: FakeClock) -> AsyncIterator[ProviderManager]:
    manager = ProviderManager(db)
    manager.stats = ProviderStats(clock=clock)
    request_logger.attach_stats(manager.stats)
    yield manager
    request_logger.attach_stats(None)
    await manager.stop()
    _reset_deps()


@pytest.fixture
async def router(
    request: pytest.FixtureRequest, db: Database, pm: ProviderManager
) -> AsyncIterator[Router]:
    """A router with the default (deny) anonymous access. Tests that send
    requests without a key are marked ``allow_unauthenticated``."""
    if request.node.get_closest_marker("allow_unauthenticated"):
        await db.set_allow_unauthenticated(True)
    rt = Router(db, pm)
    deps.init(db, pm, rt)
    yield rt
    _reset_deps()


@pytest.fixture
async def dash(db: Database, pm: ProviderManager) -> AsyncIterator[httpx.AsyncClient]:
    """The decisions routes behind the dashboard's sign-in middleware."""
    dash_deps.init(db, pm)
    invalidate_dashboard_user_count_cache()
    app = FastAPI()
    app.add_middleware(DashboardAuthMiddleware)
    app.include_router(dash_decisions_router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://dash") as c:
        yield c
    invalidate_dashboard_user_count_cache()
