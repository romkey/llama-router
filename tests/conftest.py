from __future__ import annotations

import os

import pytest
import pytest_asyncio
import sqlalchemy as sa

from llama_router.config import Settings, settings
from llama_router.database import Database

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
def _no_background_probes(monkeypatch):
    """Background endpoint probes would hit every fake upstream; tests that
    exercise them turn them back on."""
    monkeypatch.setattr(settings, "auto_probe_enabled", False)


@pytest_asyncio.fixture
async def db(tmp_path):
    if _TEST_DATABASE_URL:
        _empty_database(_TEST_DATABASE_URL)
        database = Database(_TEST_DATABASE_URL)
    else:
        database = Database(str(tmp_path / "test.db"))
    await database.connect()
    yield database
    await database.close()
