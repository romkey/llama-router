from __future__ import annotations

import pytest
import pytest_asyncio

from llama_router.config import settings
from llama_router.database import Database


@pytest.fixture(autouse=True)
def _no_background_probes(monkeypatch):
    """Background endpoint probes would hit every fake upstream; tests that
    exercise them turn them back on."""
    monkeypatch.setattr(settings, "auto_probe_enabled", False)


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "test.db"))
    await database.connect()
    yield database
    await database.close()
