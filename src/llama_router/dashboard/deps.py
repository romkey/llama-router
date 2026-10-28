"""Shared dependency accessors for dashboard routes."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

from ..config import settings
from ..database import Database
from ..provider_manager import ProviderManager
from ..registry_cache.cache import BlobCache
from ..worker_link.worker import WorkerLinkManager

if TYPE_CHECKING:
    from ..auto_benchmark import AutoBenchmarker

_db: Database | None = None
_pm: ProviderManager | None = None
_blob_cache: BlobCache | None = None
_worker_link: WorkerLinkManager | None = None
_auto_benchmark: AutoBenchmarker | None = None
_dashboard_session_secret_cache: str | None = None


def init(db: Database, pm: ProviderManager) -> None:
    global _db, _pm
    _db = db
    _pm = pm


def init_cache(cache: BlobCache) -> None:
    global _blob_cache
    _blob_cache = cache


def init_worker_link(manager: WorkerLinkManager) -> None:
    global _worker_link
    _worker_link = manager


def init_auto_benchmark(manager: AutoBenchmarker) -> None:
    global _auto_benchmark
    _auto_benchmark = manager


def get_auto_benchmark() -> AutoBenchmarker:
    """The auto-benchmarker, creating an idle (not started) one for tests."""
    global _auto_benchmark
    from ..auto_benchmark import AutoBenchmarker

    db, pm = get_db(), get_pm()
    if _auto_benchmark is None or _auto_benchmark._db is not db:
        _auto_benchmark = AutoBenchmarker(db, pm)
    return _auto_benchmark


def get_worker_link() -> WorkerLinkManager:
    """Return the worker link manager, creating an idle one for tests."""
    global _worker_link
    db = get_db()
    if _worker_link is None or _worker_link.db is not db:
        _worker_link = WorkerLinkManager(db)
    return _worker_link


def get_db() -> Database:
    if _db is None:
        raise RuntimeError("db is not initialised; call init() first")
    return _db


def get_pm() -> ProviderManager:
    if _pm is None:
        raise RuntimeError("pm is not initialised; call init() first")
    return _pm


def get_cache() -> BlobCache | None:
    return _blob_cache


def invalidate_dashboard_session_secret_cache() -> None:
    global _dashboard_session_secret_cache
    _dashboard_session_secret_cache = None


async def get_dashboard_session_secret() -> str:
    global _dashboard_session_secret_cache
    if settings.session_secret.strip():
        return settings.session_secret.strip()
    if _dashboard_session_secret_cache:
        return _dashboard_session_secret_cache
    db = get_db()
    key = "dashboard_session_secret"
    val = await db.get_app_setting(key)
    if not val:
        val = secrets.token_urlsafe(48)
        await db.set_app_setting(key, val)
    _dashboard_session_secret_cache = val
    return val
