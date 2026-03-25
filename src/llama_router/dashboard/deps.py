"""Shared dependency accessors for dashboard routes."""

from __future__ import annotations

import secrets

from ..config import settings
from ..database import Database
from ..provider_manager import ProviderManager
from ..registry_cache.cache import BlobCache

_db: Database | None = None
_pm: ProviderManager | None = None
_blob_cache: BlobCache | None = None
_dashboard_session_secret_cache: str | None = None


def init(db: Database, pm: ProviderManager) -> None:
    global _db, _pm
    _db = db
    _pm = pm


def init_cache(cache: BlobCache) -> None:
    global _blob_cache
    _blob_cache = cache


def get_db() -> Database:
    assert _db is not None
    return _db


def get_pm() -> ProviderManager:
    assert _pm is not None
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
