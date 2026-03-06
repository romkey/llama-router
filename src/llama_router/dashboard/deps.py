"""Shared dependency accessors for dashboard routes."""

from __future__ import annotations

from ..database import Database
from ..provider_manager import ProviderManager
from ..registry_cache.cache import BlobCache

_db: Database | None = None
_pm: ProviderManager | None = None
_blob_cache: BlobCache | None = None


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
