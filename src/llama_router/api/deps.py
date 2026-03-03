"""Shared dependency accessors for API routes."""

from __future__ import annotations

from ..database import Database
from ..provider_manager import ProviderManager
from ..router import Router

_db: Database | None = None
_pm: ProviderManager | None = None
_router: Router | None = None


def init(db: Database, pm: ProviderManager, router: Router) -> None:
    global _db, _pm, _router
    _db = db
    _pm = pm
    _router = router


def get_db() -> Database:
    assert _db is not None
    return _db


def get_pm() -> ProviderManager:
    assert _pm is not None
    return _pm


def get_router() -> Router:
    assert _router is not None
    return _router
