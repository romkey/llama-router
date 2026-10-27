"""Shared dependency accessors for llama.cpp API routes."""

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
    if _db is None:
        raise RuntimeError("db is not initialised; call init() first")
    return _db


def get_pm() -> ProviderManager:
    if _pm is None:
        raise RuntimeError("pm is not initialised; call init() first")
    return _pm


def get_router() -> Router:
    if _router is None:
        raise RuntimeError("router is not initialised; call init() first")
    return _router
