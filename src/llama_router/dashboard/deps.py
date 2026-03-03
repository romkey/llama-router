"""Shared dependency accessors for dashboard routes."""

from __future__ import annotations

from ..database import Database
from ..provider_manager import ProviderManager

_db: Database | None = None
_pm: ProviderManager | None = None


def init(db: Database, pm: ProviderManager) -> None:
    global _db, _pm
    _db = db
    _pm = pm


def get_db() -> Database:
    assert _db is not None
    return _db


def get_pm() -> ProviderManager:
    assert _pm is not None
    return _pm
