"""The dashboard's pages and APIs, one router per tab."""

from __future__ import annotations

from fastapi import APIRouter

from . import (
    api_keys,
    benchmarks,
    cache,
    health,
    home,
    login,
    model_aliases,
    providers,
    pulls,
    routing,
    users,
    workers,
)

router = APIRouter()
for _tab in (
    health,
    login,
    users,
    home,
    routing,
    api_keys,
    providers,
    benchmarks,
    pulls,
    cache,
    model_aliases,
    workers,
):
    router.include_router(_tab.router)
