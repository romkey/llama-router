from __future__ import annotations

import pytest

from llama_router.database import Database
from llama_router.models import BenchmarkResult, ProviderModel, ProviderStatus
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router


@pytest.mark.asyncio
async def test_route_picks_least_busy(db: Database):
    p1 = await db.add_provider("fast", "http://host1:11434")
    p2 = await db.add_provider("slow", "http://host2:11434")
    await db.update_provider_status(p1.id, ProviderStatus.IDLE)
    await db.update_provider_status(p2.id, ProviderStatus.IDLE)

    await db.set_provider_models(
        p1.id, [ProviderModel(provider_id=p1.id, name="llama3:8b")]
    )
    await db.set_provider_models(
        p2.id, [ProviderModel(provider_id=p2.id, name="llama3:8b")]
    )

    pm = ProviderManager(db)
    pm._active_requests[p1.id] = 3
    pm._active_requests[p2.id] = 0

    rt = Router(db, pm)
    result = await rt.route("llama3:8b")
    assert result is not None
    assert result.provider.id == p2.id
    assert result.resolved_model == "llama3:8b"


@pytest.mark.asyncio
async def test_route_prefers_faster_when_equal_load(db: Database):
    p1 = await db.add_provider("fast", "http://host1:11434")
    p2 = await db.add_provider("slow", "http://host2:11434")
    await db.update_provider_status(p1.id, ProviderStatus.IDLE)
    await db.update_provider_status(p2.id, ProviderStatus.IDLE)

    await db.set_provider_models(
        p1.id, [ProviderModel(provider_id=p1.id, name="llama3:8b")]
    )
    await db.set_provider_models(
        p2.id, [ProviderModel(provider_id=p2.id, name="llama3:8b")]
    )

    await db.save_benchmark(
        BenchmarkResult(
            provider_id=p1.id,
            model_name="llama3:8b",
            tokens_per_second=80.0,
            startup_time_ms=100.0,
        )
    )
    await db.save_benchmark(
        BenchmarkResult(
            provider_id=p2.id,
            model_name="llama3:8b",
            tokens_per_second=20.0,
            startup_time_ms=500.0,
        )
    )

    pm = ProviderManager(db)
    rt = Router(db, pm)
    result = await rt.route("llama3:8b")
    assert result is not None
    assert result.provider.id == p1.id


@pytest.mark.asyncio
async def test_route_returns_none_for_unknown_model(db: Database):
    pm = ProviderManager(db)
    rt = Router(db, pm)
    result = await rt.route("nonexistent-model")
    assert result is None


@pytest.mark.asyncio
async def test_route_skips_offline_providers(db: Database):
    p1 = await db.add_provider("offline-srv", "http://host1:11434")
    p2 = await db.add_provider("online-srv", "http://host2:11434")
    await db.update_provider_status(p1.id, ProviderStatus.OFFLINE)
    await db.update_provider_status(p2.id, ProviderStatus.IDLE)

    await db.set_provider_models(
        p1.id, [ProviderModel(provider_id=p1.id, name="llama3:8b")]
    )
    await db.set_provider_models(
        p2.id, [ProviderModel(provider_id=p2.id, name="llama3:8b")]
    )

    pm = ProviderManager(db)
    rt = Router(db, pm)
    result = await rt.route("llama3:8b")
    assert result is not None
    assert result.provider.id == p2.id


@pytest.mark.asyncio
async def test_route_follows_fallback_chain(db: Database):
    p1 = await db.add_provider("srv1", "http://host1:11434")
    await db.update_provider_status(p1.id, ProviderStatus.IDLE)

    await db.set_provider_models(
        p1.id, [ProviderModel(provider_id=p1.id, name="llama3:8b")]
    )

    await db.set_model_fallback("llama3:70b", "llama3:8b")

    pm = ProviderManager(db)
    rt = Router(db, pm)
    result = await rt.route("llama3:70b")
    assert result is not None
    assert result.provider.id == p1.id
    assert result.resolved_model == "llama3:8b"


@pytest.mark.asyncio
async def test_route_cascading_fallback(db: Database):
    p1 = await db.add_provider("srv1", "http://host1:11434")
    await db.update_provider_status(p1.id, ProviderStatus.IDLE)

    await db.set_provider_models(
        p1.id, [ProviderModel(provider_id=p1.id, name="llama3:8b")]
    )

    await db.set_model_fallback("llama3:405b", "llama3:70b")
    await db.set_model_fallback("llama3:70b", "llama3:8b")

    pm = ProviderManager(db)
    rt = Router(db, pm)
    result = await rt.route("llama3:405b")
    assert result is not None
    assert result.provider.id == p1.id
    assert result.resolved_model == "llama3:8b"


@pytest.mark.asyncio
async def test_route_fallback_returns_none_if_chain_exhausted(db: Database):
    await db.set_model_fallback("modelA", "modelB")
    await db.set_model_fallback("modelB", "modelC")

    pm = ProviderManager(db)
    rt = Router(db, pm)
    result = await rt.route("modelA")
    assert result is None
