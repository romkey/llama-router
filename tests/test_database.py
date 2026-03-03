from __future__ import annotations

import pytest

from llama_router.database import Database
from llama_router.models import BenchmarkResult, ProviderModel, ProviderStatus


@pytest.mark.asyncio
async def test_add_and_list_providers(db: Database):
    p = await db.add_provider("test-server", "http://localhost:11434")
    assert p.id is not None
    assert p.name == "test-server"

    providers = await db.list_providers()
    assert len(providers) == 1
    assert providers[0].name == "test-server"


@pytest.mark.asyncio
async def test_remove_provider(db: Database):
    p = await db.add_provider("to-remove", "http://localhost:11434")
    await db.remove_provider(p.id)
    providers = await db.list_providers()
    assert len(providers) == 0


@pytest.mark.asyncio
async def test_update_provider_status(db: Database):
    p = await db.add_provider("srv", "http://localhost:11434")
    await db.update_provider_status(p.id, ProviderStatus.IDLE)
    updated = await db.get_provider(p.id)
    assert updated is not None
    assert updated.status == ProviderStatus.IDLE


@pytest.mark.asyncio
async def test_provider_models(db: Database):
    p = await db.add_provider("srv", "http://localhost:11434")
    models = [
        ProviderModel(provider_id=p.id, name="llama3:8b", size=4_000_000_000),
        ProviderModel(provider_id=p.id, name="nomic-embed-text", size=500_000_000),
    ]
    await db.set_provider_models(p.id, models)
    result = await db.get_provider_models(p.id)
    assert len(result) == 2
    assert {m.name for m in result} == {"llama3:8b", "nomic-embed-text"}


@pytest.mark.asyncio
async def test_get_providers_for_model(db: Database):
    p1 = await db.add_provider("srv1", "http://host1:11434")
    p2 = await db.add_provider("srv2", "http://host2:11434")
    await db.update_provider_status(p1.id, ProviderStatus.IDLE)
    await db.update_provider_status(p2.id, ProviderStatus.IDLE)

    await db.set_provider_models(
        p1.id, [ProviderModel(provider_id=p1.id, name="llama3:8b")]
    )
    await db.set_provider_models(
        p2.id, [ProviderModel(provider_id=p2.id, name="llama3:8b")]
    )

    providers = await db.get_providers_for_model("llama3:8b")
    assert len(providers) == 2


@pytest.mark.asyncio
async def test_benchmarks(db: Database):
    p = await db.add_provider("srv", "http://localhost:11434")
    result = BenchmarkResult(
        provider_id=p.id,
        model_name="llama3:8b",
        startup_time_ms=150.0,
        tokens_per_second=42.5,
    )
    await db.save_benchmark(result)

    latest = await db.get_latest_benchmark(p.id, "llama3:8b")
    assert latest is not None
    assert latest.tokens_per_second == 42.5
    assert latest.startup_time_ms == 150.0


@pytest.mark.asyncio
async def test_list_all_models_deduplicates(db: Database):
    p1 = await db.add_provider("srv1", "http://host1:11434")
    p2 = await db.add_provider("srv2", "http://host2:11434")
    await db.update_provider_status(p1.id, ProviderStatus.IDLE)
    await db.update_provider_status(p2.id, ProviderStatus.IDLE)

    await db.set_provider_models(
        p1.id, [ProviderModel(provider_id=p1.id, name="llama3:8b")]
    )
    await db.set_provider_models(
        p2.id, [ProviderModel(provider_id=p2.id, name="llama3:8b")]
    )

    models = await db.list_all_models()
    assert len(models) == 1
    assert models[0]["name"] == "llama3:8b"
