from __future__ import annotations

import pytest

from llama_router.database import Database
from llama_router.models import (
    BenchmarkResult,
    ProviderModel,
    ProviderStatus,
    ProviderType,
    RequestLog,
)


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


@pytest.mark.asyncio
async def test_providers_for_model_protocol_filter(db: Database):
    p1 = await db.add_provider("ollama-srv", "http://host1:11434")
    p2 = await db.add_provider("lcpp-srv", "http://host2:8080", ProviderType.LLAMACPP)
    p3 = await db.add_provider(
        "both-srv", "http://host3:11434", ProviderType.BOTH, "http://host3:8080"
    )
    for p in [p1, p2, p3]:
        await db.update_provider_status(p.id, ProviderStatus.IDLE)
        await db.set_provider_models(
            p.id, [ProviderModel(provider_id=p.id, name="llama3:8b")]
        )

    all_providers = await db.get_providers_for_model("llama3:8b")
    assert len(all_providers) == 3

    ollama_providers = await db.get_providers_for_model("llama3:8b", protocol="ollama")
    assert len(ollama_providers) == 2
    names = {p.name for p in ollama_providers}
    assert names == {"ollama-srv", "both-srv"}

    lcpp_providers = await db.get_providers_for_model("llama3:8b", protocol="llamacpp")
    assert len(lcpp_providers) == 2
    names = {p.name for p in lcpp_providers}
    assert names == {"lcpp-srv", "both-srv"}


@pytest.mark.asyncio
async def test_address_crud(db: Database):
    p = await db.add_provider("srv", "http://host:11434")
    addr1 = await db.add_address(p.id, "http://host:11434", is_preferred=True)
    addr2 = await db.add_address(p.id, "http://backup:11434", is_preferred=False)

    addrs = await db.get_addresses(p.id)
    assert len(addrs) == 2
    assert addrs[0].is_preferred is True

    await db.set_address_live(addr1.id, True)
    a = await db.get_address(addr1.id)
    assert a is not None and a.is_live is True

    await db.set_address_preferred(addr2.id, True)
    a2 = await db.get_address(addr2.id)
    assert a2 is not None and a2.is_preferred is True

    await db.update_address(addr2.id, "http://new-backup:11434", is_preferred=False)
    a2 = await db.get_address(addr2.id)
    assert a2 is not None and a2.url == "http://new-backup:11434"
    assert a2.is_preferred is False

    await db.remove_address(addr2.id)
    addrs = await db.get_addresses(p.id)
    assert len(addrs) == 1


@pytest.mark.asyncio
async def test_seed_addresses_migration(db: Database):
    """Existing provider url/llamacpp_url get seeded into addresses table."""
    p = await db.add_provider(
        "srv", "http://host:11434", ProviderType.BOTH, "http://host:8080"
    )
    await db._seed_addresses()
    addrs = await db.get_addresses(p.id)
    assert len(addrs) >= 1
    assert addrs[0].url == "http://host:11434"
    assert addrs[0].llamacpp_url == "http://host:8080"
    assert addrs[0].is_preferred is True


@pytest.mark.asyncio
async def test_request_log(db: Database):
    entry = RequestLog(
        provider_id=1,
        provider_name="test-srv",
        protocol="ollama",
        endpoint="/api/chat",
        source_ip="192.168.1.10",
        model="llama3:8b",
        request_size=512,
        response_size=2048,
        duration_ms=1500.5,
        status="ok",
        streamed=True,
    )
    await db.save_request_log(entry)

    logs = await db.get_request_logs(limit=10)
    assert len(logs) == 1
    log = logs[0]
    assert log.provider_name == "test-srv"
    assert log.protocol == "ollama"
    assert log.endpoint == "/api/chat"
    assert log.source_ip == "192.168.1.10"
    assert log.model == "llama3:8b"
    assert log.request_size == 512
    assert log.response_size == 2048
    assert log.duration_ms == 1500.5
    assert log.streamed is True

    count = await db.count_request_logs()
    assert count == 1
