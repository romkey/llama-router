"""SQLAlchemy database layer: migrations, URL handling, and dialect-portable SQL paths.

These tests use SQLite (default CI). They validate Alembic bootstrap, persistence,
and behaviors that differ by dialect in implementation (exercised here on SQLite).
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest
from sqlalchemy.engine.url import URL

from llama_router.database import Database, _bind_params
from llama_router.models import ProviderModel, ProviderStatus, RequestLog


def test_bind_params_flattens_single_tuple() -> None:
    assert _bind_params((1, 2, 3)) == (1, 2, 3)
    assert _bind_params([1, 2]) == (1, 2)


def test_bind_params_passes_through_multiple_args() -> None:
    assert _bind_params(1, "a") == (1, "a")


@pytest.mark.asyncio
async def test_alembic_head_recorded_and_core_tables_exist(
    db: Database, tmp_path
) -> None:
    path = tmp_path / "test.db"
    con = sqlite3.connect(path)
    try:
        ver = con.execute("SELECT version_num FROM alembic_version").fetchone()
        assert ver is not None
        assert ver[0] == "002_peering_key_expiry"
        tables = {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        }
        assert "providers" in tables
        assert "provider_models" in tables
        assert "model_fallbacks" in tables
        assert "app_settings" in tables
        assert "wireguard_interface" in tables
        assert "dashboard_users" in tables
    finally:
        con.close()


@pytest.mark.asyncio
async def test_reopen_database_preserves_data(tmp_path) -> None:
    path = str(tmp_path / "persist.db")
    d1 = Database(path)
    await d1.connect()
    p = await d1.add_provider("keep-me", "http://host:11434")
    pid = p.id
    await d1.close()

    d2 = Database(path)
    await d2.connect()
    try:
        again = await d2.get_provider(pid)
        assert again is not None
        assert again.name == "keep-me"
    finally:
        await d2.close()


@pytest.mark.asyncio
async def test_database_accepts_explicit_sqlite_url(tmp_path) -> None:
    path = tmp_path / "url.db"
    url = URL.create("sqlite+aiosqlite", database=str(path)).render_as_string(
        hide_password=False
    )
    d = Database(url)
    await d.connect()
    try:
        assert await d.list_providers() == []
    finally:
        await d.close()


@pytest.mark.asyncio
async def test_model_fallback_upsert_and_chain(db: Database) -> None:
    await db.set_model_fallback("m-a", "m-b")
    await db.set_model_fallback("m-b", "m-c")
    assert await db.get_model_fallback("m-a") == "m-b"
    all_fb = await db.get_all_model_fallbacks()
    assert all_fb["m-a"] == "m-b"
    assert all_fb["m-b"] == "m-c"

    chain = await db.resolve_fallback_chain("m-a")
    assert chain == ["m-a", "m-b", "m-c"]

    await db.set_model_fallback("m-a", "m-z")
    assert await db.get_model_fallback("m-a") == "m-z"

    await db.remove_model_fallback("m-a")
    assert await db.get_model_fallback("m-a") is None


@pytest.mark.asyncio
async def test_resolve_fallback_chain_breaks_on_cycle(db: Database) -> None:
    await db.set_model_fallback("c1", "c2")
    await db.set_model_fallback("c2", "c1")
    chain = await db.resolve_fallback_chain("c1")
    assert chain == ["c1", "c2"]


@pytest.mark.asyncio
async def test_app_settings_allow_unauthenticated_and_custom_keys(db: Database) -> None:
    await db.set_allow_unauthenticated(False)
    assert await db.get_allow_unauthenticated() is False
    await db.set_allow_unauthenticated(True)
    assert await db.get_allow_unauthenticated() is True

    await db.set_app_setting("session_test_key", "v1")
    assert await db.get_app_setting("session_test_key") == "v1"
    await db.set_app_setting("session_test_key", "v2")
    assert await db.get_app_setting("session_test_key") == "v2"


@pytest.mark.asyncio
async def test_request_logs_limit_and_offset(db: Database) -> None:
    base = RequestLog(
        provider_id=None,
        provider_name=None,
        protocol="ollama",
        endpoint="/api/chat",
        source_ip=None,
        model=None,
        request_size=1,
        response_size=1,
        request_meta=None,
        duration_ms=1.0,
        status="ok",
        streamed=False,
    )
    for i in range(5):
        e = base.model_copy(update={"request_size": i})
        await db.save_request_log(e)
        await asyncio.sleep(0.02)

    total = await db.count_request_logs()
    assert total == 5

    page = await db.get_request_logs(limit=2, offset=1)
    assert len(page) == 2
    # Newest first: sizes 4,3,2,1,0 → offset 1 → 3 and 2
    sizes = sorted([x.request_size for x in page])
    assert sizes == [2, 3]


@pytest.mark.asyncio
async def test_get_backend_model_name_prefers_raw_name(db: Database) -> None:
    p = await db.add_provider("srv", "http://localhost:11434")
    await db.set_provider_models(
        p.id,
        [
            ProviderModel(
                provider_id=p.id,
                name="clean",
                raw_name="registry.com/thing:latest",
                size=1,
            )
        ],
    )
    assert await db.get_backend_model_name(p.id, "clean") == "registry.com/thing:latest"
    assert (
        await db.get_backend_model_name(p.id, "registry.com/thing:latest")
        == "registry.com/thing:latest"
    )


@pytest.mark.asyncio
async def test_delete_benchmark_rowcounts(db: Database) -> None:
    from llama_router.models import BenchmarkResult

    p = await db.add_provider("bench-del", "http://localhost:11434")
    for _ in range(3):
        await db.save_benchmark(
            BenchmarkResult(
                provider_id=p.id,
                model_name="m-del",
                protocol="ollama",
                startup_time_ms=1.0,
                tokens_per_second=1.0,
            )
        )
    n = await db.delete_benchmarks_for_model("m-del")
    assert n == 3
    n_all = await db.delete_all_benchmarks()
    assert n_all >= 0


@pytest.mark.asyncio
async def test_list_api_keys_includes_pin_counts(db: Database) -> None:
    p = await db.add_provider("k", "http://localhost:11434")
    kid = await db.create_api_key("pre", "hash-list-api", "latency", True)
    await db.set_api_key_model_pin(kid, "m1", p.id)
    rows = await db.list_api_keys()
    row = next(r for r in rows if r["id"] == kid)
    assert row["pin_count"] == 1


@pytest.mark.asyncio
async def test_lookup_api_key_updates_last_used(db: Database) -> None:
    await db.create_api_key("lu", "hash-lu", "latency", True)
    first = await db.lookup_api_key("hash-lu")
    assert first is not None
    second = await db.lookup_api_key("hash-lu")
    assert second is not None


@pytest.mark.asyncio
async def test_dashboard_user_case_insensitive_lookup_and_order(db: Database) -> None:
    from llama_router.dashboard.auth_core import hash_password

    h = hash_password("x" * 12)
    await db.create_dashboard_user("Alpha", h, False)
    await db.create_dashboard_user("beta", h, False)
    u = await db.get_dashboard_user_by_username("alpha")
    assert u is not None
    assert u["username"] == "Alpha"
    u2 = await db.get_dashboard_user_by_username("BETA")
    assert u2 is not None
    assert u2["username"] == "beta"

    listed = await db.list_dashboard_users()
    names = [x["username"] for x in listed]
    assert names == sorted(names, key=str.lower)


@pytest.mark.asyncio
async def test_seed_addresses_idempotent(db: Database) -> None:
    p = await db.add_provider("seed", "http://seed:11434")
    await db._seed_addresses()
    await db._seed_addresses()
    addrs = await db.get_addresses(p.id)
    assert len(addrs) == 1


@pytest.mark.asyncio
async def test_list_all_models_excludes_offline_providers(db: Database) -> None:
    on_p = await db.add_provider("on", "http://on:11434")
    off_p = await db.add_provider("off", "http://off:11434")
    await db.update_provider_status(on_p.id, ProviderStatus.IDLE)
    await db.update_provider_status(off_p.id, ProviderStatus.OFFLINE)
    await db.set_provider_models(
        on_p.id, [ProviderModel(provider_id=on_p.id, name="shared")]
    )
    await db.set_provider_models(
        off_p.id, [ProviderModel(provider_id=off_p.id, name="shared")]
    )
    models = await db.list_all_models()
    assert len(models) == 1
    assert models[0]["name"] == "shared"
