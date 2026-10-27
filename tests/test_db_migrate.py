"""Alembic sync migration entrypoint (used before the async engine starts)."""

from __future__ import annotations

import sqlite3

import pytest

from llama_router import config
from llama_router.db_migrate import run_upgrade_sync

HEAD = "017_api_key_decision_policies"


def test_run_upgrade_sync_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    dbfile = tmp_path / "migrate_twice.db"
    monkeypatch.setattr(config.settings, "database_url", "")
    monkeypatch.setattr(config.settings, "database_path", str(dbfile))
    sync_url = config.settings.sync_database_url_for_alembic()
    run_upgrade_sync(sync_url)
    run_upgrade_sync(sync_url)
    con = sqlite3.connect(dbfile)
    try:
        n = con.execute("SELECT COUNT(*) FROM alembic_version").fetchone()[0]
        assert n == 1
        assert (
            con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='providers'"
            ).fetchone()
            is not None
        )
    finally:
        con.close()


def test_run_upgrade_sync_after_dropped_alembic_version_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Schema present but no revision (e.g. lost alembic_version) must not loop on CREATE."""
    dbfile = tmp_path / "legacy_no_revision.db"
    monkeypatch.setattr(config.settings, "database_url", "")
    monkeypatch.setattr(config.settings, "database_path", str(dbfile))
    sync_url = config.settings.sync_database_url_for_alembic()
    run_upgrade_sync(sync_url)
    con = sqlite3.connect(dbfile)
    try:
        con.execute("DROP TABLE alembic_version")
        con.commit()
    finally:
        con.close()

    run_upgrade_sync(sync_url)

    con = sqlite3.connect(dbfile)
    try:
        ver = con.execute("SELECT version_num FROM alembic_version").fetchone()
        assert ver is not None
        assert ver[0] == HEAD
    finally:
        con.close()


def test_clear_false_stamp_when_table_missing_then_recreate(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Stale 001_initial stamp with a missing new-era table is repaired on startup."""
    dbfile = tmp_path / "false_stamp.db"
    monkeypatch.setattr(config.settings, "database_url", "")
    monkeypatch.setattr(config.settings, "database_path", str(dbfile))
    sync_url = config.settings.sync_database_url_for_alembic()
    run_upgrade_sync(sync_url)
    con = sqlite3.connect(dbfile)
    try:
        con.execute("DROP TABLE dashboard_users")
        con.commit()
    finally:
        con.close()

    run_upgrade_sync(sync_url)

    con = sqlite3.connect(dbfile)
    try:
        assert (
            con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='dashboard_users'"
            ).fetchone()
            is not None
        )
        ver = con.execute("SELECT version_num FROM alembic_version").fetchone()
        assert ver is not None
        assert ver[0] == HEAD
    finally:
        con.close()


def _alembic_cfg(sync_url: str):
    from pathlib import Path

    from alembic.config import Config

    import llama_router

    pkg = Path(llama_router.__file__).resolve().parent
    cfg = Config(str(pkg / "alembic.ini"))
    cfg.set_main_option("script_location", str(pkg / "alembic"))
    cfg.set_main_option("sqlalchemy.url", sync_url)
    return cfg


def test_upgrade_from_wireguard_schema_keeps_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A 0.15-era database with WireGuard peers upgrades cleanly to workers."""
    from alembic import command

    dbfile = tmp_path / "legacy_wg.db"
    monkeypatch.setattr(config.settings, "database_url", "")
    monkeypatch.setattr(config.settings, "database_path", str(dbfile))
    sync_url = config.settings.sync_database_url_for_alembic()
    command.upgrade(_alembic_cfg(sync_url), "002_peering_key_expiry")

    con = sqlite3.connect(dbfile)
    try:
        con.execute("PRAGMA foreign_keys = ON")
        con.execute(
            "INSERT INTO wireguard_peers (name, public_key, allowed_ips) "
            "VALUES ('peer', 'pk', '10.8.0.2/32')"
        )
        con.execute(
            "INSERT INTO providers (name, url, wireguard_peer_id) "
            "VALUES ('gpu-box', 'http://10.8.0.2:11434', 1)"
        )
        con.execute(
            "INSERT INTO provider_models (provider_id, name) VALUES (1, 'llama3:8b')"
        )
        con.commit()
    finally:
        con.close()

    run_upgrade_sync(sync_url)

    con = sqlite3.connect(dbfile)
    try:
        tables = {
            r[0]
            for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "workers" in tables
        assert "wireguard_peers" not in tables
        assert "wireguard_interface" not in tables
        cols = {r[1] for r in con.execute("PRAGMA table_info(providers)")}
        assert "wireguard_peer_id" not in cols
        assert con.execute("SELECT name FROM providers").fetchall() == [("gpu-box",)]
        # The provider URL moved into its first address.
        assert con.execute(
            "SELECT url, is_preferred FROM provider_addresses"
        ).fetchall() == [("http://10.8.0.2:11434", 1)]
        assert con.execute("SELECT COUNT(*) FROM provider_models").fetchone()[0] == 1
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []
        ver = con.execute("SELECT version_num FROM alembic_version").fetchone()[0]
        assert ver == HEAD
    finally:
        con.close()

    # A second start must not re-run 001 and recreate the WireGuard tables.
    run_upgrade_sync(sync_url)
    con = sqlite3.connect(dbfile)
    try:
        tables = {
            r[0]
            for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "wireguard_peers" not in tables
    finally:
        con.close()


def test_upgrade_004_moves_provider_urls_and_worker_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Provider URLs become addresses; worker:// URLs become worker_name."""
    from alembic import command

    dbfile = tmp_path / "pre_004.db"
    monkeypatch.setattr(config.settings, "database_url", "")
    monkeypatch.setattr(config.settings, "database_path", str(dbfile))
    sync_url = config.settings.sync_database_url_for_alembic()
    command.upgrade(_alembic_cfg(sync_url), "003_workers")

    con = sqlite3.connect(dbfile)
    try:
        con.executescript("""
            INSERT INTO providers (id, name, url, llamacpp_url, provider_type)
              VALUES (1, 'has-addr', 'http://a:11434', NULL, 'ollama');
            INSERT INTO provider_addresses (provider_id, url, is_preferred)
              VALUES (1, 'http://a-real:11434', 1);
            INSERT INTO providers (id, name, url, llamacpp_url, provider_type)
              VALUES (2, 'no-addr', 'http://b:11434', 'http://b:8080', 'both');
            INSERT INTO providers (id, name, url, provider_type)
              VALUES (3, 'edge', 'worker://edge', 'ollama');
            INSERT INTO provider_addresses (provider_id, url, is_preferred)
              VALUES (3, 'worker://edge', 1);
            INSERT INTO provider_models (provider_id, name) VALUES (2, 'm');
            """)
        con.commit()
    finally:
        con.close()

    # Stop at 004: 009 later splits the 'both' provider (see its own test).
    command.upgrade(_alembic_cfg(sync_url), "004_provider_api_key")

    con = sqlite3.connect(dbfile)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(providers)")}
        assert {"url", "llamacpp_url"}.isdisjoint(cols)
        assert {"worker_name", "api_key_enc", "api_key_header"} <= cols
        rows = con.execute(
            "SELECT id, name, worker_name FROM providers ORDER BY id"
        ).fetchall()
        assert rows == [
            (1, "has-addr", None),
            (2, "no-addr", None),
            (3, "edge", "edge"),
        ]
        addrs = con.execute(
            "SELECT provider_id, url, llamacpp_url FROM provider_addresses "
            "ORDER BY provider_id, id"
        ).fetchall()
        # Existing addresses are kept as-is; a provider without one gets its URL.
        assert addrs == [
            (1, "http://a-real:11434", None),
            (2, "http://b:11434", "http://b:8080"),
            (3, "worker://edge", None),
        ]
        assert con.execute("SELECT COUNT(*) FROM provider_models").fetchone()[0] == 1
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        con.close()


def test_upgrade_009_splits_both_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A 'both' provider becomes an Ollama provider plus an OpenAI one."""
    import json

    from alembic import command

    dbfile = tmp_path / "pre_009.db"
    monkeypatch.setattr(config.settings, "database_url", "")
    monkeypatch.setattr(config.settings, "database_path", str(dbfile))
    sync_url = config.settings.sync_database_url_for_alembic()
    command.upgrade(_alembic_cfg(sync_url), "008_api_key_session_affinity")

    def d(**kw) -> str:
        return json.dumps(kw)

    con = sqlite3.connect(dbfile)
    try:
        con.executemany(
            "INSERT INTO providers (id, name, provider_type, gpu_type, worker_name, "
            "api_key_enc, capabilities) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "gpu", "both", "4090", None, "enc", '{"x": 1}'),
                (2, "lcpp", "llamacpp", None, None, None, None),
                (3, "edge", "both", None, "edge", None, None),
                (4, "gpu (llama.cpp)", "ollama", None, None, None, None),
            ],
        )
        con.executemany(
            "INSERT INTO provider_addresses (provider_id, url, llamacpp_url, "
            "is_preferred) VALUES (?, ?, ?, ?)",
            [
                (1, "http://g:11434", "http://g:8080/", 1),
                (1, "http://g2:11434", None, 0),
                (2, "http://l:8080", None, 1),
                (3, "worker://edge", None, 1),
            ],
        )
        con.executemany(
            "INSERT INTO provider_models (provider_id, name, details) VALUES (?, ?, ?)",
            [
                (1, "a", d(_in_ollama=True, _in_llamacpp=False, family="llama")),
                (1, "b", d(_in_ollama=False, _in_llamacpp=True)),
                (1, "c", d(_in_ollama=True, _in_llamacpp=True)),
                (1, "d", None),
                (2, "e", d(_in_ollama=False, _in_llamacpp=True)),
            ],
        )
        con.executemany(
            "INSERT INTO benchmarks (provider_id, model_name, protocol) VALUES (?, ?, ?)",
            [(1, "a", "ollama"), (1, "b", "llamacpp"), (2, "e", "llamacpp")],
        )
        con.execute(
            "INSERT INTO api_keys (id, key_prefix, key_hash) VALUES (1, 'p', 'h')"
        )
        con.executemany(
            "INSERT INTO api_key_model_pins (api_key_id, model_name, provider_id) "
            "VALUES (1, ?, 1)",
            [("a",), ("b",)],
        )
        con.commit()
    finally:
        con.close()

    run_upgrade_sync(sync_url)

    con = sqlite3.connect(dbfile)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(providers)")}
        assert "provider_type" not in cols and "protocol" in cols
        addr_cols = {r[1] for r in con.execute("PRAGMA table_info(provider_addresses)")}
        assert "llamacpp_url" not in addr_cols

        providers = {
            r[1]: r
            for r in con.execute(
                "SELECT id, name, protocol, gpu_type, api_key_enc, capabilities "
                "FROM providers"
            )
        }
        assert {n: r[2] for n, r in providers.items()} == {
            "gpu": "ollama",
            "lcpp": "openai",
            "edge": "ollama",
            "gpu (llama.cpp)": "ollama",
            "gpu (llama.cpp) 2": "openai",
        }
        new = providers["gpu (llama.cpp) 2"]
        new_id = new[0]
        # Hardware and key carry over; the mixed probe is dropped on both halves.
        assert (new[3], new[4], new[5]) == ("4090", "enc", None)
        assert providers["gpu"][5] is None

        addrs = con.execute(
            "SELECT url, is_preferred FROM provider_addresses WHERE provider_id = ? "
            "ORDER BY id",
            (new_id,),
        ).fetchall()
        assert addrs == [("http://g:8080", 1), ("http://g2:11434", 0)]

        def models(pid: int) -> dict[str, dict]:
            return {
                name: json.loads(det) if det else {}
                for name, det in con.execute(
                    "SELECT name, details FROM provider_models WHERE provider_id = ?",
                    (pid,),
                )
            }

        assert models(1) == {"a": {"family": "llama"}, "c": {}, "d": {}}
        assert models(new_id) == {"b": {}, "c": {}}
        assert models(2) == {"e": {}}

        benches = con.execute(
            "SELECT provider_id, model_name, protocol FROM benchmarks ORDER BY id"
        ).fetchall()
        assert benches == [
            (1, "a", "ollama"),
            (new_id, "b", "openai"),
            (2, "e", "openai"),
        ]
        pins = dict(
            con.execute("SELECT model_name, provider_id FROM api_key_model_pins")
        )
        assert pins == {"a": 1, "b": new_id}
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        con.close()


def test_upgrade_012_classes_existing_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from alembic import command

    dbfile = tmp_path / "pre_012.db"
    monkeypatch.setattr(config.settings, "database_url", "")
    monkeypatch.setattr(config.settings, "database_path", str(dbfile))
    sync_url = config.settings.sync_database_url_for_alembic()
    command.upgrade(_alembic_cfg(sync_url), "011_provider_presets")
    con = sqlite3.connect(dbfile)
    try:
        con.executemany(
            "INSERT INTO providers (id, name, protocol, preset, worker_name) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (1, "lan", "ollama", None, None),
                (2, "openai", "openai", "openai", None),
                (3, "public", "openai", None, None),
                (4, "edge", "ollama", None, "edge"),
                (5, "bare", "ollama", None, None),
            ],
        )
        con.executemany(
            "INSERT INTO provider_addresses (provider_id, url, is_preferred) "
            "VALUES (?, ?, 1)",
            [
                (1, "http://192.168.1.9:11434"),
                (2, "http://10.0.0.1"),  # the preset decides, not the URL
                (3, "https://llm.example.com/v1"),
                (4, "worker://edge"),
            ],
        )
        con.commit()
    finally:
        con.close()

    run_upgrade_sync(sync_url)

    con = sqlite3.connect(dbfile)
    try:
        got = dict(con.execute("SELECT name, locality FROM providers"))
        assert got == {
            "lan": "local",
            "openai": "cloud",
            "public": "cloud",
            "edge": "local",
            "bare": "local",
        }
        for table in ("api_keys", "request_log"):
            cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
            assert "locality" in cols
    finally:
        con.close()
