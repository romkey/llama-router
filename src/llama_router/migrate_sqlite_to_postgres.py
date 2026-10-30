"""Copy llama-router data from SQLite to PostgreSQL.

Requires a sync PostgreSQL URL (``postgresql+psycopg://...``) and the optional
``postgres`` install extra (``psycopg``).

Typical use::

    llama-router-migrate-sqlite-pg \\
        --sqlite /path/to/llama_router.db \\
        --postgres postgresql+psycopg://user:pass@localhost:5432/llama_router

The target database must exist. Alembic leaves seed rows in some tables; the
tool **truncates all application tables** (not ``alembic_version``) before
copying from SQLite so the file is the single source of truth.

Steps:

1. Run Alembic migrations on PostgreSQL (idempotent).
2. Truncate application tables on PostgreSQL.
3. Copy rows in foreign-key order, preserving primary keys.
4. Reset PostgreSQL sequences so new inserts get correct IDs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Connection, Engine

from .db_migrate import run_upgrade_sync

logger = logging.getLogger(__name__)

# Foreign-key safe copy order (parents before children).
_TABLES: list[tuple[str, list[str]]] = [
    (
        "providers",
        [
            "id",
            "name",
            "protocol",
            "status",
            "machine_type",
            "gpu_type",
            "gpu_ram",
            "created_at",
            "updated_at",
            "worker_name",
            "api_key_enc",
            "api_key_header",
            "capabilities",
            "probed_at",
            "flavor",
            "preset",
            "extra_headers",
            "manual_models",
            "locality",
            "model_filter",
            "read_timeout_s",
            "auto_benchmark",
        ],
    ),
    (
        "workers",
        [
            "id",
            "name",
            "token_hash",
            "token_prefix",
            "provider_id",
            "enabled",
            "created_at",
            "last_connected_at",
            "last_remote_addr",
        ],
    ),
    (
        "provider_models",
        [
            "id",
            "provider_id",
            "name",
            "raw_name",
            "size",
            "digest",
            "modified_at",
            "details",
        ],
    ),
    (
        "benchmarks",
        [
            "id",
            "provider_id",
            "model_name",
            "protocol",
            "startup_time_ms",
            "tokens_per_second",
            "source",
            "created_at",
        ],
    ),
    (
        "request_log",
        [
            "id",
            "provider_id",
            "provider_name",
            "protocol",
            "endpoint",
            "source_ip",
            "model",
            "request_size",
            "response_size",
            "request_meta",
            "duration_ms",
            "ttfb_ms",
            "router_ms",
            "status",
            "streamed",
            "error_detail",
            "locality",
            "api_key_id",
            "prompt_tokens",
            "completion_tokens",
            "cost_usd",
            "created_at",
            "routed_model",
        ],
    ),
    (
        "model_fallbacks",
        ["id", "model_name", "fallback_model"],
    ),
    (
        "provider_addresses",
        [
            "id",
            "provider_id",
            "url",
            "is_preferred",
            "is_live",
            "created_at",
        ],
    ),
    (
        "api_keys",
        [
            "id",
            "key_prefix",
            "key_hash",
            "routing_mode",
            "allow_fallback",
            "session_affinity",
            "locality",
            "daily_spend_cap_usd",
            "created_at",
            "last_used_at",
        ],
    ),
    (
        "model_prices",
        ["id", "provider_id", "model_pattern", "input_per_mtok", "output_per_mtok"],
    ),
    (
        "model_aliases",
        ["id", "alias", "canonical", "created_at"],
    ),
    (
        "model_groups",
        ["id", "name", "description", "created_at"],
    ),
    (
        "model_group_members",
        ["id", "group_id", "model_name", "provider_id", "position", "created_at"],
    ),
    (
        "api_key_model_pins",
        ["id", "api_key_id", "model_name", "provider_id", "created_at"],
    ),
    (
        "app_settings",
        ["key", "value"],
    ),
    (
        "dashboard_users",
        [
            "id",
            "username",
            "password_hash",
            "is_admin",
            "created_at",
            "auth_source",
            "oidc_issuer",
            "oidc_subject",
            "email",
            "display_name",
            "last_login_at",
        ],
    ),
    (
        "decision_calibrations",
        [
            "id",
            "model_name",
            "question_key",
            "question_type",
            "question",
            "temperature",
            "offsets",
            "examples",
            "created_at",
            "updated_at",
        ],
    ),
    (
        "api_key_decision_policies",
        ["api_key_id", "policy", "created_at", "updated_at"],
    ),
    (
        "chat_conversations",
        [
            "id",
            "user_id",
            "title",
            "provider_id",
            "model_name",
            "created_at",
            "updated_at",
        ],
    ),
    (
        "chat_messages",
        [
            "id",
            "conversation_id",
            "role",
            "content",
            "image",
            "model_name",
            "provider_name",
            "duration_ms",
            "created_at",
        ],
    ),
    (
        "management_keys",
        [
            "id",
            "name",
            "key_prefix",
            "key_hash",
            "privileges",
            "created_by",
            "created_at",
            "expires_at",
            "last_used_at",
        ],
    ),
]

# Tables with a PostgreSQL sequence on ``id`` (reset after copy).
_SEQUENCE_TABLES = [
    "providers",
    "workers",
    "provider_models",
    "benchmarks",
    "request_log",
    "model_fallbacks",
    "provider_addresses",
    "api_keys",
    "api_key_model_pins",
    "dashboard_users",
    "model_aliases",
    "model_groups",
    "model_group_members",
    "model_prices",
    "decision_calibrations",
    "chat_conversations",
    "chat_messages",
    "management_keys",
]


def _pg_quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sqlite_url(path: Path) -> str:
    p = path.expanduser().resolve()
    return f"sqlite:///{p}"


def _normalize_postgres_url(url: str) -> str:
    u = url.strip()
    if u.startswith("postgresql+asyncpg://"):
        return "postgresql+psycopg://" + u[len("postgresql+asyncpg://") :]
    if u.startswith("postgres://"):
        return "postgresql+psycopg://" + u[len("postgres://") :]
    if not u.startswith("postgresql+"):
        raise SystemExit(
            "PostgreSQL URL must use a sync driver, e.g. postgresql+psycopg://user:pass@host/dbname"
        )
    if "asyncpg" in u.split("://", 1)[0]:
        raise SystemExit("Use postgresql+psycopg:// (sync) for this tool, not asyncpg.")
    return u


def _truncate_app_tables(pg: Connection) -> None:
    names = ", ".join(_pg_quote_ident(t) for t, _ in _TABLES)
    pg.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))


def _fetch_sqlite_rows(
    sqlite: Connection, table: str, columns: Sequence[str]
) -> list[dict[str, Any]]:
    cols = ", ".join(_pg_quote_ident(c) for c in columns)
    result = sqlite.execute(text(f"SELECT {cols} FROM {_pg_quote_ident(table)}"))  # noqa: S608 - identifiers come from the fixed table list and are quoted; values are bound
    return [dict(row._mapping) for row in result]


def _insert_rows_pg(
    pg: Connection, table: str, columns: Sequence[str], rows: list[dict[str, Any]]
) -> None:
    if not rows:
        return
    tq = _pg_quote_ident(table)
    col_sql = ", ".join(_pg_quote_ident(c) for c in columns)
    binds = [f":c{i}" for i in range(len(columns))]
    params_sql = ", ".join(binds)
    stmt = text(f"INSERT INTO {tq} ({col_sql}) VALUES ({params_sql})")  # noqa: S608 - identifiers come from the fixed table list and are quoted; values are bound
    for row in rows:
        payload = {f"c{i}": row.get(columns[i]) for i in range(len(columns))}
        pg.execute(stmt, payload)


def _reset_sequences(pg: Connection) -> None:
    for table in _SEQUENCE_TABLES:
        seq = pg.execute(
            text(f"SELECT pg_get_serial_sequence('public.{table}', 'id')")
        ).scalar()
        if seq is None:
            continue
        mx = pg.execute(
            text(f"SELECT MAX({_pg_quote_ident('id')}) FROM {_pg_quote_ident(table)}")  # noqa: S608 - identifiers come from the fixed table list and are quoted; values are bound
        ).scalar()
        if mx is None:
            pg.execute(text("SELECT setval(:seq, 1, false)"), {"seq": seq})
        else:
            pg.execute(
                text("SELECT setval(:seq, CAST(:mx AS bigint), true)"),
                {"seq": seq, "mx": mx},
            )


def migrate_sqlite_to_postgres(
    sqlite_path: Path,
    postgres_url: str,
) -> None:
    try:
        import psycopg  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "psycopg is required. Install with: pip install 'llama-router[postgres]'"
        ) from e

    pg_url = _normalize_postgres_url(postgres_url)
    sqlite_file = sqlite_path.expanduser().resolve()
    if not sqlite_file.is_file():
        raise SystemExit(f"SQLite file not found: {sqlite_file}")

    logger.info("Running Alembic migrations on PostgreSQL target")
    run_upgrade_sync(pg_url)

    sl_eng: Engine = create_engine(_sqlite_url(sqlite_file))
    pg_eng: Engine = create_engine(pg_url)

    with sl_eng.connect() as sl_conn, pg_eng.begin() as pg_conn:
        logger.info("Truncating application tables on PostgreSQL before copy")
        _truncate_app_tables(pg_conn)

        source_tables = set(inspect(sl_conn).get_table_names())
        for table, columns in _TABLES:
            if table not in source_tables:
                logger.info("Skipping %s (not in source database)", table)
                continue
            # An older source lacks columns added since; they keep their
            # defaults on the target.
            present = {c["name"] for c in inspect(sl_conn).get_columns(table)}
            cols = [c for c in columns if c in present]
            rows = _fetch_sqlite_rows(sl_conn, table, cols)
            logger.info("Copying %s (%d rows)", table, len(rows))
            _insert_rows_pg(pg_conn, table, cols, rows)

        logger.info("Resetting PostgreSQL sequences")
        _reset_sequences(pg_conn)

    logger.info("Migration finished successfully")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser(
        description="Copy llama-router data from SQLite to PostgreSQL.",
    )
    p.add_argument(
        "--sqlite",
        required=True,
        type=Path,
        help="Path to the SQLite database file (e.g. llama_router.db)",
    )
    p.add_argument(
        "--postgres",
        required=True,
        help="Sync PostgreSQL URL, e.g. postgresql+psycopg://user:pass@host:5432/dbname",
    )
    args = p.parse_args(argv)
    try:
        migrate_sqlite_to_postgres(args.sqlite, args.postgres)
    except SystemExit as e:
        if e.args:
            logger.error("%s", e.args[0])
        raise
    except Exception:
        logger.exception("Migration failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
