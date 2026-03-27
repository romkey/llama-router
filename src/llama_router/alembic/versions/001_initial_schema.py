"""Initial schema (SQLite, PostgreSQL, MariaDB / MySQL).

Revision ID: 001_initial
Revises:
Create Date: 2025-02-24
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_mysql = bind.dialect.name in ("mysql", "mariadb")

    op.create_table(
        "wireguard_interface",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("enabled", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("listen_port", sa.Integer(), nullable=False, server_default="51820"),
        sa.Column("private_key", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "address_cidr", sa.Text(), nullable=False, server_default="10.8.0.1/24"
        ),
        sa.Column("mtu", sa.Integer(), nullable=True),
        sa.Column("endpoint_public", sa.Text(), nullable=True),
        sa.Column(
            "updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.Column("peering_api_key", sa.Text(), nullable=False, server_default=""),
        sa.Column("peering_enabled", sa.Integer(), nullable=False, server_default="0"),
        sa.CheckConstraint("id = 1", name="ck_wireguard_interface_singleton"),
        if_not_exists=True,
    )

    op.create_table(
        "wireguard_peers",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("name", sa.Text(), nullable=False, server_default=""),
        sa.Column("public_key", sa.Text(), nullable=False),
        sa.Column("preshared_key", sa.Text(), nullable=True),
        sa.Column("allowed_ips", sa.Text(), nullable=False),
        sa.Column("endpoint", sa.Text(), nullable=True),
        sa.Column("persistent_keepalive", sa.Integer(), nullable=True),
        sa.Column("enabled", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        if_not_exists=True,
    )

    op.create_table(
        "providers",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("llamacpp_url", sa.Text(), nullable=True),
        sa.Column("provider_type", sa.Text(), nullable=False, server_default="ollama"),
        sa.Column("status", sa.Text(), nullable=False, server_default="unknown"),
        sa.Column("machine_type", sa.Text(), nullable=True),
        sa.Column("gpu_type", sa.Text(), nullable=True),
        sa.Column("gpu_ram", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.Column(
            "updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.Column("wireguard_peer_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["wireguard_peer_id"],
            ["wireguard_peers.id"],
            ondelete="SET NULL",
        ),
        sa.UniqueConstraint("name", name="uq_providers_name"),
        if_not_exists=True,
    )

    op.create_table(
        "provider_models",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column(
            "provider_id",
            sa.Integer(),
            sa.ForeignKey("providers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("raw_name", sa.Text(), nullable=True),
        sa.Column("size", sa.Integer(), nullable=True),
        sa.Column("digest", sa.Text(), nullable=True),
        sa.Column("modified_at", sa.Text(), nullable=True),
        sa.Column("details", sa.Text(), nullable=True),
        sa.UniqueConstraint(
            "provider_id", "name", name="uq_provider_models_provider_name"
        ),
        if_not_exists=True,
    )

    op.create_table(
        "benchmarks",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column(
            "provider_id",
            sa.Integer(),
            sa.ForeignKey("providers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("protocol", sa.Text(), nullable=True),
        sa.Column("startup_time_ms", sa.Float(), nullable=True),
        sa.Column("tokens_per_second", sa.Float(), nullable=True),
        sa.Column(
            "created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        if_not_exists=True,
    )

    op.create_table(
        "request_log",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("provider_id", sa.Integer(), nullable=True),
        sa.Column("provider_name", sa.Text(), nullable=True),
        sa.Column("protocol", sa.Text(), nullable=False),
        sa.Column("endpoint", sa.Text(), nullable=False),
        sa.Column("source_ip", sa.Text(), nullable=True),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("request_size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("response_size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("request_meta", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("status", sa.Text(), nullable=False, server_default="ok"),
        sa.Column("streamed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        if_not_exists=True,
    )
    op.create_index(
        "idx_request_log_created",
        "request_log",
        ["created_at"],
        if_not_exists=True,
    )

    op.create_table(
        "model_fallbacks",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("fallback_model", sa.Text(), nullable=False),
        sa.UniqueConstraint("model_name", name="uq_model_fallbacks_model_name"),
        if_not_exists=True,
    )

    op.create_table(
        "provider_addresses",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column(
            "provider_id",
            sa.Integer(),
            sa.ForeignKey("providers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("llamacpp_url", sa.Text(), nullable=True),
        sa.Column("is_preferred", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_live", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        if_not_exists=True,
    )

    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("key_prefix", sa.Text(), nullable=False),
        sa.Column("key_hash", sa.Text(), nullable=False),
        sa.Column("routing_mode", sa.Text(), nullable=False, server_default="latency"),
        sa.Column("allow_fallback", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.Column("last_used_at", sa.TIMESTAMP(), nullable=True),
        sa.UniqueConstraint("key_hash", name="uq_api_keys_key_hash"),
        if_not_exists=True,
    )

    op.create_table(
        "api_key_model_pins",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column(
            "api_key_id",
            sa.Integer(),
            sa.ForeignKey("api_keys.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column(
            "provider_id",
            sa.Integer(),
            sa.ForeignKey("providers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.UniqueConstraint(
            "api_key_id", "model_name", name="uq_api_key_model_pins_key_model"
        ),
        if_not_exists=True,
    )

    op.create_table(
        "app_settings",
        sa.Column("key", sa.Text(), primary_key=True),
        sa.Column("value", sa.Text(), nullable=False),
        if_not_exists=True,
    )

    op.create_table(
        "dashboard_users",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("username", sa.Text(), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("is_admin", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.UniqueConstraint("username", name="uq_dashboard_users_username"),
        if_not_exists=True,
    )

    dialect = bind.dialect.name
    if dialect == "sqlite":
        op.execute(sa.text("INSERT OR IGNORE INTO wireguard_interface (id) VALUES (1)"))
    elif is_mysql:
        op.execute(sa.text("INSERT IGNORE INTO wireguard_interface (id) VALUES (1)"))
    else:
        op.execute(
            sa.text(
                "INSERT INTO wireguard_interface (id) VALUES (1) "
                "ON CONFLICT (id) DO NOTHING"
            )
        )

    if is_mysql:
        op.execute(
            sa.text(
                "INSERT IGNORE INTO app_settings (key, value) "
                "VALUES ('allow_unauthenticated', 'true')"
            )
        )
    elif dialect == "sqlite":
        op.execute(
            sa.text(
                'INSERT OR IGNORE INTO app_settings ("key", value) '
                "VALUES ('allow_unauthenticated', 'true')"
            )
        )
    else:
        op.execute(
            sa.text(
                'INSERT INTO app_settings ("key", value) '
                "VALUES ('allow_unauthenticated', 'true') "
                'ON CONFLICT ("key") DO NOTHING'
            )
        )


def downgrade() -> None:
    op.drop_table("dashboard_users", if_exists=True)
    op.drop_table("app_settings", if_exists=True)
    op.drop_table("api_key_model_pins", if_exists=True)
    op.drop_table("api_keys", if_exists=True)
    op.drop_table("provider_addresses", if_exists=True)
    op.drop_table("model_fallbacks", if_exists=True)
    op.drop_index("idx_request_log_created", table_name="request_log", if_exists=True)
    op.drop_table("request_log", if_exists=True)
    op.drop_table("benchmarks", if_exists=True)
    op.drop_table("provider_models", if_exists=True)
    op.drop_table("providers", if_exists=True)
    op.drop_table("wireguard_peers", if_exists=True)
    op.drop_table("wireguard_interface", if_exists=True)
