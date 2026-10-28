"""Add hub worker tokens; remove WireGuard tables.

Revision ID: 003_workers
Revises: 002_peering_key_expiry
Create Date: 2026-10-25
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from llama_router.db_types import ShortText

revision = "003_workers"
down_revision = "002_peering_key_expiry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    if "workers" not in tables:
        op.create_table(
            "workers",
            sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
            sa.Column("name", sa.String(64), nullable=False),
            sa.Column("token_hash", sa.String(64), nullable=False),
            sa.Column("token_prefix", sa.Text(), nullable=False),
            sa.Column(
                "provider_id",
                sa.Integer(),
                sa.ForeignKey("providers.id", ondelete="CASCADE"),
                nullable=True,
            ),
            sa.Column("enabled", sa.Integer(), nullable=False, server_default="1"),
            sa.Column(
                "created_at",
                sa.TIMESTAMP(),
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.Column("last_connected_at", sa.TIMESTAMP(), nullable=True),
            sa.Column("last_remote_addr", sa.Text(), nullable=True),
            sa.UniqueConstraint("name", name="uq_workers_name"),
            sa.UniqueConstraint("token_hash", name="uq_workers_token_hash"),
        )

    provider_cols = {c["name"] for c in insp.get_columns("providers")}
    if "wireguard_peer_id" in provider_cols:
        if bind.dialect.name == "sqlite":
            with op.batch_alter_table("providers", recreate="always") as batch:
                batch.drop_column("wireguard_peer_id")
        else:
            for fk in insp.get_foreign_keys("providers"):
                fk_name = fk.get("name")
                if fk.get("constrained_columns") == ["wireguard_peer_id"] and fk_name:
                    op.drop_constraint(fk_name, "providers", type_="foreignkey")
            op.drop_column("providers", "wireguard_peer_id")

    if "wireguard_peers" in tables:
        op.drop_table("wireguard_peers")
    if "wireguard_interface" in tables:
        op.drop_table("wireguard_interface")


def downgrade() -> None:
    op.drop_table("workers")
    op.create_table(
        "wireguard_interface",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("enabled", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("listen_port", sa.Integer(), nullable=False, server_default="51820"),
        sa.Column("private_key", ShortText, nullable=False, server_default=""),
        sa.Column(
            "address_cidr", ShortText, nullable=False, server_default="10.8.0.1/24"
        ),
        sa.Column("mtu", sa.Integer(), nullable=True),
        sa.Column("endpoint_public", sa.Text(), nullable=True),
        sa.Column(
            "updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.Column("peering_api_key", ShortText, nullable=False, server_default=""),
        sa.Column("peering_enabled", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("peering_key_expires_at", sa.TIMESTAMP(), nullable=True),
        sa.Column(
            "peering_key_use_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("peering_key_max_uses", sa.Integer(), nullable=True),
    )
    op.create_table(
        "wireguard_peers",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("name", ShortText, nullable=False, server_default=""),
        sa.Column("public_key", sa.Text(), nullable=False),
        sa.Column("preshared_key", sa.Text(), nullable=True),
        sa.Column("allowed_ips", sa.Text(), nullable=False),
        sa.Column("endpoint", sa.Text(), nullable=True),
        sa.Column("persistent_keepalive", sa.Integer(), nullable=True),
        sa.Column("enabled", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
    )
    with op.batch_alter_table("providers") as batch:
        batch.add_column(sa.Column("wireguard_peer_id", sa.Integer(), nullable=True))
