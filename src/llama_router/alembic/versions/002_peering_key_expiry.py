"""Peering key expiry and use limits.

Revision ID: 002_peering_key_expiry
Revises: 001_initial
Create Date: 2026-03-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "002_peering_key_expiry"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {c["name"] for c in insp.get_columns("wireguard_interface")}
    if "peering_key_expires_at" not in cols:
        op.add_column(
            "wireguard_interface",
            sa.Column("peering_key_expires_at", sa.TIMESTAMP(), nullable=True),
        )
    if "peering_key_use_count" not in cols:
        op.add_column(
            "wireguard_interface",
            sa.Column(
                "peering_key_use_count",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )
    if "peering_key_max_uses" not in cols:
        op.add_column(
            "wireguard_interface",
            sa.Column("peering_key_max_uses", sa.Integer(), nullable=True),
        )


def downgrade() -> None:
    op.drop_column("wireguard_interface", "peering_key_max_uses")
    op.drop_column("wireguard_interface", "peering_key_use_count")
    op.drop_column("wireguard_interface", "peering_key_expires_at")
