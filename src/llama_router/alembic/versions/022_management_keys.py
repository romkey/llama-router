"""Management keys: tokens for the dashboard's management API.

Revision ID: 022_management_keys
Revises: 021_chat
Create Date: 2026-10-30
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "022_management_keys"
down_revision = "021_chat"
branch_labels = None
depends_on = None


def upgrade() -> None:
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "management_keys" in tables:
        return
    op.create_table(
        "management_keys",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("key_prefix", sa.String(32), nullable=False),
        # SHA-256 of the key, hex.
        sa.Column("key_hash", sa.String(64), nullable=False, unique=True),
        # Space-separated privilege names, e.g. "keys:read keys:write".
        sa.Column("privileges", sa.String(255), nullable=False),
        # Username of the administrator who issued it.
        sa.Column("created_by", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        # Unix seconds; NULL never expires. An integer compares the same way
        # on every database.
        sa.Column("expires_at", sa.BigInteger(), nullable=True),
        sa.Column("last_used_at", sa.TIMESTAMP(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("management_keys")
