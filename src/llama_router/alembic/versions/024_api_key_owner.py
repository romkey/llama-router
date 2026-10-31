"""The dashboard user who issued an API key to themselves.

Revision ID: 024_api_key_owner
Revises: 023_worker_child_providers
Create Date: 2026-10-31
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "024_api_key_owner"
down_revision = "023_worker_child_providers"
branch_labels = None
depends_on = None


def upgrade() -> None:
    insp = sa.inspect(op.get_bind())
    cols = {c["name"] for c in insp.get_columns("api_keys")}
    # NULL for keys an admin issued. No foreign key: SQLite can't add one
    # without rebuilding the table; deleting a user deletes their keys.
    if "owner_user_id" not in cols:
        op.add_column(
            "api_keys", sa.Column("owner_user_id", sa.Integer(), nullable=True)
        )
    indexes = {i["name"] for i in insp.get_indexes("api_keys")}
    if "ix_api_keys_owner_user_id" not in indexes:
        op.create_index("ix_api_keys_owner_user_id", "api_keys", ["owner_user_id"])


def downgrade() -> None:
    op.drop_index("ix_api_keys_owner_user_id", table_name="api_keys")
    with op.batch_alter_table("api_keys") as batch:
        batch.drop_column("owner_user_id")
