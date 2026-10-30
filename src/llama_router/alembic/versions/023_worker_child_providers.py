"""Providers a worker shares individually, one row per provider on the worker.

Revision ID: 023_worker_child_providers
Revises: 022_management_keys
Create Date: 2026-10-30
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "023_worker_child_providers"
down_revision = "022_management_keys"
branch_labels = None
depends_on = None


def upgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("providers")}
    # The worker's own provider row, and the provider's id on the worker.
    if "worker_parent_id" not in cols:
        op.add_column(
            "providers", sa.Column("worker_parent_id", sa.Integer(), nullable=True)
        )
    if "worker_remote_id" not in cols:
        op.add_column(
            "providers", sa.Column("worker_remote_id", sa.Integer(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("providers") as batch:
        batch.drop_column("worker_remote_id")
        batch.drop_column("worker_parent_id")
