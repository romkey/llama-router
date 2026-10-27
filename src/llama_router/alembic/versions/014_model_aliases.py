"""Model aliases: other names for a model, so a request for any of them can
reach every provider that serves it under any of them.

Revision ID: 014_model_aliases
Revises: 013_cloud_operation
Create Date: 2026-10-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "014_model_aliases"
down_revision = "013_cloud_operation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "model_aliases" in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        "model_aliases",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("alias", sa.String(255), nullable=False),
        sa.Column("canonical", sa.String(255), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.UniqueConstraint("alias", name="uq_model_aliases_alias"),
    )


def downgrade() -> None:
    op.drop_table("model_aliases")
