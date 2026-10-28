"""The model a request actually ran, when it differs from the one asked
for: a model group's member, an alias's served name or a fallback.

Revision ID: 020_request_routed_model
Revises: 019_model_groups
Create Date: 2026-10-28
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "020_request_routed_model"
down_revision = "019_model_groups"
branch_labels = None
depends_on = None


def upgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("request_log")}
    if "routed_model" not in cols:
        op.add_column(
            "request_log", sa.Column("routed_model", sa.String(255), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("request_log") as batch:
        batch.drop_column("routed_model")
