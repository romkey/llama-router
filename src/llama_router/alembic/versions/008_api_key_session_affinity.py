"""Per-API-key session affinity override.

NULL follows LLAMA_ROUTER_SESSION_AFFINITY_ENABLED; 1 forces it on, 0 off.

Revision ID: 008_api_key_session_affinity
Revises: 007_benchmark_source
Create Date: 2026-10-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "008_api_key_session_affinity"
down_revision = "007_benchmark_source"
branch_labels = None
depends_on = None


def upgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("api_keys")}
    if "session_affinity" not in cols:
        op.add_column(
            "api_keys", sa.Column("session_affinity", sa.Integer(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("api_keys") as batch:
        batch.drop_column("session_affinity")
