"""Record whether a benchmark was run by hand or by the idle auto-benchmarker.

Revision ID: 007_benchmark_source
Revises: 006_request_latency
Create Date: 2026-10-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "007_benchmark_source"
down_revision = "006_request_latency"
branch_labels = None
depends_on = None


def upgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("benchmarks")}
    if "source" not in cols:
        op.add_column("benchmarks", sa.Column("source", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("benchmarks") as batch:
        batch.drop_column("source")
