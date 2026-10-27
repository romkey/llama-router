"""Record time to first byte and router overhead per proxied request.

Revision ID: 006_request_latency
Revises: 005_provider_capabilities
Create Date: 2026-10-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "006_request_latency"
down_revision = "005_provider_capabilities"
branch_labels = None
depends_on = None


def upgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("request_log")}
    if "ttfb_ms" not in cols:
        op.add_column("request_log", sa.Column("ttfb_ms", sa.Float(), nullable=True))
    if "router_ms" not in cols:
        op.add_column("request_log", sa.Column("router_ms", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("request_log") as batch:
        batch.drop_column("router_ms")
        batch.drop_column("ttfb_ms")
