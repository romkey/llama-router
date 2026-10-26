"""Store the endpoints a provider's upstream supports, from the last probe.

Revision ID: 005_provider_capabilities
Revises: 004_provider_api_key
Create Date: 2026-10-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "005_provider_capabilities"
down_revision = "004_provider_api_key"
branch_labels = None
depends_on = None


def upgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("providers")}
    if "capabilities" not in cols:
        op.add_column("providers", sa.Column("capabilities", sa.Text(), nullable=True))
    if "probed_at" not in cols:
        op.add_column(
            "providers", sa.Column("probed_at", sa.TIMESTAMP(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("providers") as batch:
        batch.drop_column("probed_at")
        batch.drop_column("capabilities")
