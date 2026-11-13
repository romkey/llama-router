"""Provider capacity limits: ``providers.max_concurrent``.

How many requests the router sends a provider at once; more wait in line
for a free slot. Null (the default, for new and existing providers) means no
limit, as before.

Revision ID: 035_provider_max_concurrent
Revises: 034_api_key_strict_pins
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "035_provider_max_concurrent"
down_revision = "034_api_key_strict_pins"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    cols = {c["name"] for c in inspector.get_columns("providers")}
    if "max_concurrent" not in cols:
        op.add_column(
            "providers", sa.Column("max_concurrent", sa.Integer(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("providers") as batch:
        batch.drop_column("max_concurrent")
