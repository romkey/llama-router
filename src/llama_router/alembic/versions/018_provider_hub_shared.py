"""Which providers a hub may use when this router serves it as a worker.

Revision ID: 018_provider_hub_shared
Revises: 017_api_key_decision_policies
Create Date: 2026-10-28
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "018_provider_hub_shared"
down_revision = "017_api_key_decision_policies"
branch_labels = None
depends_on = None


def upgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("providers")}
    if "hub_shared" not in cols:
        # 1: a hub this router works for may route to the provider.
        op.add_column(
            "providers",
            sa.Column(
                "hub_shared",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("1"),
            ),
        )


def downgrade() -> None:
    with op.batch_alter_table("providers") as batch:
        batch.drop_column("hub_shared")
