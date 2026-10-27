"""Decision policies per API key (see decision_policy.py).

One row per key: the decision model, a score question whose levels pick the
model a request is routed to, and noul questions that reject a request
above a threshold. Stored as JSON and validated when written and read.

Revision ID: 015_api_key_decision_policies
Revises: 014_decision_calibrations
Create Date: 2026-10-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "015_api_key_decision_policies"
down_revision = "014_decision_calibrations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "api_key_decision_policies",
        sa.Column(
            "api_key_id",
            sa.Integer(),
            sa.ForeignKey("api_keys.id", ondelete="CASCADE"),
            primary_key=True,
            autoincrement=False,
        ),
        # JSON object; see DecisionPolicy.
        sa.Column("policy", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.Column(
            "updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_table("api_key_decision_policies", if_exists=True)
