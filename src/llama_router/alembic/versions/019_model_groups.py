"""Model groups: a name that routes to the best of several models, each
anywhere or on one provider.

Revision ID: 019_model_groups
Revises: 018_provider_hub_shared
Create Date: 2026-10-28
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "019_model_groups"
down_revision = "018_provider_hub_shared"
branch_labels = None
depends_on = None


def upgrade() -> None:
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "model_groups" not in tables:
        op.create_table(
            "model_groups",
            sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.TIMESTAMP(),
                nullable=False,
                server_default=sa.func.current_timestamp(),
            ),
            sa.UniqueConstraint("name", name="uq_model_groups_name"),
        )
    if "model_group_members" not in tables:
        op.create_table(
            "model_group_members",
            sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
            sa.Column(
                "group_id",
                sa.Integer(),
                sa.ForeignKey("model_groups.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("model_name", sa.String(255), nullable=False),
            # NULL: the model on any provider that serves it.
            sa.Column(
                "provider_id",
                sa.Integer(),
                sa.ForeignKey("providers.id", ondelete="CASCADE"),
                nullable=True,
            ),
            sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
            sa.Column(
                "created_at",
                sa.TIMESTAMP(),
                nullable=False,
                server_default=sa.func.current_timestamp(),
            ),
        )
        op.create_index(
            "ix_model_group_members_group_id", "model_group_members", ["group_id"]
        )


def downgrade() -> None:
    op.drop_table("model_group_members")
    op.drop_table("model_groups")
