"""Shared private model groups.

``user_model_groups.shared`` (0/1) lets the owner offer a group to every
other dashboard user; ``user_model_group_subscriptions`` records who opted
in to which shared group. Only users who opted in can use one.

Revision ID: 040_shared_user_groups
Revises: 039_user_model_groups
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "040_shared_user_groups"
down_revision = "039_user_model_groups"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    cols = {c["name"] for c in inspector.get_columns("user_model_groups")}
    if "shared" not in cols:
        op.add_column(
            "user_model_groups",
            sa.Column("shared", sa.Integer(), nullable=False, server_default="0"),
        )
    if "user_model_group_subscriptions" not in inspector.get_table_names():
        op.create_table(
            "user_model_group_subscriptions",
            sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
            sa.Column(
                "user_id",
                sa.Integer(),
                sa.ForeignKey("dashboard_users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "group_id",
                sa.Integer(),
                sa.ForeignKey("user_model_groups.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "created_at",
                sa.TIMESTAMP(),
                nullable=False,
                server_default=sa.func.current_timestamp(),
            ),
            sa.UniqueConstraint(
                "user_id", "group_id", name="uq_user_model_group_subscriptions"
            ),
        )
        op.create_index(
            "ix_user_model_group_subscriptions_group_id",
            "user_model_group_subscriptions",
            ["group_id"],
        )


def downgrade() -> None:
    op.drop_table("user_model_group_subscriptions")
    op.drop_column("user_model_groups", "shared")
