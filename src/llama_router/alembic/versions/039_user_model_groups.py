"""Private model groups: groups a dashboard user defines for their own keys.

Kept apart from ``model_groups`` so a group's name is unique per owner, not
across the router, and deleting a user takes their groups with them.

Revision ID: 039_user_model_groups
Revises: 038_audit_log
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "039_user_model_groups"
down_revision = "038_audit_log"
branch_labels = None
depends_on = None


def upgrade() -> None:
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "user_model_groups" not in tables:
        op.create_table(
            "user_model_groups",
            sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
            sa.Column(
                "owner_user_id",
                sa.Integer(),
                sa.ForeignKey("dashboard_users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.TIMESTAMP(),
                nullable=False,
                server_default=sa.func.current_timestamp(),
            ),
            sa.UniqueConstraint(
                "owner_user_id", "name", name="uq_user_model_groups_owner_name"
            ),
        )
    if "user_model_group_members" not in tables:
        op.create_table(
            "user_model_group_members",
            sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
            sa.Column(
                "group_id",
                sa.Integer(),
                sa.ForeignKey("user_model_groups.id", ondelete="CASCADE"),
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
        )
        op.create_index(
            "ix_user_model_group_members_group_id",
            "user_model_group_members",
            ["group_id"],
        )


def downgrade() -> None:
    op.drop_table("user_model_group_members")
    op.drop_table("user_model_groups")
