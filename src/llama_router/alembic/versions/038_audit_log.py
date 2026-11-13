"""Audit trail: ``audit_log``.

One row per state-changing dashboard action: when, who (a dashboard user, a
management key, or the router itself), what was done to what, a summary of
the fields changed (never secrets) and the client's address.

Revision ID: 038_audit_log
Revises: 037_api_key_lifecycle
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "038_audit_log"
down_revision = "037_api_key_lifecycle"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "audit_log" in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("actor_kind", sa.String(20), nullable=False),
        sa.Column("actor_id", sa.Integer(), nullable=True),
        sa.Column("actor_name", sa.String(200), nullable=True),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("target_kind", sa.String(50), nullable=True),
        sa.Column("target_id", sa.String(200), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("client_ip", sa.String(64), nullable=True),
    )
    op.create_index("idx_audit_log_at", "audit_log", ["at"])
    op.create_index("idx_audit_log_target", "audit_log", ["target_kind", "target_id"])


def downgrade() -> None:
    op.drop_index("idx_audit_log_target", table_name="audit_log")
    op.drop_index("idx_audit_log_at", table_name="audit_log")
    op.drop_table("audit_log")
