"""Dashboard session revocation and SSO account approval.

``session_version`` is copied into every signed session cookie; bumping it
signs the user out everywhere. ``approved`` is 0 for single sign-on accounts
waiting for an administrator (``LLAMA_ROUTER_OIDC_REQUIRE_APPROVAL``);
existing accounts stay approved.

Revision ID: 026_session_version
Revises: 024_api_key_owner
Create Date: 2026-10-31
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "026_session_version"
# Rebase onto 025_default_deny_anonymous once that migration is on main.
down_revision = "024_api_key_owner"
branch_labels = None
depends_on = None


def upgrade() -> None:
    insp = sa.inspect(op.get_bind())
    cols = {c["name"] for c in insp.get_columns("dashboard_users")}
    with op.batch_alter_table("dashboard_users") as batch:
        if "session_version" not in cols:
            batch.add_column(
                sa.Column(
                    "session_version", sa.Integer(), nullable=False, server_default="0"
                )
            )
        if "approved" not in cols:
            batch.add_column(
                sa.Column("approved", sa.Integer(), nullable=False, server_default="1")
            )


def downgrade() -> None:
    with op.batch_alter_table("dashboard_users") as batch:
        batch.drop_column("approved")
        batch.drop_column("session_version")
