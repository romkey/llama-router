"""OpenID Connect sign-in for dashboard users.

Adds where an account signs in from (``local`` or ``oidc``), the provider
identity it is bound to, and profile fields refreshed at each sign-in.
``password_hash`` becomes nullable: OIDC accounts have no password.

Revision ID: 009_dashboard_users_oidc
Revises: 007_benchmark_source
Create Date: 2026-10-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "009_dashboard_users_oidc"
down_revision = "007_benchmark_source"
branch_labels = None
depends_on = None

_NEW_COLUMNS = (
    ("auth_source", sa.String(16), dict(nullable=False, server_default="local")),
    ("oidc_issuer", sa.Text(), dict(nullable=True)),
    ("oidc_subject", sa.Text(), dict(nullable=True)),
    ("email", sa.Text(), dict(nullable=True)),
    ("display_name", sa.Text(), dict(nullable=True)),
    ("last_login_at", sa.TIMESTAMP(), dict(nullable=True)),
)
_OIDC_INDEX = "uq_dashboard_users_oidc"


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {c["name"]: c for c in insp.get_columns("dashboard_users")}
    indexes = {i["name"] for i in insp.get_indexes("dashboard_users")}
    with op.batch_alter_table("dashboard_users") as batch:
        if not cols["password_hash"]["nullable"]:
            batch.alter_column("password_hash", existing_type=sa.Text(), nullable=True)
        for name, type_, kw in _NEW_COLUMNS:
            if name not in cols:
                batch.add_column(sa.Column(name, type_, **kw))
    if _OIDC_INDEX not in indexes:
        # MySQL cannot index unbounded TEXT; the prefix lengths keep the pair
        # well under the key size limit.
        op.create_index(
            _OIDC_INDEX,
            "dashboard_users",
            ["oidc_issuer", "oidc_subject"],
            unique=True,
            mysql_length={"oidc_issuer": 255, "oidc_subject": 255},
        )


def downgrade() -> None:
    op.drop_index(_OIDC_INDEX, table_name="dashboard_users")
    op.execute(sa.text("DELETE FROM dashboard_users WHERE password_hash IS NULL"))
    with op.batch_alter_table("dashboard_users") as batch:
        for name, _, _ in reversed(_NEW_COLUMNS):
            batch.drop_column(name)
        batch.alter_column("password_hash", existing_type=sa.Text(), nullable=False)
