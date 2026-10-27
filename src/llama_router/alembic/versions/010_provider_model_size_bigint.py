"""Store model sizes as 64-bit integers.

``provider_models.size`` holds a model's size in bytes, which is usually more
than 2 GiB. INTEGER is 32-bit on PostgreSQL and MySQL, so those databases
rejected nearly every model. SQLite's INTEGER is already 64-bit and is left
alone.

Revision ID: 010_provider_model_size_bigint
Revises: 009_dashboard_users_oidc
Create Date: 2026-10-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "010_provider_model_size_bigint"
down_revision = "009_dashboard_users_oidc"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return
    op.alter_column(
        "provider_models",
        "size",
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        existing_nullable=True,
    )


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return
    op.alter_column(
        "provider_models",
        "size",
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
