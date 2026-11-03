"""Optional expiry on worker tokens.

``workers.token_expires_at`` is a Unix timestamp, like
``management_keys.expires_at``; NULL means the token never expires.

Revision ID: 031_worker_token_expiry
Revises: 030_request_log_model_index
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "031_worker_token_expiry"
down_revision = "030_request_log_model_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("workers")}
    if "token_expires_at" not in cols:
        op.add_column(
            "workers", sa.Column("token_expires_at", sa.BigInteger(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("workers") as batch:
        batch.drop_column("token_expires_at")
