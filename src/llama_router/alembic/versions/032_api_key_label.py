"""Optional label on API keys, so a key can be told apart by more than its prefix.

Revision ID: 032_api_key_label
Revises: 031_worker_token_expiry
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "032_api_key_label"
down_revision = "031_worker_token_expiry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("api_keys")}
    if "label" not in cols:
        op.add_column("api_keys", sa.Column("label", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("api_keys") as batch:
        batch.drop_column("label")
