"""Strict pins: a key whose pinned provider can't serve the model is refused.

``api_keys.strict_pins`` off (the default, for new and existing keys) keeps
today's behaviour: when the pinned provider is offline, no longer lists the
model or is outside the key's locality, another provider serves it. On, the
request is refused with 503 instead. An integer 0/1, like the table's other
flags (allow_fallback, restrict_models), so raw SQL binds the same value on
every database.

Revision ID: 034_api_key_strict_pins
Revises: 033_api_key_model_access
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "034_api_key_strict_pins"
down_revision = "033_api_key_model_access"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    cols = {c["name"] for c in inspector.get_columns("api_keys")}
    if "strict_pins" not in cols:
        op.add_column(
            "api_keys",
            sa.Column(
                "strict_pins",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )


def downgrade() -> None:
    with op.batch_alter_table("api_keys") as batch:
        batch.drop_column("strict_pins")
