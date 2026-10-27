"""Cloud presets: which preset a provider came from, extra headers, and
models added by hand.

Revision ID: 011_provider_presets
Revises: 010_provider_flavor
Create Date: 2026-10-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "011_provider_presets"
down_revision = "010_provider_flavor"
branch_labels = None
depends_on = None

_COLUMNS = ("preset", "extra_headers", "manual_models")


def upgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("providers")}
    for name in _COLUMNS:
        if name not in cols:
            op.add_column("providers", sa.Column(name, sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("providers") as batch:
        for name in reversed(_COLUMNS):
            batch.drop_column(name)
