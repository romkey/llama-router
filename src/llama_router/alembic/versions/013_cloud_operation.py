"""Cloud-aware operation: a model filter for big catalogs, a per-provider
read timeout, and a per-provider auto-benchmark override.

Revision ID: 013_cloud_operation
Revises: 012_locality
Create Date: 2026-10-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "013_cloud_operation"
down_revision = "012_locality"
branch_labels = None
depends_on = None

_COLUMNS = (
    ("model_filter", sa.Text()),
    ("read_timeout_s", sa.Float()),
    # NULL: the default for the provider's locality (on local, off cloud).
    ("auto_benchmark", sa.Integer()),
)


def upgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("providers")}
    for name, type_ in _COLUMNS:
        if name not in cols:
            op.add_column("providers", sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("providers") as batch:
        for name, _ in reversed(_COLUMNS):
            batch.drop_column(name)
