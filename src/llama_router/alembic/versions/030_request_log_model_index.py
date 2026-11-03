"""Index request_log on (model, created_at).

The dashboard's per-model counts and log filters scan the whole log
without it.

Revision ID: 030_request_log_model_index
Revises: 029_drop_bogus_benchmarks
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "030_request_log_model_index"
down_revision = "029_drop_bogus_benchmarks"
branch_labels = None
depends_on = None

INDEX = "idx_request_log_model_created"


def upgrade() -> None:
    bind = op.get_bind()
    if INDEX in {ix["name"] for ix in sa.inspect(bind).get_indexes("request_log")}:
        return
    # model is TEXT; MySQL can only index a prefix of it.
    op.create_index(
        INDEX, "request_log", ["model", "created_at"], mysql_length={"model": 191}
    )


def downgrade() -> None:
    op.drop_index(INDEX, table_name="request_log")
