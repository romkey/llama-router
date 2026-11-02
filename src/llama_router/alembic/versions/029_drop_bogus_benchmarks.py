"""Delete benchmarks with impossible speeds.

The Ollama chat benchmark divided by one nanosecond when a response had no
``eval_duration``, storing speeds in the billions of tokens per second. Delete
them so the auto-benchmarker measures those models again.

Revision ID: 029_drop_bogus_benchmarks
Revises: 028_dashboard_mfa
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "029_drop_bogus_benchmarks"
down_revision = "028_dashboard_mfa"
branch_labels = None
depends_on = None

# Matches models.MAX_TOKENS_PER_SECOND; a migration keeps its own copy.
MAX_TOKENS_PER_SECOND = 1_000_000.0


def upgrade() -> None:
    op.get_bind().execute(
        sa.text("DELETE FROM benchmarks WHERE tokens_per_second > :limit"),
        {"limit": MAX_TOKENS_PER_SECOND},
    )


def downgrade() -> None:
    pass
