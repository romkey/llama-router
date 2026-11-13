"""Per-user quotas.

``dashboard_users`` gets ``daily_token_limit``, ``requests_per_minute`` and
``daily_spend_cap_usd``; null follows the admin's default for users (itself
unlimited until set). ``api_keys`` gets ``daily_token_limit`` and
``requests_per_minute``, which may only be tighter than the owner's.
``request_log.user_id`` records whose quota a request counted against, so
revoking and reissuing a key does not reset a user's usage; it is indexed
with ``created_at`` for the daily totals.

Revision ID: 036_user_quotas
Revises: 035_provider_max_concurrent
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from alembic import op

revision = "036_user_quotas"
down_revision = "035_provider_max_concurrent"
branch_labels = None
depends_on = None

INDEX = "idx_request_log_user_created"

_USER_COLUMNS = (
    ("daily_token_limit", sa.BigInteger()),
    ("requests_per_minute", sa.Integer()),
    ("daily_spend_cap_usd", sa.Float()),
)
_KEY_COLUMNS = (
    ("daily_token_limit", sa.BigInteger()),
    ("requests_per_minute", sa.Integer()),
)


def _add(table: str, columns: tuple[tuple[str, Any], ...]) -> None:
    have = {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}
    for name, type_ in columns:
        if name not in have:
            op.add_column(table, sa.Column(name, type_, nullable=True))


def upgrade() -> None:
    _add("dashboard_users", _USER_COLUMNS)
    _add("api_keys", _KEY_COLUMNS)
    _add("request_log", (("user_id", sa.Integer()),))
    bind = op.get_bind()
    if INDEX not in {ix["name"] for ix in sa.inspect(bind).get_indexes("request_log")}:
        op.create_index(INDEX, "request_log", ["user_id", "created_at"])


def downgrade() -> None:
    op.drop_index(INDEX, table_name="request_log")
    with op.batch_alter_table("request_log") as batch:
        batch.drop_column("user_id")
    with op.batch_alter_table("api_keys") as batch:
        for name, _ in _KEY_COLUMNS:
            batch.drop_column(name)
    with op.batch_alter_table("dashboard_users") as batch:
        for name, _ in _USER_COLUMNS:
            batch.drop_column(name)
