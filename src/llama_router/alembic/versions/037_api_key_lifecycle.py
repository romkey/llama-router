"""API key expiry, renewal and pausing.

``expires_at`` (null: never) ends a key at that time (401 "API key
expired"); the key is kept. ``renewal_requested_at`` marks an owner's
request for a renewal an admin must approve; ``renewal_denied_reason`` is
shown to the owner after a denial. ``paused_at`` pauses a key (403 "API key
paused"); ``paused_by_user_id`` is who did it (null: the router itself),
``pause_locked`` 1 for an admin hold that only an admin may lift, and
``pause_reason`` says why.

Revision ID: 037_api_key_lifecycle
Revises: 036_user_quotas
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from alembic import op

revision = "037_api_key_lifecycle"
down_revision = "036_user_quotas"
branch_labels = None
depends_on = None

_COLUMNS: tuple[tuple[str, Any, dict[str, Any]], ...] = (
    ("expires_at", sa.DateTime(), {}),
    ("renewal_requested_at", sa.DateTime(), {}),
    ("renewal_denied_reason", sa.String(500), {}),
    ("paused_at", sa.DateTime(), {}),
    ("paused_by_user_id", sa.Integer(), {}),
    ("pause_locked", sa.Integer(), {"nullable": False, "server_default": "0"}),
    ("pause_reason", sa.String(500), {}),
)


def upgrade() -> None:
    have = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("api_keys")}
    for name, type_, kw in _COLUMNS:
        if name not in have:
            op.add_column(
                "api_keys", sa.Column(name, type_, **{"nullable": True, **kw})
            )


def downgrade() -> None:
    with op.batch_alter_table("api_keys") as batch:
        for name, _, _ in reversed(_COLUMNS):
            batch.drop_column(name)
