"""New installs require an API key; existing installs keep their setting.

Revision 001 seeds ``allow_unauthenticated = 'true'``, which left a fresh
router open to anyone who could reach its API ports. Alembic runs 001 and
this revision in the same upgrade on a new database, so this one turns the
seed off when nothing has been configured yet: no providers, API keys or
dashboard accounts. A router with any of those was already in use and may
rely on anonymous access, so its stored value is left alone (the dashboard
and a startup warning say when it is on).

Revision ID: 025_default_deny_anonymous
Revises: 024_api_key_owner
Create Date: 2026-10-31
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "025_default_deny_anonymous"
down_revision = "024_api_key_owner"
branch_labels = None
depends_on = None

_IN_USE_TABLES = ("providers", "api_keys", "dashboard_users")


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    for table in _IN_USE_TABLES:
        if table in tables:
            # Table names come from the constant above, not from input.
            row = bind.execute(sa.text(f"SELECT 1 FROM {table} LIMIT 1")).first()  # noqa: S608
            if row is not None:
                return
    bind.execute(
        sa.text(
            "UPDATE app_settings SET value = 'false' "
            "WHERE app_settings.key = 'allow_unauthenticated' AND value = 'true'"
        )
    )


def downgrade() -> None:
    # Which installs this revision changed is not recorded; leave the value.
    pass
