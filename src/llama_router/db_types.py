"""Column types shared by the Alembic migrations."""

from __future__ import annotations

import sqlalchemy as sa

# TEXT everywhere except MySQL and MariaDB, where it is VARCHAR(255). MySQL
# rejects TEXT columns that have a default or are part of a key (primary,
# unique or index) without a prefix length, so every such column uses this
# type. SQLite and PostgreSQL get exactly the DDL they always did.
ShortText = sa.Text().with_variant(sa.String(255), "mysql", "mariadb")
