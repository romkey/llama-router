"""Provider API keys; worker identity column; drop provider-level URLs.

Providers are reached through ``provider_addresses``. The provider's own
``url`` and ``llamacpp_url`` only seeded the first address and carried a
worker's identity as ``worker://<name>``. This revision:

1. copies any provider URL that has no address yet into ``provider_addresses``;
2. moves the worker identity into ``worker_name``;
3. adds ``api_key_enc`` and ``api_key_header`` for upstream authentication;
4. drops ``url`` and ``llamacpp_url``.

Revision ID: 004_provider_api_key
Revises: 003_workers
Create Date: 2026-10-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "004_provider_api_key"
down_revision = "003_workers"
branch_labels = None
depends_on = None

_WORKER_SCHEME = "worker://"


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {c["name"] for c in insp.get_columns("providers")}

    if "worker_name" not in cols:
        op.add_column("providers", sa.Column("worker_name", sa.Text(), nullable=True))
    if "api_key_enc" not in cols:
        op.add_column("providers", sa.Column("api_key_enc", sa.Text(), nullable=True))
    if "api_key_header" not in cols:
        op.add_column(
            "providers", sa.Column("api_key_header", sa.Text(), nullable=True)
        )

    if "url" in cols:
        rows = bind.execute(
            sa.text("SELECT id, url, llamacpp_url FROM providers")
        ).mappings()
        for row in list(rows):
            url = (row["url"] or "").strip()
            if url.startswith(_WORKER_SCHEME):
                bind.execute(
                    sa.text("UPDATE providers SET worker_name = :n WHERE id = :id"),
                    {"n": url[len(_WORKER_SCHEME) :], "id": row["id"]},
                )
            has_addr = bind.execute(
                sa.text(
                    "SELECT COUNT(*) FROM provider_addresses WHERE provider_id = :id"
                ),
                {"id": row["id"]},
            ).scalar()
            if url and not has_addr:
                bind.execute(
                    sa.text(
                        "INSERT INTO provider_addresses "
                        "(provider_id, url, llamacpp_url, is_preferred, is_live) "
                        "VALUES (:pid, :url, :lcpp, 1, 0)"
                    ),
                    {"pid": row["id"], "url": url, "lcpp": row["llamacpp_url"]},
                )

        if bind.dialect.name == "sqlite":
            with op.batch_alter_table("providers", recreate="always") as batch:
                batch.drop_column("url")
                batch.drop_column("llamacpp_url")
        else:
            op.drop_column("providers", "url")
            op.drop_column("providers", "llamacpp_url")

    existing = {ix["name"] for ix in sa.inspect(bind).get_indexes("providers")}
    if "uq_providers_worker_name" not in existing:
        op.create_index(
            "uq_providers_worker_name", "providers", ["worker_name"], unique=True
        )


def downgrade() -> None:
    op.drop_index("uq_providers_worker_name", table_name="providers")
    with op.batch_alter_table("providers") as batch:
        batch.add_column(sa.Column("url", sa.Text(), nullable=False, server_default=""))
        batch.add_column(sa.Column("llamacpp_url", sa.Text(), nullable=True))
    bind = op.get_bind()
    bind.execute(
        sa.text(
            "UPDATE providers SET url = :scheme || worker_name "
            "WHERE worker_name IS NOT NULL"
        ),
        {"scheme": _WORKER_SCHEME},
    )
    bind.execute(
        sa.text(
            "UPDATE providers SET url = COALESCE((SELECT a.url FROM provider_addresses a "
            "WHERE a.provider_id = providers.id ORDER BY a.is_preferred DESC, a.id "
            "LIMIT 1), '') WHERE worker_name IS NULL"
        )
    )
    with op.batch_alter_table("providers") as batch:
        batch.drop_column("api_key_header")
        batch.drop_column("api_key_enc")
        batch.drop_column("worker_name")
