"""Provider flavor: the server the last probe recognised.

Revision ID: 010_provider_flavor
Revises: 009_single_protocol_providers
Create Date: 2026-10-27
"""

from __future__ import annotations

import json

import sqlalchemy as sa
from alembic import op

revision = "010_provider_flavor"
down_revision = "009_single_protocol_providers"
branch_labels = None
depends_on = None

_ORDER = (
    "llama-router",
    "ollama",
    "lemonade",
    "lm-studio",
    "koboldcpp",
    "sglang",
    "tgi",
    "vllm",
    "llama.cpp",
    "litellm",
    "localai",
    "openai-compatible",
)


def upgrade() -> None:
    bind = op.get_bind()
    cols = {c["name"] for c in sa.inspect(bind).get_columns("providers")}
    if "flavor" not in cols:
        op.add_column("providers", sa.Column("flavor", sa.Text(), nullable=True))
    # Name the flavor from probes already stored.
    rows = bind.execute(
        sa.text("SELECT id, capabilities FROM providers WHERE capabilities IS NOT NULL")
    ).mappings()
    for row in list(rows):
        try:
            caps = json.loads(row["capabilities"])
        except ValueError:
            continue
        servers = caps.get("servers") or [] if isinstance(caps, dict) else []
        flavor = next((f for f in _ORDER if f in servers), None)
        if flavor:
            bind.execute(
                sa.text("UPDATE providers SET flavor = :f WHERE id = :id"),
                {"f": flavor, "id": row["id"]},
            )


def downgrade() -> None:
    with op.batch_alter_table("providers") as batch:
        batch.drop_column("flavor")
