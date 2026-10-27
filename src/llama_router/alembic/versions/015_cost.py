"""Cost: model prices per provider, token usage and cost in the request log,
and a daily spend cap per API key.

Revision ID: 015_cost
Revises: 014_model_aliases
Create Date: 2026-10-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "015_cost"
down_revision = "014_model_aliases"
branch_labels = None
depends_on = None

_LOG_COLUMNS = (
    ("api_key_id", sa.Integer()),
    ("prompt_tokens", sa.Integer()),
    ("completion_tokens", sa.Integer()),
    ("cost_usd", sa.Float()),
)


def upgrade() -> None:
    insp = sa.inspect(op.get_bind())
    if "model_prices" not in insp.get_table_names():
        op.create_table(
            "model_prices",
            sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
            sa.Column(
                "provider_id",
                sa.Integer(),
                sa.ForeignKey("providers.id", ondelete="CASCADE"),
                nullable=False,
            ),
            # A model name or fnmatch pattern; "*" prices every model.
            sa.Column("model_pattern", sa.String(255), nullable=False),
            # USD per million tokens.
            sa.Column("input_per_mtok", sa.Float(), nullable=False),
            sa.Column("output_per_mtok", sa.Float(), nullable=False),
            sa.UniqueConstraint(
                "provider_id", "model_pattern", name="uq_model_prices_provider_model"
            ),
        )
    log_cols = {c["name"] for c in insp.get_columns("request_log")}
    for name, type_ in _LOG_COLUMNS:
        if name not in log_cols:
            op.add_column("request_log", sa.Column(name, type_, nullable=True))
    key_cols = {c["name"] for c in insp.get_columns("api_keys")}
    if "daily_spend_cap_usd" not in key_cols:
        op.add_column(
            "api_keys", sa.Column("daily_spend_cap_usd", sa.Float(), nullable=True)
        )
    existing = {ix["name"] for ix in insp.get_indexes("request_log")}
    if "idx_request_log_key_created" not in existing:
        op.create_index(
            "idx_request_log_key_created",
            "request_log",
            ["api_key_id", "created_at"],
        )


def downgrade() -> None:
    op.drop_index("idx_request_log_key_created", table_name="request_log")
    with op.batch_alter_table("api_keys") as batch:
        batch.drop_column("daily_spend_cap_usd")
    with op.batch_alter_table("request_log") as batch:
        for name, _ in reversed(_LOG_COLUMNS):
            batch.drop_column(name)
    op.drop_table("model_prices")
