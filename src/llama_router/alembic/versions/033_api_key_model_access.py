"""API keys limited to chosen models and model groups.

``api_keys.restrict_models`` says whether the key is limited at all, so a
limited key whose last allowed group is deleted can use nothing rather than
everything. ``api_key_allowed_models`` holds what it may use: a model by
name, or a group by id (so renaming the group keeps the key's access).
Each row sets exactly one of the two; the repository writes them that way
(MySQL refuses a CHECK on a column with a cascading foreign key).

Revision ID: 033_api_key_model_access
Revises: 032_api_key_label
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "033_api_key_model_access"
down_revision = "032_api_key_label"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    cols = {c["name"] for c in inspector.get_columns("api_keys")}
    if "restrict_models" not in cols:
        op.add_column(
            "api_keys",
            sa.Column(
                "restrict_models",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )
    if "api_key_allowed_models" not in set(inspector.get_table_names()):
        op.create_table(
            "api_key_allowed_models",
            sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
            sa.Column(
                "api_key_id",
                sa.Integer(),
                sa.ForeignKey("api_keys.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("model_name", sa.String(255), nullable=True),
            sa.Column(
                "group_id",
                sa.Integer(),
                sa.ForeignKey("model_groups.id", ondelete="CASCADE"),
                nullable=True,
            ),
            sa.UniqueConstraint(
                "api_key_id", "model_name", name="uq_api_key_allowed_models_model"
            ),
            sa.UniqueConstraint(
                "api_key_id", "group_id", name="uq_api_key_allowed_models_group"
            ),
        )
        op.create_index(
            "ix_api_key_allowed_models_api_key_id",
            "api_key_allowed_models",
            ["api_key_id"],
        )


def downgrade() -> None:
    op.drop_table("api_key_allowed_models")
    with op.batch_alter_table("api_keys") as batch:
        batch.drop_column("restrict_models")
