"""Calibrations for decision questions (see decisions.py).

One row per model and question wording: a temperature and per-answer log
offsets that correct the shifted priors local models show on yes/no and
rubric questions.

Revision ID: 014_decision_calibrations
Revises: 013_cloud_operation
Create Date: 2026-10-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "014_decision_calibrations"
down_revision = "013_cloud_operation"
branch_labels = None
depends_on = None

# MySQL and MariaDB reject TEXT in a key without a prefix length.
_KEY_TEXT = sa.Text().with_variant(sa.String(255), "mysql", "mariadb")


def upgrade() -> None:
    op.create_table(
        "decision_calibrations",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("model_name", _KEY_TEXT, nullable=False),
        sa.Column("question_key", sa.String(64), nullable=False),
        sa.Column("question_type", sa.String(16), nullable=False),
        # The question as given, for the dashboard.
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=False),
        # JSON object: answer key -> log offset.
        sa.Column("offsets", sa.Text(), nullable=False),
        # Labelled examples it was fit from, if it was fit here.
        sa.Column("examples", sa.Integer(), nullable=True),
        sa.Column(
            "created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.Column(
            "updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.UniqueConstraint(
            "model_name", "question_key", name="uq_decision_calibrations_model_q"
        ),
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_table("decision_calibrations", if_exists=True)
