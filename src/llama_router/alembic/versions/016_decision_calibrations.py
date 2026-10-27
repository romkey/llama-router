"""Calibrations for decision questions (see decisions.py).

One row per model and question wording: a temperature and per-answer log
offsets that correct the shifted priors local models show on yes/no and
rubric questions.

Revision ID: 016_decision_calibrations
Revises: 015_cost
Create Date: 2026-10-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from llama_router.db_types import ShortText

revision = "016_decision_calibrations"
down_revision = "015_cost"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "decision_calibrations",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("model_name", ShortText, nullable=False),
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
