"""Dashboard chat: each user's conversations with models, for trying them out.

Revision ID: 021_chat
Revises: 020_request_routed_model
Create Date: 2026-10-30
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = "021_chat"
down_revision = "020_request_routed_model"
branch_labels = None
depends_on = None

# Messages carry images as data URLs; MySQL's TEXT stops at 64 KB.
_LONG_TEXT = sa.Text().with_variant(mysql.LONGTEXT(), "mysql", "mariadb")


def upgrade() -> None:
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "chat_conversations" not in tables:
        op.create_table(
            "chat_conversations",
            sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
            # NULL: made before any dashboard user existed (no sign-in).
            sa.Column(
                "user_id",
                sa.Integer(),
                sa.ForeignKey("dashboard_users.id", ondelete="CASCADE"),
                nullable=True,
            ),
            sa.Column("title", sa.String(255), nullable=False),
            sa.Column(
                "provider_id",
                sa.Integer(),
                sa.ForeignKey("providers.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("model_name", sa.String(255), nullable=True),
            sa.Column(
                "created_at",
                sa.TIMESTAMP(),
                nullable=False,
                server_default=sa.func.current_timestamp(),
            ),
            sa.Column(
                "updated_at",
                sa.TIMESTAMP(),
                nullable=False,
                server_default=sa.func.current_timestamp(),
            ),
        )
        op.create_index(
            "ix_chat_conversations_user_id", "chat_conversations", ["user_id"]
        )
    if "chat_messages" not in tables:
        op.create_table(
            "chat_messages",
            sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
            sa.Column(
                "conversation_id",
                sa.Integer(),
                sa.ForeignKey("chat_conversations.id", ondelete="CASCADE"),
                nullable=False,
            ),
            # "user", "assistant" or "error".
            sa.Column("role", sa.String(16), nullable=False),
            sa.Column("content", _LONG_TEXT, nullable=False),
            # Data URL (or upstream URL) of an attached or generated image.
            sa.Column("image", _LONG_TEXT, nullable=True),
            sa.Column("model_name", sa.String(255), nullable=True),
            sa.Column("provider_name", sa.String(255), nullable=True),
            sa.Column("duration_ms", sa.Float(), nullable=True),
            sa.Column(
                "created_at",
                sa.TIMESTAMP(),
                nullable=False,
                server_default=sa.func.current_timestamp(),
            ),
        )
        op.create_index(
            "ix_chat_messages_conversation_id", "chat_messages", ["conversation_id"]
        )


def downgrade() -> None:
    op.drop_table("chat_messages")
    op.drop_table("chat_conversations")
