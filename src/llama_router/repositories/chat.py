"""Dashboard chat conversations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import Repository, _iso


class ChatRepository(Repository):
    @staticmethod
    def _chat_owner(user_id: int | None) -> tuple[str, tuple[Any, ...]]:
        if user_id is None:
            return "user_id IS NULL", ()
        return "user_id = ?", (user_id,)

    async def create_chat_conversation(
        self,
        user_id: int | None,
        title: str,
        provider_id: int | None,
        model_name: str | None,
    ) -> int:
        return await self._insert_get_id(
            "INSERT INTO chat_conversations (user_id, title, provider_id, "
            "model_name) VALUES (?, ?, ?, ?)",
            (user_id, title, provider_id, model_name),
        )

    async def list_chat_conversations(
        self, user_id: int | None
    ) -> list[dict[str, Any]]:
        """The user's conversations, most recently used first, without
        their messages."""
        owner, params = self._chat_owner(user_id)
        rows = await self._select_all(
            "SELECT * FROM chat_conversations "  # noqa: S608 - fixed clause; values are bound
            f"WHERE {owner} ORDER BY updated_at DESC, id DESC",
            params,
        )
        return [_row_to_chat_conversation(r) for r in rows]

    async def get_chat_conversation(
        self, conversation_id: int, user_id: int | None
    ) -> dict[str, Any] | None:
        """The conversation with its messages, if it is the user's."""
        owner, params = self._chat_owner(user_id)
        row = await self._select_one(
            "SELECT * FROM chat_conversations "  # noqa: S608 - fixed clause; values are bound
            f"WHERE id = ? AND {owner}",
            (conversation_id, *params),
        )
        if row is None:
            return None
        conv = _row_to_chat_conversation(row)
        rows = await self._select_all(
            "SELECT * FROM chat_messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        )
        conv["messages"] = [_row_to_chat_message(r) for r in rows]
        return conv

    async def add_chat_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        *,
        image: str | None = None,
        model_name: str | None = None,
        provider_name: str | None = None,
        duration_ms: float | None = None,
    ) -> dict[str, Any]:
        mid = await self._insert_get_id(
            "INSERT INTO chat_messages (conversation_id, role, content, image, "
            "model_name, provider_name, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                role,
                content,
                image,
                model_name,
                provider_name,
                duration_ms,
            ),
        )
        row = await self._select_one("SELECT * FROM chat_messages WHERE id = ?", (mid,))
        if row is None:  # pragma: no cover - just inserted
            raise RuntimeError("chat message vanished")
        return _row_to_chat_message(row)

    async def touch_chat_conversation(
        self, conversation_id: int, provider_id: int | None, model_name: str | None
    ) -> None:
        """Remember the model last used and move the conversation to the top."""
        await self._execute(
            "UPDATE chat_conversations SET provider_id = ?, model_name = ?, "
            "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (provider_id, model_name, conversation_id),
        )

    async def delete_chat_conversation(
        self, conversation_id: int, user_id: int | None
    ) -> bool:
        owner, params = self._chat_owner(user_id)
        await self._execute(
            "DELETE FROM chat_messages WHERE conversation_id IN ("  # noqa: S608 - fixed clause; values are bound
            f"SELECT id FROM chat_conversations WHERE id = ? AND {owner})",
            (conversation_id, *params),
        )
        n = await self._execute_rowcount(
            f"DELETE FROM chat_conversations WHERE id = ? AND {owner}",  # noqa: S608 - fixed clause; values are bound
            (conversation_id, *params),
        )
        return n > 0


def _row_to_chat_conversation(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "provider_id": row["provider_id"],
        "model_name": row["model_name"],
        "created_at": _iso(row["created_at"]),
        "updated_at": _iso(row["updated_at"]),
    }


def _row_to_chat_message(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "role": row["role"],
        "content": row["content"],
        "image": row["image"],
        "model_name": row["model_name"],
        "provider_name": row["provider_name"],
        "duration_ms": row["duration_ms"],
        "created_at": _iso(row["created_at"]),
    }
