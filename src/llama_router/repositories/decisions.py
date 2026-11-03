"""Decision calibrations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..db_events import CHANGE_DECISIONS, _notifies
from .base import Repository, _iso, _json_or_none


class DecisionRepository(Repository):
    @_notifies(CHANGE_DECISIONS)
    async def set_decision_calibration(
        self,
        model_name: str,
        question_key: str,
        question_type: str,
        question: dict[str, Any],
        temperature: float,
        offsets: dict[str, float],
        examples: int | None = None,
    ) -> None:
        """Store the calibration for *model_name* and one question wording."""
        q = json.dumps(question, ensure_ascii=False)
        o = json.dumps(offsets)
        if self._dialect_name in ("mysql", "mariadb"):
            await self._execute(
                "INSERT INTO decision_calibrations (model_name, question_key, "
                "question_type, question, temperature, offsets, examples) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON DUPLICATE KEY UPDATE "
                "question_type = VALUES(question_type), question = VALUES(question), "
                "temperature = VALUES(temperature), offsets = VALUES(offsets), "
                "examples = VALUES(examples), updated_at = CURRENT_TIMESTAMP",
                (model_name, question_key, question_type, q, temperature, o, examples),
            )
        else:
            await self._execute(
                "INSERT INTO decision_calibrations (model_name, question_key, "
                "question_type, question, temperature, offsets, examples) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (model_name, question_key) DO UPDATE SET "
                "question_type = ?, question = ?, temperature = ?, offsets = ?, "
                "examples = ?, updated_at = CURRENT_TIMESTAMP",
                (
                    model_name,
                    question_key,
                    question_type,
                    q,
                    temperature,
                    o,
                    examples,
                    question_type,
                    q,
                    temperature,
                    o,
                    examples,
                ),
            )

    async def list_decision_calibrations(self) -> list[dict[str, Any]]:
        rows = await self._select_all(
            "SELECT * FROM decision_calibrations ORDER BY model_name, id"
        )
        return [_row_to_calibration(r) for r in rows]

    @_notifies(CHANGE_DECISIONS)
    async def delete_decision_calibration(self, calibration_id: int) -> bool:
        n = await self._execute_rowcount(
            "DELETE FROM decision_calibrations WHERE id = ?", (calibration_id,)
        )
        return n > 0


def _row_to_calibration(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        offsets = json.loads(row["offsets"] or "{}")
    except TypeError, ValueError:
        offsets = {}
    return {
        "id": row["id"],
        "model_name": row["model_name"],
        "question_key": row["question_key"],
        "question_type": row["question_type"],
        "question": _json_or_none(row["question"]),
        "temperature": float(row["temperature"]),
        "offsets": {str(k): float(v) for k, v in offsets.items()},
        "examples": row["examples"],
        "created_at": _iso(row["created_at"]),
        "updated_at": _iso(row["updated_at"]),
    }
