"""Which models and model groups an API key may use, for the dashboard API.

A request body carries it as ``allowed``: ``null`` for any model, or
``{"models": [names], "groups": [group ids]}`` to limit the key to those.
A model is allowed under any of its aliases; a group by its own name only,
not its members by theirs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from ..database import Database

MODEL_NAME_MAX = 255
# More than any real key needs; keeps one request from writing thousands of rows.
MAX_ENTRIES = 500


@dataclass(frozen=True, slots=True)
class AccessSpec:
    restricted: bool
    models: tuple[str, ...] = ()
    group_ids: tuple[int, ...] = ()


UNRESTRICTED = AccessSpec(restricted=False)


def _model_name(raw: Any) -> str:
    if not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="allowed.models must be names")
    name = raw.strip()
    if not name:
        raise HTTPException(status_code=400, detail="allowed.models has a blank name")
    if len(name) > MODEL_NAME_MAX or not name.isprintable():
        raise HTTPException(
            status_code=400,
            detail=(
                f"allowed model names must be printable and at most "
                f"{MODEL_NAME_MAX} characters"
            ),
        )
    return name


def _group_id(raw: Any) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise HTTPException(status_code=400, detail="allowed.groups must be group ids")
    return raw


async def parse_access(db: Database, raw: Any) -> AccessSpec:
    """Validate a body's ``allowed`` value. Group ids must exist; model names
    needn't be served right now (their provider may be offline)."""
    if raw is None:
        return UNRESTRICTED
    if not isinstance(raw, dict):
        raise HTTPException(
            status_code=400,
            detail="allowed must be null or {models: [...], groups: [...]}",
        )
    models_raw = raw.get("models") or []
    groups_raw = raw.get("groups") or []
    if not isinstance(models_raw, list) or not isinstance(groups_raw, list):
        raise HTTPException(
            status_code=400, detail="allowed.models and allowed.groups must be lists"
        )
    if len(models_raw) + len(groups_raw) > MAX_ENTRIES:
        raise HTTPException(
            status_code=400, detail=f"allowed may list at most {MAX_ENTRIES} entries"
        )
    models = tuple(dict.fromkeys(_model_name(m) for m in models_raw))
    group_ids = tuple(dict.fromkeys(_group_id(g) for g in groups_raw))
    if group_ids:
        known = {g.id for g in await db.list_model_groups()}
        missing = [g for g in group_ids if g not in known]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"No model group with id {', '.join(map(str, missing))}",
            )
    return AccessSpec(restricted=True, models=models, group_ids=group_ids)


async def store_access(db: Database, key_id: int, spec: AccessSpec) -> None:
    await db.set_api_key_access(
        key_id, spec.restricted, list(spec.models), list(spec.group_ids)
    )


async def access_json(db: Database, key_id: int) -> dict[str, Any] | None:
    """The key's ``allowed`` value as the API returns it: None for any model."""
    access = await db.get_api_key_access(key_id)
    if access is None or not access["restricted"]:
        return None
    return {"models": access["models"], "groups": access["groups"]}


async def with_access(db: Database, key: dict[str, Any]) -> dict[str, Any]:
    """*key* (a ``list_api_keys`` row) with its ``allowed`` value."""
    return {**key, "allowed": await access_json(db, int(key["id"]))}


async def allowable(db: Database, served: list[str]) -> dict[str, Any]:
    """What a key can be limited to: model groups (id, name, members) and
    the model names served now (*served*)."""
    groups = await db.list_model_groups()
    group_names = {g.name for g in groups}
    return {
        "groups": [
            {
                "id": g.id,
                "name": g.name,
                "members": [m.model_name for m in g.members],
            }
            for g in groups
        ],
        "models": sorted(n for n in set(served) if n not in group_names),
    }
