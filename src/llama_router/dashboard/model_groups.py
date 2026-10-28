"""Dashboard API for model groups.

A model group is a name clients ask for like a model; each request goes to
the best of its members (see ``routing_table._add_groups``). Writes go to
the database, which tells the router to rebuild its table, so a change is
routable within the router's rebuild debounce.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..database import Database
from ..models import ModelGroup
from ..routing_table import MODES, RoutingTable, member_label
from . import deps

router = APIRouter()

MAX_NAME = 255
MAX_DESCRIPTION = 1000
# Endpoints a group's current choice is shown for, in order of preference.
SHOWN_ENDPOINTS = ("/v1/chat/completions", "/api/chat")


def _table() -> RoutingTable | None:
    rt = deps.get_pm().router
    return rt.table if rt is not None else None


def group_rows(
    groups: list[ModelGroup], table: RoutingTable | None, providers: dict[int, str]
) -> list[dict[str, Any]]:
    """Groups as the dashboard shows them: members with whether anything
    serves them now, and where each routing mode would send a chat request."""
    rows = []
    for g in groups:
        members = []
        for m in g.members:
            provider = None if m.provider_id is None else providers.get(m.provider_id)
            members.append(
                {
                    "id": m.id,
                    "model_name": m.model_name,
                    "provider_id": m.provider_id,
                    "provider": provider,
                    "label": member_label(m.model_name, provider),
                    "available": _member_available(table, m.model_name, m.provider_id),
                }
            )
        choices: dict[str, str] = {}
        endpoint = None
        if table is not None:
            for ep in SHOWN_ENDPOINTS:
                if table.ranked_for(g.name, ep, "balanced"):
                    endpoint = ep
                    break
            if endpoint is not None:
                for mode in MODES:
                    ranked = table.ranked_for(g.name, endpoint, mode)
                    if ranked:
                        best = ranked[0]
                        choices[mode] = f"{best.model} on {best.provider.name}"
        rows.append(
            {
                "id": g.id,
                "name": g.name,
                "description": g.description,
                "members": members,
                "endpoint": endpoint,
                "choices": choices,
            }
        )
    return rows


def _member_available(
    table: RoutingTable | None, model: str, provider_id: int | None
) -> bool:
    if table is None:
        return False
    for (name, _path), cands in table.all_candidates.items():
        if name != model:
            continue
        if provider_id is None or any(c.provider_id == provider_id for c in cands):
            return True
    return False


async def _provider_names(db: Database) -> dict[int, str]:
    return {p.id: p.name for p in await db.list_providers() if p.id is not None}


async def _served_names(db: Database) -> set[str]:
    names: set[str] = set()
    for m in await db.get_all_provider_models():
        names.add(m.name)
        if m.raw_name:
            names.add(m.raw_name)
    return names


async def _check_group_name(db: Database, name: str, *, current: str | None) -> None:
    """Refuse a name that is empty, too long, or already means something."""
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    if len(name) > MAX_NAME:
        raise HTTPException(status_code=400, detail="name is too long")
    if name == current:
        return
    if await db.get_model_group(name) is not None:
        raise HTTPException(status_code=409, detail=f"group {name} already exists")
    aliases = await db.get_model_aliases()
    if name in aliases or name in aliases.values():
        raise HTTPException(
            status_code=409, detail=f"{name} is a model alias; choose another name"
        )
    if name in await _served_names(db):
        raise HTTPException(
            status_code=409,
            detail=f"a provider serves a model named {name}; choose another name",
        )


async def _member(db: Database, body: dict[str, Any]) -> tuple[str, int | None]:
    model = str(body.get("model_name") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model_name is required")
    if len(model) > MAX_NAME:
        raise HTTPException(status_code=400, detail="model_name is too long")
    if await db.get_model_group(model) is not None:
        raise HTTPException(
            status_code=400, detail="a member cannot be another model group"
        )
    raw = body.get("provider_id")
    if raw in (None, ""):
        return model, None
    try:
        pid = int(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail="provider_id must be a number"
        ) from exc
    if await db.get_provider(pid) is None:
        raise HTTPException(status_code=404, detail="provider not found")
    return model, pid


def _description(body: dict[str, Any]) -> str | None:
    text = str(body.get("description") or "").strip()
    if len(text) > MAX_DESCRIPTION:
        raise HTTPException(status_code=400, detail="description is too long")
    return text or None


async def _group(db: Database, group_id: int) -> ModelGroup:
    for g in await db.list_model_groups():
        if g.id == group_id:
            return g
    raise HTTPException(status_code=404, detail="group not found")


async def _body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="body must be JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    return body


@router.get("/api/model-groups")
async def list_groups() -> JSONResponse:
    """Every group with its members and current choices, plus problems the
    last routing table build found."""
    db = deps.get_db()
    table = _table()
    rows = group_rows(await db.list_model_groups(), table, await _provider_names(db))
    warnings = list(table.warnings) if table is not None else []
    return JSONResponse({"groups": rows, "warnings": warnings})


@router.post("/api/model-groups")
async def create_group(request: Request) -> JSONResponse:
    """Body: {name, description?, members?: [{model_name, provider_id?}]}."""
    body = await _body(request)
    db = deps.get_db()
    name = str(body.get("name") or "").strip()
    await _check_group_name(db, name, current=None)
    raw_members = body.get("members") or []
    if not isinstance(raw_members, list):
        raise HTTPException(status_code=400, detail="members must be a list")
    members = [await _member(db, m if isinstance(m, dict) else {}) for m in raw_members]
    if any(model == name for model, _ in members):
        raise HTTPException(status_code=400, detail="a group cannot contain itself")
    group = await db.create_model_group(name, _description(body))
    gid = group.id
    if gid is None:  # pragma: no cover - the insert returns its id
        raise HTTPException(status_code=500, detail="group was not created")
    for model, pid in members:
        await db.add_model_group_member(gid, model, pid)
    return JSONResponse({"id": gid, "name": name}, status_code=201)


@router.patch("/api/model-groups/{group_id}")
async def update_group(group_id: int, request: Request) -> JSONResponse:
    """Body: {name?, description?}. Renaming keeps the members."""
    body = await _body(request)
    db = deps.get_db()
    group = await _group(db, group_id)
    name = str(body.get("name") or group.name).strip()
    await _check_group_name(db, name, current=group.name)
    description = _description(body) if "description" in body else group.description
    await db.update_model_group(group.name, description=description, new_name=name)
    return JSONResponse({"id": group_id, "name": name, "description": description})


@router.delete("/api/model-groups/{group_id}")
async def delete_group(group_id: int) -> JSONResponse:
    db = deps.get_db()
    group = await _group(db, group_id)
    await db.delete_model_group(group.name)
    return JSONResponse({"status": "ok"})


@router.post("/api/model-groups/{group_id}/members")
async def add_member(group_id: int, request: Request) -> JSONResponse:
    """Body: {model_name, provider_id?}. Without a provider the member is the
    model on any provider that serves it."""
    db = deps.get_db()
    group = await _group(db, group_id)
    model, pid = await _member(db, await _body(request))
    if model == group.name:
        raise HTTPException(status_code=400, detail="a group cannot contain itself")
    m = await db.add_model_group_member(group_id, model, pid)
    return JSONResponse(m.model_dump(), status_code=201)


@router.delete("/api/model-groups/{group_id}/members/{member_id}")
async def remove_member(group_id: int, member_id: int) -> JSONResponse:
    db = deps.get_db()
    await _group(db, group_id)
    if not await db.remove_model_group_member(group_id, member_id):
        raise HTTPException(status_code=404, detail="member not found")
    return JSONResponse({"status": "ok"})
