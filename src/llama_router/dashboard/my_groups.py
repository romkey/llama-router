"""Private model groups dashboard users define for themselves.

A private group works like a router-wide model group (one name that routes
to the best of its members), but only for its owner: requests made with
the API keys they issued, and their dashboard chats. Its name only has to
be unique among the owner's own groups, and it wins over a model or
router-wide group of the same name for them alone.

The routing table compiles every user's groups when it is rebuilt (see
``routing_table._add_user_groups``), so a request for one is resolved with
a dictionary lookup and never reads the database.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .. import audit
from ..config import settings
from ..database import Database
from ..models import ModelGroup
from ..routing_table import RoutingTable, member_label
from . import deps
from .model_groups import (
    MAX_DESCRIPTION,
    MAX_NAME,
    SHOWN_ENDPOINTS,
    _member_available,
    _provider_names,
    _served_names,
)
from .my_keys import _owner_id
from .pages import dash_page_ctx, templates

router = APIRouter()

# Most members one private group may have.
MAX_MEMBERS = 50


def _table() -> RoutingTable | None:
    rt = deps.get_pm().router
    return rt.table if rt is not None else None


def _rows(
    groups: list[ModelGroup],
    table: RoutingTable | None,
    providers: dict[int, str],
    served: set[str],
    owner: int,
) -> list[dict[str, Any]]:
    mine = table.user_groups.get(owner, {}) if table is not None else {}
    rows = []
    for g in groups:
        members = []
        for m in g.members:
            provider = None if m.provider_id is None else providers.get(m.provider_id)
            members.append(
                {
                    "model_name": m.model_name,
                    "provider_id": m.provider_id,
                    "provider": provider,
                    "label": member_label(m.model_name, provider),
                    "available": _member_available(table, m.model_name, m.provider_id),
                }
            )
        choice = None
        key = mine.get(g.name)
        if table is not None and key is not None:
            for ep in SHOWN_ENDPOINTS:
                ranked = table.ranked_for(key, ep, "balanced")
                if ranked:
                    choice = f"{ranked[0].model} on {ranked[0].provider.name}"
                    break
        shadows = None
        if table is not None and table.is_group(g.name):
            shadows = "group"
        elif g.name in served:
            shadows = "model"
        rows.append(
            {
                "id": g.id,
                "name": g.name,
                "description": g.description,
                "members": members,
                # The balanced choice for a chat request right now; None
                # until the routing table has the group, or nothing serves it.
                "choice": choice,
                # A router-wide group or a served model this group hides for
                # its owner, if any.
                "shadows": shadows,
            }
        )
    return rows


def _name(body: dict[str, Any], current: str | None = None) -> str:
    raw = body.get("name", current)
    name = str(raw or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    if len(name) > MAX_NAME:
        raise HTTPException(status_code=400, detail="name is too long")
    return name


def _description(body: dict[str, Any], current: str | None = None) -> str | None:
    if "description" not in body:
        return current
    text = str(body.get("description") or "").strip()
    if len(text) > MAX_DESCRIPTION:
        raise HTTPException(status_code=400, detail="description is too long")
    return text or None


async def _members(
    db: Database, raw: Any, *, own_names: set[str]
) -> list[tuple[str, int | None]]:
    """Members from a request body: [{model_name, provider_id?}].

    A member is a model, anywhere or on one provider. Groups do not nest, so
    neither a router-wide group nor another of the user's groups may be one.
    A member may have the group's own name: it is then the model the group
    shadows, as member names always mean models.
    """
    if not isinstance(raw, list):
        raise HTTPException(status_code=400, detail="members must be a list")
    if len(raw) > MAX_MEMBERS:
        raise HTTPException(
            status_code=400, detail=f"a group may have at most {MAX_MEMBERS} members"
        )
    router_groups = {g.name for g in await db.list_model_groups()}
    provider_ids: set[int] | None = None
    out: list[tuple[str, int | None]] = []
    for item in raw:
        body = item if isinstance(item, dict) else {}
        model = str(body.get("model_name") or "").strip()
        if not model:
            raise HTTPException(status_code=400, detail="model_name is required")
        if len(model) > MAX_NAME:
            raise HTTPException(status_code=400, detail="model_name is too long")
        if model in router_groups or model in own_names:
            raise HTTPException(
                status_code=400, detail="a member cannot be another model group"
            )
        pid_raw = body.get("provider_id")
        pid: int | None = None
        if pid_raw not in (None, ""):
            try:
                pid = int(pid_raw)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400, detail="provider_id must be a number"
                ) from exc
            if provider_ids is None:
                provider_ids = {
                    p.id for p in await db.list_providers() if p.id is not None
                }
            if pid not in provider_ids:
                raise HTTPException(status_code=404, detail="provider not found")
        out.append((model, pid))
    return out


async def _own_group(db: Database, owner: int, group_id: int) -> ModelGroup:
    group = await db.get_user_model_group(owner, group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="group not found")
    return group


def _summary(name: str, members: list[tuple[str, int | None]] | None) -> str:
    text = f"name={name}"
    if members is not None:
        text += f" members={len(members)}"
    return text


@router.get("/my-groups", response_class=HTMLResponse)
async def my_groups_page(request: Request) -> Response:
    return templates.TemplateResponse(
        request,
        "my_groups.html",
        await dash_page_ctx(
            request,
            {
                "max_groups": settings.max_model_groups_per_user,
                "max_members": MAX_MEMBERS,
            },
        ),
    )


@router.get("/api/my-groups")
async def list_my_groups(request: Request) -> JSONResponse:
    """The user's private groups, with their members and current choices,
    and the model names they can add."""
    owner = _owner_id(request)
    db = deps.get_db()
    served = await _served_names(db)
    providers = await _provider_names(db)
    rows = _rows(
        await db.list_user_model_groups(owner), _table(), providers, served, owner
    )
    return JSONResponse(
        {
            "groups": rows,
            "models": sorted(served),
            "providers": [
                {"id": pid, "name": name}
                for pid, name in sorted(providers.items(), key=lambda p: p[1])
            ],
            "limits": {
                "max_groups": settings.max_model_groups_per_user,
                "max_members": MAX_MEMBERS,
            },
        }
    )


@router.post("/api/my-groups")
async def create_my_group(request: Request) -> JSONResponse:
    """Body: {name, description?, members?: [{model_name, provider_id?}]}."""
    owner = _owner_id(request)
    body = await deps.json_body(request)
    db = deps.get_db()
    name = _name(body)
    existing = await db.list_user_model_groups(owner)
    own_names = {g.name for g in existing}
    if name in own_names:
        raise HTTPException(status_code=409, detail=f"you already have a group {name}")
    limit = settings.max_model_groups_per_user
    if limit > 0 and len(existing) >= limit:
        raise HTTPException(
            status_code=409,
            detail=f"You already have {limit} model groups; delete one to add another",
        )
    members = await _members(db, body.get("members") or [], own_names=own_names)
    group = await db.create_user_model_group(owner, name, _description(body), members)
    await audit.record(
        request,
        "my_model_group.create",
        ("my_model_group", group.id),
        _summary(name, members),
    )
    return JSONResponse({"id": group.id, "name": name}, status_code=201)


@router.patch("/api/my-groups/{group_id:int}")
async def update_my_group(group_id: int, request: Request) -> JSONResponse:
    """Body: {name?, description?, members?}. Members, when given, replace
    the group's members in the order given."""
    owner = _owner_id(request)
    body = await deps.json_body(request)
    db = deps.get_db()
    group = await _own_group(db, owner, group_id)
    name = _name(body, group.name)
    others = {g.name for g in await db.list_user_model_groups(owner)} - {group.name}
    if name in others:
        raise HTTPException(status_code=409, detail=f"you already have a group {name}")
    members = (
        await _members(db, body["members"], own_names=others)
        if "members" in body
        else None
    )
    description = _description(body, group.description)
    await db.update_user_model_group(
        owner, group_id, name=name, description=description, members=members
    )
    await audit.record(
        request,
        "my_model_group.update",
        ("my_model_group", group_id),
        _summary(name, members),
    )
    return JSONResponse({"id": group_id, "name": name, "description": description})


@router.delete("/api/my-groups/{group_id:int}")
async def delete_my_group(group_id: int, request: Request) -> JSONResponse:
    owner = _owner_id(request)
    if not await deps.get_db().delete_user_model_group(owner, group_id):
        raise HTTPException(status_code=404, detail="group not found")
    return JSONResponse({"status": "ok"})
