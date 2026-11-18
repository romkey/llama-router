"""Private model groups dashboard users define for themselves.

A private group works like a router-wide model group (one name that routes
to the best of its members), but only for its owner: requests made with
the API keys they issued, and their dashboard chats. Its name only has to
be unique among the owner's own groups, and it wins over a model or
router-wide group of the same name for them alone.

An owner may share a group with every other user. Sharing alone changes
nothing for anyone: a user who opts in to a shared group can then use it,
just like one of their own (which still wins a name both have).

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
from ..routing_table import RoutingTable, member_label, user_group_key
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


def _member_rows(
    group: ModelGroup, table: RoutingTable | None, providers: dict[int, str]
) -> list[dict[str, Any]]:
    rows = []
    for m in group.members:
        provider = None if m.provider_id is None else providers.get(m.provider_id)
        rows.append(
            {
                "model_name": m.model_name,
                "provider_id": m.provider_id,
                "provider": provider,
                "label": member_label(m.model_name, provider),
                "available": _member_available(table, m.model_name, m.provider_id),
            }
        )
    return rows


def _choice(table: RoutingTable | None, key: str | None) -> str | None:
    """The balanced choice for a chat request to table key *key* right now;
    None until the routing table has the group, or nothing serves it."""
    if table is None or key is None:
        return None
    for ep in SHOWN_ENDPOINTS:
        ranked = table.ranked_for(key, ep, "balanced")
        if ranked:
            return f"{ranked[0].model} on {ranked[0].provider.name}"
    return None


def _shadows(table: RoutingTable | None, served: set[str], name: str) -> str | None:
    """What a private group called *name* hides for whoever uses it: a
    router-wide "group", a served "model", or nothing."""
    if table is not None and table.is_group(name):
        return "group"
    if name in served:
        return "model"
    return None


def _rows(
    groups: list[ModelGroup],
    table: RoutingTable | None,
    providers: dict[int, str],
    served: set[str],
    owner: int,
    opted_in: dict[int, int],
) -> list[dict[str, Any]]:
    mine = table.user_groups.get(owner, {}) if table is not None else {}
    return [
        {
            "id": g.id,
            "name": g.name,
            "description": g.description,
            "shared": g.shared,
            # How many other users opted in to it.
            "opted_in": opted_in.get(g.id or 0, 0) if g.shared else 0,
            "members": _member_rows(g, table, providers),
            "choice": _choice(table, mine.get(g.name)),
            # A router-wide group or a served model this group hides for
            # its owner, if any.
            "shadows": _shadows(table, served, g.name),
        }
        for g in groups
    ]


def _owner_names(users: list[dict[str, Any]]) -> dict[int, str]:
    return {
        int(u["id"]): str(u.get("display_name") or u.get("username") or "?")
        for u in users
    }


def _name(body: dict[str, Any], current: str | None = None) -> str:
    raw = body.get("name", current)
    name = str(raw or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    if len(name) > MAX_NAME:
        raise HTTPException(status_code=400, detail="name is too long")
    return name


def _shared(body: dict[str, Any], current: bool = False) -> bool:
    if "shared" not in body:
        return current
    value = body["shared"]
    if not isinstance(value, bool):
        raise HTTPException(status_code=400, detail="shared must be true or false")
    return value


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


def _summary(
    name: str, members: list[tuple[str, int | None]] | None, shared: bool
) -> str:
    text = f"name={name}"
    if members is not None:
        text += f" members={len(members)}"
    return text + f" shared={str(shared).lower()}"


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
    opted_in: dict[int, int] = {}
    for _user, gid in await db.list_user_model_group_subscriptions():
        opted_in[gid] = opted_in.get(gid, 0) + 1
    rows = _rows(
        await db.list_user_model_groups(owner),
        _table(),
        providers,
        served,
        owner,
        opted_in,
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
    """Body: {name, description?, shared?, members?: [{model_name, provider_id?}]}."""
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
    shared = _shared(body)
    group = await db.create_user_model_group(
        owner, name, _description(body), members, shared=shared
    )
    await audit.record(
        request,
        "my_model_group.create",
        ("my_model_group", group.id),
        _summary(name, members, shared),
    )
    return JSONResponse({"id": group.id, "name": name}, status_code=201)


@router.patch("/api/my-groups/{group_id:int}")
async def update_my_group(group_id: int, request: Request) -> JSONResponse:
    """Body: {name?, description?, shared?, members?}. Members, when given,
    replace the group's members in the order given. Unsharing a group drops
    everyone's opt-in to it."""
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
    shared = _shared(body, group.shared)
    await db.update_user_model_group(
        owner,
        group_id,
        name=name,
        description=description,
        members=members,
        shared=shared,
    )
    await audit.record(
        request,
        "my_model_group.update",
        ("my_model_group", group_id),
        _summary(name, members, shared),
    )
    return JSONResponse(
        {"id": group_id, "name": name, "description": description, "shared": shared}
    )


@router.delete("/api/my-groups/{group_id:int}")
async def delete_my_group(group_id: int, request: Request) -> JSONResponse:
    owner = _owner_id(request)
    if not await deps.get_db().delete_user_model_group(owner, group_id):
        raise HTTPException(status_code=404, detail="group not found")
    return JSONResponse({"status": "ok"})


def _shared_by_others(groups: list[ModelGroup], user: int) -> dict[int, ModelGroup]:
    return {
        g.id: g
        for g in groups
        if g.shared and g.id is not None and g.owner_user_id != user
    }


@router.get("/api/shared-groups")
async def list_shared_groups(request: Request) -> JSONResponse:
    """Other users' shared groups, whether the user opted in to each, and
    what using one would change for them."""
    user = _owner_id(request)
    db = deps.get_db()
    groups = await db.list_user_model_groups()
    shared = _shared_by_others(groups, user)
    own_names = {g.name for g in groups if g.owner_user_id == user}
    subscribed = [gid for _u, gid in await db.list_user_model_group_subscriptions(user)]
    mine = set(subscribed)
    # Name -> the opted-in shared group that has it for the user: the
    # earliest opt-in, as in the routing table.
    winners: dict[str, int] = {}
    for gid in subscribed:
        if gid in shared:
            winners.setdefault(shared[gid].name, gid)
    owners = _owner_names(await db.list_dashboard_users())
    table = _table()
    served = await _served_names(db)
    providers = await _provider_names(db)
    rows = []
    for gid, g in sorted(
        shared.items(),
        key=lambda kv: (kv[1].name, owners.get(kv[1].owner_user_id or 0, "")),
    ):
        key = user_group_key(g.owner_user_id or 0, g.name)
        opted = gid in mine
        # Which of the user's groups has this name instead, if any: their
        # own, or another shared group they opted in to. Either way opting
        # in to this one is refused, or (if already in) it isn't used.
        hidden_by = None
        if g.name in own_names:
            hidden_by = "own"
        elif winners.get(g.name, gid) != gid:
            hidden_by = "shared"
        rows.append(
            {
                "id": gid,
                "name": g.name,
                "description": g.description,
                "owner": owners.get(g.owner_user_id or 0, "?"),
                "members": _member_rows(g, table, providers),
                "opted_in": opted,
                "choice": _choice(table, key) if opted and hidden_by is None else None,
                "shadows": _shadows(table, served, g.name),
                "hidden_by": hidden_by,
            }
        )
    return JSONResponse({"groups": rows})


@router.post("/api/shared-groups/{group_id:int}/opt-in")
async def opt_in_shared_group(group_id: int, request: Request) -> JSONResponse:
    """Use another user's shared group. Refused when the user already has a
    group of that name, their own or another shared one they opted in to."""
    user = _owner_id(request)
    db = deps.get_db()
    groups = await db.list_user_model_groups()
    group = _shared_by_others(groups, user).get(group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="shared group not found")
    if any(g.owner_user_id == user and g.name == group.name for g in groups):
        raise HTTPException(
            status_code=409, detail=f"you already have a group {group.name}"
        )
    by_id = {g.id: g for g in groups}
    for _u, gid in await db.list_user_model_group_subscriptions(user):
        other = by_id.get(gid)
        if gid != group_id and other is not None and other.name == group.name:
            raise HTTPException(
                status_code=409,
                detail=f"you already use a shared group {group.name}",
            )
    if not await db.subscribe_user_model_group(user, group_id):
        raise HTTPException(status_code=404, detail="shared group not found")
    await audit.record(
        request,
        "my_model_group.opt_in",
        ("my_model_group", group_id),
        f"name={group.name}",
    )
    return JSONResponse({"id": group_id, "opted_in": True})


@router.delete("/api/shared-groups/{group_id:int}/opt-in")
async def opt_out_shared_group(group_id: int, request: Request) -> JSONResponse:
    user = _owner_id(request)
    await deps.get_db().unsubscribe_user_model_group(user, group_id)
    return JSONResponse({"id": group_id, "opted_in": False})
