"""Audit tab: the trail of state-changing dashboard actions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .. import deps, management_keys
from ..pages import dash_page_ctx, templates

router = APIRouter()

_FILTER_PARAMS = (
    "actor_kind",
    "actor_name",
    "action_prefix",
    "target_kind",
    "target_id",
)


def _admin_or_auditor(request: Request) -> None:
    """Admins, and management keys with ``audit:read`` (the middleware has
    already checked the privilege for those)."""
    user = getattr(request.state, "dashboard_user", None)
    if user is None:
        raise HTTPException(status_code=403, detail="Forbidden")
    if user.is_admin or user.auth_source == management_keys.AUTH_SOURCE:
        return
    raise HTTPException(status_code=403, detail="Forbidden")


async def _entries(request: Request) -> list[dict[str, Any]]:
    q = request.query_params
    filters = {name: q.get(name) for name in _FILTER_PARAMS}
    since = None
    days = q.get("days")
    if days:
        try:
            since = datetime.now(UTC) - timedelta(days=float(days))
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="days must be a number"
            ) from exc
    try:
        limit = int(q.get("limit") or 200)
        before = int(q["before"]) if q.get("before") else None
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="limit and before must be numbers"
        ) from exc
    return await deps.get_db().list_audit_log(
        limit=limit, before_id=before, since=since, **filters
    )


@router.get("/api/audit")
async def api_audit(request: Request) -> JSONResponse:
    """Audit entries, newest first. Query: actor_kind, actor_name,
    action_prefix, target_kind, target_id, days, limit (max 1000), before
    (an entry id, for the next page)."""
    _admin_or_auditor(request)
    return JSONResponse(jsonable_encoder({"entries": await _entries(request)}))


@router.get("/audit", response_class=HTMLResponse)
async def audit_page(request: Request) -> Response:
    user = getattr(request.state, "dashboard_user", None)
    if user is None or not user.is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    return templates.TemplateResponse(
        request,
        "audit.html",
        await dash_page_ctx(
            request,
            {
                "entries": await _entries(request),
                "filters": {
                    n: request.query_params.get(n) or ""
                    for n in (*_FILTER_PARAMS, "days")
                },
            },
        ),
    )
