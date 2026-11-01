"""Management keys: tokens for the dashboard's management API.

An administrator issues a management key with a set of privileges. A
script sends it as ``Authorization: Bearer lrm_...`` to the dashboard port
and may then call the ``/api/...`` endpoints those privileges cover, such
as issuing and revoking API keys. Everything else is refused, including
the dashboard pages, user accounts and management keys themselves, so a
management key can never widen its own access.

Keys are random 256-bit tokens, so they are stored as a plain SHA-256: a
slow hash protects guessable passwords, not tokens like these.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from secrets import token_urlsafe
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from ..auth import key_prefix
from ..database import Database
from . import deps

logger = logging.getLogger(__name__)

KEY_PREFIX = "lrm_"
AUTH_SOURCE = "management_key"

# Privilege name -> what it allows, as shown in the dashboard. A ``:write``
# privilege includes the ``:read`` one of the same area.
PRIVILEGES: dict[str, str] = {
    "keys:read": "List API keys with their settings, pins and decision policies",
    "keys:write": "Issue, configure and revoke API keys",
    "models:read": "Read model aliases, fallbacks, groups and the routing table",
    "models:write": "Change model aliases, fallbacks and groups; rebuild routing",
    "status:read": "Read router status (GET /api/status)",
}

# Area -> path prefixes it covers. A prefix matches itself and anything
# below it ("/api/keys" covers "/api/keys/3/pins", not "/api/keysets").
_AREAS: dict[str, tuple[str, ...]] = {
    "keys": ("/api/keys",),
    "models": (
        "/api/model-aliases",
        "/api/fallbacks",
        "/api/model-groups",
        "/api/routing-table",
        "/api/routing/explain",
    ),
}
# Area -> exact paths, read only.
_EXACT_READ: dict[str, tuple[str, ...]] = {"status": ("/api/status",)}

_READ_METHODS = frozenset({"GET", "HEAD"})
_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

MAX_NAME_LENGTH = 64
MAX_EXPIRY_DAYS = 3650
# last_used_at is written at most this often per key.
_TOUCH_INTERVAL_S = 60.0
_touched: dict[int, float] = {}


@dataclass(frozen=True)
class ManagementKey:
    id: int
    name: str
    privileges: frozenset[str]


def generate_management_key() -> str:
    return KEY_PREFIX + token_urlsafe(32)


def management_key_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def bearer_token(request: Request) -> str | None:
    """The request's management key, if its Authorization header carries one."""
    raw = (request.headers.get("authorization") or "").strip()
    if raw[:7].lower() != "bearer ":
        return None
    token = raw[7:].strip()
    return token if token.startswith(KEY_PREFIX) else None


def required_privilege(method: str, path: str) -> str | None:
    """The privilege a management key needs for this request, or None if no
    management key may make it."""
    method = method.upper()
    if method in _READ_METHODS:
        kind = "read"
    elif method in _WRITE_METHODS:
        kind = "write"
    else:
        return None
    for area, prefixes in _AREAS.items():
        for prefix in prefixes:
            if path == prefix or path.startswith(prefix + "/"):
                return f"{area}:{kind}"
    if kind == "read":
        for area, paths in _EXACT_READ.items():
            if path in paths:
                return f"{area}:read"
    return None


def has_privilege(granted: frozenset[str], needed: str) -> bool:
    if needed in granted:
        return True
    area, _, kind = needed.partition(":")
    return kind == "read" and f"{area}:write" in granted


def parse_privileges(values: Any) -> list[str]:
    """Validate a list of privilege names from a request body."""
    if not isinstance(values, list) or not values:
        raise HTTPException(
            status_code=400, detail="privileges must be a non-empty list"
        )
    out: set[str] = set()
    for v in values:
        name = str(v).strip().lower()
        if name not in PRIVILEGES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown privilege {v!r}; known: " + ", ".join(PRIVILEGES),
            )
        out.add(name)
    return sorted(out)


async def authenticate(db: Database, token: str) -> ManagementKey | None:
    """The live (known, unexpired) management key for *token*."""
    row = await db.lookup_management_key(management_key_hash(token))
    if row is None:
        return None
    expires = row["expires_at"]
    if expires is not None and expires <= int(time.time()):
        return None
    key_id = int(row["id"])
    now = time.monotonic()
    if now - _touched.get(key_id, 0.0) >= _TOUCH_INTERVAL_S:
        _touched[key_id] = now
        await db.touch_management_key(key_id)
    return ManagementKey(
        id=key_id,
        name=str(row["name"]),
        privileges=frozenset(p for p in row["privileges"] if p in PRIVILEGES),
    )


# --- Administration (signed-in administrators only) ---

router = APIRouter()


def _require_admin_user(request: Request) -> str:
    """Return the administrator's username.

    Management keys are refused even though the middleware already keeps
    them away from these paths.
    """
    user = getattr(request.state, "dashboard_user", None)
    if user is None or not user.is_admin or user.auth_source == AUTH_SOURCE:
        raise HTTPException(status_code=403, detail="Forbidden")
    return str(user.username)


@router.get("/api/management-keys")
async def api_list_management_keys(request: Request) -> JSONResponse:
    _require_admin_user(request)
    keys = await deps.get_db().list_management_keys()
    return JSONResponse(
        jsonable_encoder({"keys": keys, "privileges": PRIVILEGES}),
    )


@router.post("/api/management-keys")
async def api_create_management_key(request: Request) -> JSONResponse:
    """Body: {name, privileges: [...], expires_in_days: int or null}.

    The key is in the response once and never again.
    """
    admin = _require_admin_user(request)
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="JSON body required") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    name = str(body.get("name") or "").strip()
    if not name or len(name) > MAX_NAME_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"name is required (at most {MAX_NAME_LENGTH} characters)",
        )
    privileges = parse_privileges(body.get("privileges"))
    expires_at = _expiry(body.get("expires_in_days"))
    plaintext = generate_management_key()
    key_id = await deps.get_db().create_management_key(
        name=name,
        key_prefix=key_prefix(plaintext),
        key_hash=management_key_hash(plaintext),
        privileges=privileges,
        created_by=admin,
        expires_at=expires_at,
    )
    logger.info(
        "Management key %d (%s) issued by %s with %s",
        key_id,
        name,
        admin,
        " ".join(privileges),
    )
    return JSONResponse(
        {
            "id": key_id,
            "name": name,
            "management_key": plaintext,
            "privileges": privileges,
            "expires_at": expires_at,
        }
    )


@router.delete("/api/management-keys/{key_id}")
async def api_delete_management_key(key_id: int, request: Request) -> JSONResponse:
    admin = _require_admin_user(request)
    if not await deps.get_db().delete_management_key(key_id):
        raise HTTPException(status_code=404, detail="Management key not found")
    _touched.pop(key_id, None)
    logger.info("Management key %d revoked by %s", key_id, admin)
    return JSONResponse({"ok": True})


def _expiry(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        days = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail="expires_in_days must be a whole number"
        ) from exc
    if not 1 <= days <= MAX_EXPIRY_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"expires_in_days must be between 1 and {MAX_EXPIRY_DAYS}",
        )
    return int(time.time()) + days * 86400
