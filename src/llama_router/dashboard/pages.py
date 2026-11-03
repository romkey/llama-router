"""Templates, page context and helpers shared by the dashboard's pages."""

from __future__ import annotations

import time
import weakref
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import ParseResult, quote, urlparse

from fastapi import HTTPException, Request
from fastapi.responses import (
    RedirectResponse,
)
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from .. import __version__
from ..database import Database
from . import deps

# Counting the whole request log is a full scan, and the dashboard asks on
# every refresh; a page count up to this stale is fine.
LOG_COUNT_TTL = 30.0
_log_counts: weakref.WeakKeyDictionary[
    Database, dict[int | None, tuple[int, float]]
] = weakref.WeakKeyDictionary()


async def _log_count(db: Database, owner_user_id: int | None = None) -> int:
    """db.count_request_logs, cached for LOG_COUNT_TTL seconds."""
    counts = _log_counts.setdefault(db, {})
    now = time.monotonic()
    hit = counts.get(owner_user_id)
    if hit is not None and hit[1] > now:
        return hit[0]
    value = await db.count_request_logs(owner_user_id=owner_user_id)
    counts[owner_user_id] = (value, now + LOG_COUNT_TTL)
    return value


def _prune_completed(store: dict[str, Any], max_age_seconds: float = 600) -> None:
    now = time.monotonic()
    stale = [
        k
        for k, v in store.items()
        if v.get("status") in ("done", "failed")
        and v.get("completed_at") is not None
        and now - v["completed_at"] > max_age_seconds
    ]
    for k in stale:
        del store[k]


def _ollama_library_slug(model_name: str) -> str:
    """Extract an Ollama library slug from raw model names/URLs.

    Handles cache/registry-prefixed names like
    ``host:9200/library/llama3.2:latest`` by removing host and ``library/``
    prefix, then stripping tag/digest for a stable library page URL.
    """
    name = model_name.strip()
    if not name:
        return ""

    if "://" in name:
        name = name.split("://", 1)[1]

    parts = name.split("/")
    if len(parts) > 1:
        first = parts[0]
        if "." in first or ":" in first or first == "localhost":
            name = "/".join(parts[1:])

    if name.startswith("library/"):
        name = name[len("library/") :]

    if "@" in name:
        name = name.split("@", 1)[0]

    slash_idx = name.rfind("/")
    colon_idx = name.rfind(":")
    if colon_idx > slash_idx:
        name = name[:colon_idx]

    return name


def _ollama_library_url(model_name: str) -> str:
    """Build an Ollama model library URL from a model name."""
    slug = _ollama_library_slug(model_name)
    if not slug:
        return "https://ollama.com/library"
    return f"https://ollama.com/library/{quote(slug, safe='/')}"


def _parse_utc(value: str | datetime | None) -> datetime | str | None:
    """A stored timestamp as an aware UTC datetime; text it cannot read as is."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        for pattern in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
        ):
            try:
                return datetime.strptime(value, pattern).replace(tzinfo=UTC)
            except ValueError:
                continue
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value


def _localtime(value: str | datetime | None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Jinja2 filter: convert a UTC timestamp to the local timezone (honours TZ)."""
    parsed = _parse_utc(value)
    if parsed is None:
        return "—"
    if isinstance(parsed, str):
        return parsed
    return parsed.astimezone().strftime(fmt)


def _time_tag(value: str | datetime | None, relative: bool = False) -> Markup | str:
    """Jinja2 filter: a ``<time>`` the page shows as ``YYYY-MM-DD HH:MM``.

    base.html rewrites it in the browser's zone (or as "3 days ago" when
    *relative*), with the full timestamp in its title. The text here, in the
    server's zone, is what shows without JavaScript.
    """
    parsed = _parse_utc(value)
    if parsed is None:
        return "—"
    if isinstance(parsed, str):
        return parsed
    local = parsed.astimezone()
    return Markup('<time datetime="{}" data-local{} title="{}">{}</time>').format(
        parsed.astimezone(UTC).isoformat(),
        Markup(" data-relative") if relative else "",
        local.strftime("%Y-%m-%d %H:%M:%S %Z"),
        local.strftime("%Y-%m-%d %H:%M"),
    )


_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
templates.env.globals["version"] = __version__
templates.env.globals["ollama_library_url"] = _ollama_library_url
templates.env.filters["localtime"] = _localtime
templates.env.filters["when"] = _time_tag


def template_auth_ctx(request: Request) -> dict[str, Any]:
    user = getattr(request.state, "dashboard_user", None)
    if user is None:
        return {
            "read_only": True,
            "show_api_keys": False,
            "show_workers": False,
            "show_log": False,
            "show_cache": False,
            "show_users_tab": False,
            "dash_user": None,
        }
    adm = user.is_admin
    return {
        "read_only": not adm,
        "show_api_keys": adm,
        "show_workers": adm,
        "show_log": True,
        "show_cache": adm,
        "show_users_tab": adm,
        "dash_user": user,
    }


def merge_dash_template_ctx(request: Request, ctx: dict[str, Any]) -> dict[str, Any]:
    merged = template_auth_ctx(request)
    merged.update(ctx)
    merged["show_cache"] = merged["show_cache"] and deps.get_cache() is not None
    return merged


async def dash_page_ctx(request: Request, ctx: dict[str, Any]) -> dict[str, Any]:
    """merge_dash_template_ctx plus the Log tab's count, which every page shows."""
    merged = merge_dash_template_ctx(request, ctx)
    if merged["show_log"] and "log_total" not in merged:
        user = merged["dash_user"]
        owner = None if user.is_admin else user.id
        merged["log_total"] = await _log_count(deps.get_db(), owner)
    return merged


def _provider_url(provider_id: int, query: str = "") -> str:
    """Same-site URL for a provider detail page; the id only ever forms the suffix."""
    out = "/providers/" + str(int(provider_id))
    if query:
        out += "?" + query
    return out


def _provider_error(provider_id: int, msg: str) -> RedirectResponse:
    """Back to a provider's page with *msg* shown, after a failed form."""
    return RedirectResponse(
        url=_provider_url(provider_id, "error=" + quote(msg, safe="")),
        status_code=303,
    )


def _same_origin_referer_path(request: Request, fallback: str) -> str:
    raw = (request.headers.get("referer") or "").strip()
    if not raw:
        return fallback
    try:
        ref = urlparse(raw)
        base = urlparse(str(request.base_url))
    except ValueError:
        return fallback
    if ref.scheme.lower() not in ("http", "https"):
        return fallback

    def _host_port(u: ParseResult) -> tuple[str, int]:
        host = (u.hostname or "").lower()
        port = u.port
        if port is None:
            port = 443 if u.scheme.lower() == "https" else 80
        return host, port

    try:
        rh, rp = _host_port(ref)
        bh, bp = _host_port(base)
    except ValueError:  # a port that is not a number
        return fallback
    if not rh or rh != bh or rp != bp:
        return fallback
    path = ref.path or "/"
    if not path.startswith("/") or path.startswith("//"):
        return fallback
    if ref.query:
        return f"{path}?{ref.query}"
    return path


def _require_admin(request: Request) -> None:
    user = getattr(request.state, "dashboard_user", None)
    if not user or not user.is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")


def _is_admin(request: Request) -> bool:
    user = getattr(request.state, "dashboard_user", None)
    return bool(user and user.is_admin)
