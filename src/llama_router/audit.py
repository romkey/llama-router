"""The audit trail: who changed what in the dashboard, and when.

Every state-changing dashboard route writes one entry. :data:`ACTIONS`
names each route's action and what it acts on; :class:`AuditMiddleware`
writes the entry once the route has answered successfully, unless the route
wrote a more detailed one itself with :func:`record` (old and new values from
:func:`changes`). Routes that change nothing an admin needs to see (signing
in, the user's own chats) are in :data:`EXEMPT`. A test walks every non-GET
dashboard route and fails on one in neither.

Only the dashboard app records; the API apps never do. Summaries name fields
with their old and new values, but never a secret: keys, tokens, passwords
and provider credentials are recorded as "changed".
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any
from urllib.parse import parse_qsl

from fastapi import Request

from .client_address import client_ip
from .database import Database

logger = logging.getLogger(__name__)

ACTOR_USER = "user"
ACTOR_MANAGEMENT_KEY = "management_key"
ACTOR_SYSTEM = "system"

# Field names whose values are never written to the audit trail.
_SECRET_PARTS = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "credential",
    "key_hash",
    "private",
    "recovery",
    # Provider headers may carry credentials (Authorization: ...).
    "header",
)
# Bodies larger than this are not read for the generic summary.
_MAX_BODY = 64 * 1024
_MAX_SUMMARY = 2000
_STATE_KEY = "audit_recorded"

# "module.function" of each state-changing route -> (action, target kind).
# The target id is the route's first path parameter, when it has one.
ACTIONS: dict[str, tuple[str, str | None]] = {
    # Users and sign-in security
    "users.users_add": ("user.create", "user"),
    "users.users_delete": ("user.delete", "user"),
    "users.users_set_password": ("user.password", "user"),
    "users.account_change_password": ("account.password", "user"),
    "users.users_approve": ("user.approve", "user"),
    "users.users_unapprove": ("user.unapprove", "user"),
    "users.users_limits": ("user.limits", "user"),
    "users.users_quota_defaults": ("system.user_quota_defaults", "system"),
    "users.users_pause_keys": ("user.pause_keys", "user"),
    "users.users_sign_out": ("user.sign_out", "user"),
    "login.logout_everywhere": ("account.sign_out_everywhere", "user"),
    "mfa.registration_verify": ("account.second_factor_add", "user"),
    "mfa.regenerate_codes": ("account.recovery_codes", "user"),
    "mfa.remove_credential": ("account.second_factor_remove", "user"),
    # API keys
    "api_keys.api_set_lifecycle_settings": ("system.key_lifecycle", "system"),
    "api_keys.api_renew_key": ("api_key.renew", "api_key"),
    "api_keys.api_approve_renewal": ("api_key.renewal_approve", "api_key"),
    "api_keys.api_deny_renewal": ("api_key.renewal_deny", "api_key"),
    "api_keys.api_pause_key": ("api_key.pause", "api_key"),
    "api_keys.api_unpause_key": ("api_key.unpause", "api_key"),
    "api_keys.api_generate_key": ("api_key.create", "api_key"),
    "api_keys.api_update_key": ("api_key.update", "api_key"),
    "api_keys.api_regenerate_key": ("api_key.regenerate", "api_key"),
    "api_keys.api_set_key_spend_cap": ("api_key.update", "api_key"),
    "api_keys.api_set_key_locality": ("api_key.update", "api_key"),
    "api_keys.api_delete_key": ("api_key.delete", "api_key"),
    "api_keys.api_set_key_pin": ("api_key.pin_set", "api_key"),
    "api_keys.api_delete_key_pin": ("api_key.pin_remove", "api_key"),
    "api_keys.api_set_allow_unauthenticated": ("system.anonymous_access", "system"),
    "decisions.api_set_key_policy": ("api_key.policy_set", "api_key"),
    "decisions.api_delete_key_policy": ("api_key.policy_remove", "api_key"),
    "my_keys.issue_my_key": ("api_key.create", "api_key"),
    "my_keys.edit_my_key": ("api_key.update", "api_key"),
    "my_keys.renew_my_key": ("api_key.renew", "api_key"),
    "my_keys.pause_my_key": ("api_key.pause", "api_key"),
    "my_keys.unpause_my_key": ("api_key.unpause", "api_key"),
    "my_keys.set_my_key_pin": ("api_key.pin_set", "api_key"),
    "my_keys.delete_my_key_pin": ("api_key.pin_remove", "api_key"),
    "my_keys.regenerate_my_key": ("api_key.regenerate", "api_key"),
    "my_keys.revoke_my_key": ("api_key.delete", "api_key"),
    "my_keys.set_self_service_defaults": ("system.self_service_defaults", "system"),
    # Management keys
    "management_keys.api_create_management_key": (
        "management_key.create",
        "management_key",
    ),
    "management_keys.api_delete_management_key": (
        "management_key.delete",
        "management_key",
    ),
    # Providers and prices
    "providers.api_set_price": ("provider.price_set", "provider"),
    "providers.api_delete_price": ("provider.price_remove", "provider"),
    "providers.add_provider": ("provider.create", "provider"),
    "providers.api_probe_provider": ("provider.probe", "provider"),
    "providers.edit_provider": ("provider.update", "provider"),
    "providers.remove_provider": ("provider.delete", "provider"),
    "providers.refresh_provider": ("provider.refresh", "provider"),
    "providers.delete_model": ("provider.model_delete", "provider"),
    "providers.add_address": ("provider.address_add", "provider"),
    "providers.save_provider_address": ("provider.address_save", "provider"),
    "providers.edit_address": ("provider.address_update", "provider"),
    "providers.remove_address": ("provider.address_remove", "provider"),
    "providers.toggle_preferred": ("provider.address_preferred", "provider"),
    # Benchmarks, pulls and the cache
    "benchmarks.api_start_benchmark": ("benchmark.start", "provider"),
    "benchmarks.benchmark_model": ("benchmark.start", "provider"),
    "benchmarks.delete_benchmark": ("benchmark.delete", "benchmark"),
    "benchmarks.delete_benchmarks_for_model": ("benchmark.delete_model", "model"),
    "benchmarks.delete_all_benchmarks": ("benchmark.delete_all", "system"),
    "benchmarks.api_auto_benchmark_configure": ("system.auto_benchmark", "system"),
    "benchmarks.api_auto_benchmark_check": ("benchmark.auto_check", "system"),
    "benchmarks.api_fill_missing_benchmarks": ("benchmark.fill_missing", "system"),
    "pulls.api_pull_model": ("model.pull", "provider"),
    "pulls.api_hf_pull": ("model.pull", "provider"),
    "pulls.api_fill_missing_pulls": ("model.pull_missing", "system"),
    "pulls.pull_model_legacy": ("model.pull", "provider"),
    "pulls.pull_model_all_legacy": ("model.pull_all", "model"),
    "cache.api_cache_toggle": ("system.registry_cache", "system"),
    "cache.api_cache_clear": ("cache.clear", "system"),
    "cache.api_cache_model": ("cache.model", "model"),
    # Models: fallbacks, aliases, groups, routing
    "model_aliases.set_fallback": ("model.fallback_set", "model"),
    "model_aliases.remove_fallback": ("model.fallback_remove", "model"),
    "model_aliases.set_model_alias": ("model.alias_set", "model"),
    "model_aliases.delete_model_alias": ("model.alias_remove", "model"),
    "model_aliases.delete_model_all_providers": ("model.delete_everywhere", "model"),
    "model_groups.create_group": ("model_group.create", "model_group"),
    "model_groups.update_group": ("model_group.update", "model_group"),
    "model_groups.delete_group": ("model_group.delete", "model_group"),
    "model_groups.add_member": ("model_group.member_add", "model_group"),
    "model_groups.set_members": ("model_group.members_set", "model_group"),
    "model_groups.remove_member": ("model_group.member_remove", "model_group"),
    "routing.api_rebuild_routing_table": ("routing.rebuild", "system"),
    "decisions.api_set_calibration": ("decision.calibration_set", "calibration"),
    "decisions.api_fit_calibration": ("decision.calibration_fit", "calibration"),
    "decisions.api_delete_calibration": ("decision.calibration_remove", "calibration"),
    # Workers and their tokens
    "workers.api_create_worker": ("worker.create", "worker"),
    "workers.api_rotate_worker_token": ("worker.token_rotate", "worker"),
    "workers.api_set_worker_enabled": ("worker.enabled", "worker"),
    "workers.api_delete_worker": ("worker.delete", "worker"),
    "workers.api_set_hub_providers": ("worker_link.shared_providers", "system"),
    "workers.api_new_router_id": ("system.router_id", "system"),
    "workers.api_worker_link_expose": ("worker_link.expose", "system"),
    "workers.api_worker_link_save": ("worker_link.update", "system"),
}

# Non-GET routes that write no audit entry, and why.
EXEMPT: dict[str, str] = {
    "login.login_submit": "signing in",
    "login.logout_post": "signing out",
    "mfa.registration_options": "a challenge; registration_verify records",
    "mfa.authentication_options": "signing in",
    "mfa.authentication_verify": "signing in",
    "mfa.recovery_login": "signing in",
    "providers.api_probe_url": "reads a URL; changes nothing",
    "pulls.api_hf_files": "lists a repository's files",
    "decisions.api_try_decision": "asks a model; changes nothing",
    "chat.send": "the user's own chat",
    "chat.stream": "the user's own chat",
    "chat.delete_conversation": "the user's own chat",
}


def route_name(endpoint: Any) -> str:
    module = str(getattr(endpoint, "__module__", "")).rsplit(".", 1)[-1]
    return f"{module}.{getattr(endpoint, '__name__', '')}"


def is_secret(name: str) -> bool:
    lowered = name.lower()
    if lowered.endswith("_limit"):
        return False  # a usage cap (daily_token_limit), not a credential
    return any(part in lowered for part in _SECRET_PARTS)


def _show(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, default=str, sort_keys=True)
    else:
        text = str(value)
    return text if len(text) <= 120 else text[:117] + "..."


def changes(
    old: Mapping[str, Any] | None,
    new: Mapping[str, Any],
    fields: Iterable[str] | None = None,
) -> str:
    """``field: old -> new`` for each of *fields* (default: *new*'s) whose
    value changed; secrets as ``field: changed``."""
    old = old or {}
    parts = []
    for name in fields if fields is not None else new:
        if name not in new:
            continue
        before, after = old.get(name), new[name]
        if before == after:
            continue
        if is_secret(name):
            parts.append(f"{name}: changed")
        elif name not in old:
            parts.append(f"{name}: {_show(after)}")
        else:
            parts.append(f"{name}: {_show(before)} -> {_show(after)}")
    return "; ".join(parts) or "no change"


def fields_summary(values: Mapping[str, Any]) -> str:
    """``field: value`` for each submitted field, secrets as "changed"."""
    parts = []
    for name, value in values.items():
        if is_secret(name):
            if value not in (None, "", [], {}):
                parts.append(f"{name}: changed")
        else:
            parts.append(f"{name}: {_show(value)}")
    return "; ".join(parts)


def _actor(state: Any) -> tuple[str, int | None, str | None]:
    user = getattr(state, "dashboard_user", None)
    key = getattr(state, "management_key", None)
    if key is not None:
        return ACTOR_MANAGEMENT_KEY, int(key.id), str(key.name)
    if user is None:
        return ACTOR_SYSTEM, None, None
    return ACTOR_USER, int(user.id) or None, str(user.username)


async def record(
    request: Request | None,
    action: str,
    target: tuple[str | None, Any] = (None, None),
    summary: str | None = None,
    *,
    db: Database | None = None,
) -> None:
    """Write one audit entry for *request*'s actor (the router itself
    without a request). *target* is (kind, id). Never raises: a failed write
    is logged, and the action it records still stands."""
    if db is None:
        from .dashboard import deps

        db = deps.get_db()
    if request is not None:
        state = request.state
        kind, actor_id, name = _actor(state)
        ip = client_ip(request)
        setattr(state, _STATE_KEY, True)
    else:
        kind, actor_id, name, ip = ACTOR_SYSTEM, None, "router", None
    target_kind, target_id = target
    text = (summary or "")[:_MAX_SUMMARY] or None
    try:
        await db.add_audit_entry(
            actor_kind=kind,
            actor_id=actor_id,
            actor_name=name,
            action=action,
            target_kind=target_kind,
            target_id=None if target_id is None else str(target_id),
            summary=text,
            client_ip=ip,
        )
    except Exception:
        logger.exception("Could not write audit entry %s", action)


def _body_fields(body: bytes, content_type: str) -> dict[str, Any]:
    if not body or len(body) > _MAX_BODY:
        return {}
    try:
        if "json" in content_type:
            value = json.loads(body)
            return dict(value) if isinstance(value, dict) else {}
        if "x-www-form-urlencoded" in content_type:
            return dict(parse_qsl(body.decode(), keep_blank_values=True))
    except ValueError, UnicodeDecodeError:
        return {}
    return {}


class AuditMiddleware:
    """Writes the audit entry of a successful state-changing request whose
    route did not write its own. Pure ASGI: it neither buffers responses nor
    wraps streams; it keeps a copy of a small request body for the
    summary."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(
        self, scope: MutableMapping[str, Any], receive: Any, send: Any
    ) -> None:
        if scope["type"] != "http" or scope.get("method") in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return
        chunks: list[bytes] = []
        size = 0
        status = 0

        async def keep(message: Any = None) -> Any:
            nonlocal size
            message = await receive()
            if message.get("type") == "http.request" and size <= _MAX_BODY:
                body = message.get("body", b"")
                size += len(body)
                if size <= _MAX_BODY:
                    chunks.append(body)
            return message

        failed_form = False

        async def watch(message: Any) -> None:
            nonlocal status, failed_form
            if message.get("type") == "http.response.start":
                status = int(message.get("status", 0))
                for name, value in message.get("headers") or ():
                    # Forms answer a refused change by redirecting with
                    # ?error=...: nothing changed.
                    if name.lower() == b"location" and b"error=" in value:
                        failed_form = True
            await send(message)

        await self.app(scope, keep, watch)
        if not 200 <= status < 400 or failed_form:
            return
        if scope.get("endpoint") is None:
            return  # turned away before routing (sign-in, forbidden)
        state = scope.get("state")
        if isinstance(state, Mapping) and state.get(_STATE_KEY):
            return
        name = route_name(scope.get("endpoint"))
        found = ACTIONS.get(name)
        if found is None:
            if name not in EXEMPT:
                logger.warning("No audit action for dashboard route %s", name)
            return
        action, target_kind = found
        params = scope.get("path_params") or {}
        target_id = next(iter(params.values()), None) if params else None
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", ())}
        fields = _body_fields(b"".join(chunks), headers.get("content-type", ""))
        summary = fields_summary(fields)
        path = scope.get("path", "")
        text = f"{scope.get('method')} {path}" + (f": {summary}" if summary else "")
        await record(Request(scope), action, (target_kind, target_id), text)
