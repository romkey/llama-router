"""Models a client can ask for: served models plus usable model groups."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .database import Database
from .models import ProviderStatus
from .router import Router
from .routing_table import RoutingInputs


async def listed_models(
    db: Database,
    router: Router | None,
    *,
    hub_only: bool = False,
    provider_id: int | None = None,
) -> list[dict[str, Any]]:
    """Rows for ``/api/tags`` and ``/v1/models``, sorted by name.

    Served models and groups come from the inputs of the current routing
    table, so listing reads no rows. A group appears once at least one member
    can be served (for a hub, by a provider shared with it). Every group
    hides a served model of the same name, usable or not, because routing
    sends that name to the group. With *provider_id* (a hub asking about one
    provider) only that provider's models are listed, and no groups.
    """
    exclude = router.hub_loopback_ids() if hub_only and router else None
    table = await router.current_table() if router is not None else None
    inputs = router.inputs if router is not None else None
    if inputs is None:
        models = await db.list_all_models(
            hub_only=hub_only, exclude_ids=exclude, only_id=provider_id
        )
    else:
        models = served_models(
            inputs, hub_only=hub_only, exclude_ids=exclude, only_id=provider_id
        )
    if table is None or not table.groups or provider_id is not None:
        return models
    rows = [m for m in models if m["name"] not in table.groups]
    rows += table.group_listing(hub_only)
    rows.sort(key=lambda m: str(m["name"]))
    return rows


def served_models(
    inputs: RoutingInputs,
    *,
    hub_only: bool = False,
    exclude_ids: Iterable[int] | None = None,
    only_id: int | None = None,
) -> list[dict[str, Any]]:
    """What ``Database.list_all_models`` returns, built from routing inputs:
    models of online providers, one row per name (the lowest model ID wins).
    """
    skip = set(exclude_ids or ())
    best: dict[str, Any] = {}
    for p in inputs.providers:
        if p.id is None or p.status == ProviderStatus.OFFLINE:
            continue
        if (hub_only and not p.hub_shared) or p.id in skip:
            continue
        if only_id is not None and p.id != only_id:
            continue
        for m in inputs.models.get(p.id, []):
            cur = best.get(m.name)
            if cur is None or (m.id or 0) < (cur.id or 0):
                best[m.name] = m
    return [
        {
            "name": m.name,
            "size": m.size,
            "digest": m.digest,
            "modified_at": m.modified_at,
            "details": m.details or {},
        }
        for _, m in sorted(best.items())
    ]
