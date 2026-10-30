"""Models a client can ask for: served models plus usable model groups."""

from __future__ import annotations

from typing import Any

from .database import Database
from .router import Router


async def listed_models(
    db: Database,
    router: Router | None,
    *,
    hub_only: bool = False,
    provider_id: int | None = None,
) -> list[dict[str, Any]]:
    """Rows for ``/api/tags`` and ``/v1/models``, sorted by name.

    Groups come from the current routing table, so listing them reads no
    extra rows. A group appears once at least one member can be served
    (for a hub, by a provider shared with it). Every group hides a served
    model of the same name, usable or not, because routing sends that
    name to the group. With *provider_id* (a hub asking about one provider)
    only that provider's models are listed, and no groups.
    """
    models = await db.list_all_models(
        hub_only=hub_only,
        exclude_ids=router.hub_loopback_ids() if hub_only and router else None,
        only_id=provider_id,
    )
    table = router.table if router is not None else None
    if table is None or not table.groups or provider_id is not None:
        return models
    rows = [m for m in models if m["name"] not in table.groups]
    rows += table.group_listing(hub_only)
    rows.sort(key=lambda m: str(m["name"]))
    return rows
