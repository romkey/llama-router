"""Models a client can ask for: served models plus usable model groups."""

from __future__ import annotations

from typing import Any

from .database import Database
from .router import Router


async def listed_models(
    db: Database, router: Router | None, *, hub_only: bool = False
) -> list[dict[str, Any]]:
    """Rows for ``/api/tags`` and ``/v1/models``, sorted by name.

    Groups come from the current routing table, so listing them reads no
    extra rows. A group appears once at least one member can be served
    (for a hub, by a provider shared with it). A group hides a served
    model of the same name, as routing does.
    """
    models = await db.list_all_models(hub_only=hub_only)
    table = router.table if router is not None else None
    if table is None:
        return models
    groups = table.group_listing(hub_only)
    if not groups:
        return models
    names = {g["name"] for g in groups}
    rows = [m for m in models if m["name"] not in names] + groups
    rows.sort(key=lambda m: str(m["name"]))
    return rows
