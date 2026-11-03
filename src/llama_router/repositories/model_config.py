"""Prices, fallbacks, aliases and model groups."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..db_events import (
    CHANGE_ALIASES,
    CHANGE_FALLBACKS,
    CHANGE_GROUPS,
    CHANGE_PRICES,
    _notifies,
)
from ..db_sql import qmark, result_mappings
from ..models import ModelGroup, ModelGroupMember
from .base import _KEEP, Repository, _Keep


class ModelConfigRepository(Repository):
    async def list_model_prices(
        self, provider_id: int | None = None
    ) -> list[dict[str, Any]]:
        if provider_id is None:
            rows = await self._select_all(
                "SELECT * FROM model_prices ORDER BY provider_id, model_pattern"
            )
        else:
            rows = await self._select_all(
                "SELECT * FROM model_prices WHERE provider_id = ? "
                "ORDER BY model_pattern",
                (provider_id,),
            )
        return [dict(r) for r in rows]

    @_notifies(CHANGE_PRICES)
    async def set_model_price(
        self,
        provider_id: int,
        model_pattern: str,
        input_per_mtok: float,
        output_per_mtok: float,
    ) -> None:
        if self._dialect_name in ("mysql", "mariadb"):
            await self._execute(
                "INSERT INTO model_prices (provider_id, model_pattern, input_per_mtok, "
                "output_per_mtok) VALUES (?, ?, ?, ?) ON DUPLICATE KEY UPDATE "
                "input_per_mtok = VALUES(input_per_mtok), "
                "output_per_mtok = VALUES(output_per_mtok)",
                (provider_id, model_pattern, input_per_mtok, output_per_mtok),
            )
        else:
            await self._execute(
                "INSERT INTO model_prices (provider_id, model_pattern, input_per_mtok, "
                "output_per_mtok) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (provider_id, model_pattern) DO UPDATE SET "
                "input_per_mtok = ?, output_per_mtok = ?",
                (
                    provider_id,
                    model_pattern,
                    input_per_mtok,
                    output_per_mtok,
                    input_per_mtok,
                    output_per_mtok,
                ),
            )

    @_notifies(CHANGE_PRICES)
    async def remove_model_price(self, provider_id: int, price_id: int) -> bool:
        return bool(
            await self._execute_rowcount(
                "DELETE FROM model_prices WHERE id = ? AND provider_id = ?",
                (price_id, provider_id),
            )
        )

    @_notifies(CHANGE_FALLBACKS)
    async def set_model_fallback(self, model_name: str, fallback_model: str) -> None:
        if self._dialect_name in ("mysql", "mariadb"):
            await self._execute(
                "INSERT INTO model_fallbacks (model_name, fallback_model) "
                "VALUES (?, ?) "
                "ON DUPLICATE KEY UPDATE fallback_model = VALUES(fallback_model)",
                (model_name, fallback_model),
            )
        else:
            await self._execute(
                "INSERT INTO model_fallbacks (model_name, fallback_model) "
                "VALUES (?, ?) "
                "ON CONFLICT (model_name) DO UPDATE SET fallback_model = ?",
                (model_name, fallback_model, fallback_model),
            )

    @_notifies(CHANGE_FALLBACKS)
    async def remove_model_fallback(self, model_name: str) -> None:
        await self._execute(
            "DELETE FROM model_fallbacks WHERE model_name = ?", (model_name,)
        )

    async def get_model_fallback(self, model_name: str) -> str | None:
        row = await self._select_one(
            "SELECT fallback_model FROM model_fallbacks WHERE model_name = ?",
            (model_name,),
        )
        return row["fallback_model"] if row else None

    async def get_all_model_fallbacks(self) -> dict[str, str]:
        rows = await self._select_all(
            "SELECT model_name, fallback_model FROM model_fallbacks ORDER BY model_name"
        )
        return {r["model_name"]: r["fallback_model"] for r in rows}

    @_notifies(CHANGE_ALIASES)
    async def set_model_alias(self, alias: str, canonical: str) -> None:
        """Make *alias* another name for *canonical*."""
        if self._dialect_name in ("mysql", "mariadb"):
            await self._execute(
                "INSERT INTO model_aliases (alias, canonical) VALUES (?, ?) "
                "ON DUPLICATE KEY UPDATE canonical = VALUES(canonical)",
                (alias, canonical),
            )
        else:
            await self._execute(
                "INSERT INTO model_aliases (alias, canonical) VALUES (?, ?) "
                "ON CONFLICT (alias) DO UPDATE SET canonical = ?",
                (alias, canonical, canonical),
            )

    @_notifies(CHANGE_ALIASES)
    async def remove_model_alias(self, alias: str) -> bool:
        return bool(
            await self._execute_rowcount(
                "DELETE FROM model_aliases WHERE alias = ?", (alias,)
            )
        )

    async def get_model_aliases(self) -> dict[str, str]:
        """alias -> canonical name."""
        rows = await self._select_all(
            "SELECT alias, canonical FROM model_aliases ORDER BY canonical, alias"
        )
        return {r["alias"]: r["canonical"] for r in rows}

    async def list_model_groups(self) -> list[ModelGroup]:
        """Every group with its members, in name then position order."""
        groups = await self._select_all(
            "SELECT id, name, description FROM model_groups ORDER BY name"
        )
        members = await self._select_all(
            "SELECT id, group_id, model_name, provider_id, position "
            "FROM model_group_members ORDER BY group_id, position, id"
        )
        by_group: dict[int, list[ModelGroupMember]] = {}
        for m in members:
            by_group.setdefault(int(m["group_id"]), []).append(_row_to_member(m))
        return [
            ModelGroup(
                id=int(g["id"]),
                name=g["name"],
                description=g["description"],
                members=by_group.get(int(g["id"]), []),
            )
            for g in groups
        ]

    async def get_model_group(self, name: str) -> ModelGroup | None:
        for g in await self.list_model_groups():
            if g.name == name:
                return g
        return None

    @_notifies(CHANGE_GROUPS)
    async def create_model_group(
        self, name: str, description: str | None = None
    ) -> ModelGroup:
        gid = await self._insert_get_id(
            "INSERT INTO model_groups (name, description) VALUES (?, ?)",
            (name, description),
        )
        return ModelGroup(id=gid, name=name, description=description)

    @_notifies(CHANGE_GROUPS)
    async def update_model_group(
        self,
        name: str,
        *,
        description: str | _Keep | None = _KEEP,
        new_name: str | None = None,
    ) -> bool:
        """Rename the group and/or set its description. A field left out is
        unchanged; ``description=None`` clears it."""
        sets: list[str] = []
        params: list[Any] = []
        if not isinstance(description, _Keep):
            sets.append("description = ?")
            params.append(description)
        if new_name is not None and new_name != name:
            sets.append("name = ?")
            params.append(new_name)
        if not sets:
            return await self.get_model_group(name) is not None
        n = await self._execute_rowcount(
            f"UPDATE model_groups SET {', '.join(sets)} WHERE name = ?",  # noqa: S608 - fixed column names; values are bound
            (*params, name),
        )
        return n > 0

    @_notifies(CHANGE_GROUPS)
    async def delete_model_group(self, name: str) -> bool:
        row = await self._select_one(
            "SELECT id FROM model_groups WHERE name = ?", (name,)
        )
        if row is None:
            return False
        # Explicit, so MySQL without enforced foreign keys and SQLite with
        # them off end up the same.
        await self._execute(
            "DELETE FROM model_group_members WHERE group_id = ?", (row["id"],)
        )
        await self._execute("DELETE FROM model_groups WHERE id = ?", (row["id"],))
        return True

    @_notifies(CHANGE_GROUPS)
    async def add_model_group_member(
        self, group_id: int, model_name: str, provider_id: int | None = None
    ) -> ModelGroupMember:
        """Add a member, or return the existing one with the same model and
        provider."""
        rows = await self._select_all(
            "SELECT id, group_id, model_name, provider_id, position "
            "FROM model_group_members WHERE group_id = ?",
            (group_id,),
        )
        for r in rows:
            if r["model_name"] == model_name and r["provider_id"] == provider_id:
                return _row_to_member(r)
        position = max((int(r["position"]) for r in rows), default=-1) + 1
        mid = await self._insert_get_id(
            "INSERT INTO model_group_members "
            "(group_id, model_name, provider_id, position) VALUES (?, ?, ?, ?)",
            (group_id, model_name, provider_id, position),
        )
        return ModelGroupMember(
            id=mid,
            group_id=group_id,
            model_name=model_name,
            provider_id=provider_id,
            position=position,
        )

    @_notifies(CHANGE_GROUPS)
    async def remove_model_group_member(self, group_id: int, member_id: int) -> bool:
        n = await self._execute_rowcount(
            "DELETE FROM model_group_members WHERE id = ? AND group_id = ?",
            (member_id, group_id),
        )
        return n > 0

    @_notifies(CHANGE_GROUPS)
    async def set_model_group_members(
        self, group_id: int, members: list[tuple[str, int | None]]
    ) -> None:
        """Make the group's members exactly ``members`` (model, provider id)
        in one transaction. Members that stay keep their id and position; new
        ones go after them in the order given."""
        wanted = list(dict[str, Any].fromkeys(members))
        sf = self._require_session_factory()
        async with sf() as session, session.begin():
            st, bd = qmark(
                "SELECT id, model_name, provider_id, position "
                "FROM model_group_members WHERE group_id = ?",
                group_id,
            )
            rows = list(result_mappings(await session.execute(st, bd)))
            have = {(r["model_name"], r["provider_id"]): r for r in rows}
            for key, r in have.items():
                if key not in wanted:
                    st, bd = qmark(
                        "DELETE FROM model_group_members WHERE id = ?", r["id"]
                    )
                    await session.execute(st, bd)
            position = max((int(r["position"]) for r in rows), default=-1)
            for model_name, provider_id in wanted:
                if (model_name, provider_id) in have:
                    continue
                position += 1
                st, bd = qmark(
                    "INSERT INTO model_group_members "
                    "(group_id, model_name, provider_id, position) "
                    "VALUES (?, ?, ?, ?)",
                    group_id,
                    model_name,
                    provider_id,
                    position,
                )
                await session.execute(st, bd)

    async def resolve_fallback_chain(
        self, model_name: str, max_depth: int = 10
    ) -> list[str]:
        query = """
        WITH RECURSIVE chain(depth, resolved_name) AS (
            SELECT 0, ?
            UNION ALL
            SELECT c.depth + 1, mf.fallback_model
            FROM chain c
            INNER JOIN model_fallbacks mf ON mf.model_name = c.resolved_name
            WHERE c.depth < ? AND mf.fallback_model IS NOT NULL
        )
        SELECT resolved_name FROM chain ORDER BY depth
        """
        rows = await self._select_all(query, (model_name, max_depth))
        chain: list[str] = []
        seen: set[str] = set()
        for row in rows:
            name = row["resolved_name"]
            if name in seen:
                break
            seen.add(name)
            chain.append(name)
        return chain


def _row_to_member(row: Mapping[str, Any]) -> ModelGroupMember:
    pid = row["provider_id"]
    return ModelGroupMember(
        id=int(row["id"]),
        group_id=int(row["group_id"]),
        model_name=row["model_name"],
        provider_id=None if pid is None else int(pid),
        position=int(row["position"]),
    )
