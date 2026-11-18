"""Private model groups: groups a dashboard user defines for their own keys.

Every method that changes one group takes the owner too, so a user can only
ever change their own groups, whatever id they send. An owner may share a
group; other users then opt in to it (a subscription) to use it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..db_events import CHANGE_USER_GROUPS, _notifies
from ..db_sql import qmark, result_first
from ..models import ModelGroup, ModelGroupMember
from .base import Repository

_MEMBER_COLUMNS = "m.id, m.group_id, m.model_name, m.provider_id, m.position"


class UserGroupRepository(Repository):
    async def list_user_model_groups(
        self, owner_user_id: int | None = None
    ) -> list[ModelGroup]:
        """One user's groups (every user's when *owner_user_id* is None),
        with their members, in owner, name then position order."""
        where = "" if owner_user_id is None else " WHERE owner_user_id = ?"
        member_where = "" if owner_user_id is None else " WHERE g.owner_user_id = ?"
        params: tuple[Any, ...] = () if owner_user_id is None else (owner_user_id,)
        groups = await self._select_all(
            "SELECT id, owner_user_id, name, description, shared FROM user_model_groups"  # noqa: S608 - fixed SQL fragments; values are bound
            f"{where} ORDER BY owner_user_id, name",
            params,
        )
        members = await self._select_all(
            f"SELECT {_MEMBER_COLUMNS} FROM user_model_group_members m "  # noqa: S608 - fixed SQL fragments; values are bound
            f"JOIN user_model_groups g ON g.id = m.group_id{member_where} "
            "ORDER BY m.group_id, m.position, m.id",
            params,
        )
        by_group: dict[int, list[ModelGroupMember]] = {}
        for m in members:
            by_group.setdefault(int(m["group_id"]), []).append(_row_to_member(m))
        return [
            ModelGroup(
                id=int(g["id"]),
                name=g["name"],
                description=g["description"],
                owner_user_id=int(g["owner_user_id"]),
                shared=bool(g["shared"]),
                members=by_group.get(int(g["id"]), []),
            )
            for g in groups
        ]

    async def get_user_model_group(
        self, owner_user_id: int, group_id: int
    ) -> ModelGroup | None:
        for g in await self.list_user_model_groups(owner_user_id):
            if g.id == group_id:
                return g
        return None

    async def count_user_model_groups(self, owner_user_id: int) -> int:
        row = await self._select_one(
            "SELECT COUNT(*) AS n FROM user_model_groups WHERE owner_user_id = ?",
            (owner_user_id,),
        )
        return int(row["n"]) if row else 0

    @_notifies(CHANGE_USER_GROUPS)
    async def create_user_model_group(
        self,
        owner_user_id: int,
        name: str,
        description: str | None = None,
        members: list[tuple[str, int | None]] | None = None,
        *,
        shared: bool = False,
    ) -> ModelGroup:
        """Create the group with *members* (model, provider id) in one
        transaction, so the router never builds a table with half of it."""
        wanted = list(dict.fromkeys(members or []))
        sf = self._require_session_factory()
        async with sf() as session, session.begin():
            st, bd = qmark(
                "INSERT INTO user_model_groups "
                "(owner_user_id, name, description, shared) VALUES (?, ?, ?, ?)",
                owner_user_id,
                name,
                description,
                int(shared),
            )
            await session.execute(st, bd)
            st, bd = qmark(
                "SELECT id FROM user_model_groups WHERE owner_user_id = ? AND name = ?",
                owner_user_id,
                name,
            )
            row = result_first(await session.execute(st, bd))
            if row is None:  # pragma: no cover - the insert just made it
                raise RuntimeError("private model group was not created")
            gid = int(row["id"])
            for position, (model_name, provider_id) in enumerate(wanted):
                st, bd = qmark(
                    "INSERT INTO user_model_group_members "
                    "(group_id, model_name, provider_id, position) VALUES (?, ?, ?, ?)",
                    gid,
                    model_name,
                    provider_id,
                    position,
                )
                await session.execute(st, bd)
        return ModelGroup(
            id=gid,
            name=name,
            description=description,
            owner_user_id=owner_user_id,
            shared=shared,
            members=[
                ModelGroupMember(group_id=gid, model_name=m, provider_id=p, position=i)
                for i, (m, p) in enumerate(wanted)
            ],
        )

    @_notifies(CHANGE_USER_GROUPS)
    async def update_user_model_group(
        self,
        owner_user_id: int,
        group_id: int,
        *,
        name: str,
        description: str | None,
        members: list[tuple[str, int | None]] | None = None,
        shared: bool | None = None,
    ) -> bool:
        """Set the group's name and description and, when given, whether it
        is shared and its members (exactly those, in that order), in one
        transaction. Unsharing a group drops everyone's opt-in to it. False
        when the user has no such group."""
        sf = self._require_session_factory()
        async with sf() as session, session.begin():
            # Looked up first: MySQL counts only rows an UPDATE changed.
            st, bd = qmark(
                "SELECT id FROM user_model_groups WHERE id = ? AND owner_user_id = ?",
                group_id,
                owner_user_id,
            )
            if result_first(await session.execute(st, bd)) is None:
                return False
            st, bd = qmark(
                "UPDATE user_model_groups SET name = ?, description = ? WHERE id = ?",
                name,
                description,
                group_id,
            )
            await session.execute(st, bd)
            if shared is not None:
                st, bd = qmark(
                    "UPDATE user_model_groups SET shared = ? WHERE id = ?",
                    int(shared),
                    group_id,
                )
                await session.execute(st, bd)
                if not shared:
                    st, bd = qmark(
                        "DELETE FROM user_model_group_subscriptions WHERE group_id = ?",
                        group_id,
                    )
                    await session.execute(st, bd)
            if members is not None:
                st, bd = qmark(
                    "DELETE FROM user_model_group_members WHERE group_id = ?",
                    group_id,
                )
                await session.execute(st, bd)
                for position, (model_name, provider_id) in enumerate(
                    dict.fromkeys(members)
                ):
                    st, bd = qmark(
                        "INSERT INTO user_model_group_members "
                        "(group_id, model_name, provider_id, position) "
                        "VALUES (?, ?, ?, ?)",
                        group_id,
                        model_name,
                        provider_id,
                        position,
                    )
                    await session.execute(st, bd)
        return True

    @_notifies(CHANGE_USER_GROUPS)
    async def delete_user_model_group(self, owner_user_id: int, group_id: int) -> bool:
        sf = self._require_session_factory()
        async with sf() as session, session.begin():
            st, bd = qmark(
                "SELECT id FROM user_model_groups WHERE id = ? AND owner_user_id = ?",
                group_id,
                owner_user_id,
            )
            if result_first(await session.execute(st, bd)) is None:
                return False
            # Explicit, so MySQL without enforced foreign keys and SQLite with
            # them off end up the same.
            for sql in (
                "DELETE FROM user_model_group_subscriptions WHERE group_id = ?",
                "DELETE FROM user_model_group_members WHERE group_id = ?",
                "DELETE FROM user_model_groups WHERE id = ?",
            ):
                st, bd = qmark(sql, group_id)
                await session.execute(st, bd)
        return True

    async def list_user_model_group_subscriptions(
        self, user_id: int | None = None
    ) -> list[tuple[int, int]]:
        """(user id, group id) of each opt-in to a shared group (one user's
        when *user_id* is given), oldest first. Only groups still shared."""
        where = "" if user_id is None else " AND s.user_id = ?"
        rows = await self._select_all(
            "SELECT s.user_id, s.group_id FROM user_model_group_subscriptions s "  # noqa: S608 - fixed SQL fragments; values are bound
            "JOIN user_model_groups g ON g.id = s.group_id "
            f"WHERE g.shared = 1{where} ORDER BY s.id",
            () if user_id is None else (user_id,),
        )
        return [(int(r["user_id"]), int(r["group_id"])) for r in rows]

    @_notifies(CHANGE_USER_GROUPS)
    async def subscribe_user_model_group(self, user_id: int, group_id: int) -> bool:
        """Opt *user_id* in to another user's shared group. False when there
        is no such shared group, or it is the user's own. Opting in twice is
        the same as once."""
        sf = self._require_session_factory()
        async with sf() as session, session.begin():
            st, bd = qmark(
                "SELECT id FROM user_model_groups "
                "WHERE id = ? AND shared = 1 AND owner_user_id <> ?",
                group_id,
                user_id,
            )
            if result_first(await session.execute(st, bd)) is None:
                return False
            st, bd = qmark(
                "SELECT id FROM user_model_group_subscriptions "
                "WHERE user_id = ? AND group_id = ?",
                user_id,
                group_id,
            )
            if result_first(await session.execute(st, bd)) is None:
                st, bd = qmark(
                    "INSERT INTO user_model_group_subscriptions (user_id, group_id) "
                    "VALUES (?, ?)",
                    user_id,
                    group_id,
                )
                await session.execute(st, bd)
        return True

    @_notifies(CHANGE_USER_GROUPS)
    async def unsubscribe_user_model_group(self, user_id: int, group_id: int) -> None:
        sf = self._require_session_factory()
        async with sf() as session, session.begin():
            st, bd = qmark(
                "DELETE FROM user_model_group_subscriptions "
                "WHERE user_id = ? AND group_id = ?",
                user_id,
                group_id,
            )
            await session.execute(st, bd)


def _row_to_member(row: Mapping[str, Any]) -> ModelGroupMember:
    pid = row["provider_id"]
    return ModelGroupMember(
        id=int(row["id"]),
        group_id=int(row["group_id"]),
        model_name=row["model_name"],
        provider_id=None if pid is None else int(pid),
        position=int(row["position"]),
    )
