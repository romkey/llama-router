"""Dashboard users."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..db_events import CHANGE_API_KEYS, _notifies
from ..db_sql import qmark
from .base import Repository, _bind_params


class DashboardUserRepository(Repository):
    async def count_dashboard_users(self) -> int:
        row = await self._select_one("SELECT COUNT(*) AS c FROM dashboard_users")
        return int(row["c"]) if row else 0

    async def list_dashboard_users(self) -> list[dict[str, Any]]:
        if self._dialect_name == "sqlite":
            order_sql = "ORDER BY username COLLATE NOCASE ASC"
        else:
            order_sql = "ORDER BY LOWER(username) ASC"
        rows = await self._select_all(
            f"SELECT {_DASHBOARD_USER_COLUMNS} FROM dashboard_users {order_sql}"  # noqa: S608 - column names are whitelisted / fixed SQL fragments; values are bound
        )
        users = [_row_to_dashboard_user(r) for r in rows]
        for u in users:
            u["has_password"] = bool(u.pop("password_hash"))
        return users

    async def get_dashboard_user_by_username(
        self, username: str
    ) -> dict[str, Any] | None:
        u = username.strip()
        if self._dialect_name == "sqlite":
            row = await self._select_one(
                f"SELECT {_DASHBOARD_USER_COLUMNS} "  # noqa: S608 - column names are whitelisted / fixed SQL fragments; values are bound
                "FROM dashboard_users WHERE username = ? COLLATE NOCASE",
                (u,),
            )
        else:
            row = await self._select_one(
                f"SELECT {_DASHBOARD_USER_COLUMNS} "  # noqa: S608 - column names are whitelisted / fixed SQL fragments; values are bound
                "FROM dashboard_users WHERE LOWER(username) = LOWER(?)",
                (u,),
            )
        return _row_to_dashboard_user(row) if row else None

    async def get_dashboard_user_by_id(self, user_id: int) -> dict[str, Any] | None:
        row = await self._select_one(
            f"SELECT {_DASHBOARD_USER_COLUMNS} FROM dashboard_users WHERE id = ?",  # noqa: S608 - column names are whitelisted / fixed SQL fragments; values are bound
            (user_id,),
        )
        return _row_to_dashboard_user(row) if row else None

    async def get_dashboard_user_by_oidc(
        self, issuer: str, subject: str
    ) -> dict[str, Any] | None:
        row = await self._select_one(
            f"SELECT {_DASHBOARD_USER_COLUMNS} FROM dashboard_users "  # noqa: S608 - column names are whitelisted / fixed SQL fragments; values are bound
            "WHERE oidc_issuer = ? AND oidc_subject = ?",
            (issuer, subject),
        )
        return _row_to_dashboard_user(row) if row else None

    async def create_dashboard_user(
        self, username: str, password_hash: str, is_admin: bool
    ) -> int:
        floor = self._floor_sql()
        sql = (
            "INSERT INTO dashboard_users (username, password_hash, is_admin, "  # noqa: S608 - fixed SQL fragment; values are bound
            f"session_version) VALUES (?, ?, ?, {floor})"
        )
        return await self._insert_get_id(
            sql, (username.strip(), password_hash, int(is_admin), _SESSION_FLOOR_KEY)
        )

    def _cast_sql(self, expr: str, to: str) -> str:
        mysql = self._dialect_name in ("mysql", "mariadb")
        if to == "int":
            return f"CAST({expr} AS {'SIGNED' if mysql else 'INTEGER'})"
        return f"CAST({expr} AS {'CHAR' if mysql else 'TEXT'})"

    def _floor_sql(self) -> str:
        """SQL for the session_version a new account starts at (binds the key).

        The floor is above the version of every account ever deleted, so a
        cookie left over from a deleted account never matches a new one, even
        with the same id (SQLite reuses the highest) and the same username.
        It is read inside the INSERT, so a concurrent delete can't slip
        between reading it and using it.
        """
        value = self._cast_sql("value", "int")
        return (
            f"COALESCE((SELECT {value} FROM app_settings "  # noqa: S608 - fixed SQL fragment; the key is bound
            "WHERE app_settings.key = ?), 0)"
        )

    async def create_oidc_dashboard_user(
        self,
        username: str,
        issuer: str,
        subject: str,
        *,
        email: str | None,
        display_name: str | None,
        is_admin: bool,
        approved: bool = True,
    ) -> int:
        """Create an account that signs in only through OpenID Connect.

        An unapproved account exists but cannot use the dashboard until an
        administrator approves it.
        """
        floor = self._floor_sql()
        sql = (
            "INSERT INTO dashboard_users (username, password_hash, is_admin, "  # noqa: S608 - fixed SQL fragment; values are bound
            "auth_source, oidc_issuer, oidc_subject, email, display_name, "
            "last_login_at, approved, session_version) "
            f"VALUES (?, NULL, ?, 'oidc', ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, {floor})"
        )
        return await self._insert_get_id(
            sql,
            (
                username.strip(),
                int(is_admin),
                issuer,
                subject,
                email,
                display_name,
                int(approved),
                _SESSION_FLOOR_KEY,
            ),
        )

    async def update_oidc_dashboard_user(
        self,
        user_id: int,
        *,
        email: str | None,
        display_name: str | None,
        is_admin: bool,
    ) -> None:
        """Refresh an OIDC account from the identity provider at sign-in.

        A change of the admin flag ends the account's other sessions.
        """
        # session_version is assigned before is_admin: MySQL evaluates SET
        # left to right and would otherwise compare against the new value.
        await self._execute(
            "UPDATE dashboard_users SET "
            "session_version = session_version "
            "+ CASE WHEN is_admin <> ? THEN 1 ELSE 0 END, "
            "email = ?, display_name = ?, is_admin = ?, "
            "last_login_at = CURRENT_TIMESTAMP WHERE id = ?",
            (int(is_admin), email, display_name, int(is_admin), user_id),
        )

    async def touch_dashboard_user_login(self, user_id: int) -> None:
        await self._execute(
            "UPDATE dashboard_users SET last_login_at = CURRENT_TIMESTAMP WHERE id = ?",
            (user_id,),
        )

    async def update_dashboard_user_password(
        self, user_id: int, password_hash: str
    ) -> None:
        """Set the password and end every session of the account."""
        await self._execute(
            "UPDATE dashboard_users SET password_hash = ?, "
            "session_version = session_version + 1 WHERE id = ?",
            (password_hash, user_id),
        )

    async def bump_dashboard_user_session_version(self, user_id: int) -> None:
        """Invalidate every session cookie issued to the account so far."""
        await self._execute(
            "UPDATE dashboard_users SET session_version = session_version + 1 "
            "WHERE id = ?",
            (user_id,),
        )

    async def unapprove_dashboard_user(self, user_id: int) -> bool:
        """Turn the account back to waiting for approval; its sessions end."""
        return bool(
            await self._execute_rowcount(
                "UPDATE dashboard_users SET approved = 0, "
                "session_version = session_version + 1 WHERE id = ?",
                (user_id,),
            )
        )

    async def approve_dashboard_user(self, user_id: int) -> bool:
        return bool(
            await self._execute_rowcount(
                "UPDATE dashboard_users SET approved = 1 WHERE id = ?", (user_id,)
            )
        )

    # The per-user quota columns; null follows the admin's default.
    QUOTA_FIELDS = ("daily_token_limit", "requests_per_minute", "daily_spend_cap_usd")

    @_notifies(CHANGE_API_KEYS)
    async def set_dashboard_user_limits(
        self, user_id: int, limits: dict[str, Any]
    ) -> bool:
        """Set the user's quotas in *limits* (``QUOTA_FIELDS``; None follows
        the default). Their keys pick the change up on the next request."""
        unknown = set(limits) - set(self.QUOTA_FIELDS)
        if unknown:
            raise ValueError(f"unknown limits: {sorted(unknown)}")
        if not limits:
            return await self.get_dashboard_user_by_id(user_id) is not None
        names = sorted(limits)
        assignments = ", ".join(f"{n} = ?" for n in names)
        sql = "UPDATE dashboard_users SET " + assignments + " WHERE id = ?"  # noqa: S608 - names from QUOTA_FIELDS
        params = (*[limits[n] for n in names], user_id)
        if await self._execute_rowcount(sql, params):
            return True
        return await self.get_dashboard_user_by_id(user_id) is not None

    @_notifies(CHANGE_API_KEYS)
    async def delete_dashboard_user(self, user_id: int) -> None:
        """Delete the user and the API keys they issued to themselves."""
        # Raise the floor new accounts start at above this account's version,
        # so its cookies stay dead if the id and username come back. One
        # transaction, and the floor only ever grows (GREATEST in SQL), so
        # concurrent deletes can't lower it.
        if self._dialect_name in ("mysql", "mariadb"):
            seed = "INSERT IGNORE INTO app_settings (`key`, value) VALUES (?, '0')"
        else:
            seed = (
                "INSERT INTO app_settings (\"key\", value) VALUES (?, '0') "
                'ON CONFLICT ("key") DO NOTHING'
            )
        greatest = "MAX" if self._dialect_name == "sqlite" else "GREATEST"
        current = self._cast_sql("value", "int")
        deleted = (
            "COALESCE((SELECT session_version + 1 FROM dashboard_users "
            "WHERE id = ?), 0)"
        )
        raise_floor = (
            "UPDATE app_settings SET value = "  # noqa: S608 - fixed SQL fragments; values are bound
            f"{self._cast_sql(f'{greatest}({current}, {deleted})', 'text')} "
            "WHERE app_settings.key = ?"
        )
        statements: list[tuple[str, tuple[Any, ...]]] = [
            (seed, (_SESSION_FLOOR_KEY,)),
            (raise_floor, (user_id, _SESSION_FLOOR_KEY)),
            ("DELETE FROM api_keys WHERE owner_user_id = ?", (user_id,)),
            ("DELETE FROM dashboard_users WHERE id = ?", (user_id,)),
        ]
        sf = self._require_session_factory()
        async with sf() as session, session.begin():
            for sql, params in statements:
                stmt, bind = qmark(sql, *_bind_params(params))
                await session.execute(stmt, bind)

    async def count_dashboard_admins(self) -> int:
        """Administrators who can use the dashboard (approved accounts only)."""
        row = await self._select_one(
            "SELECT COUNT(*) AS c FROM dashboard_users "
            "WHERE is_admin = 1 AND approved = 1"
        )
        return int(row["c"]) if row else 0


_SESSION_FLOOR_KEY = "dashboard_session_version_floor"


_DASHBOARD_USER_COLUMNS = (
    "id, username, password_hash, is_admin, created_at, auth_source, "
    "oidc_issuer, oidc_subject, email, display_name, last_login_at, "
    "mfa_enabled, "
    "session_version, approved, "
    "daily_token_limit, requests_per_minute, daily_spend_cap_usd"
)


def _row_to_dashboard_user(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "username": row["username"],
        "password_hash": row["password_hash"],
        "is_admin": bool(row["is_admin"]),
        "created_at": row["created_at"],
        "auth_source": row["auth_source"] or "local",
        "oidc_issuer": row["oidc_issuer"],
        "oidc_subject": row["oidc_subject"],
        "email": row["email"],
        "display_name": row["display_name"],
        "last_login_at": row["last_login_at"],
        "mfa_enabled": bool(row["mfa_enabled"]),
        "session_version": int(row["session_version"] or 0),
        "approved": bool(row["approved"]),
        "daily_token_limit": _opt_int(row.get("daily_token_limit")),
        "requests_per_minute": _opt_int(row.get("requests_per_minute")),
        "daily_spend_cap_usd": (
            None
            if row.get("daily_spend_cap_usd") is None
            else float(row["daily_spend_cap_usd"])
        ),
    }


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)
