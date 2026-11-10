"""API keys, auth settings and app settings."""

from __future__ import annotations

import json
import logging
from typing import Any

from ..db_events import CHANGE_API_KEYS, CHANGE_AUTH_SETTINGS, _notifies
from .base import Repository, _tristate

logger = logging.getLogger(__name__)

# Settings ``update_api_key`` may change: everything chosen at creation
# except the secret and the owner.
EDITABLE_FIELDS = frozenset(
    {
        "label",
        "routing_mode",
        "allow_fallback",
        "session_affinity",
        "locality",
        "daily_spend_cap_usd",
    }
)


class ApiKeyRepository(Repository):
    @_notifies(CHANGE_API_KEYS)
    async def create_api_key(
        self,
        key_prefix: str,
        key_hash: str,
        routing_mode: str,
        allow_fallback: bool,
        session_affinity: bool | None = None,
        locality: str | None = None,
        daily_spend_cap_usd: float | None = None,
        owner_user_id: int | None = None,
        label: str | None = None,
    ) -> int:
        """*owner_user_id* is the dashboard user who issued the key to
        themselves; None for a key an admin issued. *label* is a free-text
        name shown beside the prefix."""
        return await self._insert_get_id(
            "INSERT INTO api_keys "
            "(key_prefix, key_hash, routing_mode, allow_fallback, session_affinity, "
            "locality, daily_spend_cap_usd, owner_user_id, label) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key_prefix,
                key_hash,
                routing_mode,
                int(allow_fallback),
                None if session_affinity is None else int(session_affinity),
                locality,
                daily_spend_cap_usd,
                owner_user_id,
                label,
            ),
        )

    @_notifies(CHANGE_API_KEYS)
    async def update_api_key(
        self,
        key_id: int,
        fields: dict[str, Any],
        *,
        owner_user_id: int | None = None,
    ) -> bool:
        """Change the settings in *fields* (keys from ``EDITABLE_FIELDS``);
        the others stay. With *owner_user_id*, only a key that user issued to
        themselves matches. Returns whether the key was found."""
        unknown = set(fields) - EDITABLE_FIELDS
        if unknown:
            raise ValueError(f"not editable: {', '.join(sorted(unknown))}")
        values = dict(fields)
        if "allow_fallback" in values:
            values["allow_fallback"] = int(bool(values["allow_fallback"]))
        if values.get("session_affinity") is not None:
            values["session_affinity"] = int(bool(values["session_affinity"]))
        names = sorted(values)
        where = " WHERE id = ?"
        params: list[Any] = [values[n] for n in names] + [key_id]
        if owner_user_id is not None:
            where += " AND owner_user_id = ?"
            params.append(owner_user_id)
        if not names:
            # Nothing to change: report whether the key exists.
            return await self._key_exists(key_id, owner_user_id)
        # Column names come from EDITABLE_FIELDS, never from the caller.
        assignments = ", ".join(f"{n} = ?" for n in names)
        sql = "UPDATE api_keys SET " + assignments + where  # noqa: S608
        if await self._execute_rowcount(sql, tuple(params)):
            return True
        # MySQL counts only rows whose values changed; look before saying 404.
        return await self._key_exists(key_id, owner_user_id)

    async def _key_exists(self, key_id: int, owner_user_id: int | None) -> bool:
        return await self.get_api_key(key_id, owner_user_id=owner_user_id) is not None

    @_notifies(CHANGE_API_KEYS)
    async def regenerate_api_key(
        self,
        key_id: int,
        key_prefix: str,
        key_hash: str,
        *,
        owner_user_id: int | None = None,
    ) -> bool:
        """Give the key a new secret; its id and settings stay. The old
        secret stops working at once."""
        sql = "UPDATE api_keys SET key_prefix = ?, key_hash = ? WHERE id = ?"
        params: tuple[Any, ...] = (key_prefix, key_hash, key_id)
        if owner_user_id is not None:
            sql += " AND owner_user_id = ?"
            params += (owner_user_id,)
        return bool(await self._execute_rowcount(sql, params))

    async def set_api_key_spend_cap(self, key_id: int, cap_usd: float | None) -> bool:
        """USD a key may spend per UTC day before it is limited to local."""
        return await self.update_api_key(key_id, {"daily_spend_cap_usd": cap_usd})

    async def set_api_key_locality(self, key_id: int, locality: str | None) -> bool:
        """Limit a key to local or cloud providers (None: either)."""
        return await self.update_api_key(key_id, {"locality": locality})

    async def get_api_key(
        self, key_id: int, *, owner_user_id: int | None = None
    ) -> dict[str, Any] | None:
        """One key in the shape ``list_api_keys`` returns, or None."""
        keys = await self._list_api_keys(key_id=key_id, owner_user_id=owner_user_id)
        return keys[0] if keys else None

    async def list_api_keys(
        self, owner_user_id: int | None = None
    ) -> list[dict[str, Any]]:
        """All keys, or only those *owner_user_id* issued to themselves."""
        return await self._list_api_keys(owner_user_id=owner_user_id)

    async def _list_api_keys(
        self, *, key_id: int | None = None, owner_user_id: int | None = None
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if key_id is not None:
            conditions.append("k.id = ?")
            params.append(key_id)
        if owner_user_id is not None:
            conditions.append("k.owner_user_id = ?")
            params.append(owner_user_id)
        where = ("WHERE " + " AND ".join(conditions) + " ") if conditions else ""
        rows = await self._select_all(
            "SELECT "
            "k.id, k.key_prefix, k.routing_mode, k.allow_fallback, k.session_affinity, "
            "k.locality, k.daily_spend_cap_usd, k.created_at, k.last_used_at, "
            "k.owner_user_id, k.label, u.username AS owner_username, "
            "COUNT(p.id) AS pin_count "
            "FROM api_keys k "
            "LEFT JOIN api_key_model_pins p ON p.api_key_id = k.id "
            "LEFT JOIN dashboard_users u ON u.id = k.owner_user_id "
            + where
            + "GROUP BY k.id, k.key_prefix, k.routing_mode, k.allow_fallback, "
            "k.session_affinity, k.locality, k.daily_spend_cap_usd, k.created_at, "
            "k.last_used_at, k.owner_user_id, k.label, u.username "
            "ORDER BY k.created_at DESC, k.id DESC",
            tuple(params),
        )
        return [
            {
                "id": int(r["id"]),
                "key_prefix": r["key_prefix"],
                "routing_mode": r["routing_mode"],
                "allow_fallback": bool(r["allow_fallback"]),
                "session_affinity": _tristate(r["session_affinity"]),
                "locality": r["locality"],
                "daily_spend_cap_usd": r["daily_spend_cap_usd"],
                "created_at": r["created_at"],
                "last_used_at": r["last_used_at"],
                "owner_user_id": (
                    None if r["owner_user_id"] is None else int(r["owner_user_id"])
                ),
                "owner_username": r["owner_username"],
                "label": r["label"],
                "pin_count": int(r["pin_count"] or 0),
            }
            for r in rows
        ]

    @_notifies(CHANGE_API_KEYS)
    async def delete_api_key(self, key_id: int) -> None:
        await self._execute("DELETE FROM api_keys WHERE id = ?", (key_id,))

    @_notifies(CHANGE_API_KEYS)
    async def delete_owned_api_key(self, key_id: int, owner_user_id: int) -> bool:
        """Delete *key_id* only if *owner_user_id* issued it to themselves."""
        return bool(
            await self._execute_rowcount(
                "DELETE FROM api_keys WHERE id = ? AND owner_user_id = ?",
                (key_id, owner_user_id),
            )
        )

    async def set_api_key_hash(self, key_id: int, key_hash: str) -> None:
        """Replace a key's stored fingerprint (re-hashing with a new pepper, or
        upgrading a legacy SHA-256 one)."""
        await self._execute(
            "UPDATE api_keys SET key_hash = ? WHERE id = ?", (key_hash, key_id)
        )

    async def lookup_api_key(self, key_hash: str) -> dict[str, Any] | None:
        row = await self._select_one(
            "SELECT id, key_prefix, routing_mode, allow_fallback, session_affinity, "
            "locality, daily_spend_cap_usd FROM api_keys "
            "WHERE key_hash = ?",
            (key_hash,),
        )
        if not row:
            return None
        await self._execute(
            "UPDATE api_keys SET last_used_at = CURRENT_TIMESTAMP WHERE id = ?",
            (row["id"],),
        )
        return {
            "id": int(row["id"]),
            "key_prefix": row["key_prefix"],
            "routing_mode": row["routing_mode"],
            "allow_fallback": bool(row["allow_fallback"]),
            "session_affinity": _tristate(row["session_affinity"]),
            "locality": row["locality"],
            "daily_spend_cap_usd": row["daily_spend_cap_usd"],
        }

    async def touch_api_key(self, key_id: int) -> None:
        await self._execute(
            "UPDATE api_keys SET last_used_at = CURRENT_TIMESTAMP WHERE id = ?",
            (key_id,),
        )

    @_notifies(CHANGE_API_KEYS)
    async def set_api_key_model_pin(
        self, key_id: int, model_name: str, provider_id: int
    ) -> None:
        if self._dialect_name in ("mysql", "mariadb"):
            await self._execute(
                "INSERT INTO api_key_model_pins (api_key_id, model_name, provider_id) "
                "VALUES (?, ?, ?) "
                "ON DUPLICATE KEY UPDATE provider_id = VALUES(provider_id)",
                (key_id, model_name, provider_id),
            )
        elif self._dialect_name == "postgresql":
            await self._execute(
                "INSERT INTO api_key_model_pins (api_key_id, model_name, provider_id) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT (api_key_id, model_name) "
                "DO UPDATE SET provider_id = EXCLUDED.provider_id",
                (key_id, model_name, provider_id),
            )
        else:
            await self._execute(
                "INSERT INTO api_key_model_pins (api_key_id, model_name, provider_id) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT (api_key_id, model_name) "
                "DO UPDATE SET provider_id = excluded.provider_id",
                (key_id, model_name, provider_id),
            )

    @_notifies(CHANGE_API_KEYS)
    async def remove_api_key_model_pin(self, key_id: int, model_name: str) -> None:
        await self._execute(
            "DELETE FROM api_key_model_pins WHERE api_key_id = ? AND model_name = ?",
            (key_id, model_name),
        )

    async def get_api_key_decision_policy(self, key_id: int) -> dict[str, Any] | None:
        """The key's decision policy as stored (a JSON object), or None."""
        row = await self._select_one(
            "SELECT policy FROM api_key_decision_policies WHERE api_key_id = ?",
            (key_id,),
        )
        if not row:
            return None
        try:
            policy = json.loads(row["policy"])
        except TypeError, ValueError:
            logger.warning("Decision policy of API key %s is not JSON", key_id)
            return None
        return policy if isinstance(policy, dict) else None

    @_notifies(CHANGE_API_KEYS)
    async def set_api_key_decision_policy(
        self, key_id: int, policy: dict[str, Any]
    ) -> None:
        data = json.dumps(policy, ensure_ascii=False)
        if self._dialect_name in ("mysql", "mariadb"):
            await self._execute(
                "INSERT INTO api_key_decision_policies (api_key_id, policy) "
                "VALUES (?, ?) ON DUPLICATE KEY UPDATE policy = VALUES(policy), "
                "updated_at = CURRENT_TIMESTAMP",
                (key_id, data),
            )
        else:
            await self._execute(
                "INSERT INTO api_key_decision_policies (api_key_id, policy) "
                "VALUES (?, ?) ON CONFLICT (api_key_id) DO UPDATE SET "
                "policy = ?, updated_at = CURRENT_TIMESTAMP",
                (key_id, data, data),
            )

    @_notifies(CHANGE_API_KEYS)
    async def delete_api_key_decision_policy(self, key_id: int) -> bool:
        n = await self._execute_rowcount(
            "DELETE FROM api_key_decision_policies WHERE api_key_id = ?", (key_id,)
        )
        return n > 0

    async def list_api_key_model_pins(self, key_id: int) -> list[dict[str, Any]]:
        rows = await self._select_all(
            "SELECT p.model_name, p.provider_id, pr.name AS provider_name "
            "FROM api_key_model_pins p "
            "JOIN providers pr ON pr.id = p.provider_id "
            "WHERE p.api_key_id = ? "
            "ORDER BY p.model_name ASC",
            (key_id,),
        )
        return [
            {
                "model_name": r["model_name"],
                "provider_id": int(r["provider_id"]),
                "provider_name": r["provider_name"],
            }
            for r in rows
        ]

    async def get_allow_unauthenticated(self) -> bool:
        row = await self._select_one(
            # Qualified so MySQL accepts the reserved word without quoting.
            "SELECT value FROM app_settings "
            "WHERE app_settings.key = 'allow_unauthenticated'"
        )
        if not row:
            # Fail closed: anonymous access is something to opt into.
            return False
        return str(row["value"]).lower() in ("1", "true", "yes", "on")

    @_notifies(CHANGE_AUTH_SETTINGS)
    async def set_allow_unauthenticated(self, allow: bool) -> None:
        val = "true" if allow else "false"
        if self._dialect_name in ("mysql", "mariadb"):
            await self._execute(
                "INSERT INTO app_settings (`key`, value) "
                "VALUES ('allow_unauthenticated', ?) "
                "ON DUPLICATE KEY UPDATE value = VALUES(value)",
                (val,),
            )
        elif self._dialect_name == "postgresql":
            await self._execute(
                'INSERT INTO app_settings ("key", value) '
                "VALUES ('allow_unauthenticated', ?) "
                'ON CONFLICT ("key") DO UPDATE SET value = EXCLUDED.value',
                (val,),
            )
        else:
            await self._execute(
                'INSERT INTO app_settings ("key", value) '
                "VALUES ('allow_unauthenticated', ?) "
                'ON CONFLICT ("key") DO UPDATE SET value = excluded.value',
                (val,),
            )

    async def get_app_setting(self, key: str) -> str | None:
        row = await self._select_one(
            "SELECT value FROM app_settings WHERE app_settings.key = ?", (key,)
        )
        return str(row["value"]) if row else None

    async def set_app_setting(self, key: str, value: str) -> None:
        if self._dialect_name in ("mysql", "mariadb"):
            await self._execute(
                "INSERT INTO app_settings (`key`, value) VALUES (?, ?) "
                "ON DUPLICATE KEY UPDATE value = VALUES(value)",
                (key, value),
            )
        elif self._dialect_name == "postgresql":
            await self._execute(
                'INSERT INTO app_settings ("key", value) VALUES (?, ?) '
                'ON CONFLICT ("key") DO UPDATE SET value = EXCLUDED.value',
                (key, value),
            )
        else:
            await self._execute(
                'INSERT INTO app_settings ("key", value) VALUES (?, ?) '
                'ON CONFLICT ("key") DO UPDATE SET value = excluded.value',
                (key, value),
            )

    async def add_app_setting(self, key: str, value: str) -> str:
        """Store *value* unless *key* already has one; return the stored value."""
        if self._dialect_name in ("mysql", "mariadb"):
            await self._execute(
                "INSERT IGNORE INTO app_settings (`key`, value) VALUES (?, ?)",
                (key, value),
            )
        else:
            await self._execute(
                'INSERT INTO app_settings ("key", value) VALUES (?, ?) '
                'ON CONFLICT ("key") DO NOTHING',
                (key, value),
            )
        stored = await self.get_app_setting(key)
        if stored:
            return stored
        # An empty value was stored before: replace it.
        await self.set_app_setting(key, value)
        return value
