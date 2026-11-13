"""Parsing of dashboard form and JSON values."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from ..router import ROUTING_MODES
from ..upstream_url import LOCALITIES


def _tristate(value: Any) -> bool | None:
    """Parse an optional on/off flag: None, "", "default" mean unset."""
    if value is None or isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("", "default", "none", "null"):
        return None
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise HTTPException(
        status_code=400, detail="session_affinity must be on, off or default"
    )


def _utc_day_start() -> datetime:
    now = datetime.now(UTC)
    return now.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)


def _positive_int(value: Any, what: str) -> int | None:
    """A whole number of at least 1 from a form or JSON value; blank (or
    null) is None."""
    if isinstance(value, bool):
        raise HTTPException(status_code=400, detail=f"{what} must be a whole number")
    raw = str(value if value is not None else "").strip()
    if not raw:
        return None
    try:
        number = int(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"{what} must be a whole number"
        ) from exc
    if not 1 <= number <= 10**12:
        raise HTTPException(status_code=400, detail=f"{what} is out of range")
    return number


def _usd(value: Any, what: str) -> float | None:
    """A non-negative USD amount from a form or JSON value; blank is None."""
    raw = str(value if value is not None else "").strip().lstrip("$")
    if not raw:
        return None
    try:
        amount = float(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{what} must be a number") from exc
    if amount < 0 or amount > 1_000_000:
        raise HTTPException(status_code=400, detail=f"{what} is out of range")
    return amount


def _parse_locality(value: str | None) -> str:
    v = (value or "").strip().lower()
    if v not in LOCALITIES:
        raise ValueError("locality must be local or cloud")
    return v


KEY_LABEL_MAX = 100


def _key_label(value: Any) -> str | None:
    """An API key's optional label from a request body; blank means none."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="label must be text")
    label = value.strip()
    if len(label) > KEY_LABEL_MAX:
        raise HTTPException(
            status_code=400,
            detail=f"label must be at most {KEY_LABEL_MAX} characters",
        )
    return label or None


def _key_locality(value: Any) -> str | None:
    """An API key's locality from a request body: local, cloud, or any."""
    v = str(value or "").strip().lower()
    if v in ("", "any", "none", "either"):
        return None
    if v not in LOCALITIES:
        raise HTTPException(
            status_code=400, detail="locality must be local, cloud or any"
        )
    return v


def _model_filter(text: str | None) -> str | None:
    """Model filter patterns, normalised to one per line; None for none."""
    patterns = _model_lines(text)
    return "\n".join(patterns) or None


def _timeout(value: str | None) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError as exc:
        raise ValueError("read timeout must be a number of seconds") from exc
    if not 1 <= seconds <= 3600:
        raise ValueError("read timeout must be between 1 and 3600 seconds")
    return seconds


def _max_concurrent(value: str | None) -> int | None:
    """A provider's concurrent-request limit; None (blank) for no limit."""
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        limit = int(raw)
    except ValueError as exc:
        raise ValueError("max concurrent requests must be a whole number") from exc
    if not 1 <= limit <= 10_000:
        raise ValueError("max concurrent requests must be between 1 and 10000")
    return limit


def _model_lines(text: str | None) -> list[str]:
    """Model names, one per line or comma-separated."""
    out: list[str] = []
    for part in (text or "").replace(",", "\n").splitlines():
        name = part.strip()
        if name and name not in out:
            if len(name) > 200:
                raise ValueError("model name too long")
            out.append(name)
    return out


ADMIN_KEY_FIELDS = (
    "label",
    "routing_mode",
    "allow_fallback",
    "session_affinity",
    "locality",
    "daily_spend_cap_usd",
    "strict_pins",
    "daily_token_limit",
    "requests_per_minute",
)
# Locality and the spend cap of a self-issued key are an administrator's to
# set, so its owner can't widen them. Strict pins only ever refuse, so the
# owner may turn them on or off.
OWNER_KEY_FIELDS = (
    "label",
    "routing_mode",
    "allow_fallback",
    "session_affinity",
    "strict_pins",
    # Owners may only tighten their own quotas on a key (see my_keys).
    "daily_token_limit",
    "requests_per_minute",
)


def _routing_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode not in ROUTING_MODES:
        raise HTTPException(status_code=400, detail="Invalid routing_mode")
    return mode


def _flag(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise HTTPException(status_code=400, detail=f"{name} must be true or false")


_KEY_PARSERS: dict[str, Any] = {
    "label": _key_label,
    "routing_mode": _routing_mode,
    "allow_fallback": lambda v: _flag(v, "allow_fallback"),
    "session_affinity": _tristate,
    "locality": _key_locality,
    "daily_spend_cap_usd": lambda v: _usd(v, "daily spend cap"),
    "strict_pins": lambda v: _flag(v, "strict_pins"),
    "daily_token_limit": lambda v: _positive_int(v, "daily_token_limit"),
    "requests_per_minute": lambda v: _positive_int(v, "requests_per_minute"),
}


def strict_pins_flag(body: dict[str, Any]) -> bool:
    """``strict_pins`` from a new key's body: off when absent."""
    return _flag(body["strict_pins"], "strict_pins") if "strict_pins" in body else False


def key_edits(body: dict[str, Any], allowed: tuple[str, ...]) -> dict[str, Any]:
    """The API key settings an edit request changes, validated.

    Only fields present in *body* are returned (absent means unchanged). A
    field outside *allowed* that is not a key setting at all is ignored; one
    that is a setting the caller may not change is a 400, so the client
    learns rather than having it silently dropped.
    """
    forbidden = sorted(k for k in body if k in _KEY_PARSERS and k not in allowed)
    if forbidden:
        raise HTTPException(
            status_code=400,
            detail=f"You can't change {', '.join(forbidden)} on this key",
        )
    return {name: _KEY_PARSERS[name](body[name]) for name in allowed if name in body}
