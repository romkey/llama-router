"""Parsing of dashboard form and JSON values."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

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
