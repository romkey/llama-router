"""Create the first dashboard administrator from environment variables.

The dashboard has no web form for the first account: anyone who reached it
before the operator did could have claimed it. Instead the operator sets
``LLAMA_ROUTER_ADMIN_USERNAME`` and ``LLAMA_ROUTER_ADMIN_PASSWORD`` and the
account is created at startup while there are no accounts.
"""

from __future__ import annotations

import logging

from ..database import Database
from .auth_core import hash_password, normalize_username

logger = logging.getLogger(__name__)

MIN_LENGTH = 8  # Also in the error message below.


async def ensure_initial_admin(db: Database, username: str, password: str) -> bool:
    """Create the administrator if there are no accounts. True if created."""
    if await db.count_dashboard_users() > 0:
        return False
    if not username.strip() and not password:
        logger.warning(
            "No dashboard accounts: set LLAMA_ROUTER_ADMIN_USERNAME and "
            "LLAMA_ROUTER_ADMIN_PASSWORD and restart to create the first "
            "administrator. The dashboard stays locked until then."
        )
        return False
    try:
        name = normalize_username(username)
    except ValueError as exc:
        logger.error("LLAMA_ROUTER_ADMIN_USERNAME is not usable: %s", exc)
        return False
    if len(password) < MIN_LENGTH:
        # No arguments: CodeQL treats anything near the password as secret.
        logger.error(
            "LLAMA_ROUTER_ADMIN_PASSWORD must be at least 8 characters; "
            "no administrator created"
        )
        return False
    await db.create_dashboard_user(name, hash_password(password), True)
    logger.info(
        "Created dashboard administrator %s from LLAMA_ROUTER_ADMIN_USERNAME; "
        "the variables are ignored from now on",
        name,
    )
    return True
