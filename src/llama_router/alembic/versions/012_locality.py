"""Locality: providers are local or cloud, API keys may require one, and the
request log records which a request went to.

Existing providers are classed from their preset or first address.

Revision ID: 012_locality
Revises: 011_provider_presets
Create Date: 2026-10-27
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

import sqlalchemy as sa
from alembic import op

from llama_router.db_types import ShortText

revision = "012_locality"
down_revision = "011_provider_presets"
branch_labels = None
depends_on = None

_LOCAL_SUFFIXES = (".local", ".lan", ".home", ".internal", ".home.arpa", ".ts.net")
_LOCAL_NETWORKS = (
    "127.0.0.0/8",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "169.254.0.0/16",
    "100.64.0.0/10",
    "::1/128",
    "fc00::/7",
    "fe80::/10",
)


def _guess(url: str) -> str:
    # A frozen copy of upstream_url.guess_locality: migrations must not change
    # when the application code does.
    raw = (url or "").strip()
    if raw.startswith("worker://"):
        return "local"
    host = (urlsplit(raw).hostname or "").lower().rstrip(".")
    if not host:
        return "local"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if "." not in host or host.endswith(_LOCAL_SUFFIXES) or host == "localhost":
            return "local"
        return "cloud"
    if any(ip in ipaddress.ip_network(n) for n in _LOCAL_NETWORKS):
        return "local"
    return "cloud"


def _cols(table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    if "locality" not in _cols("providers"):
        op.add_column(
            "providers",
            sa.Column("locality", ShortText, nullable=False, server_default="local"),
        )
        rows = bind.execute(
            sa.text("SELECT id, preset, worker_name FROM providers")
        ).mappings()
        for row in list(rows):
            if row["worker_name"] is not None:
                continue
            if row["preset"]:
                locality = "cloud"
            else:
                url = bind.execute(
                    sa.text(
                        "SELECT url FROM provider_addresses WHERE provider_id = :id "
                        "ORDER BY is_preferred DESC, id LIMIT 1"
                    ),
                    {"id": row["id"]},
                ).scalar()
                locality = _guess(url or "")
            if locality != "local":
                bind.execute(
                    sa.text("UPDATE providers SET locality = :l WHERE id = :id"),
                    {"l": locality, "id": row["id"]},
                )
    if "locality" not in _cols("api_keys"):
        op.add_column("api_keys", sa.Column("locality", sa.Text(), nullable=True))
    if "locality" not in _cols("request_log"):
        op.add_column("request_log", sa.Column("locality", sa.Text(), nullable=True))


def downgrade() -> None:
    for table in ("request_log", "api_keys", "providers"):
        with op.batch_alter_table(table) as batch:
            batch.drop_column("locality")
