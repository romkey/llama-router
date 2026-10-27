"""One protocol per provider.

``providers.provider_type`` (``ollama``, ``llamacpp`` or ``both``) becomes
``providers.protocol`` (``ollama`` or ``openai``), and addresses lose their
second ``llamacpp_url``. This revision:

1. adds ``protocol``: ``llamacpp`` becomes ``openai``;
2. splits each direct ``both`` provider in two. The original keeps its id and
   name and speaks Ollama. A new ``"<name> (llama.cpp)"`` provider speaks the
   OpenAI-style API at each address's ``llamacpp_url`` (or its ``url``), with
   the same hardware details and upstream key. Models found only on the
   llama.cpp side move to it, models found on both are copied, and
   ``llamacpp`` benchmarks and pins to moved models follow them. Both lose
   their stored probe, which mixed the two upstreams;
3. turns ``both`` workers into Ollama workers: a worker relaying its own
   router reaches every route through the Ollama port;
4. renames the ``llamacpp`` benchmark protocol to ``openai`` and drops the
   per-model ``_in_ollama`` / ``_in_llamacpp`` markers;
5. drops ``providers.provider_type`` and ``provider_addresses.llamacpp_url``.

Downgrade restores the columns but does not merge split providers.

Revision ID: 009_single_protocol_providers
Revises: 010_provider_model_size_bigint
Create Date: 2026-10-27
"""

from __future__ import annotations

import json

import sqlalchemy as sa
from alembic import op

from llama_router.db_types import ShortText

revision = "009_single_protocol_providers"
down_revision = "010_provider_model_size_bigint"
branch_labels = None
depends_on = None

_MARKERS = ("_in_ollama", "_in_llamacpp")


def _details(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        val = json.loads(raw)
    except ValueError:
        return {}
    return val if isinstance(val, dict) else {}


def _dump(details: dict) -> str | None:
    clean = {k: v for k, v in details.items() if k not in _MARKERS}
    return json.dumps(clean) if clean else None


def _free_name(bind, base: str) -> str:
    name = base
    n = 2
    while bind.execute(
        sa.text("SELECT 1 FROM providers WHERE name = :n"), {"n": name}
    ).first():
        name = f"{base} {n}"
        n += 1
    return name


def _split(bind, row) -> None:
    """Move the llama.cpp half of a ``both`` provider into its own provider."""
    old_id = row["id"]
    name = _free_name(bind, f"{row['name']} (llama.cpp)")
    bind.execute(
        sa.text(
            "INSERT INTO providers (name, protocol, status, machine_type, gpu_type, "
            "gpu_ram, api_key_enc, api_key_header) VALUES "
            "(:name, 'openai', 'unknown', :mt, :gt, :gr, :key, :hdr)"
        ),
        {
            "name": name,
            "mt": row["machine_type"],
            "gt": row["gpu_type"],
            "gr": row["gpu_ram"],
            "key": row["api_key_enc"],
            "hdr": row["api_key_header"],
        },
    )
    new_id = bind.execute(
        sa.text("SELECT id FROM providers WHERE name = :n"), {"n": name}
    ).scalar_one()

    addrs = bind.execute(
        sa.text(
            "SELECT url, llamacpp_url, is_preferred FROM provider_addresses "
            "WHERE provider_id = :id ORDER BY id"
        ),
        {"id": old_id},
    ).mappings()
    for a in list(addrs):
        url = (a["llamacpp_url"] or "").strip() or a["url"]
        bind.execute(
            sa.text(
                "INSERT INTO provider_addresses "
                "(provider_id, url, is_preferred, is_live) "
                "VALUES (:pid, :url, :pref, 0)"
            ),
            {"pid": new_id, "url": url.rstrip("/"), "pref": a["is_preferred"]},
        )

    moved: set[str] = set()
    models = bind.execute(
        sa.text(
            "SELECT id, name, raw_name, size, digest, modified_at, details "
            "FROM provider_models WHERE provider_id = :id"
        ),
        {"id": old_id},
    ).mappings()
    for m in list(models):
        details = _details(m["details"])
        in_lcpp = bool(details.get("_in_llamacpp"))
        # Rows written before the markers existed count as Ollama-side.
        in_ollama = bool(details.get("_in_ollama", not in_lcpp))
        if not in_lcpp:
            continue
        bind.execute(
            sa.text(
                "INSERT INTO provider_models (provider_id, name, raw_name, size, "
                "digest, modified_at, details) VALUES "
                "(:pid, :name, :raw, :size, :digest, :mod, :details)"
            ),
            {
                "pid": new_id,
                "name": m["name"],
                "raw": m["raw_name"],
                "size": m["size"],
                "digest": m["digest"],
                "mod": m["modified_at"],
                "details": _dump(details),
            },
        )
        if not in_ollama:
            bind.execute(
                sa.text("DELETE FROM provider_models WHERE id = :id"), {"id": m["id"]}
            )
            moved.add(m["name"])
            if m["raw_name"]:
                moved.add(m["raw_name"])

    bind.execute(
        sa.text(
            "UPDATE benchmarks SET provider_id = :new "
            "WHERE provider_id = :old AND protocol = 'llamacpp'"
        ),
        {"new": new_id, "old": old_id},
    )
    for model in moved:
        bind.execute(
            sa.text(
                "UPDATE api_key_model_pins SET provider_id = :new "
                "WHERE provider_id = :old AND model_name = :m"
            ),
            {"new": new_id, "old": old_id, "m": model},
        )
    bind.execute(
        sa.text(
            "UPDATE providers SET capabilities = NULL, probed_at = NULL "
            "WHERE id IN (:old, :new)"
        ),
        {"old": old_id, "new": new_id},
    )


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {c["name"] for c in insp.get_columns("providers")}
    addr_cols = {c["name"] for c in insp.get_columns("provider_addresses")}

    if "protocol" not in cols:
        op.add_column(
            "providers",
            sa.Column("protocol", ShortText, nullable=False, server_default="ollama"),
        )

    if "provider_type" in cols:
        bind.execute(
            sa.text(
                "UPDATE providers SET protocol = 'openai' "
                "WHERE provider_type = 'llamacpp'"
            )
        )
        both = bind.execute(
            sa.text(
                "SELECT id, name, machine_type, gpu_type, gpu_ram, worker_name, "
                "api_key_enc, api_key_header FROM providers "
                "WHERE provider_type = 'both' ORDER BY id"
            )
        ).mappings()
        for row in list(both):
            if row["worker_name"] is None and "llamacpp_url" in addr_cols:
                _split(bind, row)
            bind.execute(
                sa.text("UPDATE providers SET protocol = 'ollama' WHERE id = :id"),
                {"id": row["id"]},
            )

    bind.execute(
        sa.text("UPDATE benchmarks SET protocol = 'openai' WHERE protocol = 'llamacpp'")
    )

    rows = bind.execute(
        sa.text(
            "SELECT id, details FROM provider_models "
            # "_" matches any character here; that only widens the scan.
            "WHERE details LIKE '%_in_ollama%' OR details LIKE '%_in_llamacpp%'"
        )
    ).mappings()
    for r in list(rows):
        bind.execute(
            sa.text("UPDATE provider_models SET details = :d WHERE id = :id"),
            {"d": _dump(_details(r["details"])), "id": r["id"]},
        )

    sqlite = bind.dialect.name == "sqlite"
    if "provider_type" in cols:
        if sqlite:
            with op.batch_alter_table("providers", recreate="always") as batch:
                batch.drop_column("provider_type")
        else:
            op.drop_column("providers", "provider_type")
    if "llamacpp_url" in addr_cols:
        if sqlite:
            with op.batch_alter_table("provider_addresses", recreate="always") as batch:
                batch.drop_column("llamacpp_url")
        else:
            op.drop_column("provider_addresses", "llamacpp_url")


def downgrade() -> None:
    with op.batch_alter_table("provider_addresses") as batch:
        batch.add_column(sa.Column("llamacpp_url", sa.Text(), nullable=True))
    with op.batch_alter_table("providers") as batch:
        batch.add_column(
            sa.Column(
                "provider_type", ShortText, nullable=False, server_default="ollama"
            )
        )
    bind = op.get_bind()
    bind.execute(
        sa.text(
            "UPDATE providers SET provider_type = 'llamacpp' WHERE protocol = 'openai'"
        )
    )
    bind.execute(
        sa.text("UPDATE benchmarks SET protocol = 'llamacpp' WHERE protocol = 'openai'")
    )
    with op.batch_alter_table("providers") as batch:
        batch.drop_column("protocol")
