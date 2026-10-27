"""Data behind the dashboard provider filter pills."""

from __future__ import annotations

import json
import re

import pytest

from llama_router.models import ProviderModel, ProviderProtocol


async def _seed(db):
    p = await db.add_provider("gpu-1", ProviderProtocol.OLLAMA)
    await db.add_address(p.id, "http://10.0.0.5:11434", True)
    await db.set_provider_models(
        p.id,
        [
            ProviderModel(provider_id=p.id, name="qwen3:8b"),
            ProviderModel(provider_id=p.id, name="llama3.2:latest"),
        ],
    )
    return p


@pytest.mark.asyncio
async def test_status_api_carries_filter_fields(dash_client) -> None:
    client, db = dash_client
    await _seed(db)
    data = (await client.get("/api/status")).json()
    p = data["providers"][0]
    assert p["model_names"] == ["llama3.2:latest", "qwen3:8b"]
    assert p["is_worker"] is False
    assert p["protocol"] == "ollama"
    assert "status_detail" in p


@pytest.mark.asyncio
async def test_dashboard_embeds_provider_seed(dash_client) -> None:
    client, db = dash_client
    await _seed(db)
    html = (await client.get("/")).text
    m = re.search(
        r'<script type="application/json" id="provider-filter-seed">(.*?)</script>',
        html,
        re.S,
    )
    assert m, "filter seed missing"
    seed = json.loads(m.group(1))
    assert [s["name"] for s in seed] == ["gpu-1"]
    assert 'id="provider-filter-groups"' in html
