"""Live check of each cloud preset against the real service.

Skipped unless that preset's LLAMA_ROUTER_PROVIDER_<ID>_API_KEY is set (and,
for Azure, ..._RESOURCE). Run it when a preset may have gone stale:

    LLAMA_ROUTER_PROVIDER_OPENAI_API_KEY=sk-... pytest tests/test_cloud_presets_live.py
"""

from __future__ import annotations

import os

import pytest

from llama_router.cloud_presets import PRESETS
from llama_router.llamacpp_client import OpenAIClient
from llama_router.ollama_client import OllamaClient
from llama_router.provider_secrets import auth_headers


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.id)
async def test_preset_lists_models(preset) -> None:
    key = os.environ.get(preset.env_var, "").strip()
    if not key:
        pytest.skip(f"{preset.env_var} not set")
    prefix = preset.env_var[: -len("API_KEY")]
    params = {k: os.environ.get(prefix + k.upper(), "") for k in preset.url_params}
    headers = {**preset.extra_headers, **auth_headers(key, preset.api_key_header)}
    url = preset.url(**params)
    if preset.protocol == "ollama":
        client = OllamaClient(url, headers=headers)
        models = await client.get_tags()
    else:
        client = OpenAIClient(url, headers=headers)
        models = await client.get_models()
    await client.close()
    assert models or preset.models, f"{preset.id} listed no models"
