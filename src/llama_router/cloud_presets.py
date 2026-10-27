"""Ready-made settings for common cloud LLM services.

Choosing a preset in the Add Provider dialog fills in the URL, protocol, key
header and any extra headers; the user supplies a name and a key. The same
table lets the probe recognise a service by its hostname, which is cheaper and
more reliable than reading its responses.

``checked`` records when the entry was last compared with the service's own
documentation; entries without a date come from general knowledge and should
be confirmed with ``tests/test_cloud_presets_live.py`` and a real key.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlsplit


@dataclass(frozen=True)
class CloudPreset:
    id: str
    name: str
    base_url: str
    protocol: str = "openai"
    # Comma-separated headers that carry the key; Authorization gets "Bearer".
    api_key_header: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Hostnames (fnmatch patterns) that identify the service.
    hosts: tuple[str, ...] = ()
    docs_url: str = ""
    key_url: str = ""
    # Models to add by hand: for services without a model list, or whose list
    # leaves out models it serves.
    models: tuple[str, ...] = ()
    # Default discovery filter, for services that list very many models.
    model_filter: str = ""
    # Placeholders in base_url the user must fill in, e.g. {"resource": "..."}.
    url_params: dict[str, str] = field(default_factory=dict)
    checked: str | None = None

    @property
    def env_var(self) -> str:
        """Environment variable whose value provisions this preset at startup."""
        slug = re.sub(r"[^A-Z0-9]+", "_", self.id.upper())
        return f"LLAMA_ROUTER_PROVIDER_{slug}_API_KEY"

    def url(self, **params: str) -> str:
        url = self.base_url
        for key in self.url_params:
            value = (params.get(key) or "").strip()
            if not value or not re.fullmatch(r"[A-Za-z0-9-]+", value):
                raise ValueError(f"{key} is required (letters, digits and -)")
            url = url.replace("{" + key + "}", value)
        return url

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["models"] = list(self.models)
        d["hosts"] = list(self.hosts)
        d["env_var"] = self.env_var
        return d


PRESETS: tuple[CloudPreset, ...] = (
    CloudPreset(
        id="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        hosts=("api.openai.com",),
        docs_url="https://platform.openai.com/docs/api-reference",
        key_url="https://platform.openai.com/api-keys",
    ),
    CloudPreset(
        id="openrouter",
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        extra_headers={"X-Title": "llama-router"},
        hosts=("openrouter.ai",),
        docs_url="https://openrouter.ai/docs",
        key_url="https://openrouter.ai/keys",
        model_filter=(
            "openai/gpt-*, anthropic/claude-*, google/gemini-*, deepseek/*, "
            "meta-llama/llama-*, mistralai/*, qwen/*, x-ai/grok-*"
        ),
    ),
    CloudPreset(
        id="anthropic",
        name="Anthropic",
        base_url="https://api.anthropic.com/v1",
        # Bearer for the OpenAI-compatible routes, x-api-key for /v1/models.
        api_key_header="Authorization, x-api-key",
        extra_headers={"anthropic-version": "2023-06-01"},
        hosts=("api.anthropic.com",),
        docs_url="https://platform.claude.com/docs/en/api/openai-sdk",
        key_url="https://platform.claude.com/settings/keys",
        checked="2026-10-27",
    ),
    CloudPreset(
        id="perplexity",
        name="Perplexity",
        base_url="https://api.perplexity.ai/v1",
        hosts=("api.perplexity.ai",),
        docs_url="https://docs.perplexity.ai",
        key_url="https://www.perplexity.ai/settings/api",
        # Its model list covers the Agent API; Sonar models are added by hand.
        models=("sonar", "sonar-pro", "sonar-reasoning-pro", "sonar-deep-research"),
        checked="2026-10-27",
    ),
    CloudPreset(
        id="groq",
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        hosts=("api.groq.com",),
        docs_url="https://console.groq.com/docs/openai",
        key_url="https://console.groq.com/keys",
    ),
    CloudPreset(
        id="together",
        name="Together AI",
        base_url="https://api.together.xyz/v1",
        hosts=("api.together.xyz", "api.together.ai"),
        docs_url="https://docs.together.ai",
        key_url="https://api.together.ai/settings/api-keys",
    ),
    CloudPreset(
        id="fireworks",
        name="Fireworks AI",
        base_url="https://api.fireworks.ai/inference/v1",
        hosts=("api.fireworks.ai",),
        docs_url="https://docs.fireworks.ai",
        key_url="https://fireworks.ai/account/api-keys",
    ),
    CloudPreset(
        id="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        hosts=("api.deepseek.com",),
        docs_url="https://api-docs.deepseek.com",
        key_url="https://platform.deepseek.com/api_keys",
    ),
    CloudPreset(
        id="mistral",
        name="Mistral AI",
        base_url="https://api.mistral.ai/v1",
        hosts=("api.mistral.ai",),
        docs_url="https://docs.mistral.ai/api",
        key_url="https://console.mistral.ai/api-keys",
    ),
    CloudPreset(
        id="gemini",
        name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        hosts=("generativelanguage.googleapis.com",),
        docs_url="https://ai.google.dev/gemini-api/docs/openai",
        key_url="https://aistudio.google.com/apikey",
    ),
    CloudPreset(
        id="xai",
        name="xAI",
        base_url="https://api.x.ai/v1",
        hosts=("api.x.ai",),
        docs_url="https://docs.x.ai",
        key_url="https://console.x.ai",
    ),
    CloudPreset(
        id="cerebras",
        name="Cerebras",
        base_url="https://api.cerebras.ai/v1",
        hosts=("api.cerebras.ai",),
        docs_url="https://inference-docs.cerebras.ai",
        key_url="https://cloud.cerebras.ai",
    ),
    CloudPreset(
        id="deepinfra",
        name="DeepInfra",
        base_url="https://api.deepinfra.com/v1/openai",
        hosts=("api.deepinfra.com",),
        docs_url="https://deepinfra.com/docs/openai_api",
        key_url="https://deepinfra.com/dash/api_keys",
    ),
    CloudPreset(
        id="ollama-cloud",
        name="Ollama Cloud",
        base_url="https://ollama.com",
        protocol="ollama",
        hosts=("ollama.com",),
        docs_url="https://docs.ollama.com/cloud",
        key_url="https://ollama.com/settings/keys",
        checked="2026-10-27",
    ),
    CloudPreset(
        id="azure-openai",
        name="Azure OpenAI",
        base_url="https://{resource}.openai.azure.com/openai/v1",
        api_key_header="api-key",
        hosts=("*.openai.azure.com", "*.cognitiveservices.azure.com"),
        docs_url=(
            "https://learn.microsoft.com/azure/ai-foundry/openai/api-version-lifecycle"
        ),
        key_url="https://portal.azure.com",
        url_params={"resource": "Azure OpenAI resource name"},
        checked="2026-10-27",
    ),
)

_BY_ID = {p.id: p for p in PRESETS}


def get_preset(preset_id: str | None) -> CloudPreset | None:
    return _BY_ID.get((preset_id or "").strip().lower())


def preset_for_url(url: str) -> CloudPreset | None:
    """The preset whose service *url* points at, by hostname."""
    host = (urlsplit((url or "").strip()).hostname or "").lower()
    if not host:
        return None
    for p in PRESETS:
        if any(fnmatch.fnmatch(host, pattern) for pattern in p.hosts):
            return p
    return None


def env_provisioned() -> list[tuple[CloudPreset, str, dict[str, str]]]:
    """Presets whose ``LLAMA_ROUTER_PROVIDER_<ID>_API_KEY`` is set.

    A preset with URL placeholders also needs ``..._<ID>_<PARAM>`` (for Azure
    OpenAI, ``LLAMA_ROUTER_PROVIDER_AZURE_OPENAI_RESOURCE``).
    """
    out = []
    for p in PRESETS:
        key = os.environ.get(p.env_var, "").strip()
        if not key:
            continue
        prefix = p.env_var[: -len("API_KEY")]
        params = {
            name: os.environ.get(prefix + name.upper(), "").strip()
            for name in p.url_params
        }
        out.append((p, key, params))
    return out
