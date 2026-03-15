from __future__ import annotations

import pytest

from llama_router.dashboard.routes import _ollama_library_slug, _ollama_library_url


@pytest.mark.parametrize(
    ("model_name", "expected_slug"),
    [
        ("llama3.2:latest", "llama3.2"),
        ("nomic-embed-text", "nomic-embed-text"),
        ("myorg/custom-model:Q4_K_M", "myorg/custom-model"),
        ("myorg/custom-model@sha256:abcdef123456", "myorg/custom-model"),
        # Cache/registry-prefixed names should normalize to just model path.
        ("127.0.0.1:7444/library/llama3.2:latest", "llama3.2"),
        ("cache-host.local:7444/library/myorg/model:Q8_0", "myorg/model"),
        ("registry.example.com/myorg/model:latest", "myorg/model"),
        # Full URL forms should normalize too.
        ("http://cache-host.local:7444/library/llama3.2:latest", "llama3.2"),
    ],
)
def test_ollama_library_slug_normalizes_model_names(
    model_name: str, expected_slug: str
) -> None:
    assert _ollama_library_slug(model_name) == expected_slug


def test_ollama_library_url_uses_normalized_slug() -> None:
    model_name = "127.0.0.1:7444/library/myorg/model:Q4_K_M"
    assert _ollama_library_url(model_name) == "https://ollama.com/library/myorg/model"


def test_ollama_library_url_empty_name_falls_back_to_library_root() -> None:
    assert _ollama_library_url("   ") == "https://ollama.com/library"
