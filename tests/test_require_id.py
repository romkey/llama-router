from __future__ import annotations

import pytest

from llama_router.models import Provider, ProviderAddress


def test_require_id_returns_the_saved_id() -> None:
    assert Provider(id=3, name="gpu").require_id() == 3
    assert ProviderAddress(id=4, provider_id=3, url="http://x").require_id() == 4


def test_require_id_rejects_unsaved_rows() -> None:
    with pytest.raises(ValueError, match="has no id"):
        Provider(name="gpu").require_id()
    with pytest.raises(ValueError, match="has no id"):
        ProviderAddress(provider_id=3, url="http://x").require_id()
