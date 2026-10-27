"""Provider detail page markup."""

from __future__ import annotations

from html.parser import HTMLParser

import pytest

from llama_router.models import ProviderProtocol


class _EditButtons(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.buttons: list[dict[str, str | None]] = []

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag == "button" and "data-edit-address" in d:
            self.buttons.append(d)


@pytest.mark.asyncio
async def test_edit_address_button_carries_address_in_data_attributes(
    dash_client,
) -> None:
    client, db = dash_client
    provider = await db.add_provider("gpu-1", ProviderProtocol.OLLAMA)
    await db.add_address(provider.id, "http://10.0.0.5:11434", True)

    resp = await client.get(f"/providers/{provider.id}")
    assert resp.status_code == 200
    parser = _EditButtons()
    parser.feed(resp.text)

    assert len(parser.buttons) == 1
    btn = parser.buttons[0]
    # No inline handler: JSON quotes inside onclick="..." broke the attribute.
    assert "onclick" not in btn
    assert btn["data-addr-url"] == "http://10.0.0.5:11434"
    assert "data-addr-lcpp" not in btn
    assert btn["data-addr-preferred"] == "true"
    assert btn["data-addr-id"].isdigit()
