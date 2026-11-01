"""X-Forwarded-For is believed only from trusted proxies."""

from __future__ import annotations

import pytest
from starlette.requests import Request

from llama_router.client_address import client_ip, parse_trusted_proxies
from llama_router.config import settings


def _request(peer: str, forwarded: str | None = None) -> Request:
    headers = []
    if forwarded is not None:
        headers.append((b"x-forwarded-for", forwarded.encode()))
    return Request(
        {"type": "http", "headers": headers, "client": (peer, 1234), "path": "/"}
    )


def test_header_ignored_without_trusted_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "trusted_proxies", "")
    assert client_ip(_request("203.0.113.9", "1.2.3.4")) == "203.0.113.9"


def test_header_ignored_from_untrusted_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "trusted_proxies", "10.0.0.0/8")
    assert client_ip(_request("203.0.113.9", "1.2.3.4")) == "203.0.113.9"


def test_trusted_proxy_header_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "trusted_proxies", "10.0.0.0/8, 127.0.0.1")
    assert client_ip(_request("10.1.2.3", "198.51.100.7")) == "198.51.100.7"


def test_rightmost_untrusted_hop_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A client cannot spoof by prepending its own entries."""
    monkeypatch.setattr(settings, "trusted_proxies", "10.0.0.0/8")
    req = _request("10.0.0.1", "6.6.6.6, 198.51.100.7, 10.0.0.2")
    assert client_ip(req) == "198.51.100.7"


def test_non_ip_values_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "trusted_proxies", "10.0.0.1")
    assert client_ip(_request("10.0.0.1", "<script>")) == "10.0.0.1"
    assert client_ip(_request("10.0.0.1", "1.2.3.4, evil")) == "10.0.0.1"


def test_ipv6(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "trusted_proxies", "::1")
    assert client_ip(_request("::1", "2001:db8::5")) == "2001:db8::5"


def test_invalid_entries_are_skipped() -> None:
    nets = parse_trusted_proxies("10.0.0.0/8,nonsense 192.168.1.1")
    assert [str(n) for n in nets] == ["10.0.0.0/8", "192.168.1.1/32"]
