from __future__ import annotations

import pytest

from llama_router.wireguard_config import (
    generate_wireguard_private_key,
    is_valid_wg_key_b64,
    public_key_from_private,
    render_wg_quick_config,
)


def test_generate_and_derive_public_roundtrip() -> None:
    priv = generate_wireguard_private_key()
    assert is_valid_wg_key_b64(priv)
    pub = public_key_from_private(priv)
    assert is_valid_wg_key_b64(pub)
    pub2 = public_key_from_private(priv)
    assert pub == pub2


def test_render_disabled_is_comment_only() -> None:
    text = render_wg_quick_config(
        {"enabled": False, "private_key": generate_wireguard_private_key()},
        [],
    )
    assert "[Interface]" not in text
    assert "llama-router" in text


def test_render_enabled_minimal_peer() -> None:
    priv = generate_wireguard_private_key()
    priv_peer = generate_wireguard_private_key()
    pub_peer = public_key_from_private(priv_peer)
    text = render_wg_quick_config(
        {
            "enabled": True,
            "private_key": priv,
            "address_cidr": "10.8.0.1/24",
            "listen_port": 51820,
            "mtu": None,
        },
        [
            {
                "enabled": True,
                "public_key": pub_peer,
                "allowed_ips": "10.8.0.2/32",
                "preshared_key": "",
                "endpoint": "",
                "persistent_keepalive": None,
            }
        ],
    )
    assert "[Interface]" in text
    assert f"PrivateKey = {priv}" in text
    assert "Address = 10.8.0.1/24" in text
    assert "ListenPort = 51820" in text
    assert "[Peer]" in text
    assert f"PublicKey = {pub_peer}" in text
    assert "AllowedIPs = 10.8.0.2/32" in text


def test_render_invalid_private_raises() -> None:
    with pytest.raises(ValueError, match="Invalid interface private key"):
        render_wg_quick_config(
            {
                "enabled": True,
                "private_key": "not-a-key====",
                "address_cidr": "10.0.0.1/24",
            },
            [],
        )


def test_render_peer_keepalive_and_endpoint() -> None:
    priv = generate_wireguard_private_key()
    pub_peer = public_key_from_private(generate_wireguard_private_key())
    text = render_wg_quick_config(
        {
            "enabled": True,
            "private_key": priv,
            "address_cidr": "10.1.0.1/24",
            "listen_port": 12345,
            "mtu": 1280,
        },
        [
            {
                "enabled": True,
                "public_key": pub_peer,
                "allowed_ips": "10.1.0.2/32,192.168.0.0/24",
                "endpoint": "peer.example:51820",
                "persistent_keepalive": 25,
            }
        ],
    )
    assert "MTU = 1280" in text
    assert "Endpoint = peer.example:51820" in text
    assert "PersistentKeepalive = 25" in text
