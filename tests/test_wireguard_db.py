from __future__ import annotations

import pytest

from llama_router.wireguard_config import (
    generate_wireguard_private_key,
    public_key_from_private,
)


@pytest.mark.asyncio
async def test_wireguard_interface_singleton_and_peers(db):
    iface = await db.get_wireguard_interface()
    assert iface["id"] == 1
    assert "address_cidr" in iface

    priv = generate_wireguard_private_key()
    await db.update_wireguard_interface(
        enabled=True,
        listen_port=51999,
        address_cidr="10.99.0.1/24",
        mtu=1400,
        endpoint_public="vpn.test:51999",
        new_private_key=priv,
    )
    iface2 = await db.get_wireguard_interface()
    assert iface2["enabled"] is True
    assert iface2["listen_port"] == 51999
    assert iface2["public_key"] == public_key_from_private(priv)

    pub_peer = public_key_from_private(generate_wireguard_private_key())
    pid = await db.add_wireguard_peer(
        name="peer-a",
        public_key=pub_peer,
        allowed_ips="10.99.0.2/32",
        endpoint="remote:51820",
        persistent_keepalive=25,
        enabled=True,
    )
    peers = await db.list_wireguard_peers()
    assert len(peers) == 1
    assert peers[0]["id"] == pid
    assert peers[0]["name"] == "peer-a"

    await db.remove_wireguard_peer(pid)
    assert await db.list_wireguard_peers() == []

    cfg = await db.get_wireguard_peering_config()
    assert "peering_enabled" in cfg
    assert "peering_api_key" in cfg
    await db.set_wireguard_peering_config(True, "test-secret-key")
    cfg2 = await db.get_wireguard_peering_config()
    assert cfg2["peering_enabled"] is True
    assert cfg2["peering_api_key"] == "test-secret-key"
    assert cfg2.get("peering_key_use_count") == 0
    assert "peering_key_expires_at" in cfg2


@pytest.mark.asyncio
async def test_increment_peering_key_use_count(db) -> None:
    await db.set_wireguard_peering_config(
        True,
        "k",
        peering_key_max_uses=99,
        reset_peering_key_use_count=True,
    )
    assert await db.increment_peering_key_use_count() == 1
    assert await db.increment_peering_key_use_count() == 2
