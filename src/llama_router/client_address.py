"""The client's IP address, honouring X-Forwarded-For only from trusted proxies.

Anyone can send an ``X-Forwarded-For`` header, so it is believed only when
the TCP peer is listed in ``LLAMA_ROUTER_TRUSTED_PROXIES``. The header is
then read from the right, skipping trusted proxies, and the first other
address is the client. Anything that is not an IP literal is ignored.
"""

from __future__ import annotations

import ipaddress
import logging
from functools import lru_cache

from starlette.requests import HTTPConnection

from .config import settings

logger = logging.getLogger(__name__)

Network = ipaddress.IPv4Network | ipaddress.IPv6Network


@lru_cache(maxsize=8)
def parse_trusted_proxies(value: str) -> tuple[Network, ...]:
    """Networks from a comma- or space-separated list of IPs and CIDRs."""
    nets: list[Network] = []
    for item in value.replace(",", " ").split():
        try:
            nets.append(ipaddress.ip_network(item, strict=False))
        except ValueError:
            logger.warning("Ignoring invalid LLAMA_ROUTER_TRUSTED_PROXIES entry %r", item)
    return tuple(nets)


def _ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(value.strip())
    except ValueError:
        return None


def _trusted(
    addr: ipaddress.IPv4Address | ipaddress.IPv6Address, nets: tuple[Network, ...]
) -> bool:
    return any(addr in net for net in nets)


def client_ip(request: HTTPConnection) -> str:
    """The address the request came from, for logs and per-client limits."""
    peer = request.client.host if request.client else ""
    nets = parse_trusted_proxies(settings.trusted_proxies)
    peer_addr = _ip(peer)
    forwarded = request.headers.get("x-forwarded-for")
    if not forwarded or not nets or peer_addr is None or not _trusted(peer_addr, nets):
        return peer or "unknown"
    for hop in reversed(forwarded.split(",")):
        addr = _ip(hop)
        if addr is None:
            break  # garbage from here on; fall back to the last good hop
        peer_addr = addr
        if not _trusted(addr, nets):
            break
    return str(peer_addr)
