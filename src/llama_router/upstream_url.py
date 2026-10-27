"""Where an OpenAI-style upstream's API lives under its address URL.

Two conventions meet here. Local servers are entered as the server root
(``http://gpu:8080``) and serve the API under ``/v1``. Cloud services are
entered the way their SDKs take them, as the API base itself
(``https://api.openai.com/v1``, ``https://openrouter.ai/api/v1``,
``https://generativelanguage.googleapis.com/v1beta/openai``). A URL whose path
ends in a version segment, optionally followed by ``openai``, is an API base;
any other URL is a server root with the API at ``/v1`` below it.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

_VERSION = re.compile(r"^v\d+[a-z0-9]*$")
_PREFIX_WORDS = {"api", "openai", "inference"}


def split_api_base(url: str) -> tuple[str, str]:
    """Return ``(root_url, api_path)`` for an address URL.

    ``root_url`` is where server-native routes such as llama.cpp's ``/health``
    and ``/props`` live. ``api_path`` is the path that replaces ``/v1`` in
    OpenAI-style routes, relative to the URL's origin.
    """
    raw = (url or "").strip().rstrip("/")
    parts = urlsplit(raw)
    origin = f"{parts.scheme}://{parts.netloc}" if parts.scheme else ""
    segs = [s for s in parts.path.split("/") if s]
    version_at = None
    for i in range(len(segs) - 1, -1, -1):
        if _VERSION.match(segs[i]):
            version_at = i
            break
    if version_at is not None and all(s == "openai" for s in segs[version_at + 1 :]):
        root = segs[:version_at]
        while root and root[-1] in _PREFIX_WORDS:
            root = root[:-1]
        root_path = "/" + "/".join(root) if root else ""
        return origin + root_path, "/" + "/".join(segs)
    root_path = "/" + "/".join(segs) if segs else ""
    return origin + root_path, root_path + "/v1"


def api_path_for(url: str, path: str) -> str:
    """Map an OpenAI route written as ``/v1/...`` onto *url*'s API base.

    The result is relative to the URL's origin.
    """
    _, api = split_api_base(url)
    if path == "/v1" or path.startswith("/v1/"):
        return api + path[3:]
    return path


LOCAL = "local"
CLOUD = "cloud"
LOCALITIES = (LOCAL, CLOUD)

_LOCAL_SUFFIXES = (".local", ".lan", ".home", ".internal", ".home.arpa", ".ts.net")
_LOCAL_NETWORKS = tuple(
    ipaddress.ip_network(n)
    for n in (
        "127.0.0.0/8",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",
        "100.64.0.0/10",  # carrier-grade NAT, used by Tailscale
        "::1/128",
        "fc00::/7",
        "fe80::/10",
    )
)


def guess_locality(url: str) -> str:
    """``local`` for an address on this machine or a private network, else
    ``cloud``.

    Private, loopback, link-local and Tailscale (100.64.0.0/10) addresses,
    hostnames without a dot, and names under .local, .lan, .home, .internal,
    .home.arpa or .ts.net count as local. A home server reached through a
    public name guesses wrong; the provider's setting can be changed.
    """
    raw = (url or "").strip()
    if raw.startswith("worker://"):
        return LOCAL
    host = (urlsplit(raw).hostname or "").lower().rstrip(".")
    if not host:
        return LOCAL
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if "." not in host or host.endswith(_LOCAL_SUFFIXES) or host == "localhost":
            return LOCAL
        return CLOUD
    if any(ip in net for net in _LOCAL_NETWORKS):
        return LOCAL
    return CLOUD
