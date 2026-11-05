"""Serve the site icons (favicon, touch icons, web manifest) without a login.

Browsers and iOS fetch these at fixed root paths, often before or without
a session, so the auth middleware lets them through (``ICON_PATHS``).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import Response

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_MEDIA_TYPES = {
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".webmanifest": "application/manifest+json",
}

# Every file in static/ is a public icon; the set is fixed at import time
# so a request path can never reach outside that directory.
_FILES = {
    f"/{p.name}": p for p in sorted(_STATIC_DIR.iterdir()) if p.suffix in _MEDIA_TYPES
}
ICON_PATHS = frozenset(_FILES)

router = APIRouter()


async def _serve(path: str) -> Response:
    file = _FILES[path]
    return Response(
        file.read_bytes(),
        media_type=_MEDIA_TYPES[file.suffix],
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _register(path: str) -> None:
    async def handler() -> Response:
        return await _serve(path)

    router.add_api_route(path, handler, methods=["GET"], include_in_schema=False)


for _path in _FILES:
    _register(_path)
