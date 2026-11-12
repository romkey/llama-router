"""Reading a client's JSON request body."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request

from .json_io import loads


async def json_object(request: Request) -> dict[str, Any]:
    """The request's JSON body, which must be an object.

    Answers 400 for a body that is not JSON (or nests too deeply to parse), a
    JSON value other than an object, or a ``model`` that is not a string, so
    handlers can index the body without failing on what a client sent.
    A large body is parsed in a worker thread (see json_io).
    """
    try:
        body: Any = await loads(await request.body())
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise HTTPException(status_code=400, detail="JSON body required") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    if "model" in body and not isinstance(body["model"], str):
        raise HTTPException(status_code=400, detail="model must be a string")
    return body
