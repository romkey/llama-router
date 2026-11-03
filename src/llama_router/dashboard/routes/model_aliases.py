"""Models tab: fallbacks, aliases and removing a model everywhere."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import (
    JSONResponse,
    RedirectResponse,
)

from .. import deps

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/fallbacks")
async def set_fallback(request: Request):
    """Set or update a model fallback. Body: {model: str, fallback: str}."""
    body = await deps.json_body(request)
    model = body.get("model", "").strip()
    fallback = body.get("fallback", "").strip()
    if not model or not fallback:
        raise HTTPException(status_code=400, detail="model and fallback are required")
    if model == fallback:
        raise HTTPException(
            status_code=400, detail="A model cannot be its own fallback"
        )
    db = deps.get_db()
    await db.set_model_fallback(model, fallback)
    return JSONResponse({"status": "ok", "model": model, "fallback": fallback})


@router.delete("/api/fallbacks/{model_name:path}")
async def remove_fallback(model_name: str):
    """Remove a model fallback."""
    db = deps.get_db()
    await db.remove_model_fallback(model_name)
    return JSONResponse({"status": "ok"})


def suggest_aliases(
    names_by_provider: dict[int, set[str]], aliases: dict[str, str]
) -> list[dict[str, str]]:
    """Likely aliases between providers' model names.

    ``vendor/model`` (OpenRouter, Together, ...) and ``model:latest``
    (Ollama) are suggested as aliases of ``model`` when another provider
    serves that bare name. Existing aliases are not suggested again.
    """
    everywhere: dict[str, set[int]] = {}
    for pid, names in names_by_provider.items():
        for n in names:
            everywhere.setdefault(n, set()).add(pid)
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for name, pids in sorted(everywhere.items()):
        candidates = []
        if "/" in name:
            candidates.append(name.rsplit("/", 1)[1])
        if name.endswith(":latest"):
            candidates.append(name[: -len(":latest")])
        for bare in candidates:
            if not bare or bare == name or name in aliases or name in seen:
                continue
            if everywhere.get(bare, set()) - pids:
                out.append({"alias": name, "canonical": bare})
                seen.add(name)
    return out


@router.get("/api/model-aliases")
async def list_model_aliases():
    """All aliases (alias -> canonical) and suggested new ones."""
    db = deps.get_db()
    aliases = await db.get_model_aliases()
    names: dict[int, set[str]] = {}
    for m in await db.get_all_provider_models():
        names.setdefault(m.provider_id, set()).add(m.name)
    return JSONResponse(
        {"aliases": aliases, "suggestions": suggest_aliases(names, aliases)}
    )


@router.post("/api/model-aliases")
async def set_model_alias(request: Request):
    """Body: {alias, canonical}. The alias then reaches every provider of the
    canonical name and of its other aliases."""
    body = await deps.json_body(request)
    alias = str(body.get("alias") or "").strip()
    canonical = str(body.get("canonical") or "").strip()
    if not alias or not canonical:
        raise HTTPException(status_code=400, detail="alias and canonical are required")
    if alias == canonical:
        raise HTTPException(status_code=400, detail="a model cannot alias itself")
    if len(alias) > 255 or len(canonical) > 255:
        raise HTTPException(status_code=400, detail="model name too long")
    db = deps.get_db()
    aliases = await db.get_model_aliases()
    # Refuse a loop: following canonical's chain must never reach alias.
    cur, seen = canonical, set()
    while cur not in seen:
        if cur == alias:
            raise HTTPException(status_code=400, detail="that alias would form a loop")
        seen.add(cur)
        if cur not in aliases:
            break
        cur = aliases[cur]
    await db.set_model_alias(alias, canonical)
    return JSONResponse({"alias": alias, "canonical": canonical})


@router.delete("/api/model-aliases/{alias:path}")
async def delete_model_alias(alias: str):
    if not await deps.get_db().remove_model_alias(alias):
        raise HTTPException(status_code=404, detail="alias not found")
    return JSONResponse({"status": "ok"})


@router.get("/api/fallbacks")
async def list_fallbacks():
    """Return all configured model fallbacks."""
    db = deps.get_db()
    fallbacks = await db.get_all_model_fallbacks()
    return JSONResponse(fallbacks)


@router.post("/models/delete-all")
async def delete_model_all_providers(model: str = Form(...)):
    pm = deps.get_pm()
    infos = await pm.list_provider_infos()
    targets = [
        i
        for i in infos
        if i.provider.is_ollama and any(m.name == model for m in i.models)
    ]

    async def _delete_one(info):
        provider_id = info.provider.require_id()
        try:
            await pm.delete_remote_model(provider_id, model)
        except Exception:
            logger.debug(
                "delete_remote_model failed for provider %d model %s",
                provider_id,
                model,
            )

    await asyncio.gather(*[_delete_one(i) for i in targets])
    return RedirectResponse(url="/#models-pane", status_code=303)
