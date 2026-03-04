from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..models import ProviderType
from . import deps

from .. import __version__

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
templates.env.globals["version"] = __version__

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    pm = deps.get_pm()
    db = deps.get_db()
    infos = await pm.list_provider_infos()
    all_models = await db.list_all_models()
    all_benchmarks = await db.get_all_benchmarks()

    benchmarks_by_model: dict[str, list[dict]] = {}
    for b in all_benchmarks:
        benchmarks_by_model.setdefault(b["model_name"], []).append(b)

    ollama_providers = [i for i in infos if i.provider.supports_ollama]
    ollama_count = len(ollama_providers)
    model_provider_counts: dict[str, int] = {}
    for info in ollama_providers:
        for m in info.models:
            model_provider_counts[m.name] = model_provider_counts.get(m.name, 0) + 1

    log_page = int(request.query_params.get("log_page", "1"))
    log_per_page = 50
    log_total = await db.count_request_logs()
    log_entries = await db.get_request_logs(
        limit=log_per_page, offset=(log_page - 1) * log_per_page
    )
    log_pages = max(1, (log_total + log_per_page - 1) // log_per_page)

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "providers": infos,
            "models": all_models,
            "benchmarks_by_model": benchmarks_by_model,
            "ollama_count": ollama_count,
            "model_provider_counts": model_provider_counts,
            "log_entries": log_entries,
            "log_page": log_page,
            "log_pages": log_pages,
            "log_total": log_total,
        },
    )


@router.get("/providers/{provider_id}", response_class=HTMLResponse)
async def provider_detail(request: Request, provider_id: int):
    pm = deps.get_pm()
    db = deps.get_db()
    info = await pm.get_provider_info(provider_id)
    if not info:
        raise HTTPException(status_code=404, detail="Provider not found")

    local_names = {m.name for m in info.models}
    all_models = await db.list_all_models()
    missing_models = [m for m in all_models if m["name"] not in local_names]

    return templates.TemplateResponse(
        "provider_detail.html",
        {"request": request, "info": info, "missing_models": missing_models},
    )


@router.post("/providers/add")
async def add_provider(
    name: str = Form(...),
    url: str = Form(...),
    provider_type: str = Form("ollama"),
    llamacpp_url: Optional[str] = Form(None),
):
    pm = deps.get_pm()
    ptype = ProviderType(provider_type)
    lcpp_url = llamacpp_url if llamacpp_url else None
    try:
        await pm.add_provider(name, url, ptype, lcpp_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RedirectResponse(url="/", status_code=303)


@router.post("/providers/{provider_id}/edit")
async def edit_provider(
    provider_id: int,
    name: str = Form(...),
    url: str = Form(...),
    provider_type: str = Form("ollama"),
    llamacpp_url: Optional[str] = Form(None),
):
    pm = deps.get_pm()
    ptype = ProviderType(provider_type)
    lcpp_url = llamacpp_url if llamacpp_url else None
    try:
        await pm.update_provider(provider_id, name, url, ptype, lcpp_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RedirectResponse(url=f"/providers/{provider_id}", status_code=303)


@router.post("/providers/{provider_id}/remove")
async def remove_provider(provider_id: int):
    pm = deps.get_pm()
    await pm.remove_provider(provider_id)
    return RedirectResponse(url="/", status_code=303)


@router.post("/providers/{provider_id}/refresh")
async def refresh_provider(provider_id: int):
    pm = deps.get_pm()
    await pm.refresh_provider(provider_id)
    return RedirectResponse(url=f"/providers/{provider_id}", status_code=303)


@router.post("/providers/{provider_id}/benchmark/{model_name:path}")
async def benchmark_model(provider_id: int, model_name: str):
    pm = deps.get_pm()
    result = await pm.benchmark_provider(provider_id, model_name)
    if not result:
        raise HTTPException(status_code=500, detail="Benchmark failed")
    return RedirectResponse(url=f"/providers/{provider_id}", status_code=303)


@router.post("/providers/{provider_id}/delete-model/{model_name:path}")
async def delete_model(provider_id: int, model_name: str):
    pm = deps.get_pm()
    try:
        await pm.delete_remote_model(provider_id, model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return RedirectResponse(url=f"/providers/{provider_id}", status_code=303)


@router.post("/providers/{provider_id}/addresses/add")
async def add_address(
    provider_id: int,
    url: str = Form(...),
    llamacpp_url: Optional[str] = Form(None),
    is_preferred: Optional[str] = Form(None),
):
    pm = deps.get_pm()
    lcpp = llamacpp_url if llamacpp_url else None
    try:
        await pm.add_address(provider_id, url, lcpp, is_preferred=bool(is_preferred))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RedirectResponse(url=f"/providers/{provider_id}", status_code=303)


@router.post("/providers/{provider_id}/addresses/{address_id}/edit")
async def edit_address(
    provider_id: int,
    address_id: int,
    url: str = Form(...),
    llamacpp_url: Optional[str] = Form(None),
    is_preferred: Optional[str] = Form(None),
):
    pm = deps.get_pm()
    lcpp = llamacpp_url if llamacpp_url else None
    try:
        await pm.update_address(address_id, url, lcpp, is_preferred=bool(is_preferred))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RedirectResponse(url=f"/providers/{provider_id}", status_code=303)


@router.post("/providers/{provider_id}/addresses/{address_id}/remove")
async def remove_address(provider_id: int, address_id: int):
    pm = deps.get_pm()
    await pm.remove_address(address_id)
    return RedirectResponse(url=f"/providers/{provider_id}", status_code=303)


@router.post("/providers/{provider_id}/addresses/{address_id}/toggle-preferred")
async def toggle_preferred(provider_id: int, address_id: int):
    pm = deps.get_pm()
    await pm.toggle_address_preferred(address_id)
    return RedirectResponse(url=f"/providers/{provider_id}", status_code=303)


@router.post("/providers/{provider_id}/pull")
async def pull_model(provider_id: int, model: str = Form(...)):
    pm = deps.get_pm()
    client = pm.get_client(provider_id)
    try:
        async for _ in client.pull_stream(model):
            pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    await pm.refresh_provider(provider_id)
    return RedirectResponse(url=f"/providers/{provider_id}", status_code=303)


@router.post("/models/pull-all")
async def pull_model_all_providers(model: str = Form(...)):
    pm = deps.get_pm()
    infos = await pm.list_provider_infos()
    ollama_infos = [i for i in infos if i.provider.supports_ollama]

    async def _pull_one(info):
        assert info.provider.id is not None
        client = pm.get_ollama_client(info.provider.id)
        try:
            async for _ in client.pull_stream(model):
                pass
            await pm.refresh_provider(info.provider.id)
        except Exception:
            pass

    await asyncio.gather(*[_pull_one(i) for i in ollama_infos])
    return RedirectResponse(url="/#models-pane", status_code=303)


@router.post("/models/delete-all")
async def delete_model_all_providers(model: str = Form(...)):
    pm = deps.get_pm()
    infos = await pm.list_provider_infos()
    targets = [
        i
        for i in infos
        if i.provider.supports_ollama and any(m.name == model for m in i.models)
    ]

    async def _delete_one(info):
        assert info.provider.id is not None
        try:
            await pm.delete_remote_model(info.provider.id, model)
        except Exception:
            pass

    await asyncio.gather(*[_delete_one(i) for i in targets])
    return RedirectResponse(url="/#models-pane", status_code=303)
