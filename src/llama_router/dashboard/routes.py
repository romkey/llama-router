from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from . import deps

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    pm = deps.get_pm()
    infos = await pm.list_provider_infos()
    all_models = await deps.get_db().list_all_models()
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "providers": infos, "models": all_models},
    )


@router.get("/providers/{provider_id}", response_class=HTMLResponse)
async def provider_detail(request: Request, provider_id: int):
    pm = deps.get_pm()
    info = await pm.get_provider_info(provider_id)
    if not info:
        raise HTTPException(status_code=404, detail="Provider not found")
    return templates.TemplateResponse(
        "provider_detail.html",
        {"request": request, "info": info},
    )


@router.post("/providers/add")
async def add_provider(name: str = Form(...), url: str = Form(...)):
    pm = deps.get_pm()
    try:
        await pm.add_provider(name, url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RedirectResponse(url="/", status_code=303)


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
