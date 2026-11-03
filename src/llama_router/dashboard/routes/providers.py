"""Providers tab and the provider detail page."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)

from ...cloud_presets import get_preset
from ...config import settings
from ...httpx_errors import describe_httpx_error
from ...models import ProviderProtocol
from ...probe import FAMILY_LABELS, BlockedAddressError, probe_url
from ...provider_manager import preset_settings
from ...provider_secrets import format_extra_headers, parse_extra_headers
from .. import deps
from ..auth_core import _cookie_secure
from ..form_values import (
    _model_filter,
    _model_lines,
    _parse_locality,
    _timeout,
    _tristate,
    _usd,
)
from ..pages import _provider_error, _provider_url, dash_page_ctx, templates

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/providers/{provider_id}/prices")
async def api_list_prices(provider_id: int):
    """Prices set by hand, and how many models the provider's list prices."""
    db = deps.get_db()
    if await db.get_provider(provider_id) is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    from ...pricing import listed_price

    listed = [
        {
            "model": m.name,
            "input_per_mtok": p.input_per_mtok,
            "output_per_mtok": p.output_per_mtok,
        }
        for m in await db.get_provider_models(provider_id)
        if (p := listed_price(m)) is not None
    ]
    return JSONResponse(
        {"prices": await db.list_model_prices(provider_id), "listed": listed}
    )


@router.post("/api/providers/{provider_id}/prices")
async def api_set_price(provider_id: int, request: Request):
    """Body: {model_pattern, input_per_mtok, output_per_mtok} in USD per
    million tokens. "*" prices every model of the provider."""
    db = deps.get_db()
    if await db.get_provider(provider_id) is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    body = await deps.json_body(request)
    pattern = str(body.get("model_pattern") or "").strip()
    if not pattern or len(pattern) > 255:
        raise HTTPException(status_code=400, detail="model_pattern is required")
    inp = _usd(body.get("input_per_mtok"), "input price")
    out = _usd(body.get("output_per_mtok"), "output price")
    if inp is None or out is None:
        raise HTTPException(status_code=400, detail="both prices are required")
    await db.set_model_price(provider_id, pattern, inp, out)
    return JSONResponse(
        {"model_pattern": pattern, "input_per_mtok": inp, "output_per_mtok": out}
    )


@router.delete("/api/providers/{provider_id}/prices/{price_id}")
async def api_delete_price(provider_id: int, price_id: int):
    if not await deps.get_db().remove_model_price(provider_id, price_id):
        raise HTTPException(status_code=404, detail="price not found")
    return JSONResponse({"status": "ok"})


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

    cache = deps.get_cache()
    cached_models: set[str] = set()
    cached_only_models: list[dict] = []
    if cache:
        cached_models = await asyncio.to_thread(cache.cached_models)
        provider_and_missing_names = local_names | {m["name"] for m in missing_models}
        for detail in await asyncio.to_thread(cache.cached_model_details):
            if detail["name"] not in provider_and_missing_names:
                total_size = sum(b["size"] for b in detail["blobs"])
                cached_only_models.append({"name": detail["name"], "size": total_size})
        cached_only_models.sort(key=lambda m: m["name"])

    return templates.TemplateResponse(
        request,
        "provider_detail.html",
        await dash_page_ctx(
            request,
            {
                "info": info,
                "extra_headers_text": format_extra_headers(info.provider.extra_headers),
                "manual_models_text": "\n".join(info.provider.manual_models or []),
                "missing_models": missing_models,
                "cached_models": cached_models,
                "cached_only_models": cached_only_models,
                "cache_external_host_set": bool(
                    (settings.cache_external_host or "").strip()
                ),
                "cache_enabled": settings.cache_enabled,
                "family_labels": FAMILY_LABELS,
            },
        ),
    )


# A failed Add Provider form comes back to the dashboard with its fields in
# this short-lived cookie, so the dialog reopens filled in. The API key and
# extra headers can hold secrets and are left out.
_ADD_FORM_COOKIE = "add_provider_form"
_ADD_FORM_FIELDS = (
    "name",
    "url",
    "protocol",
    "machine_type",
    "gpu_type",
    "gpu_ram",
    "api_key_header",
    "preset",
    "preset_resource",
    "manual_models",
    "locality",
    "model_filter",
    "auto_benchmark",
)
_ADD_FORM_MAX = 3000


def _add_provider_failed(
    request: Request, msg: str, form: dict[str, str | None]
) -> RedirectResponse:
    """Back to the dashboard with *msg*, reopening the Add Provider dialog."""
    resp = RedirectResponse(
        url="/?add_provider=1&error=" + quote(msg, safe=""), status_code=303
    )
    kept = {k: v for k, v in form.items() if k in _ADD_FORM_FIELDS and v}
    raw = base64.urlsafe_b64encode(json.dumps(kept).encode()).decode()
    if len(raw) <= _ADD_FORM_MAX:
        resp.set_cookie(
            _ADD_FORM_COOKIE,
            raw,
            max_age=300,
            httponly=True,
            samesite="strict",
            secure=_cookie_secure(request),
        )
    return resp


def _clear_add_form(request: Request, resp: Response) -> None:
    """Expire the prefill cookie with the attributes it was set with."""
    resp.delete_cookie(
        _ADD_FORM_COOKIE,
        secure=_cookie_secure(request),
        httponly=True,
        samesite="strict",
    )


def _add_provider_prefill(request: Request) -> dict[str, str]:
    raw = request.cookies.get(_ADD_FORM_COOKIE, "")
    if not raw or len(raw) > _ADD_FORM_MAX:
        return {}
    try:
        data = json.loads(base64.urlsafe_b64decode(raw.encode()))
    except binascii.Error, ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        k: v for k, v in data.items() if k in _ADD_FORM_FIELDS and isinstance(v, str)
    }


@router.post("/providers/add")
async def add_provider(
    request: Request,
    name: str = Form(...),
    url: str = Form(""),
    protocol: str = Form("ollama"),
    machine_type: str | None = Form(None),
    gpu_type: str | None = Form(None),
    gpu_ram: str | None = Form(None),
    api_key: str | None = Form(None),
    api_key_header: str | None = Form(None),
    preset: str | None = Form(None),
    preset_resource: str | None = Form(None),
    extra_headers: str | None = Form(None),
    manual_models: str | None = Form(None),
    locality: str | None = Form(None),
    model_filter: str | None = Form(None),
    auto_benchmark: str | None = Form(None),
):
    """Create a provider; *url* becomes its first address. It is probed in the
    background, so the stored endpoint list does not wait on the dialog.

    With *preset*, the cloud preset's URL, protocol, key header and extra
    headers fill whatever the form left blank.
    """
    form = {
        "name": name,
        "url": url,
        "protocol": protocol,
        "machine_type": machine_type,
        "gpu_type": gpu_type,
        "gpu_ram": gpu_ram,
        "api_key_header": api_key_header,
        "preset": preset,
        "preset_resource": preset_resource,
        "manual_models": manual_models,
        "locality": locality,
        "model_filter": model_filter,
        "auto_benchmark": auto_benchmark,
    }
    pm = deps.get_pm()
    settings_: dict[str, Any] = {}
    try:
        proto = ProviderProtocol.parse(protocol)
        chosen = get_preset(preset) if preset else None
        if preset and chosen is None:
            raise ValueError("unknown preset")
        if chosen is not None:
            settings_ = preset_settings(chosen)
            proto = ProviderProtocol(chosen.protocol)
            if chosen.url_params or not url.strip():
                url = chosen.url(resource=preset_resource or "")
            if not (api_key_header or "").strip():
                api_key_header = chosen.api_key_header
        if extra_headers is not None and extra_headers.strip():
            settings_["extra_headers"] = parse_extra_headers(extra_headers)
        if manual_models is not None and manual_models.strip():
            settings_["manual_models"] = _model_lines(manual_models)
        if not url.strip():
            raise ValueError("URL is required")
        if (locality or "auto") != "auto":
            settings_["locality"] = _parse_locality(locality)
        if model_filter is not None and model_filter.strip():
            settings_["model_filter"] = _model_filter(model_filter)
        if _tristate(auto_benchmark) is not None:
            settings_["auto_benchmark"] = _tristate(auto_benchmark)
    except ValueError as exc:
        return _add_provider_failed(request, str(exc) or "Invalid provider", form)
    try:
        await pm.add_provider(
            name,
            url,
            proto,
            machine_type=machine_type or None,
            gpu_type=gpu_type or None,
            gpu_ram=gpu_ram or None,
            api_key=(api_key or "").strip() or None,
            api_key_header=api_key_header or None,
            provider_settings=settings_ or None,
        )
    except BlockedAddressError as exc:
        return _add_provider_failed(request, str(exc), form)
    except Exception:
        logger.exception("add_provider failed")
        return _add_provider_failed(request, "Could not add provider", form)
    resp = RedirectResponse(url="/", status_code=303)
    _clear_add_form(request, resp)
    return resp


@router.post("/api/providers/probe")
async def api_probe_url(request: Request):
    """Probe a URL before a provider is saved (Add Provider dialog)."""
    body = await deps.json_body(request)
    url = str(body.get("url") or "")
    api_key = (str(body.get("api_key") or "")).strip() or None
    header = (str(body.get("api_key_header") or "")).strip() or None
    try:
        headers = parse_extra_headers(str(body.get("extra_headers") or ""))
        chosen = get_preset(str(body.get("preset") or ""))
        if chosen is not None:
            headers = {**chosen.extra_headers, **headers}
            header = header or chosen.api_key_header
        if api_key:
            from ...provider_secrets import auth_headers

            headers.update(auth_headers(api_key, header))
        result = await probe_url(url, headers=headers or None, key_sent=bool(api_key))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    data = result.to_dict()
    data["family_labels"] = FAMILY_LABELS
    return JSONResponse(data)


@router.post("/api/providers/{provider_id}/probe")
async def api_probe_provider(provider_id: int):
    """Probe a saved provider and store the endpoints its upstream supports."""
    pm = deps.get_pm()
    try:
        caps = await pm.probe_provider(provider_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if caps is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    return JSONResponse(caps)


@router.post("/providers/{provider_id}/edit")
async def edit_provider(
    provider_id: int,
    name: str = Form(...),
    protocol: str = Form("ollama"),
    machine_type: str | None = Form(None),
    gpu_type: str | None = Form(None),
    gpu_ram: str | None = Form(None),
    api_key: str | None = Form(None),
    api_key_header: str | None = Form(None),
    clear_api_key: str | None = Form(None),
    extra_headers: str | None = Form(None),
    manual_models: str | None = Form(None),
    settings_form: str | None = Form(None),
    locality: str | None = Form(None),
    model_filter: str | None = Form(None),
    read_timeout_s: str | None = Form(None),
    auto_benchmark: str | None = Form(None),
):
    """Edit provider details. A blank API key keeps the stored one.

    The extra headers and added models are written only when the form says
    it carries them (*settings_form*): an empty field then clears them.
    """
    pm = deps.get_pm()
    settings_: dict[str, Any] = {}
    try:
        proto = ProviderProtocol.parse(protocol)
        if settings_form:
            settings_["extra_headers"] = parse_extra_headers(extra_headers) or None
            settings_["manual_models"] = _model_lines(manual_models) or None
            settings_["model_filter"] = _model_filter(model_filter)
            settings_["read_timeout_s"] = _timeout(read_timeout_s)
            settings_["auto_benchmark"] = _tristate(auto_benchmark)
        if locality:
            settings_["locality"] = _parse_locality(locality)
    except ValueError as exc:
        return _provider_error(provider_id, str(exc) or "Invalid protocol")
    try:
        await pm.update_provider(
            provider_id,
            name,
            proto,
            machine_type=machine_type or None,
            gpu_type=gpu_type or None,
            gpu_ram=gpu_ram or None,
            api_key=(api_key or "").strip() or None,
            api_key_header=api_key_header,
            clear_api_key=bool(clear_api_key),
            provider_settings=settings_ or None,
        )
    except ValueError as exc:
        return _provider_error(provider_id, str(exc) or "Invalid provider")
    except Exception:
        logger.exception("update_provider failed (provider_id=%s)", provider_id)
        return _provider_error(provider_id, "Could not update provider")
    return RedirectResponse(url=_provider_url(provider_id), status_code=303)


@router.post("/providers/{provider_id}/remove")
async def remove_provider(provider_id: int):
    pm = deps.get_pm()
    await pm.remove_provider(provider_id)
    return RedirectResponse(url="/", status_code=303)


@router.post("/providers/{provider_id}/refresh")
async def refresh_provider(provider_id: int):
    pm = deps.get_pm()
    await pm.refresh_provider(provider_id)
    return RedirectResponse(url=_provider_url(provider_id), status_code=303)


@router.post("/providers/{provider_id}/delete-model/{model_name:path}")
async def delete_model(provider_id: int, model_name: str):
    pm = deps.get_pm()
    try:
        await pm.delete_remote_model(provider_id, model_name)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            msg = f"Model {model_name} not found on backend (removed from local list)"
            return _provider_error(provider_id, msg)
        logger.error(
            "Delete model: backend HTTP %s for provider_id=%s model=%r",
            exc.response.status_code,
            provider_id,
            model_name,
        )
        return _provider_error(provider_id, "Could not delete model on backend")
    except httpx.HTTPError as exc:
        logger.error(
            "Delete model failed (provider_id=%s, model=%r): %s",
            provider_id,
            model_name,
            describe_httpx_error(exc),
        )
        return _provider_error(provider_id, "Could not delete model on backend")
    except Exception:
        logger.exception(
            "Delete model failed (provider_id=%s, model=%r)",
            provider_id,
            model_name,
        )
        return _provider_error(provider_id, "Could not delete model on backend")
    return RedirectResponse(url=_provider_url(provider_id), status_code=303)


@router.post("/providers/{provider_id}/addresses/add")
async def add_address(
    provider_id: int,
    url: str = Form(...),
    is_preferred: str | None = Form(None),
):
    pm = deps.get_pm()
    try:
        await pm.add_address(provider_id, url, is_preferred=bool(is_preferred))
    except BlockedAddressError as exc:
        return _provider_error(provider_id, str(exc))
    except Exception:
        logger.exception("add_address failed (provider_id=%s)", provider_id)
        return _provider_error(provider_id, "Could not add address")
    return RedirectResponse(url=_provider_url(provider_id), status_code=303)


@router.post("/providers/{provider_id}/addresses/save")
async def save_provider_address(
    provider_id: int,
    url: str = Form(...),
    is_preferred: str | None = Form(None),
    address_id: str | None = Form(None),
):
    """Add or update an address; used by dashboard forms with a stable action URL."""
    db = deps.get_db()
    pm = deps.get_pm()
    pref = bool(is_preferred)
    raw_id = (address_id or "").strip()
    try:
        if not raw_id:
            await pm.add_address(provider_id, url, is_preferred=pref)
        else:
            aid = int(raw_id)
            addr = await db.get_address(aid)
            if addr is None or addr.provider_id != provider_id:
                return _provider_error(provider_id, "Invalid address")
            await pm.update_address(aid, url, is_preferred=pref)
    except BlockedAddressError as exc:
        return _provider_error(provider_id, str(exc))
    except ValueError:
        return _provider_error(provider_id, "Invalid address")
    except Exception:
        logger.exception(
            "save_provider_address failed (provider_id=%s address_id=%r)",
            provider_id,
            raw_id or None,
        )
        return _provider_error(provider_id, "Could not save address")
    return RedirectResponse(url=_provider_url(provider_id), status_code=303)


@router.post("/providers/{provider_id}/addresses/{address_id}/edit")
async def edit_address(
    provider_id: int,
    address_id: int,
    url: str = Form(...),
    is_preferred: str | None = Form(None),
):
    pm = deps.get_pm()
    try:
        await pm.update_address(address_id, url, is_preferred=bool(is_preferred))
    except BlockedAddressError as exc:
        return _provider_error(provider_id, str(exc))
    except Exception:
        logger.exception(
            "update_address failed (provider_id=%s address_id=%s)",
            provider_id,
            address_id,
        )
        return _provider_error(provider_id, "Could not update address")
    return RedirectResponse(url=_provider_url(provider_id), status_code=303)


@router.post("/providers/{provider_id}/addresses/{address_id}/remove")
async def remove_address(provider_id: int, address_id: int):
    pm = deps.get_pm()
    await pm.remove_address(address_id)
    return RedirectResponse(url=_provider_url(provider_id), status_code=303)


@router.post("/providers/{provider_id}/addresses/{address_id}/toggle-preferred")
async def toggle_preferred(provider_id: int, address_id: int):
    pm = deps.get_pm()
    await pm.toggle_address_preferred(address_id)
    return RedirectResponse(url=_provider_url(provider_id), status_code=303)
