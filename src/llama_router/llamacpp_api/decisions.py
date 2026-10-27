"""Decision requests in the shape of TypeSafe's Jev API.

``POST /v1/systemone`` (Jev's route) and ``POST /v1/decisions`` take a state
and typed questions and answer each with a probability distribution; see
:mod:`llama_router.decisions`. An upstream that serves either route gets the
request as it came, on the route it serves (the bodies are the same). Any other provider that serves chat completions
answers it from first-token logprobs, with the stored calibration for the
model and question applied.

A provider that answers without logprobs is marked in the routing stats, so
the routing table stops offering it for decisions, and the request is routed
once more.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .. import decision_service as svc
from .. import decisions as d
from ..auth import routing_preferences_from_request
from ..endpoints import normalize_endpoint
from ..httpx_errors import describe_httpx_error, forward_upstream_http_error
from ..request_logger import log_request_background
from ..routing_table import native_decision_path
from ..worker_link.protocol import refused as worker_refused
from . import deps

logger = logging.getLogger(__name__)
router = APIRouter()


def _validation_detail(exc: ValidationError) -> list[dict[str, Any]]:
    return [
        {"loc": list(e.get("loc", ())), "msg": e.get("msg", "")}
        for e in exc.errors(include_url=False, include_context=False)
    ]


async def _pass_through(pm, provider, path: str, body: dict) -> dict | None:
    """Send the request on as it came; None when a worker will not relay it.

    A worker older than decision support refuses the route with 403. Its
    stored probe result can still say otherwise, so the request is emulated
    instead and the worker is probed again.
    """
    try:
        return await svc.native(pm, provider.id, path, body)
    except httpx.HTTPStatusError as exc:
        resp = exc.response
        if not (provider.is_worker and worker_refused(resp.status_code, resp.text)):
            raise
    logger.info(
        "Worker %s does not relay %s; answering from logprobs instead",
        provider.name,
        path,
    )
    pm.schedule_probe(provider.id, "worker refused decisions")
    return None


@router.post("/v1/systemone")
@router.post("/systemone")
@router.post("/v1/decisions")
@router.post("/decisions")
async def decide(request: Request):
    endpoint_path = request.url.path
    path = normalize_endpoint(endpoint_path)
    assert path is not None
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="JSON body required")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    try:
        req = d.DecisionRequest.model_validate(body)
    except ValidationError as exc:
        return JSONResponse(
            status_code=400, content={"detail": _validation_detail(exc)}
        )

    rt = deps.get_router()
    pm = deps.get_pm()
    db = deps.get_db()
    prefs = await routing_preferences_from_request(db, request)

    for attempt in range(2):
        result = await rt.route(req.model, path, prefs)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"No available provider for model '{req.model}'",
            )
        provider = result.provider
        assert provider.id is not None
        start = time.monotonic()
        log = dict(
            provider=provider,
            protocol="v1",
            endpoint=endpoint_path,
            request=request,
            model=req.model,
            routed_model=result.model_key,
            request_body=body,
        )
        calibrated = 0
        native_path = native_decision_path(provider, path)
        try:
            out = None
            if native_path is not None:
                out = await _pass_through(
                    pm, provider, native_path, {**body, "model": result.backend_model}
                )
            if out is None:
                inputs = rt.inputs
                names = svc.model_names(
                    (
                        result.model_key,
                        result.resolved_model,
                        req.model,
                        result.backend_model,
                    ),
                    inputs.aliases if inputs else None,
                )
                cal = await svc.calibrations(db).for_request(names, req)
                calibrated = len(cal)
                answered = await svc.emulate(
                    pm, provider.id, result.backend_model, req, cal
                )
                out = {"model": result.resolved_model, **answered}
                pm.stats.record_logprobs(provider.id, ok=True)
        except d.NoLogprobs:
            pm.stats.record_logprobs(provider.id, ok=False)
            log_request_background(
                db,
                **log,
                duration_ms=(time.monotonic() - start) * 1000,
                status="error",
                error_detail="backend returned no logprobs; not used for decisions",
                sample=False,
            )
            logger.warning(
                "Provider %s returned no logprobs for a decision question; "
                "routing decisions elsewhere",
                provider.name,
            )
            if attempt == 0:
                await rt.rebuild()
                continue
            raise HTTPException(
                status_code=502,
                detail="No provider for this model returned logprobs",
            )
        except d.DecisionError as exc:
            log_request_background(
                db,
                **log,
                duration_ms=(time.monotonic() - start) * 1000,
                status="error",
                error_detail=str(exc)[:500],
                sample=False,
            )
            raise HTTPException(status_code=502, detail=str(exc))
        except httpx.HTTPStatusError as exc:
            log_request_background(
                db,
                **log,
                duration_ms=(time.monotonic() - start) * 1000,
                status="error",
                error_detail=(
                    f"HTTP {exc.response.status_code}: {exc.response.text[:400]}"
                ),
            )
            return forward_upstream_http_error(exc)
        except httpx.HTTPError as exc:
            detail = describe_httpx_error(exc)
            log_request_background(
                db,
                **log,
                duration_ms=(time.monotonic() - start) * 1000,
                status="error",
                error_detail=detail[:500],
            )
            logger.error(
                "Upstream failure on %s (model=%r, provider=%s): %s",
                endpoint_path,
                req.model,
                provider.name,
                detail,
            )
            raise HTTPException(status_code=502, detail="upstream request failed")

        log_request_background(
            db,
            **log,
            response_body=out,
            duration_ms=(time.monotonic() - start) * 1000,
            request_meta=svc.summarize(out, calibrated),
        )
        return JSONResponse(content=out)
    raise AssertionError("unreachable")  # pragma: no cover
