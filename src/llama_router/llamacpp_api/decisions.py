"""Decision requests in the shape of TypeSafe's Jev API.

``POST /v1/systemone`` (Jev's route) and ``POST /v1/decisions`` take a state
and typed questions and answer each with a probability distribution; see
:mod:`llama_router.decisions`. An upstream that serves either route gets the
request as it came, on the route it serves (the bodies are the same). Any other
provider that serves chat completions answers it from first-token logprobs,
with the stored calibration for the model and question applied.

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
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from .. import decision_service as svc
from .. import decisions as d
from ..auth import routing_preferences_from_request
from ..endpoints import normalize_endpoint
from ..httpx_errors import describe_httpx_error, forward_upstream_http_error
from ..limits import upstream_slot
from ..request_body import json_object
from ..request_logger import log_request_background
from ..router import RoutingPreferences
from . import deps

logger = logging.getLogger(__name__)
router = APIRouter()


def _validation_detail(exc: ValidationError) -> list[dict[str, Any]]:
    return [
        {"loc": list(e.get("loc", ())), "msg": e.get("msg", "")}
        for e in exc.errors(include_url=False, include_context=False)
    ]


@router.post("/v1/systemone")
@router.post("/systemone")
@router.post("/v1/decisions")
@router.post("/decisions")
async def decide(request: Request) -> Response:
    prefs = await routing_preferences_from_request(deps.get_db(), request)
    body = await json_object(request)
    slot = await upstream_slot(request, prefs)
    async with slot:
        return await serve(request, body, request.url.path, prefs)


async def serve(
    request: Request,
    body: Any,
    endpoint_path: str,
    prefs: RoutingPreferences | None,
) -> Response:
    """Answer a decision request as *endpoint_path* would. The dashboard's
    try form calls this too, with no key preferences."""
    path = normalize_endpoint(endpoint_path)
    if path is None:
        raise ValueError(f"not a decision endpoint: {endpoint_path!r}")
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

    for attempt in range(2):
        result = await rt.route(req.model, path, prefs)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"No available provider for model '{req.model}'",
            )
        # One request is a call per question: refuse it whole when its
        # estimated cost would pass the key's daily spend cap.
        await svc.check_spend_cap(db, rt, prefs, result, req)
        provider = result.provider
        provider_id = provider.require_id()
        start = time.monotonic()
        log: dict[str, Any] = {
            "provider": provider,
            "protocol": "v1",
            "endpoint": endpoint_path,
            "request": request,
            "model": req.model,
            "routed_model": result.model_key,
            "request_body": body,
        }
        calibrated = 0
        try:
            inputs = rt.inputs
            out, calibrated = await svc.answer_routed(
                pm, db, result, req, path, body, inputs.aliases if inputs else None
            )
        except d.NoLogprobsError as exc:
            pm.stats.record_logprobs(provider_id, ok=False)
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
            ) from exc
        except d.DecisionError as exc:
            log_request_background(
                db,
                **log,
                duration_ms=(time.monotonic() - start) * 1000,
                status="error",
                error_detail=str(exc)[:500],
                sample=False,
            )
            raise HTTPException(status_code=502, detail=str(exc)) from exc
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
            raise HTTPException(
                status_code=502, detail="upstream request failed"
            ) from exc

        log_request_background(
            db,
            **log,
            response_body=out,
            duration_ms=(time.monotonic() - start) * 1000,
            request_meta=svc.summarize(out, calibrated),
        )
        return JSONResponse(content=out)
    raise AssertionError("unreachable")  # pragma: no cover
