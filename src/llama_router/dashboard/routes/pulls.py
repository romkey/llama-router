"""Model pulls from Ollama and Hugging Face (Models tab)."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import (
    JSONResponse,
    RedirectResponse,
)

from ... import hf_download
from ...background import spawn
from ...client_address import client_ip
from ...config import settings
from ...httpx_errors import describe_httpx_error
from ...models import RequestLog
from .. import deps
from ..pages import _is_admin, _provider_url

logger = logging.getLogger(__name__)
router = APIRouter()


_active_pulls: dict[str, dict] = {}
_active_fill_pulls: dict[str, dict] = {}


async def _pull_one_ollama(
    *,
    pid: int,
    idx: int,
    total: int,
    model: str,
    pull_api: str,
    cache_url: str | None,
    pm: Any,
    db: Any,
    pull_entry: dict,
    progress_lock: asyncio.Lock,
    source_ip: str | None = None,
    log_endpoint: str = "/api/pull/provider",
    request_meta: str | None = None,
) -> None:
    """Pull *model* on one Ollama provider; updates *pull_entry* under the lock."""
    provider = await db.get_provider(pid)
    pname = provider.name if provider else str(pid)
    prefix = f"[{idx + 1}/{total}] {pname}" if total > 1 else pname

    async with progress_lock:
        pull_entry["progress"] = f"{prefix}: starting…"
    logger.info("Pull %s starting on provider %s (id=%d)", model, pname, pid)
    start = time.monotonic()
    last_total_bytes: list[int] = [0]

    def _on_progress(info: dict, _pfx: str = prefix) -> None:
        text = info.get("status", "")
        pct = info.get("percent")
        completed = info.get("completed") or 0
        total = info.get("total") or 0
        if total:
            last_total_bytes[0] = total
        elapsed = time.monotonic() - start
        speed_str = ""
        if elapsed > 0 and completed > 0:
            speed_mbps = (completed / (1024 * 1024)) / elapsed
            speed_str = f" {speed_mbps:.1f} MB/s"
        if pct is not None:
            msg = f"{_pfx}: {text} {pct}%{speed_str}"
        else:
            msg = f"{_pfx}: {text}{speed_str}"

        async def _write() -> None:
            async with progress_lock:
                pull_entry["progress"] = msg

        try:
            asyncio.get_running_loop().create_task(_write())
        except RuntimeError:
            pull_entry["progress"] = msg

    try:
        client = pm.get_ollama_client(pid)
        await client.pull_model(
            model,
            cache_registry_url=cache_url,
            progress_callback=_on_progress,
        )
        await pm.refresh_provider(pid)
        duration = (time.monotonic() - start) * 1000
        total_bytes = last_total_bytes[0]
        speed_mbps = (
            (total_bytes / (1024 * 1024)) / (duration / 1000)
            if total_bytes and duration > 0
            else 0
        )
        done_msg = f"{prefix}: done ({duration / 1000:.0f}s"
        if speed_mbps > 0:
            done_msg += f", {speed_mbps:.1f} MB/s"
        done_msg += ")"
        async with progress_lock:
            pull_entry["completed"].append(pid)
            pull_entry["progress"] = done_msg
        logger.info(
            "Pull %s succeeded on provider %s in %.1fs",
            model,
            pname,
            duration / 1000,
        )
        log_kw: dict[str, Any] = {
            "provider_id": pid,
            "provider_name": pname,
            "protocol": "ollama",
            "endpoint": log_endpoint,
            "model": model,
            "request_size": 0,
            "response_size": 0,
            "duration_ms": duration,
            "status": "ok",
        }
        if source_ip is not None:
            log_kw["source_ip"] = source_ip
        meta = request_meta or ""
        if speed_mbps > 0:
            meta = f"{meta} {speed_mbps:.1f} MB/s".strip()
        if meta:
            log_kw["request_meta"] = meta
        await db.save_request_log(RequestLog(**log_kw))
    except Exception as exc:
        duration = (time.monotonic() - start) * 1000
        logger.error(
            "Pull %s FAILED on provider %s after %.1fs: %s",
            model,
            pname,
            duration / 1000,
            exc,
        )
        async with progress_lock:
            pull_entry["failed"].append(pid)
            pull_entry["progress"] = f"{prefix}: FAILED"
        log_kw = {
            "provider_id": pid,
            "provider_name": pname,
            "protocol": "ollama",
            "endpoint": log_endpoint,
            "model": model,
            "request_size": 0,
            "response_size": 0,
            "duration_ms": duration,
            "status": "error",
            "error_detail": "pull failed",
        }
        if source_ip is not None:
            log_kw["source_ip"] = source_ip
        if request_meta is not None:
            log_kw["request_meta"] = request_meta
        await db.save_request_log(RequestLog(**log_kw))


def _cache_registry_url() -> str | None:
    """Return the cache registry URL if cache is enabled, else None.

    Uses cache_external_host (the address backends can reach) rather than
    cache_host (the bind address).
    """
    if not settings.cache_enabled:
        return None
    host = settings.cache_external_host
    if not host:
        logger.warning(
            "LLAMA_ROUTER_CACHE_EXTERNAL_HOST is not set — cache pulls will "
            "use 127.0.0.1 which only works if Ollama runs on the same host"
        )
        host = "127.0.0.1"
    return f"http://{host}:{settings.cache_port}"


@router.post("/api/pull")
async def api_pull_model(request: Request):
    """Start a pull as a background task and return immediately."""
    body = await deps.json_body(request)
    model = body.get("model")
    provider_id = body.get("provider_id")
    pull_api = (body.get("pull_api") or "auto").strip().lower()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    if pull_api not in {"auto", "ollama", "openai", "llamacpp"}:
        raise HTTPException(status_code=400, detail="Invalid pull_api value")
    if pull_api in {"openai", "llamacpp"}:
        raise HTTPException(
            status_code=400,
            detail="pulling over the OpenAI API is not supported; use ollama",
        )

    pm = deps.get_pm()
    pull_id = str(uuid.uuid4())

    if provider_id is not None:
        try:
            provider_ids = [int(provider_id)]
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid provider_id") from exc
    else:
        infos = await pm.list_provider_infos()
        provider_ids = [
            i.provider.id
            for i in infos
            if i.provider.is_ollama
            and not i.provider.lacks("/api/pull")
            and i.provider.id is not None
        ]

    _active_pulls[pull_id] = {
        "model": model,
        "pull_api": pull_api,
        "provider_ids": provider_ids,
        "status": "pulling",
        "completed": [],
        "failed": [],
        "progress": "",
    }

    cache_url = _cache_registry_url()
    db = deps.get_db()
    source_ip = client_ip(request)

    progress_lock = asyncio.Lock()

    async def _run_pull():
        pull_entry = _active_pulls[pull_id]
        n = len(provider_ids)
        try:
            await asyncio.gather(
                *[
                    _pull_one_ollama(
                        pid=pid,
                        idx=i,
                        total=n,
                        model=model,
                        pull_api=pull_api,
                        cache_url=cache_url,
                        pm=pm,
                        db=db,
                        pull_entry=pull_entry,
                        progress_lock=progress_lock,
                        source_ip=source_ip,
                        log_endpoint="/api/pull",
                        request_meta=f"pull_api={pull_api}",
                    )
                    for i, pid in enumerate(provider_ids)
                ]
            )
        finally:
            async with progress_lock:
                pull_entry["status"] = "done"
                pull_entry["completed_at"] = time.monotonic()

    spawn(_run_pull())

    return JSONResponse({"pull_id": pull_id, "status": "pulling"})


@router.get("/api/hf/env")
async def api_hf_env(request: Request):
    """Hugging Face settings the hf CLI will see (secrets masked). Admins only:
    it shows the router's environment (proxies, paths)."""
    if not _is_admin(request):
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        hf_download.hf_executable()
        available = True
    except hf_download.HfError:
        available = False
    return JSONResponse(
        {
            "hf_available": available,
            "models_dir": str(hf_download.models_root()),
            "env": hf_download.env_summary(),
        }
    )


@router.post("/api/hf/files")
async def api_hf_files(request: Request):
    """List the files a repository and pattern select, without downloading."""
    body = await deps.json_body(request)
    try:
        repo = hf_download.validate_repo(str(body.get("repo") or ""))
        pattern = hf_download.validate_pattern(body.get("pattern"))
        revision = hf_download.validate_revision(body.get("revision"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        files = await hf_download.plan(repo, pattern, revision)
    except hf_download.HfError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    models = hf_download.model_files([f["file"] for f in files])
    suggested = (
        hf_download.default_ollama_name(repo, models[0]) if len(models) == 1 else None
    )
    return JSONResponse(
        {
            "repo": repo,
            "pattern": pattern,
            "files": files,
            "model_files": models,
            "split": hf_download.split_parts(models),
            "suggested_name": suggested,
        }
    )


async def _hf_deliver_ollama(
    *,
    pm: Any,
    db: Any,
    pid: int,
    path: Path,
    digest: str,
    model_name: str,
    entry: dict,
    lock: asyncio.Lock,
) -> None:
    provider = await db.get_provider(pid)
    pname = provider.name if provider else str(pid)
    start = time.monotonic()
    status, detail = "ok", None
    try:
        client = pm.get_ollama_client(pid)
        if await client.has_blob(digest):
            async with lock:
                entry["progress"] = f"{pname}: blob already present"
        else:

            def on_sent(sent: int, total: int) -> None:
                pct = int(sent * 100 / total) if total else 0
                entry["progress"] = f"{pname}: uploading {pct}%"

            await client.push_blob(digest, str(path), on_sent)
        async with lock:
            entry["progress"] = f"{pname}: creating {model_name}"
        await client.create_from_files(model_name, {path.name: digest})
        await pm.refresh_provider(pid)
        async with lock:
            entry["completed"].append(pid)
    except Exception as exc:
        status = "error"
        detail = (
            describe_httpx_error(exc) if isinstance(exc, httpx.HTTPError) else str(exc)
        )
        logger.error("HF push of %s to %s failed: %s", model_name, pname, detail)
        async with lock:
            entry["failed"].append(pid)
            entry["progress"] = f"{pname}: failed: {detail}"
    await db.save_request_log(
        RequestLog(
            provider_id=pid,
            provider_name=pname,
            protocol="system",
            endpoint="hf-pull",
            source_ip="internal",
            model=model_name,
            duration_ms=(time.monotonic() - start) * 1000,
            status=status,
            error_detail=detail,
        )
    )


@router.post("/api/hf/pull")
async def api_hf_pull(request: Request):
    """Download from Hugging Face, then push to Ollama providers and/or refresh
    llama.cpp providers that share the models directory."""
    body = await deps.json_body(request)
    try:
        repo = hf_download.validate_repo(str(body.get("repo") or ""))
        pattern = hf_download.validate_pattern(body.get("pattern"))
        revision = hf_download.validate_revision(body.get("revision"))
        name_raw = str(body.get("model_name") or "").strip()
        model_name = hf_download.validate_ollama_name(name_raw) if name_raw else None
        push_ids = [int(x) for x in body.get("push_provider_ids") or []]
        refresh_ids = [int(x) for x in body.get("refresh_provider_ids") or []]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    pm = deps.get_pm()
    db = deps.get_db()
    for pid in push_ids + refresh_ids:
        target = await db.get_provider(pid)
        if target is None:
            raise HTTPException(status_code=400, detail=f"unknown provider {pid}")
        if pid in push_ids and target.is_worker:
            # The worker link buffers whole request bodies; a multi-GB upload
            # through it would exhaust memory on the hub.
            raise HTTPException(
                status_code=400,
                detail=f"{target.name} is a worker; pull the model on the worker "
                "itself instead of pushing it through the hub",
            )

    pull_id = str(uuid.uuid4())
    label = model_name or f"{repo} ({pattern})"
    entry = {
        "model": label,
        "pull_api": "huggingface",
        "provider_ids": push_ids + refresh_ids,
        "status": "pulling",
        "completed": [],
        "failed": [],
        "progress": "starting download",
    }
    _active_pulls[pull_id] = entry
    lock = asyncio.Lock()

    async def _run() -> None:
        try:

            def on_progress(text: str) -> None:
                entry["progress"] = f"{repo}: {text}"

            paths = await hf_download.download(repo, pattern, revision, on_progress)
            models = [
                p
                for p in paths
                if p.name in hf_download.model_files([q.name for q in paths])
            ]
            if push_ids:
                # A refusal here fails only the Ollama targets; shared-volume
                # refreshes below still run, since the files are on disk.
                refusal: str | None = None
                if not models:
                    refusal = "no GGUF model file was downloaded"
                elif hf_download.split_parts([p.name for p in models]):
                    refusal = (
                        "split GGUF files cannot be pushed to Ollama; "
                        "use a single-file quant or the shared volume"
                    )
                elif len(models) > 1:
                    refusal = (
                        f"the pattern matched {len(models)} GGUF files; "
                        "narrow it to one quant to push to Ollama"
                    )
                if refusal is not None:
                    logger.warning("HF push of %s refused: %s", repo, refusal)
                    async with lock:
                        entry["failed"].extend(push_ids)
                        entry["progress"] = f"failed: {refusal}"
                else:
                    path = models[0]
                    name = model_name or hf_download.default_ollama_name(
                        repo, path.name
                    )
                    entry["model"] = name
                    entry["progress"] = f"hashing {path.name}"
                    digest = await hf_download.sha256_file(path)
                    await asyncio.gather(
                        *(
                            _hf_deliver_ollama(
                                pm=pm,
                                db=db,
                                pid=pid,
                                path=path,
                                digest=digest,
                                model_name=name,
                                entry=entry,
                                lock=lock,
                            )
                            for pid in push_ids
                        )
                    )
            for pid in refresh_ids:
                try:
                    await pm.refresh_provider(pid)
                    async with lock:
                        entry["completed"].append(pid)
                except Exception:
                    logger.exception("refresh after HF download failed (%s)", pid)
                    async with lock:
                        entry["failed"].append(pid)
            if not (push_ids or refresh_ids):
                entry["progress"] = (
                    f"saved {len(paths)} file(s) in the models directory"
                )
        except Exception as exc:
            logger.error("HF download of %s failed: %s", repo, exc)
            async with lock:
                done = set(entry["completed"]) | set(entry["failed"])
                entry["failed"].extend(
                    p for p in entry["provider_ids"] if p not in done
                )
                if not entry["provider_ids"]:
                    entry["failed"].append(0)
                entry["progress"] = f"failed: {exc}"
        finally:
            async with lock:
                entry["status"] = "done"
                entry["completed_at"] = time.monotonic()

    spawn(_run())
    return JSONResponse({"pull_id": pull_id, "status": "pulling"})


@router.get("/api/pulls")
async def api_active_pulls():
    """Return all active/recent pull operations."""
    return JSONResponse(
        {
            pid: {
                "model": p["model"],
                "pull_api": p.get("pull_api", "auto"),
                "status": p["status"],
                "total": len(p["provider_ids"]),
                "completed": len(p["completed"]),
                "failed": len(p["failed"]),
                "progress": p.get("progress", ""),
            }
            for pid, p in _active_pulls.items()
        }
    )


@router.get("/api/pulls/{pull_id}")
async def api_pull_status(pull_id: str):
    """Check status of a specific pull."""
    entry = _active_pulls.get(pull_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Pull not found")
    return JSONResponse(
        {
            "pull_id": pull_id,
            "model": entry["model"],
            "pull_api": entry.get("pull_api", "auto"),
            "status": entry["status"],
            "total": len(entry["provider_ids"]),
            "completed": len(entry["completed"]),
            "failed": len(entry["failed"]),
            "progress": entry.get("progress", ""),
        }
    )


@router.post("/api/pulls/fill-missing")
async def api_fill_missing_pulls():
    """Pull missing models to all Ollama-capable providers."""
    pm = deps.get_pm()
    cache_url = _cache_registry_url()

    infos = await pm.list_provider_infos()
    all_model_names = sorted({m.name for info in infos for m in info.models})

    provider_missing: dict[int, list[str]] = {}
    provider_names: dict[int, str] = {}
    for info in infos:
        provider = info.provider
        if not provider.is_ollama or provider.lacks("/api/pull") or provider.id is None:
            continue
        existing_ollama = {m.name for m in info.models}
        missing = [name for name in all_model_names if name not in existing_ollama]
        if missing:
            provider_missing[provider.id] = missing
            provider_names[provider.id] = provider.name

    job_id = str(uuid.uuid4())
    total = sum(len(v) for v in provider_missing.values())
    _active_fill_pulls[job_id] = {
        "status": "running",
        "total": total,
        "completed": 0,
        "failed": 0,
        "progress": "starting…",
        "errors": [],
    }

    async def _run() -> None:
        job = _active_fill_pulls[job_id]
        lock = asyncio.Lock()

        async def _run_provider(pid: int, models: list[str]) -> None:
            pname = provider_names.get(pid, str(pid))
            try:
                client = pm.get_ollama_client(pid)
            except Exception as exc:
                async with lock:
                    job["failed"] += len(models)
                    job["errors"].append(f"{pname}: no ollama client ({exc})")
                return

            for idx, model in enumerate(models, start=1):
                async with lock:
                    job["progress"] = f"{pname}: pulling {idx}/{len(models)} {model}"
                try:
                    await client.pull_model(model, cache_registry_url=cache_url)
                    async with lock:
                        job["completed"] += 1
                except Exception as exc:
                    async with lock:
                        job["failed"] += 1
                        if len(job["errors"]) < 20:
                            job["errors"].append(f"{pname} {model}: {exc}")
            try:
                await pm.refresh_provider(pid)
            except Exception:
                logger.exception(
                    "Refresh failed after bulk pull for provider %s", pname
                )

        try:
            await asyncio.gather(
                *[
                    _run_provider(pid, models)
                    for pid, models in provider_missing.items()
                ]
            )
            job["status"] = "done"
            job["completed_at"] = time.monotonic()
            if total == 0:
                job["progress"] = "No missing ollama models to pull."
            else:
                job["progress"] = (
                    f"Done: {job['completed']} pulled, {job['failed']} failed."
                )
        except Exception as exc:
            job["status"] = "failed"
            job["completed_at"] = time.monotonic()
            job["progress"] = "Bulk pull failed."
            job["errors"].append(str(exc))
            logger.exception("Bulk fill-missing pull job failed")

    spawn(_run())
    return JSONResponse({"job_id": job_id, "status": "running"})


@router.get("/api/pulls/fill-missing/{job_id}")
async def api_fill_missing_pulls_status(job_id: str):
    job = _active_fill_pulls.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Pull fill job not found")
    return JSONResponse({"job_id": job_id, **job})


@router.post("/providers/{provider_id}/pull")
async def pull_model_legacy(provider_id: int, model: str = Form(...)):
    """Legacy form-based pull — redirects immediately, pull runs in background."""
    pm = deps.get_pm()
    db = deps.get_db()
    pull_id = str(uuid.uuid4())
    _active_pulls[pull_id] = {
        "model": model,
        "provider_ids": [provider_id],
        "status": "pulling",
        "completed": [],
        "failed": [],
        "progress": "",
    }
    cache_url = _cache_registry_url()
    progress_lock = asyncio.Lock()

    async def _run():
        pull_entry = _active_pulls[pull_id]
        try:
            await _pull_one_ollama(
                pid=provider_id,
                idx=0,
                total=1,
                model=model,
                pull_api="auto",
                cache_url=cache_url,
                pm=pm,
                db=db,
                pull_entry=pull_entry,
                progress_lock=progress_lock,
                source_ip=None,
                log_endpoint="/api/pull",
                request_meta=None,
            )
        finally:
            async with progress_lock:
                pull_entry["status"] = "done"
                pull_entry["completed_at"] = time.monotonic()

    spawn(_run())
    return RedirectResponse(url=_provider_url(provider_id), status_code=303)


@router.post("/models/pull-all")
async def pull_model_all_legacy(model: str = Form(...)):
    """Legacy form-based pull-all — redirects immediately, pull runs in background."""
    pm = deps.get_pm()
    db = deps.get_db()
    infos = await pm.list_provider_infos()
    provider_ids = [
        i.provider.id
        for i in infos
        if i.provider.is_ollama
        and not i.provider.lacks("/api/pull")
        and i.provider.id is not None
    ]
    pull_id = str(uuid.uuid4())
    _active_pulls[pull_id] = {
        "model": model,
        "provider_ids": provider_ids,
        "status": "pulling",
        "completed": [],
        "failed": [],
        "progress": "",
    }
    cache_url = _cache_registry_url()
    progress_lock = asyncio.Lock()

    async def _run():
        pull_entry = _active_pulls[pull_id]
        n = len(provider_ids)
        try:
            await asyncio.gather(
                *[
                    _pull_one_ollama(
                        pid=pid,
                        idx=i,
                        total=n,
                        model=model,
                        pull_api="auto",
                        cache_url=cache_url,
                        pm=pm,
                        db=db,
                        pull_entry=pull_entry,
                        progress_lock=progress_lock,
                        source_ip=None,
                        log_endpoint="/api/pull",
                        request_meta=None,
                    )
                    for i, pid in enumerate(provider_ids)
                ]
            )
        finally:
            async with progress_lock:
                pull_entry["status"] = "done"
                pull_entry["completed_at"] = time.monotonic()

    spawn(_run())
    return RedirectResponse(url="/#models-pane", status_code=303)
