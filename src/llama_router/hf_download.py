"""Download GGUF models from Hugging Face with the ``hf`` CLI.

Downloads land in ``LLAMA_ROUTER_HF_MODELS_DIR``, one folder per repository.
From there a model reaches providers in one of two ways:

- **Push (Ollama):** the router uploads the GGUF as a blob
  (``POST /api/blobs/<digest>``) and creates the model (``POST /api/create``).
  This works for remote Ollama servers with no shared filesystem.
- **Shared volume (llama.cpp):** when the models directory is also mounted
  into a llama.cpp server's ``--models-dir`` (router mode), the file appears
  there and the router only refreshes the provider to discover it.

The ``hf`` subprocess inherits the router's environment, so the standard
Hugging Face variables apply unchanged: ``HF_ENDPOINT`` (a mirror or proxy),
``HF_TOKEN`` (gated and private repos), ``HF_HOME``, ``HF_HUB_CACHE``,
``HF_XET_CACHE``, ``HF_DEBUG``, ``HF_HUB_VERBOSITY``,
``HF_HUB_DOWNLOAD_TIMEOUT``, ``HF_HUB_ETAG_TIMEOUT``, ``HF_HUB_OFFLINE``,
``HF_HUB_DISABLE_TELEMETRY``, ``HF_HUB_DISABLE_XET``,
``HF_XET_HIGH_PERFORMANCE``, ``HF_HUB_DISABLE_SYMLINKS``, and the proxy and
CA variables ``HTTP_PROXY``, ``HTTPS_PROXY``, ``NO_PROXY``,
``REQUESTS_CA_BUNDLE`` and ``SSL_CERT_FILE``.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import settings

logger = logging.getLogger(__name__)

_REPO_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
)
_PATTERN_RE = re.compile(r"^[A-Za-z0-9._*?\[\]/-]{1,200}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9._/-]{1,100}$")
_OLLAMA_NAME_RE = re.compile(
    r"^[a-z0-9][a-z0-9._-]{0,79}(/[a-z0-9][a-z0-9._-]{0,79})?(:[A-Za-z0-9._-]{1,128})?$"
)
_SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
_PERCENT_RE = re.compile(r"(\d{1,3})%\|")
_QUANT_RE = re.compile(r"((?:IQ|Q)\d+(?:_[A-Z0-9]+)*|F16|BF16|F32)", re.IGNORECASE)

# Variables shown on the dashboard (values masked where secret).
HF_ENV_VARS = [
    "HF_ENDPOINT",
    "HF_TOKEN",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_XET_CACHE",
    "HF_DEBUG",
    "HF_HUB_VERBOSITY",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "HF_HUB_ETAG_TIMEOUT",
    "HF_HUB_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
    "HF_HUB_DISABLE_XET",
    "HF_XET_HIGH_PERFORMANCE",
    "HF_HUB_DISABLE_SYMLINKS",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    # Deprecated aliases still honoured by huggingface_hub.
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_CACHE",
]
_SECRET_VARS = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}


class HfError(Exception):
    """A Hugging Face download could not be planned or completed."""


def validate_repo(repo: str) -> str:
    repo = (repo or "").strip()
    if repo.startswith("hf.co/"):
        repo = repo[len("hf.co/") :]
    if not _REPO_RE.fullmatch(repo) or ".." in repo:
        raise ValueError("repository must look like owner/name")
    return repo


def validate_pattern(pattern: str | None) -> str:
    pattern = (pattern or "").strip() or "*.gguf"
    if not _PATTERN_RE.fullmatch(pattern) or ".." in pattern:
        raise ValueError("file pattern may use letters, digits, . _ - / * ? [ ]")
    return pattern


def validate_revision(revision: str | None) -> str | None:
    revision = (revision or "").strip()
    if not revision:
        return None
    if not _REVISION_RE.fullmatch(revision) or ".." in revision:
        raise ValueError("revision may use letters, digits, . _ - /")
    return revision


def validate_ollama_name(name: str) -> str:
    name = (name or "").strip()
    if not _OLLAMA_NAME_RE.fullmatch(name):
        raise ValueError(
            "model name must be lowercase like name:tag or namespace/name:tag"
        )
    return name


def default_ollama_name(repo: str, filename: str) -> str:
    """``owner/Qwen3-8B-GGUF`` + ``Qwen3-8B-Q4_K_M.gguf`` -> ``qwen3-8b:q4_k_m``."""
    base = repo.split("/", 1)[1].lower()
    for suffix in ("-gguf", "_gguf", ".gguf"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    base = re.sub(r"[^a-z0-9._-]+", "-", base).strip("-._") or "model"
    stem = Path(filename).name
    m = _QUANT_RE.search(_SPLIT_RE.sub(".gguf", stem))
    tag = m.group(1).lower() if m else "latest"
    return validate_ollama_name(f"{base[:80]}:{tag}")


def env_summary() -> list[dict[str, str | bool]]:
    """Which Hugging Face variables are set, with secrets masked."""
    out: list[dict[str, str | bool]] = []
    for var in HF_ENV_VARS:
        val = os.environ.get(var)
        if not val:
            out.append({"name": var, "set": False, "value": ""})
        elif var in _SECRET_VARS:
            out.append({"name": var, "set": True, "value": "set (hidden)"})
        else:
            out.append({"name": var, "set": True, "value": val})
    return out


def models_root() -> Path:
    root = Path(settings.hf_models_dir).absolute()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _path_under(base: Path, *parts: str) -> Path:
    root = str(base.absolute())
    candidate = os.path.normpath(base.absolute().joinpath(*parts))
    if candidate != root and not candidate.startswith(root + os.sep):
        raise ValueError("path escapes the models directory")
    return Path(candidate)


def local_dir_for(repo: str) -> Path:
    """One folder per repository: ``owner__name``."""
    return _path_under(models_root(), validate_repo(repo).replace("/", "__"))


def hf_executable() -> str:
    found = shutil.which("hf")
    if found:
        return found
    sibling = Path(sys.executable).with_name("hf")
    if sibling.exists():
        return str(sibling)
    raise HfError("the hf command is not installed (pip install huggingface_hub)")


def model_files(files: list[str]) -> list[str]:
    """GGUF files that are models, not vision projectors."""
    return [
        f
        for f in files
        if f.lower().endswith(".gguf") and not Path(f).name.lower().startswith("mmproj")
    ]


def split_parts(files: list[str]) -> bool:
    return any(_SPLIT_RE.search(f) for f in files)


async def _run(
    args: list[str], on_progress: Callable[[str], None] | None = None
) -> tuple[int, str, str]:
    """Run ``hf`` with *args*; stream stderr lines to *on_progress*."""
    proc = await asyncio.create_subprocess_exec(
        hf_executable(),
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "HF_HUB_DISABLE_UPDATE_CHECK": "1", "NO_COLOR": "1"},
    )
    assert proc.stdout is not None and proc.stderr is not None
    err_tail: list[str] = []

    async def read_stderr() -> None:
        buf = b""
        while True:
            chunk = await proc.stderr.read(4096)
            if not chunk:
                break
            buf += chunk
            # tqdm redraws with \r; treat both \r and \n as line ends.
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                err_tail.append(line)
                del err_tail[:-20]
                if on_progress:
                    on_progress(line)

    stdout_task = asyncio.create_task(proc.stdout.read())
    await read_stderr()
    out = (await stdout_task).decode("utf-8", "replace")
    code = await proc.wait()
    return code, out, "\n".join(err_tail)


def _error_from(stderr: str) -> str:
    for line in reversed(stderr.splitlines()):
        line = line.strip()
        if line and not line.startswith("Warning:") and "%|" not in line:
            return line[:300]
    return "hf exited with an error"


async def plan(repo: str, pattern: str, revision: str | None) -> list[dict]:
    """Files *pattern* selects in *repo*, from ``hf download --dry-run --json``."""
    args = ["download", repo, "--include", pattern, "--dry-run", "--json"]
    if revision:
        args += ["--revision", revision]
    code, out, err = await _run(args)
    if code != 0:
        raise HfError(_error_from(err))
    # hf already applied --include; re-filtering could only drop files it chose.
    return _parse_plan(out)


def _parse_plan(out: str) -> list[dict]:
    """File list from ``--dry-run --json`` output, tolerant of its shape.

    Accepts the whole output or its last line as JSON, a list or an object
    wrapping one (``files``/``items``/``data``), and entries keyed ``file``,
    ``path``, ``filename`` or ``rfilename``.
    """
    text = out.strip()
    if not text:
        return []
    data: Any = None
    for candidate in (text, text.splitlines()[-1]):
        try:
            data = json.loads(candidate)
            break
        except ValueError:
            continue
    if isinstance(data, dict):
        data = next(
            (
                data[k]
                for k in ("files", "items", "data")
                if isinstance(data.get(k), list)
            ),
            [],
        )
    if not isinstance(data, list):
        return []
    files = []
    for d in data:
        if isinstance(d, str):
            files.append({"file": d, "size": ""})
        elif isinstance(d, dict):
            name = next(
                (d[k] for k in ("file", "path", "filename", "rfilename") if d.get(k)),
                None,
            )
            if name:
                files.append({"file": str(name), "size": str(d.get("size") or "")})
    return files


async def download(
    repo: str,
    pattern: str,
    revision: str | None,
    on_progress: Callable[[str], None] | None = None,
) -> list[Path]:
    """Download the matching files and return their local paths."""
    target = local_dir_for(repo)
    target.mkdir(parents=True, exist_ok=True)
    args = ["download", repo, "--include", pattern, "--local-dir", str(target)]
    if revision:
        args += ["--revision", revision]

    def progress(line: str) -> None:
        m = _PERCENT_RE.search(line)
        if on_progress and m:
            on_progress(f"downloading {m.group(1)}%")

    # The files this run selects, so leftovers from earlier downloads of the
    # same repository (other quants, other revisions) are never returned.
    wanted = [f["file"] for f in await plan(repo, pattern, revision)]
    started = time.time()
    code, _out, err = await _run(args, progress)
    if code != 0:
        raise HfError(_error_from(err))
    found = []
    for name in wanted:
        path = _path_under(target, name)
        if path.is_file():
            found.append(path)
    if not found:
        # The plan could not be read; fall back to matching files this run
        # wrote or refreshed (hf touches files it verifies or downloads).
        for path in sorted(target.rglob("*")):
            rel = path.relative_to(target).as_posix()
            if (
                path.is_file()
                and not rel.startswith(".cache/")
                # Same rule as hf's --include: fnmatchcase on the full path.
                and fnmatch.fnmatchcase(rel, pattern)
                and path.stat().st_mtime >= started - 1
            ):
                found.append(path)
    if not found:
        raise HfError(f"no files in {repo} match {pattern}")
    return found


async def sha256_file(path: Path) -> str:
    def _hash() -> str:
        h = hashlib.sha256()
        with Path(path).open("rb") as fh:
            for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest()

    return await asyncio.to_thread(_hash)
