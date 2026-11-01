"""Hugging Face downloads: validation, the hf subprocess, and delivery."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import sys
import textwrap

import httpx
import pytest

from llama_router import hf_download
from llama_router.config import settings
from llama_router.dashboard import routes

FAKE_HF = textwrap.dedent("""\
    #!{python}
    # Stand-in for the hf CLI. Repo "owner/two-quants-GGUF" holds two quants;
    # any other repo holds one. "owner/missing" fails like a 404.
    import json, os, sys
    args = sys.argv[1:]
    assert args[0] == "download", args
    repo = args[1]
    def opt(name):
        for a in args:
            if a.startswith(name + "="):
                return a.split("=", 1)[1]
        return args[args.index(name) + 1] if name in args else None
    if repo == "owner/missing":
        sys.stderr.write("Warning: unauthenticated\\nRepository Not Found for url\\n")
        sys.exit(1)
    files = ["Model-Q4_K_M.gguf", "mmproj-Model-F16.gguf", "README.md"]
    if repo == "owner/two-quants-GGUF":
        files = ["Model-Q4_K_M.gguf", "Model-Q8_0.gguf"]
    import fnmatch
    pattern = opt("--include") or "*"
    picked = [f for f in files if fnmatch.fnmatch(f, pattern)]
    if "--dry-run" in args:
        print(json.dumps([{{"file": f, "size": "1.2M"}} for f in picked]))
        sys.exit(0)
    out = opt("--local-dir")
    os.makedirs(out, exist_ok=True)
    for f in picked:
        sys.stderr.write(f"{{f}}:  50%|#####     | 1/2 [00:00<00:00]\\r")
        with open(os.path.join(out, f), "wb") as fh:
            fh.write(b"GGUF" + f.encode())
    sys.stderr.write("100%|##########| 2/2\\n")
    print(out)
    """)


@pytest.fixture
def fake_hf(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "hf"
    script.write_text(FAKE_HF.format(python=sys.executable))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setattr(settings, "hf_models_dir", str(tmp_path / "models"))
    return tmp_path / "models"


@pytest.mark.parametrize(
    ("repo", "file", "name"),
    [
        ("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf", "qwen3-8b:q4_k_m"),
        (
            "bartowski/Llama-3.2-1B-Instruct-GGUF",
            "x-IQ4_XS.gguf",
            "llama-3.2-1b-instruct:iq4_xs",
        ),
        ("owner/thing", "thing-f16.gguf", "thing:f16"),
        ("owner/thing", "weights.gguf", "thing:latest"),
    ],
)
def test_default_ollama_name(repo, file, name) -> None:
    assert hf_download.default_ollama_name(repo, file) == name


@pytest.mark.parametrize(
    "bad", ["", "noslash", "a/b/c", "../etc", "owner/..", "owner/na me", "-x/y"]
)
def test_validate_repo_rejects(bad) -> None:
    with pytest.raises(ValueError, match="repository must look"):
        hf_download.validate_repo(bad)


def test_validate_repo_accepts_hf_co_prefix() -> None:
    assert hf_download.validate_repo("hf.co/Qwen/Qwen3-8B-GGUF") == "Qwen/Qwen3-8B-GGUF"


@pytest.mark.parametrize(
    "bad", ["../x.gguf", "a b", "x;rm", "$(x)", "--local-dir=/etc", "-h"]
)
def test_validate_pattern_rejects(bad) -> None:
    with pytest.raises(ValueError, match="file pattern may use"):
        hf_download.validate_pattern(bad)


def test_model_files_skip_projectors_and_detect_splits() -> None:
    files = ["a-Q4.gguf", "mmproj-a.gguf", "README.md"]
    assert hf_download.model_files(files) == ["a-Q4.gguf"]
    assert hf_download.split_parts(["m-00001-of-00003.gguf"])
    assert not hf_download.split_parts(["m-Q4_K_M.gguf"])


def test_env_summary_masks_token(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example")
    monkeypatch.setenv("HF_DEBUG", "")
    env = {v["name"]: v for v in hf_download.env_summary()}
    assert env["HF_TOKEN"]["value"] == "set (hidden)"
    assert "hf_secret" not in json.dumps(env)
    assert env["HF_ENDPOINT"]["value"] == "https://mirror.example"
    assert env["HF_DEBUG"]["set"] is False


def test_local_dir_stays_under_root(fake_hf) -> None:
    d = hf_download.local_dir_for("owner/model-GGUF")
    assert d.parent == fake_hf.resolve()
    assert d.name == "owner__model-GGUF"


async def test_plan_and_download_with_cli(fake_hf) -> None:
    files = await hf_download.plan("owner/model-GGUF", "*.gguf", None)
    assert [f["file"] for f in files] == ["Model-Q4_K_M.gguf", "mmproj-Model-F16.gguf"]

    seen: list[str] = []
    paths = await hf_download.download(
        "owner/model-GGUF", "*Q4_K_M*.gguf", None, seen.append
    )
    assert [p.name for p in paths] == ["Model-Q4_K_M.gguf"]
    assert paths[0].read_bytes().startswith(b"GGUF")
    assert "downloading 50%" in seen

    with pytest.raises(hf_download.HfError, match="Repository Not Found"):
        await hf_download.plan("owner/missing", "*.gguf", None)


def _ollama_handler(created: list[dict], blobs: dict[str, bytes]):
    def handler(request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.0"})
        if path == "/api/tags":
            names = [c["model"] for c in created]
            return httpx.Response(200, json={"models": [{"name": n} for n in names]})
        if path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        if path.startswith("/api/blobs/"):
            digest = path.rsplit("/", 1)[1]
            if method == "HEAD":
                return httpx.Response(200 if digest in blobs else 404)
            blobs[digest] = request.read()
            return httpx.Response(201)
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "Model-Q8_0"}]})
        if path == "/api/create" and method == "POST":
            created.append(json.loads(request.content))
            return httpx.Response(200, json={"status": "success"})
        return httpx.Response(404)

    return handler


async def _wait_done(pull_id: str) -> dict:
    for _ in range(200):
        entry = routes._active_pulls[pull_id]
        if entry["status"] == "done":
            return entry
        await asyncio.sleep(0.05)
    raise AssertionError("pull did not finish")


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_hf_pull_pushes_blob_and_creates_model(
    dash_client,
    httpx_mock,
    fake_hf,
) -> None:
    created: list[dict] = []
    blobs: dict[str, bytes] = {}
    httpx_mock.add_callback(_ollama_handler(created, blobs), is_reusable=True)
    client, db = dash_client
    await client.post("/providers/add", data={"name": "gpu", "url": "http://gpu:11434"})
    pid = (await db.list_providers())[0].id

    r = await client.post(
        "/api/hf/pull",
        json={
            "repo": "owner/Model-GGUF",
            "pattern": "*Q4_K_M*.gguf",
            "push_provider_ids": [pid],
        },
    )
    assert r.status_code == 200
    entry = await _wait_done(r.json()["pull_id"])
    assert entry["completed"] == [pid]
    assert entry["failed"] == []
    assert entry["model"] == "model:q4_k_m"

    expected = "sha256:" + hashlib.sha256(b"GGUFModel-Q4_K_M.gguf").hexdigest()
    assert blobs == {expected: b"GGUFModel-Q4_K_M.gguf"}
    assert created == [
        {
            "model": "model:q4_k_m",
            "files": {"Model-Q4_K_M.gguf": expected},
            "stream": False,
        }
    ]
    names = [m.name for m in await db.get_provider_models(pid)]
    assert "model:q4_k_m" in names
    logs = [e for e in await db.get_request_logs(50, 0) if e.endpoint == "hf-pull"]
    assert [(e.status, e.model) for e in logs] == [("ok", "model:q4_k_m")]


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_hf_pull_refuses_ambiguous_push(
    dash_client,
    httpx_mock,
    fake_hf,
) -> None:
    httpx_mock.add_callback(_ollama_handler([], {}), is_reusable=True)
    client, db = dash_client
    await client.post("/providers/add", data={"name": "gpu", "url": "http://gpu:11434"})
    pid = (await db.list_providers())[0].id
    r = await client.post(
        "/api/hf/pull",
        json={"repo": "owner/two-quants-GGUF", "push_provider_ids": [pid]},
    )
    entry = await _wait_done(r.json()["pull_id"])
    assert entry["failed"] == [pid]
    assert "narrow it to one quant" in entry["progress"]


async def test_hf_routes_validate_input(dash_client, fake_hf) -> None:
    client, _db = dash_client
    bad = await client.post("/api/hf/pull", json={"repo": "../etc"})
    assert bad.status_code == 400
    bad = await client.post(
        "/api/hf/pull", json={"repo": "owner/x", "model_name": "Bad Name"}
    )
    assert bad.status_code == 400
    unknown = await client.post(
        "/api/hf/pull", json={"repo": "owner/x", "push_provider_ids": [999]}
    )
    assert unknown.status_code == 400

    files = await client.post("/api/hf/files", json={"repo": "owner/Model-GGUF"})
    assert files.status_code == 200
    body = files.json()
    assert body["model_files"] == ["Model-Q4_K_M.gguf"]
    assert body["suggested_name"] == "model:q4_k_m"

    env = await client.get("/api/hf/env")
    assert env.status_code == 200
    assert env.json()["hf_available"] is True


async def test_download_ignores_leftovers_from_earlier_runs(fake_hf) -> None:
    target = hf_download.local_dir_for("owner/Model-GGUF")
    target.mkdir(parents=True)
    (target / "Old-Q8_0.gguf").write_bytes(b"GGUF old")  # from an earlier run
    paths = await hf_download.download("owner/Model-GGUF", "*.gguf", None)
    assert sorted(p.name for p in paths) == [
        "Model-Q4_K_M.gguf",
        "mmproj-Model-F16.gguf",
    ]
    # One model file after dropping the projector, so a push is not ambiguous.
    assert hf_download.model_files([p.name for p in paths]) == ["Model-Q4_K_M.gguf"]


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_refused_push_still_refreshes_shared_volume_providers(
    dash_client,
    httpx_mock,
    fake_hf,
) -> None:
    httpx_mock.add_callback(_ollama_handler([], {}), is_reusable=True)
    client, db = dash_client
    await client.post("/providers/add", data={"name": "oll", "url": "http://oll:11434"})
    await client.post(
        "/providers/add",
        data={"name": "lcpp", "url": "http://lcpp:8080", "protocol": "openai"},
    )
    by_name = {p.name: p.id for p in await db.list_providers()}
    r = await client.post(
        "/api/hf/pull",
        json={
            "repo": "owner/two-quants-GGUF",
            "push_provider_ids": [by_name["oll"]],
            "refresh_provider_ids": [by_name["lcpp"]],
        },
    )
    entry = await _wait_done(r.json()["pull_id"])
    assert entry["failed"] == [by_name["oll"]]
    assert entry["completed"] == [by_name["lcpp"]]


@pytest.mark.parametrize(
    "out",
    [
        '[{"file": "a.gguf", "size": "1M"}]',
        '{"files": [{"path": "a.gguf"}]}',
        '[\n  {\n    "rfilename": "a.gguf"\n  }\n]',
        'noise line\n["a.gguf"]',
    ],
)
def test_parse_plan_accepts_several_shapes(out) -> None:
    assert [f["file"] for f in hf_download._parse_plan(out)] == ["a.gguf"]


async def test_download_falls_back_when_plan_is_unreadable(
    fake_hf, monkeypatch
) -> None:
    async def empty_plan(repo, pattern, revision):
        return []

    monkeypatch.setattr(hf_download, "plan", empty_plan)
    paths = await hf_download.download("owner/model-GGUF", "*Q4_K_M*.gguf", None)
    assert [p.name for p in paths] == ["Model-Q4_K_M.gguf"]


async def test_hf_pull_refuses_worker_push(dash_client, fake_hf) -> None:
    client, db = dash_client
    from llama_router.models import ProviderProtocol

    w = await db.add_provider("edge", ProviderProtocol.OLLAMA, worker_name="edge")
    r = await client.post(
        "/api/hf/pull", json={"repo": "owner/x", "push_provider_ids": [w.id]}
    )
    assert r.status_code == 400
    assert "worker" in r.json()["detail"]


async def test_plan_keeps_nested_files_hf_selected(fake_hf, monkeypatch) -> None:
    async def run(args, on_progress=None):
        return 0, '[{"file": "Model-Q4.gguf"}, {"file": "sub/Model-Q8.gguf"}]', ""

    monkeypatch.setattr(hf_download, "_run", run)
    files = await hf_download.plan("owner/model-GGUF", "*.gguf", None)
    assert [f["file"] for f in files] == ["Model-Q4.gguf", "sub/Model-Q8.gguf"]


@pytest.mark.parametrize("bad", ["--local-dir=/etc", "-x", "a b"])
def test_validate_revision_rejects(bad) -> None:
    with pytest.raises(ValueError, match="revision may use"):
        hf_download.validate_revision(bad)


async def test_values_are_passed_in_option_equals_form(monkeypatch) -> None:
    seen: list[list[str]] = []

    async def run(args, on_progress=None):
        seen.append(args)
        return 0, "[]", ""

    monkeypatch.setattr(hf_download, "_run", run)
    await hf_download.plan("owner/model-GGUF", "*.gguf", "main")
    assert seen[0] == [
        "download",
        "owner/model-GGUF",
        "--include=*.gguf",
        "--dry-run",
        "--json",
        "--revision=main",
    ]


def test_path_under_accepts_base_with_parent_segments(tmp_path):
    (tmp_path / "a").mkdir()
    base = tmp_path / "a" / ".." / "models"
    assert hf_download._path_under(base, "x", "y") == tmp_path / "models" / "x" / "y"
