# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Releases before 0.22.0 are
described in their git tags and GitHub Releases.

## [Unreleased]

## [0.23.0] - 2026-10-30

### Added

- **Chat tab.** The dashboard's new **Chat** tab (`/chat`) talks to one
  model on one provider, for trying models out. Attach or paste an image
  to ask about it, or switch to **Image** to ask the model for one. Each
  dashboard user keeps their own chats, and viewers may chat too.
  Requests go straight to the chosen provider and show in the request log.

## [0.22.1] - 2026-10-29

### Added

- `LLAMA_ROUTER_QUIET_HEALTH_CHECKS=true` leaves successful `GET /healthz`
  and `GET /health` requests out of the access log. Failed ones are still
  logged.

## [0.22.0] - 2026-10-28

### Added

- **Model groups.** A group is a name such as `coding` or `vision` that
  clients ask for like a model. Each request goes to the best member
  model that can serve it now, scored by the key's routing mode. A member
  is a model on any provider, or a model on one provider. Manage groups
  under Model Groups on the Models tab or at `/api/model-groups`; see
  "Model Groups" in the README.
- Groups are precomputed in the routing table, so routing a group costs
  the same as routing a model.
- Session affinity binds a conversation to the member model as well as
  the provider, so a group conversation keeps running the same model.
- `/api/tags` and `/v1/models` list groups that have a servable member.
- The request log records the model that actually ran when it differs
  from the one asked for: a group's member, an alias or a fallback.
- Metrics `llama_router_model_group_routes_total` and
  `llama_router_model_group_available`.

### Fixed

- `migrate-sqlite-to-postgres` no longer fails on a SQLite file that
  predates a column; the column keeps its default on PostgreSQL.

## [0.21.3] - 2026-10-28

### Fixed

- Releases no longer fail with "failed to reserve cache" when CI on main
  builds the image at the same time. The release build writes its own
  layer cache, and a cache error no longer fails a release or CI.

## [0.21.2] - 2026-10-28

### Added

- `GET /metrics` on the dashboard port: Prometheus metrics for providers,
  addresses, benchmarks, live request outcomes, workers and the routing
  table. Scrapers authenticate with `LLAMA_ROUTER_METRICS_TOKEN`; see
  `docs/metrics.md`.
- The Add Provider dialog has a "Benchmark automatically" choice, so a
  provider that charges per token can be excluded from the idle
  auto-benchmarker before its first benchmark runs. It was already on the
  provider page.
- A worker can choose which of its providers its hub may use. The
  Workers tab lists them as Local, Cloud and Remotes with all/none quick
  picks; the hub's jobs route only to the ticked providers and it lists
  only their models. See `docs/workers.md`.

## [0.21.1] - 2026-10-28

### Added

- `GET /healthz` on the dashboard port: liveness with no sign-in or
  database work, used by the production compose health check.
- `llama-router --version`.
- `docker-compose.prod.yaml` runs the released image from ghcr.io with a
  read-only filesystem, no capabilities and `no-new-privileges`.
- `docker-compose.test.yaml` and `docker-compose.lint.yaml` run tests and
  lint as CI does, with nothing but Docker installed.
- Pre-commit hooks: ruff, mypy, uv lock, yamllint, actionlint, shellcheck,
  hadolint, gitleaks, and checks that `.env.example` is complete and no
  `.env` value is committed.
- Release workflow: run it from the Actions tab to open a release PR;
  merging it tags the release, pushes amd64 and arm64 images and creates
  the GitHub Release. Pushing a `vX.Y.Z` tag by hand still works.
- `docs/configuration.md` describes every environment variable;
  `docs/overview.md` maps the code.
- MIT `LICENSE`, `SECURITY.md`, `AGENTS.md`, Renovate.

### Changed

- **The production image runs as user 10001**, not root. A data volume
  created by an older image is owned by root; fix it once with
  `docker compose -f docker-compose.prod.yaml run --rm --user root llama-router chown -R 10001:10001 /app/data`.
- **Compose files are renamed:** `docker-compose.yml` is now
  `docker-compose.yaml` and runs your working tree for development;
  deployments should use `docker-compose.prod.yaml`.
  `docker-compose.dev.yml`'s mock OIDC provider moved into
  `docker-compose.yaml` (`docker compose --profile oidc up -d mock-oidc`).
- The image sets `LLAMA_ROUTER_DATABASE_PATH`, `LLAMA_ROUTER_CACHE_DIR` and
  `LLAMA_ROUTER_HF_MODELS_DIR` under `/app/data`, so `.env` need not.
- `.env.example` lists every variable, all commented out.
- Requires Python 3.14. Builds with uv (`uv_build`); development
  dependencies are a PEP 735 group (`pip install -e . --group dev`), not
  the `dev` extra.
- ruff replaces black for formatting and linting; mypy runs in strict mode.
- SIGTERM and SIGINT now shut every server down and close the database
  and worker links before exiting. Previously the process could be killed
  before cleanup ran.
- CI is split into CI (tests, wheel install with pip, image build) and
  Lint; images are pushed only by the release workflow.
