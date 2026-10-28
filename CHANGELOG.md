# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Releases before 0.22.0 are
described in their git tags and GitHub Releases.

## [Unreleased]

### Added

- `GET /metrics` on the dashboard port: Prometheus metrics for providers,
  addresses, benchmarks, live request outcomes, workers and the routing
  table. Scrapers authenticate with `LLAMA_ROUTER_METRICS_TOKEN`; see
  `docs/metrics.md`.

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
