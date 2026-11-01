# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Releases before 0.22.0 are
described in their git tags and GitHub Releases.

## [Unreleased]

### Security

- **Registry cache hardening.** The cache now binds `127.0.0.1` by default and
  `docker-compose.prod.yaml` no longer publishes port 9200; set
  `LLAMA_ROUTER_CACHE_HOST=0.0.0.0` and publish the port to serve LAN Ollama
  backends. Optional `LLAMA_ROUTER_CACHE_TOKEN` requires a bearer token,
  `LLAMA_ROUTER_CACHE_MAX_BYTES` evicts least recently used blobs, and the
  cache answers 503 while toggled off in the dashboard. Blobs are hashed while
  streaming and never committed on a digest mismatch; manifests fetched by
  digest are verified. The blob `Location` header no longer trusts the Host
  header, pre-cache names are validated, and the cache's OpenAPI docs are off.
  Manifests move to `manifests/<name>/_refs/<tag>`: `library/` models migrate
  at startup, others are fetched again on the next pull.

### Fixed

- **Readable error rows in the request log.** Error highlights use soft burgundy
  with blue links and neutral details, keeping timestamps, model names, metadata
  and expanded error details readable both normally and on hover.

- **Consistent dashboard navigation.** Users stays after Chat on every page,
  and Workers and Cache keep the same visibility and order across the dashboard,
  Users, Chat, My API keys, Decisions and provider details. Cache is shown only
  when a cache is available.

## [0.26.1] - 2026-10-31

### Fixed

- **Streamed chats no longer end in a transfer-length error.** When the
  upstream refused a streaming request (unknown model, server error), the
  router had already sent `200` and then cut the body short, which Open WebUI
  showed as "Response payload is not completed: TransferEncodingError". The
  router now waits for the upstream to answer and passes its real status and
  message through.

## [0.26.0] - 2026-10-31

### Fixed

- **API keys from OpenAI-compatible clients.** Clients can send their key as
  `Authorization: Bearer lr_...` as well as `API-KEY: lr_...`, so Open WebUI,
  the OpenAI SDKs and other clients that only send Bearer can connect.
- **New API keys stay on screen.** A key generated on the API Keys tab
  appears in a dialog that stays open until you close it, so there is time
  to copy it.

### Changed

- **Models tab sub-tabs.** Models, Fallbacks, Aliases and Groups are tabs of
  their own on the Models tab, each with a count. The address names the
  sub-tab, so a reload after an edit returns to it.
- **Choose a group's models from a list.** **Edit members** opens a checklist
  of models with a live filter. Pick **All providers** to add a model on any
  provider, or one provider to add it only there; unchecking removes it.
  Members the list doesn't show are listed below it, with a remove button.
  A new group opens the list straight away. The remove button on a member is
  now easy to see.

### Added

- **Users issue their own API keys.** Any signed-in dashboard user, admin or
  viewer, local or SSO, can issue, list and revoke their own keys for model
  access on the new **My API keys** page (`/my-keys`, linked in the top
  bar), or through `GET`/`POST /api/my-keys` and `DELETE /api/my-keys/{id}`.
  A card on the API Keys tab sets the providers (local, cloud or either) and
  daily spend cap that self-issued keys get; until an admin changes it they
  are **local only**, so users can't give themselves cloud spending. The
  admin key table gains an **Owner** column, and deleting a user deletes the
  keys they issued. Admins read and set the limits with
  `GET`/`PUT /api/keys/self-service`. Adds migration `024_api_key_owner`.
- `PUT /api/model-groups/{id}/members` replaces a group's members in one
  transaction; members that stay keep their place.

## [0.25.2] - 2026-10-31

### Fixed

- **Chat without choosing a provider.** The Chat tab's provider list starts
  with **Any provider**, which offers every model, model group and alias and
  lets the router pick where the request goes, as it would for an API
  request. With a provider chosen, the model list now also offers the groups
  and aliases that reach it. A reply from a group or alias names the model
  that answered.

## [0.25.1] - 2026-10-31

### Fixed

- **SQLite foreign keys.** Foreign keys were enforced on only one of the
  database's connections, so deleting a provider could leave its models,
  benchmarks and other rows behind. They are now enforced on every
  connection.

- **Copying SQLite to PostgreSQL with leftover rows.**
  `llama-router-migrate-sqlite-pg` no longer fails on rows whose parent was
  deleted (such as models of a removed provider). It skips them, or clears
  the reference where the column may be empty, and logs how many.

## [0.25.0] - 2026-10-31

### Added

- **Usage by key and provider.** The Keys tab has a table of requests,
  errors, tokens and cost per API key and provider for the last 24 hours,
  7 days, 30 days or all time, read from the request log.
  `GET /api/keys/usage` returns the same data for scripts (`keys:read`).

- **Share providers with a hub one by one.** A worker can now choose, on
  its **Workers** tab or with `LLAMA_ROUTER_WORKER_EXPOSE=individual`, to
  have each provider it shares appear on the hub as a provider of its own,
  named `<worker>/<provider>`. Each one has its own models, status, health
  checks, benchmarks, routing scores and request log entries on the hub.
  The default is unchanged: the worker appears as one provider. See
  "Sharing providers one by one" in docs/workers.md.

- **Router IDs and loop protection.** Each router now has a random ID,
  shown on the **Workers** tab (with a *New ID* button for a router started
  from a copy of another's data). A hub refuses a worker that is itself, and
  a worker leaves out its remotes that dial in from the hub it serves, so two
  routers that are each other's hub no longer offer the hub its own models.
  Requests a router relays to a local provider carry the routers they have
  passed through in `X-Llama-Router-Path`. A router that finds its own ID
  there, or a path of eight routers, answers `508 Loop Detected` instead of
  routing the request round again. See "Router IDs and loops" in
  docs/workers.md.

### Fixed

- **PostgreSQL and MySQL in Docker.** The Docker image now includes the
  PostgreSQL and MySQL drivers, so `LLAMA_ROUTER_DATABASE_URL` and
  `llama-router-migrate-sqlite-pg` work with the released image without
  rebuilding it.

## [0.24.0] - 2026-10-30

### Added

- **Management keys.** Administrators can issue management keys
  (`lrm_...`) on the **Users** page, each limited to the privileges they
  pick: `keys:read`, `keys:write`, `models:read`, `models:write` and
  `status:read` (the same status a dashboard viewer sees), with an optional expiry. A script sends one as
  `Authorization: Bearer` to the dashboard's `/api/...` endpoints, for
  example to issue API keys for new users, without a dashboard account.
  See "Management Keys" in the README.
- `GET /api/keys` lists API keys and their settings (never the keys).

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
