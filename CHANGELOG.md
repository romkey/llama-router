# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Releases before 0.22.0 are
described in their git tags and GitHub Releases.

## [Unreleased]

### Security

- **Request limits.** Request bodies over `LLAMA_ROUTER_MAX_BODY_BYTES`
  (10 MiB; `LLAMA_ROUTER_MAX_MEDIA_BODY_BYTES`, 50 MiB, for `/v1/audio/` and
  `/v1/images/`) get 413 on the API and dashboard ports, including chunked
  bodies. `n`/`best_of` are capped at 4 and output-token fields at
  `LLAMA_ROUTER_MAX_TOKENS_CAP` (32768). Upstream calls are limited globally
  (256) and per API key (32); a request waits up to 30 s for a slot, then gets
  429. Each key may hold 16 streams at once. Every proxied request must finish
  within `LLAMA_ROUTER_REQUEST_DEADLINE_SECONDS` (1800 s): 504, or a stream is
  ended. The API key is now checked before the body is parsed, so a request
  with a bad key and a bad body gets 401. See docs/configuration.md.

- **X-Forwarded-For is only believed from trusted proxies.** The request log
  (and the dashboard's pull log) used the header from anyone. Set
  `LLAMA_ROUTER_TRUSTED_PROXIES` (IPs or CIDRs) to the reverse proxies in
  front of the router; until then the TCP peer is logged. Values that are not
  IP addresses are ignored.
- **Request log retention.** Rows older than
  `LLAMA_ROUTER_REQUEST_LOG_RETENTION_DAYS` (default 30; `0` keeps them) are
  deleted hourly. Upgrades with an older log lose rows past that age.
- **404s without an API key are no longer logged**, so scanners probing random
  paths cannot fill the request log.
- **Anonymous session affinity is scoped by client address**, and each API key
  (or anonymous client) holds at most
  `LLAMA_ROUTER_SESSION_AFFINITY_MAX_SESSIONS_PER_KEY` sessions (default 200).
- **`GET /v1/audio/voices` needs an API key** like the other endpoints, and its
  answer is cached for 30 seconds.
- **Client disconnects and long-answer timeouts no longer trip the circuit
  breaker.** A stream the client abandons is logged as `cancelled` and not
  sampled; a read timeout on a request asking for more than
  `LLAMA_ROUTER_CIRCUIT_TIMEOUT_MAX_TOKENS` output tokens (default 4096) is
  blamed on the request.
- **Legacy SHA-256 API key hashes are upgraded to PBKDF2** the next time the key
  is used.
- **Hugging Face download values cannot become `hf` options:** patterns and
  revisions starting with `-` are refused, and values are passed as
  `--include=...` / `--revision=...`.
- **No `/docs`, `/redoc` or `/openapi.json` on the Ollama and OpenAI API
  ports.** The dashboard keeps its own.
- **No more fixed fallback secret.** Without `LLAMA_ROUTER_SESSION_SECRET`,
  provider API keys were encrypted, and client API keys peppered, with a key
  written in the source. A random secret is now generated once into
  `secret.key` beside the database (mode 0600, `LLAMA_ROUTER_SECRET_KEY_PATH`
  to move it) and reused on later starts. Existing provider keys are
  re-encrypted at startup and existing client keys re-hashed on first use, so
  nothing needs to be re-entered. Back up `secret.key` with the database.
- **Worker hub token encrypted at rest.** A token saved in the dashboard is
  stored encrypted; a plain one from an older release is encrypted on first
  read.
- **Sentry scrubbing.** Events no longer include local variables, request
  bodies, headers, cookies or query strings, and values under keys that look
  like keys, tokens, secrets or passwords are masked.
- **Upstream errors are sanitised before they reach the client.** For 400,
  404, 422 and 429 only the error's type, code and message are passed on
  (and `Retry-After` on a 429); other upstream errors get a generic body. An
  upstream 401 or 403, meaning a provider refused the router's own key, is
  now answered with 502 so it is not mistaken for a problem with the client's
  key.
- **Router-only request fields are not forwarded.** `provider`, `models`,
  `route`, `transforms`, `extra_body`, `base_url` and `api_key` are removed
  from `/v1/*` request bodies before they go upstream.
- **Malformed request bodies get 400.** The `/api/*` and `/v1/*` inference
  routes answer 400 for a body that is not JSON, not a JSON object, or has a
  `model` that is not a string, instead of failing with 500.
- **`X-Llama-Router-Path` is checked.** Only well-formed router IDs are kept
  from it and passed on, and a header longer than 512 characters is refused
  with 508.
- **Dashboard escapes server and user data.** Model names, provider names,
  addresses, pull and benchmark progress and error messages are HTML-escaped
  wherever the dashboard and provider pages build HTML in JavaScript, and
  names passed to button handlers can no longer break out of them. Pinned
  route model names must be printable and at most 255 characters.
- **Chat follows the self-service limits.** A non-admin's chats may use only
  the providers that keys they issue themselves may use (the locality an
  admin sets under API Keys). Chat spend isn't tracked against a cap, so
  while a daily spend cap is set non-admins chat on local providers only.
  Choosing a provider of another locality is refused.
- **Per-user limits.** `LLAMA_ROUTER_MAX_API_KEYS_PER_USER` (default 20)
  caps the keys a user issues themselves; `LLAMA_ROUTER_MAX_CHAT_CONVERSATIONS_PER_USER`
  (default 500) caps the conversations a user keeps. 0 turns either off.
- **Hugging Face environment is admin-only.** `GET /api/hf/env` now needs an
  admin, and hides `user:password@` in proxy URLs.
- **No cloud metadata addresses.** Provider URLs (add, address add and edit,
  and the probe) that are or resolve to `169.254.0.0/16` or `fd00:ec2::254`
  are refused. Private and loopback addresses are still allowed.
- **Addresses hidden from non-admins.** `/api/status`, the dashboard and the
  provider page show `(hidden)` instead of provider address URLs to users
  who aren't admins, and leave out a worker's remote address.

- **Decision requests are bounded.** A decision request may now ask at most 8
  questions (was 32) about a state of at most 32,000 characters (was 128,000),
  set with `LLAMA_ROUTER_DECISION_MAX_QUESTIONS` and
  `LLAMA_ROUTER_DECISION_MAX_STATE_CHARS`. A key's policy may have one gate
  fewer than that. Emulated questions are asked at most
  `LLAMA_ROUTER_DECISION_MAX_CONCURRENCY` (4) at a time, and a key with a daily
  spend cap is refused with 429 before any call is made when the request's
  estimated cost would pass the cap.

- **Decision policy gates see the whole request and fail closed.** Gates (and
  tiers) now read messages of every role, system, assistant and tool included,
  whatever their case, and tool-call arguments, not only user messages. When a policy with gates cannot
  be decided (no provider, an error, a timeout), the request is refused with
  503 instead of passing unchecked; `LLAMA_ROUTER_DECISION_POLICY_FAIL_OPEN=true`
  restores the old behaviour. Tier-only policies still fall back to the
  requested model.

- **Signed release images.** The release workflow signs each image with
  cosign keyless signing (by digest, every platform); the README shows how to
  verify one with `cosign verify`.
- **GitHub Actions pinned to commit SHAs.** Every action in the workflows is
  pinned to a full commit SHA with its version in a comment, and Renovate
  keeps the digests current.
- **No network tools in the production image.** curl, wget, nc, dig, host,
  ping, traceroute, ip and ps are gone from the released image (the
  development image keeps them). The README shows Python one-liners and a
  sidecar container for checking reachability instead.
- **The first dashboard administrator comes from the environment.** With no
  dashboard accounts the dashboard used to be open to anyone, and whoever
  reached it first could create the administrator. Now it is locked and only
  shows how to set `LLAMA_ROUTER_ADMIN_USERNAME` and
  `LLAMA_ROUTER_ADMIN_PASSWORD`; llama-router creates that account at startup
  while there are none. SSO can create the first account only for a member of
  `LLAMA_ROUTER_OIDC_ADMIN_GROUP`. **Breaking:** set the variables before
  starting a new install, or after clearing the accounts.
- **New installs require API keys.** Requests without a key are refused unless
  *Accept requests without an API key* is turned on (llama-router warns at
  startup while it is). Routers that already have providers, API keys or
  dashboard accounts keep their setting. A missing setting now means "refuse".
- **A wrong API key is always refused.** A request that sends a key that does
  not resolve gets 401 even when requests without a key are accepted, instead
  of quietly being served as anonymous.

### Fixed

- **Daily spend caps hold under concurrency and unknown usage.** A request to
  a priced cloud model now holds its estimated cost against the key's cap
  while it runs, so simultaneous requests can no longer all pass a cap they
  exceed together; the hold becomes the actual cost when the request is
  logged, or is released if it failed. A priced cloud response that reports
  no usage is charged an estimate (prompt size plus `max_tokens`, or
  `LLAMA_ROUTER_SPEND_ESTIMATE_COMPLETION_TOKENS`) instead of nothing.
  Streamed `/v1/chat/completions` and `/v1/completions` requests to cloud
  providers ask for usage, and the extra usage event is hidden from clients
  that did not ask for it. Capped keys can no longer send image or audio
  requests to unpriced cloud models.

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
