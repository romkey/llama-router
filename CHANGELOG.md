# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Releases before 0.22.0 are
described in their git tags and GitHub Releases.

## [Unreleased]

### Added

- `llama_router_event_loop_lag_seconds`, a histogram of how late the event
  loop wakes a task sleeping a fixed interval: how long any request may wait
  behind others.
- A router that finds another process's live lease in the database logs an
  error at startup naming the state each process keeps for itself (API key
  cache, spend totals, in-flight counts, affinity, limits, 429 cooldowns,
  login throttle, worker hub sockets). It only warns.
- docs/configuration.md explains why llama-router runs as one process, which
  per-key limits provide fairness, and when to revisit that.

### Changed

- JSON request bodies over 1 MB are parsed, sent upstream and measured for
  the request log in a worker thread, so a large image or document no longer
  holds the event loop. Bad bodies get the same 400s as before.

### Fixed

- Latency scoring no longer treats a loaded model with no measured time to
  first token as instant. Cloud models always count as loaded, so an untried
  provider used to outrank measured ones. Such a candidate now counts as the
  median time to first token measured on the model's other providers for
  that endpoint, or 500 ms when none is measured; a real sample replaces the
  estimate on the next rescore. **Explain** shows it as `ttft_estimate_ms`.

## [0.30.1] - 2026-11-11

### Added

- Reranking: `POST /v1/rerank` on the OpenAI API, also answered on
  `/rerank`, `/v1/reranking`, `/reranking` and `/v2/rerank`. Requests go to a
  provider whose probe found a rerank route (llama.cpp, vLLM and others), on
  the route it serves. Reranking timings are kept as their own endpoint kind,
  and workers run rerank jobs.

## [0.30.0] - 2026-11-10

### Added

- API keys can be edited after they are created: label, routing priority,
  fallback, session affinity, providers and daily spend cap, from **Edit** on
  the API Keys tab or `PATCH /api/keys/{id}`. Users can edit the label,
  routing priority, fallback and session affinity of keys they issued
  themselves on **My API keys** (`PATCH /api/my-keys/{id}`).
- **Regenerate secret** gives a key a new secret, shown once, keeping its id,
  settings, pins, decision policy and usage history; the old secret stops
  working at once (`POST /api/keys/{id}/regenerate`,
  `POST /api/my-keys/{id}/regenerate`).

- API keys can be limited to chosen models and model groups, when creating
  or editing them, by admins and by users for their own keys. A model is
  allowed under any of its aliases; a group by its own name only, not its
  members by theirs. Other models are refused with HTTP 403, fallbacks and
  chaos mode stay within the list, and model listings show only what the key
  may use.

### Deprecated

- `POST /api/keys/{id}/spend-cap` and `POST /api/keys/{id}/locality`; use
  `PATCH /api/keys/{id}`.

## [0.29.1] - 2026-11-05

### Added

- The dashboard has a favicon, Apple touch icons (including `/apple-touch-icon.png`
  and `/apple-touch-icon-precomposed.png`), Android icons and a web manifest,
  served at the site root without a login.

### Fixed

- A worker whose hub URL reaches a port with no worker endpoint (for example
  a proxy pointing at the hub's dashboard port) no longer reports "hub
  rejected the token": it says the URL has no worker endpoint and how to fix
  it. The hub now refuses a bad or expired token with HTTP 401, and a worker
  connection while the hub is disabled with HTTP 403, each with the reason.

## [0.29.0] - 2026-11-03

### Added

- Worker tokens can expire. Set "Expires after (days)" when creating a worker
  or when rotating its token (or `expires_in_days` on `POST /api/workers` and
  `POST /api/workers/{id}/rotate`); the hub refuses an expired token and
  closes a connected worker's session when its token expires.
- API keys can have an optional label, set when issuing a key on the API
  Keys tab or under My API keys (`label` on `POST /api/keys/generate` and
  `POST /api/my-keys`, up to 100 characters) and shown under the prefix. An
  upgrade adds the column.
- Recovery codes have Copy and Download buttons, and generating new ones
  asks first.
- Dashboard users with a local account can change their own password from
  their Account page (`POST /account/password` with the current and new
  password). It signs the account out everywhere else and keeps this session;
  wrong guesses count toward the sign-in lockout.
- Dashboard chat streams replies as the model writes them
  (`POST /api/chat/stream`, newline-delimited JSON events), with a Stop button
  that keeps what has arrived so far. Replies render fenced code blocks,
  inline code, bold and italic.

### Security

- Secrets are derived per purpose with HKDF from `SESSION_SECRET` (or the
  generated `secret.key`): one key signs session cookies, one encrypts stored
  provider keys and the worker hub token, one peppers client API key hashes.
  Client API keys are now stored as HMAC-SHA256 (they are 32 random bytes, so
  PBKDF2 only cost CPU); existing PBKDF2 and SHA-256 hashes still match and
  are replaced on first use. Provider keys and the hub token encrypted by
  earlier releases still decrypt and are re-encrypted with the derived key
  (provider keys at startup).
- **`SESSION_SECRET` must now be at least 32 characters**; the router refuses
  to start with a shorter one. Existing data survives the upgrade when the
  secret is unchanged. To replace a short secret without re-entering provider
  keys or reissuing client keys, set the old value as the new
  `SESSION_SECRET_PREVIOUS` alongside the new `SESSION_SECRET`; data stored
  under it moves to the new secret. With a configured `SESSION_SECRET`,
  everyone signed in to the dashboard has to sign in again once after
  upgrading (cookies are now signed with the derived key). Deployments without
  `SESSION_SECRET` keep their sessions.
- Every dashboard response now sends `X-Frame-Options: DENY`, a
  `Content-Security-Policy` with `frame-ancestors 'none'`, `object-src 'none'`
  and `base-uri 'self'`, `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: same-origin` and `Cache-Control: no-store`, plus
  `Strict-Transport-Security` when the dashboard is served over HTTPS.
- Non-admin dashboard users get 403 from `/metrics` (a metrics token still
  works), `/docs`, `/redoc` and `/openapi.json`.
- A provider's status no longer includes its full URL after a connection
  error, only the path, so non-admins can't see internal hostnames or
  credentials in it.
- Dashboard chat shows non-admins a generic message when the provider fails;
  admins and the request log keep the upstream detail.
- The router warns once when a sign-in arrives with `X-Forwarded-For` from a
  peer not in `LLAMA_ROUTER_TRUSTED_PROXIES`; the README now says the setting
  is required behind a reverse proxy.
- `/api/tags`, `/api/version`, `/api/ps`, `/v1/models` and `/v1/models/{name}`
  now need an API key, like every other API route, unless the authentication
  policy accepts requests without one. `/health` stays public.
- A request with an invalid API key no longer costs a PBKDF2 hash every time.
  Keys not shaped like the router's own (`lrk_` and 43 characters) are refused
  without hashing, keys the database does not know are remembered and refused
  for 30 seconds, and at most four hashes run at once.
- A `Referer` with a port that is not a number no longer makes "run
  benchmarks" fail with a server error; it redirects to the fallback page.
- Provider addresses must be plain `http://` or `https://` URLs without
  credentials in them, when adding a provider and when adding or editing an
  address. Alibaba Cloud's metadata address (`100.100.100.200`) is refused
  like the other cloud metadata services.
- A JSON body nested too deeply to parse gets 400 instead of 500, and the
  request log's summary of a body stops looking 64 levels down.
- At most 64 KB of an upstream error body is read, by the API routes and by
  endpoint probes.
- Creating a dashboard user or setting a user's password no longer blocks the
  server while the password is hashed.

### Fixed

- Two routers sharing a database that both start without
  `LLAMA_ROUTER_SESSION_SECRET` now keep the same generated dashboard session
  secret, instead of the later one overwriting the earlier one's (which
  signed out everyone signed in through the other).
- A management key's "last used" time is recorded on its first use even
  when the host started less than a minute ago.
- Two routers (or a router and a `llama-router` command) opening a new
  SQLite database at the same moment no longer fail with "database is
  locked" while one of them switches it to WAL.
- Cancelling a job on a worker whose socket has stalled no longer waits
  forever; the cancel message is given up on after 5 seconds.
- Editing a provider or its addresses no longer cuts off requests running on
  it: they finish on the old connection, which closes once they are done.
- An upstream that times out now gets 504 and one that cannot be reached 502,
  each with a JSON `error`, instead of a bare 500. A provider with no address
  left answers 503 `{"error": "provider has no reachable address"}`.
- `/v1/audio/transcriptions` and `/v1/images/edits` send the provider its own
  name for the model (a model group or alias used to be passed on as is) and
  drop router-only fields such as `provider`, as the JSON routes do.

### Changed

- `provider_manager.py` is split up. `ClientPool` (`client_pool.py`) holds
  each provider's clients, picks the address they use, adds upstream auth
  headers, and closes a replaced client once its last request ends. Health
  checks and endpoint probes move to `provider_health.py`, worker links and
  their catalogs to `worker_sync.py`, and benchmarks to
  `provider_benchmark.py`; `ProviderManager` combines them, so callers are
  unchanged. The new modules pass `mypy --strict`.
- `database.py` is split into per-domain repositories under
  `src/llama_router/repositories/` (providers, benchmarks, request log, model
  config, decisions, chat, workers, API keys, management keys, dashboard
  users) on a shared query base; the change reasons and `@_notifies` live in
  `db_events.py`. `Database` still combines them, so callers are unchanged.
  These modules now pass `mypy --strict`.
- The ten JSON-body proxy endpoints (`/api/chat`, `/api/generate`,
  `/api/embeddings`, `/api/embed`, `/v1/chat/completions`, `/v1/completions`,
  `/v1/embeddings`, `/v1/responses`, `/v1/images/generations`,
  `/v1/audio/speech`) share one path, `proxy_json` in `proxy.py`, instead of
  ten copies of routing, slots, logging and upstream-error handling. No
  change in behavior.
- `dashboard/routes.py` is split into one router per tab under
  `dashboard/routes/` (health, login, users, home, routing, API keys,
  providers, benchmarks, pulls, cache, model aliases, workers). Shared page
  helpers are in `dashboard/pages.py` and form parsing in
  `dashboard/form_values.py`; the session cookie and redirect helpers moved
  to `auth_core.py`. Paths and behaviour are unchanged.
- Less database and disk work behind the dashboard and the registry cache.
  `/api/status`, `/api/ps`, `/metrics`, `/health` and `/v1/audio/voices` read
  every provider's models, addresses and benchmarks in four queries instead
  of three per provider; `/v1/models` and `/api/tags` list models from the
  routing table's inputs instead of querying. The registry cache writes and
  hashes blobs in a worker thread, 4 MB at a time, and the dashboard reads
  cache statistics off the event loop. Usage per key and the per-model
  request counts only read request log rows inside
  `LLAMA_ROUTER_REQUEST_LOG_RETENTION_DAYS`, so "all" means the retained log
  even before the hourly purge runs. Session affinity finds a key's oldest
  session without scanning every session, and
  `LLAMA_ROUTER_SESSION_AFFINITY_MAX_SESSIONS` now defaults to 10000. A
  streamed response arms one deadline timer instead of one per chunk.
- More tests at security edges and failure paths: hostile `Referer`
  headers, WebAuthn sign-count replay, worker sockets that stall, drop or
  send garbage, a failed routing-table build, a client disconnecting
  mid-stream (stream counts, provider slots and the upstream are all
  released), `run()` under several settings, health checks surviving a
  crashing probe, and Hypothesis round trips of the Ollama/OpenAI
  translation. Hypothesis is a new dev dependency.
- Dashboard errors: a page that fails in the browser now shows an error page
  with a link back instead of raw JSON; `/api/*` still answers in JSON.
  A failed Add Provider form returns to the dashboard with the message and
  reopens the dialog with what was typed (except the API key and extra
  headers); failed provider and address edits return to the provider page
  with the message. A password sign-in from an address other than
  `LLAMA_ROUTER_DASHBOARD_WEBAUTHN_ORIGIN` returns to the sign-in page and
  names the configured address instead of answering 403.
- The "Accept requests without an API key" switch now asks for confirmation
  before turning on, reports success or failure, and turns itself back if
  the change fails. While it is on, the dashboard header shows an "Open
  access" badge. The switch and the limits for keys users issue themselves
  share a new Security card under API Keys.
- The dashboard stops refreshing provider status while its tab is hidden,
  and when the session has expired it shows a "sign in again" banner and
  stops instead of polling forever.
- Dashboard JSON endpoints share one body parser: a body that is not a JSON
  object, or a `provider_id` that is not a number in `/api/benchmark` and
  `/api/pull`, now gets 400 instead of 500.
- The request log is written by one background writer in batches (up to 200
  rows or every 100 ms) instead of a task and a transaction per request. Its
  queue holds 10,000 rows; when the database falls further behind, rows are
  dropped and counted in the new `request_log_dropped_total` metric.
  Shutdown writes what is queued before closing the database.
- SQLite databases use WAL journaling with `synchronous=NORMAL` and a
  five-second busy timeout, so dashboard reads no longer wait on log writes,
  and skip the per-checkout connection ping.
- An upgrade adds an index on the request log's model and time.
- The dashboard's request log page count is cached for 30 seconds instead of
  counting the whole log on every refresh.
- A request's logged size is the body the client sent, not the parsed body
  serialised again.
- Dashboard accessibility and structure. Status is never shown by colour
  alone: each provider state and address liveness has its own icon and
  text, and an offline provider always says why. Icon-only buttons and
  close buttons have labels; each page has one `h1`, cards use `h2`. Chat
  conversations are buttons and their delete button shows on keyboard
  focus; a log row's error detail opens from a button that reports whether
  it is open. Confirmations name what they remove.
- Dashboard layout: the provider counts sit in the Providers tab; the API
  Keys tab is split into Keys and a new Routing tab (preview, table,
  explorer); the menu groups Operate tabs, then Access (API Keys, Routing,
  Users) and Tools (Decisions, Chat) menus so it fits one row; the Log count
  shows on every page. Users' actions are in a menu per row, the last
  sign-in is relative and the sign-in detail is a tooltip. The Add Provider
  dialog scrolls and keeps room for the probe result.
- Dates on the dashboard all read `YYYY-MM-DD HH:MM` in the browser's time
  zone, with the full time on hover.
- "Profile" is now "Account"; the Workers card "Serve a hub from this
  router" is now "Connect to a hub (run as a worker)".
- Test infrastructure: CI runs the whole suite against PostgreSQL and MySQL
  as well as SQLite, checks there is a single Alembic head, and runs the
  tests in parallel. Coverage now includes the Alembic migrations (floor
  raised to 85%), new tests downgrade every migration and upgrade again, and
  check that `llama-router-migrate-sqlite-pg` copies every table and column.
  Tests start from a copy of one migrated database instead of migrating each
  time, and the timer tests use a fake clock instead of waiting. The
  `009_single_protocol_providers` and `010_provider_flavor` migration files
  are renamed `010a_` and `010b_` so they sort in the order they run; their
  revision ids are unchanged.

### Fixed

- On PostgreSQL, request log retention no longer fails with "can't subtract
  offset-naive and offset-aware datetimes", so old request logs are deleted.
- The "accept requests without an API key" setting is re-read every five
  minutes, so routers sharing a database see a change made on another one.
- A management key used in the router's first minute after boot now records
  its last-used time.
- If one of the router's servers cannot bind its port, the others are stopped
  and the router exits with an error instead of running without it.
- Benchmarks of Ollama-style providers that leave out `eval_duration`, such as
  another llama-router in front of llama.cpp, no longer report billions of
  tokens per second. The benchmark falls back to the reported or measured
  time, the router's Ollama translation now reports `eval_duration` and
  `prompt_eval_duration` (from llama-server's timings when present), speeds
  over a million tokens per second are not stored, and an upgrade deletes the
  bad benchmarks already stored so they are measured again.
- A hub client that falls more than `LLAMA_ROUTER_WORKER_RESPONSE_BUFFER_BYTES`
  behind a worker's response no longer stalls every other job on that worker:
  the hub fails and cancels only that job instead of pausing the socket.
- Workers decode the request path before checking it against the allowed
  endpoints, and refuse encoded slashes (`%2F`), `.` and `..` segments and
  doubly encoded paths, so an encoded path cannot reach an endpoint the
  worker does not serve.
- The routing table no longer stalls request handling while it is rebuilt,
  and is rebuilt far less often. Building runs in a worker thread; a live
  request or health-check sample only rescores the existing candidates
  instead of re-reading the database, and only when it could change an
  ordering (a moving average moved over 10 %, a failure rate moved, or a
  circuit or warm flag flipped). Per-locality and hub rankings are built on
  first use. A provider flipping between busy and idle no longer reloads the
  table every health-check cycle.

## [0.28.0] - 2026-11-02

### Added

- Password + WebAuthn second factor for local dashboard accounts, with multiple
  named authenticators, single-use recovery codes and an operator recovery
  command. `LLAMA_ROUTER_DASHBOARD_REQUIRE_SECOND_FACTOR` enforces enrollment
  and second-factor login for all local users, including admins, and rejects
  existing password-only sessions. Configure a stable HTTPS origin with
  `LLAMA_ROUTER_DASHBOARD_WEBAUTHN_ORIGIN`; SSO and API-key authentication stay
  unchanged.
- Non-admin dashboard users can view request logs and counts for their own
  current API keys. Backend error details remain admin-only.

### Changed

- Open the profile page from the dashboard username. Authenticator management,
  registration and recovery codes use separate cards.
- Move Decisions from the top banner to the shared dashboard navigation,
  immediately after Models.

### Fixed

- SQLite no longer reuses deleted API-key IDs, preventing new keys from
  inheriting access to retained request logs for deleted keys.

## [0.27.0] - 2026-11-01

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
- **Password sign-in backs off.** After five failures in a row from one
  address or for one username, sign-in answers HTTP 429 for a delay that
  doubles each time, up to five minutes. Password checks run off the event
  loop, and unknown usernames take as long as real ones.
- **Sessions can be revoked.** Session cookies carry a per-account version.
  Changing a password, a change of an SSO account's admin flag, **Sign out
  everywhere** (next to Log out) and an administrator's **Sign out** on the
  Users page end the account's existing sessions. SSO sessions last at most
  12 hours. Existing cookies are not accepted after upgrading: everyone signs
  in once more.
- **SSO without group limits is flagged**, with an error at startup and a
  warning on the Users page. The new `LLAMA_ROUTER_OIDC_REQUIRE_APPROVAL`
  creates new SSO accounts pending until an administrator approves them.
- **Cross-site requests refused.** The dashboard rejects POSTs (sign-in
  included) whose `Sec-Fetch-Site` or `Origin` names another site.
  `LLAMA_ROUTER_DASHBOARD_TRUSTED_ORIGINS` lists exceptions.
- **Secure cookies behind HTTPS proxies.** Session cookies are marked Secure
  whenever the request came over HTTPS or with `X-Forwarded-Proto: https`.
- **Hub headers from clients are ignored.** `X-Llama-Router-Via` and
  `X-Llama-Router-Provider` now count only on jobs relayed by the router's
  own worker link, proven by the per-process token in `X-Llama-Router-Relay`
  (sent only to the default upstreams, accepted only from a loopback
  address). One middleware checks it for both uses (hub narrowing and
  keyless relayed jobs) and strips all three headers from every request. A job the hub sends to one provider still obeys
  the worker key's model pins.
- **Worker links are bounded.** A worker refuses (503) new jobs while more
  than four times its concurrency are pending, and caps request bodies held
  at once (`LLAMA_ROUTER_WORKER_MAX_BODY_BYTES_IN_FLIGHT`, 256 MiB). The hub
  buffers at most `LLAMA_ROUTER_WORKER_RESPONSE_BUFFER_BYTES` (16 MiB) per
  job, stops reading from the worker while a client falls behind and fails a
  job whose client stops reading for 30 seconds, and cuts off responses over
  `LLAMA_ROUTER_WORKER_MAX_RESPONSE_BYTES` (1 GiB).
- **Worker provider lists are bounded.** The hub applies a worker's catalog
  at most every `LLAMA_ROUTER_WORKER_CATALOG_MIN_INTERVAL_SECONDS` (5),
  using only the latest, caps shared providers per worker at
  `LLAMA_ROUTER_WORKER_MAX_SHARED_PROVIDERS` (256), and deletes providers a
  worker stopped sharing after `LLAMA_ROUTER_WORKER_UNSHARED_PURGE_HOURS`
  (24; `0` keeps them), or at once when they would go over the cap.
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
