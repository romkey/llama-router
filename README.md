# llama-router

An LLM router that aggregates multiple Ollama and OpenAI-compatible backends (llama.cpp, vLLM, LiteLLM and others) behind unified APIs.

## Features

- **Unified Ollama API** — Ollama-compatible API on port 11434 that routes to the best available backend
- **Unified OpenAI API** — OpenAI-compatible API on port 8080 that routes to any backend
- **Web Dashboard** — Bootstrap-based management UI on port 80 with provider status, model inventory, benchmarks, request log, and cache management
- **Auto-Discovery** — Probes each backend to recognise the server (Ollama, llama.cpp, vLLM, LiteLLM, Lemonade, LM Studio, SGLang, TGI, KoboldCpp, LocalAI or a cloud service), its protocol, its endpoints and its cheapest health check, and lists its models
- **Health Checking** — Periodic monitoring of backend availability, including per-address liveness for multi-address providers
- **Benchmarking** — Measure startup time and tokens/sec per model per provider, by hand or automatically while providers are idle
- **Decision Models** — A [Jev](https://en.wikipedia.org/wiki/Jev_(AI_model))-compatible `/v1/systemone` API that answers yes/no, choice and score questions with probabilities, from any local chat model or an upstream that serves it natively
- **Precomputed Routing** — Routing tables are rebuilt in the background from models, liveness, benchmarks, measured round trips and live request outcomes, so picking a backend costs microseconds and no database work
- **Session Affinity** — Each coding-agent conversation stays on the backend that served it, so the backend's prompt cache stays warm
- **Model Fallbacks** — Configure cascading fallback models so requests are transparently rerouted when a model is unavailable
- **API Key Routing Policies** — Per-key balanced/latency/throughput/chaos routing, fallback control, and optional model-to-provider pinning
- **Hub and Worker Links** — Member routers dial out to a hub over one WebSocket and serve its jobs, with no inbound ports or VPN
- **OCI Registry Cache** — Built-in pull-through cache for Ollama model downloads, serving cached layers at LAN speed

## Quick Start

```bash
docker compose up -d
```

Open http://localhost to access the dashboard and add your backends.

### Dashboard login lockout (forgot password)

After you create at least one account under **Users**, the dashboard requires sign-in. Passwords are stored hashed; they cannot be read back. If you lose the password, clear the dashboard user table and start over (you will briefly have an open dashboard until you create a new user).

**Docker Compose** (default database path from `docker-compose.yml`):

```bash
docker compose exec llama-router python -c "import sqlite3; c=sqlite3.connect('/app/data/llama_router.db'); c.execute('DELETE FROM dashboard_users'); c.commit(); c.close()"
docker compose restart llama-router
```

The first command removes **all** dashboard accounts. The restart clears the in-process user-count cache so the UI immediately returns to bootstrap mode (no login required). Then open **Users** and create a new administrator.

If you set `LLAMA_ROUTER_DATABASE_PATH` to something else, use that path inside the `sqlite3.connect(...)` call. For a local SQLite file (no Docker), run the same `DELETE FROM dashboard_users` against your database file with any SQLite client.

To remove a single user by name instead of wiping everyone:

```bash
docker compose exec llama-router python -c "import sqlite3; c=sqlite3.connect('/app/data/llama_router.db'); c.execute('DELETE FROM dashboard_users WHERE username = ?', ('myuser',)); c.commit(); c.close()"
docker compose restart llama-router
```

### Single sign-on (OpenID Connect)

The dashboard can sign people in through an OpenID Connect provider such as Authentik, Keycloak, Okta, Entra ID or dex. Members of an admin group become administrators. Everyone else who signs in is a read-only viewer. Local accounts keep working alongside SSO unless you turn password sign-in off.

Register a confidential client at your provider with this redirect URI:

```
https://<dashboard host>/login/oidc/callback
```

Then set:

| Variable | Default | Meaning |
|---|---|---|
| `LLAMA_ROUTER_OIDC_ISSUER` | empty | Issuer URL. Empty turns SSO off. |
| `LLAMA_ROUTER_OIDC_CLIENT_ID` | empty | Client ID. |
| `LLAMA_ROUTER_OIDC_CLIENT_SECRET` | empty | Client secret. |
| `LLAMA_ROUTER_OIDC_ADMIN_GROUP` | empty | Members are administrators. |
| `LLAMA_ROUTER_OIDC_ALLOWED_GROUP` | empty | If set, only members (and admins) may sign in. |
| `LLAMA_ROUTER_OIDC_GROUPS_CLAIM` | `groups` | Claim holding the groups. Dots reach nested claims. |
| `LLAMA_ROUTER_OIDC_SCOPES` | `openid profile email groups` | Requested scopes. |
| `LLAMA_ROUTER_OIDC_REDIRECT_BASE_URL` | empty | External dashboard URL, e.g. `https://llm.example.org`. Empty uses the request's host and `X-Forwarded-*` headers. |
| `LLAMA_ROUTER_OIDC_DISPLAY_NAME` | `single sign-on` | Button text: "Sign in with ...". |
| `LLAMA_ROUTER_OIDC_ONLY` | `false` | Hide the password form and refuse password sign-in. |
| `LLAMA_ROUTER_OIDC_LOGOUT_AT_IDP` | `false` | Also end the session at the provider on logout. |
| `LLAMA_ROUTER_OIDC_ALLOW_INSECURE` | `false` | Allow an `http://` issuer, for local testing only. |

Behind HTTPS, also set `LLAMA_ROUTER_DASHBOARD_COOKIE_SECURE=true`. The **Users** page shows the issuer, callback URL and group rules in effect, and marks SSO accounts.

How accounts behave:

- **First sign-in creates the account.** It is matched to the provider by issuer and subject, never by email, so SSO cannot take over a local account. If the username is taken, a number is appended.
- **Roles follow the provider.** The admin flag is recomputed from the groups claim at every sign-in. A removed admin loses access at their next sign-in. Sessions last up to 14 days.
- **Groups.** If the ID token has no groups claim, the userinfo endpoint is asked.
- **No passwords.** SSO accounts cannot use password sign-in and cannot be given a password.
- **Removing access.** Deleting an SSO account on the **Users** page signs it out, but the account is recreated at the person's next sign-in. To keep someone out, remove them at the provider or set `LLAMA_ROUTER_OIDC_ALLOWED_GROUP`.
- **Bootstrap.** With no accounts at all, the dashboard is open as before. The first SSO sign-in creates the first account. If an admin group is set, that first person must be in it. Otherwise the first account is an administrator.
- **Break-glass.** Keep one local administrator, or use the lockout steps above, in case the provider is unreachable.

Provider notes:

- **Authentik:** add the `groups` scope mapping to the provider. Group names appear as-is.
- **Keycloak:** add a *Group Membership* mapper with "Full group path" off, or use realm roles with `LLAMA_ROUTER_OIDC_GROUPS_CLAIM=realm_access.roles`.
- **Okta:** add a `groups` claim with a filter on the authorization server.
- **Entra ID:** group claims carry object IDs, not names, so use the group's ID as the admin group. Users in many groups get an overage marker instead of a list, so app roles with `LLAMA_ROUTER_OIDC_GROUPS_CLAIM=roles` are more reliable.
- **Google:** has no groups claim. Leave the admin group empty so the first account administers, and promote others by hand.

To try it locally, start the mock provider in `docker-compose.dev.yml`. It shows a form where you type any subject and claims, such as `{"preferred_username": "ada", "groups": ["router-admins"]}`:

```bash
docker compose -f docker-compose.dev.yml --profile oidc up -d mock-oidc
```

Then run the router on the host with SSO pointed at it:

```bash
LLAMA_ROUTER_OIDC_ISSUER=http://localhost:18080/default LLAMA_ROUTER_OIDC_ALLOW_INSECURE=true LLAMA_ROUTER_OIDC_CLIENT_ID=llama-router LLAMA_ROUTER_OIDC_CLIENT_SECRET=dev LLAMA_ROUTER_OIDC_ADMIN_GROUP=router-admins LLAMA_ROUTER_DASHBOARD_PORT=8099 uv run llama-router
```

### Connecting routers: hubs and workers

One llama-router can serve as a **hub** for others. A **worker** is a llama-router that opens a single outbound WebSocket to the hub and serves the inference jobs the hub sends down it. The worker needs no open ports, port forwarding, or VPN, so it works from a home network, behind NAT, or while the machine is on another tailnet or VPN. Nothing on the worker's network becomes reachable from the hub.

**On the hub:** open the dashboard **Workers** tab, enter a name, and click **Create worker token**. The token is shown once. The worker also appears on the **Providers** tab and is routed to like any other provider once it connects.

**On the worker:** open its **Workers** tab, paste the hub URL and token under **Serve a hub from this router**, and save. Or set the environment variables:

```bash
LLAMA_ROUTER_WORKER_HUB_URL=https://llm.example.org
LLAMA_ROUTER_WORKER_TOKEN=lrw_...
```

The hub listens at `/worker/ws` on the Ollama API port (11434). A hub URL of `https://host`, `wss://host`, or `http://host:11434` all work. Use TLS when workers connect over the internet; a reverse proxy in front of port 11434 is the easiest way.

**What a worker serves.** Jobs run against the worker's own Ollama and OpenAI APIs, so the worker shares whatever providers it has. Only chat, generate, completions, embeddings, responses, images, audio, decisions, and model listing are allowed. Model pulls and deletes from the hub are refused unless `WORKER_ALLOW_ADMIN=true`. To point a worker straight at a local Ollama instead of its own router, set `WORKER_OLLAMA_URL`.

**Reliability.** The worker reconnects with capped exponential backoff and jitter, and WebSocket pings detect dead links within about 40 seconds. A revoked or rotated token stops reconnect attempts for five minutes at a time. Each machine needs its own token; a second connection with the same token replaces the first.

**Privacy.** A worker sees the prompts it serves. Make that clear to whoever uses the hub.

**Loops.** Do not add the hub as a provider on a worker that relays to its own router, or a request can bounce between them. If a worker also wants to use the hub's models, point the worker's upstream at its local Ollama with `WORKER_OLLAMA_URL`.

See [docs/workers.md](docs/workers.md) for the wire protocol.

## Configuration

All settings are configured via environment variables with the prefix `LLAMA_ROUTER_`.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_PATH` | `llama_router.db` | Path to SQLite database when `DATABASE_URL` is unset |
| `DATABASE_URL` | *(empty)* | Async SQLAlchemy URL (e.g. `sqlite+aiosqlite:///…`, `postgresql+asyncpg://…`, `mysql+asyncmy://…`). When set, overrides `DATABASE_PATH`. Use optional install extras `postgres` or `mysql` for drivers; Alembic migrations use matching sync drivers (`psycopg`, `pymysql`) automatically. |
| `DASHBOARD_HOST` | `0.0.0.0` | Dashboard bind address |
| `DASHBOARD_PORT` | `80` | Dashboard port |
| `API_HOST` | `0.0.0.0` | Ollama API bind address |
| `API_PORT` | `11434` | Ollama API port |
| `LLAMACPP_HOST` | `0.0.0.0` | OpenAI API bind address (the name predates other backends) |
| `LLAMACPP_PORT` | `8080` | OpenAI API port |
| `HEALTH_CHECK_INTERVAL_SECONDS` | `30` | Seconds between health checks |
| `AUTO_PROBE_ENABLED` | `true` | Probe a provider's endpoints on add, edit, address change and recovery. Probes never run inference |
| `OLLAMA_API_TRANSLATION` | `true` | Serve Ollama API requests from OpenAI-compatible providers too, translating both ways |
| `PROBE_MAX_AGE_HOURS` | `24` | Probe again when the last probe is older than this; `0` turns periodic re-probing off |
| `SESSION_AFFINITY_ENABLED` | `true` | Keep each conversation on one provider. An API key can override this |
| `SESSION_AFFINITY_TTL_SECONDS` | `1800` | A session forgets its provider after this long without a request |
| `SESSION_AFFINITY_MAX_SESSIONS` | `1000` | Most sessions remembered at once; the least recently used are dropped first |
| `WORKER_HUB_ENABLED` | `true` | Accept worker connections at `/worker/ws` on the Ollama API port |
| `WORKER_HUB_URL` | *(empty)* | Hub to connect to as a worker. With `WORKER_TOKEN`, overrides the dashboard setting |
| `WORKER_TOKEN` | *(empty)* | Worker token issued by the hub |
| `WORKER_OLLAMA_URL` | *(this router)* | Ollama API the worker relays jobs to; `none` disables Ollama jobs |
| `WORKER_LLAMACPP_URL` | *(this router)* | OpenAI-style API the worker relays jobs to; `none` disables those jobs |
| `WORKER_UPSTREAM_API_KEY` | *(empty)* | `API-KEY` the worker sends to its upstream when that requires one |
| `WORKER_MAX_CONCURRENCY` | `4` | Jobs the worker runs at once; the hub queues the rest |
| `WORKER_ALLOW_ADMIN` | `false` | Let the hub pull and delete models on this worker |
| `TZ` | (system) | Timezone for dashboard timestamps (e.g. `America/New_York`) |

### Migrating data from SQLite to PostgreSQL

Use the bundled CLI after creating an empty PostgreSQL database and granting a user DDL rights (Alembic will create tables).

1. Install the Postgres extra: `pip install 'llama-router[postgres]'` (provides `psycopg` for this tool and optional runtime use).
2. Run:

```bash
llama-router-migrate-sqlite-pg \
  --sqlite /path/to/llama_router.db \
  --postgres postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE
```

The tool applies Alembic on the target, truncates all application tables (not `alembic_version`), copies rows in foreign-key order so IDs stay stable, then resets sequences. Your SQLite file should already match the current schema (run the app against SQLite once so migrations are applied there, or use a backup from a recent version).

You can pass `postgresql+asyncpg://…` and it will be rewritten to `postgresql+psycopg://…` for the sync copy step.

### Sentry (Optional)

Set a DSN to enable Sentry reporting for unhandled exceptions and performance telemetry.

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTRY_DSN` | *(empty)* | Sentry DSN. When empty, Sentry is disabled |
| `SENTRY_ENVIRONMENT` | `production` | Sentry environment name |
| `SENTRY_TRACES_SAMPLE_RATE` | `0.0` | Performance transaction sampling rate (`0.0` to `1.0`) |
| `SENTRY_PROFILES_SAMPLE_RATE` | `0.0` | Profiling sampling rate (`0.0` to `1.0`) |
| `SENTRY_SEND_DEFAULT_PII` | `false` | Include user/IP headers where available |

### OCI Registry Cache

The cache acts as a pull-through proxy for the Ollama model registry (`registry.ollama.ai`). When enabled, model pulls are routed through the cache so that large blob layers are stored locally on first download. Subsequent pulls of the same model — on any provider — are served from the local cache at LAN speed instead of re-downloading from the internet.

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_ENABLED` | `true` | Enable the pull-through cache |
| `CACHE_DIR` | `./model_cache` | Directory to store cached blobs and manifests |
| `CACHE_HOST` | `0.0.0.0` | Cache registry bind address |
| `CACHE_PORT` | `9200` | Cache registry port |
| `CACHE_EXTERNAL_HOST` | *(none)* | **Required for cache.** The hostname or IP that Ollama backends use to reach the cache (e.g. `192.168.1.50`, `llama-router.local`). Falls back to `127.0.0.1` if unset, which only works when Ollama runs on the same machine. |
| `CACHE_MANIFEST_TTL_HOURS` | `240` | Hours before re-fetching a cached manifest (default 10 days) |

#### How the cache works

1. You pull a model through the dashboard or the Ollama API (e.g. `llama3:8b`).
2. llama-router rewrites the pull request so Ollama fetches layers from `http://<cache-external-host>:9200/...` instead of `registry.ollama.ai`.
3. The cache checks its local blob store (content-addressed by SHA256 digest):
   - **Cache hit** — the blob is streamed directly from disk at LAN speed.
   - **Cache miss** — the blob is fetched from the upstream registry, streamed to the Ollama client, and simultaneously saved to disk for future requests.
4. Manifests are cached with a configurable TTL (default 10 days). Blobs never expire because they are immutable content-addressed data.

#### Requirements

- The cache runs as an HTTP server on port 9200 (configurable). This port does **not** need to be exposed externally — it only needs to be reachable from the Ollama backends on your LAN.
- In Docker, the cache directory should be on a persistent volume with enough space for the models you pull (models range from ~2 GB to ~200+ GB).

#### Disabling the cache

Set `LLAMA_ROUTER_CACHE_ENABLED=false` to disable the cache entirely. Pulls will go directly to the upstream registry as usual.

#### Clearing the cache

Use the **Cache** tab on the dashboard, or call the API directly:

```bash
curl -X POST http://localhost/api/cache/clear
```

## Cloud Services

The **Add Provider** dialog starts from a self-hosted server or one of these presets: OpenAI, OpenRouter, Anthropic, Perplexity, Groq, Together AI, Fireworks AI, DeepSeek, Mistral AI, Google Gemini, xAI, Cerebras, DeepInfra, Ollama Cloud and Azure OpenAI. A preset fills in the URL, protocol, key header and any extra headers; you add a name and a key.

To attach a service without the dashboard, set its key in the environment. At startup the router creates a provider for each preset whose key is set, or updates the key of the one it created before:

```bash
LLAMA_ROUTER_PROVIDER_OPENAI_API_KEY=sk-...
LLAMA_ROUTER_PROVIDER_OPENROUTER_API_KEY=sk-or-...
LLAMA_ROUTER_PROVIDER_AZURE_OPENAI_API_KEY=...
LLAMA_ROUTER_PROVIDER_AZURE_OPENAI_RESOURCE=my-resource
```

An OpenAI-compatible address can be the server root (`http://gpu:8080`, API under `/v1`) or the API base itself (`https://openrouter.ai/api/v1`); a URL ending in a version segment is taken as the API base.

Each provider can also carry extra headers (one `Name: value` per line) and models added by hand, for services whose model list is missing or incomplete. Added models stay listed if the service's own list fails.

`tests/test_cloud_presets_live.py` checks each preset against the real service when its key variable is set.

### Running cloud providers

- **Benchmarks cost money.** The idle auto-benchmarker and **Fill missing benchmarks** skip cloud providers; a provider's **Auto-benchmark** setting overrides that either way. A manual benchmark on a cloud provider asks first. Cloud benchmarks record no cold start, and cloud models always count as loaded.
- **429 means busy, not broken.** When any provider answers HTTP 429, it is routed around for its `Retry-After` period (10 seconds without one, at most five minutes) as long as another provider can take the request. A 429 does not count toward its failure rate.
- **Big catalogs.** A provider's **Model filter** (patterns such as `anthropic/claude-*`, one per line) keeps only the listed models that match; models added by hand are always kept. The OpenRouter preset starts with a filter for the main model families.
- **Slow responses.** A provider's **Read timeout** replaces the default 120 seconds.

## Health Check

All three API ports expose a `GET /health` endpoint that returns JSON:

```json
{"status": "ok", "version": "0.7.8", "providers": 3, "providers_online": 2}
```

The dashboard port (80) includes provider counts; the Ollama (11434) and OpenAI (8080) ports return a minimal response. The Docker Compose file includes a health check configuration using this endpoint.

## Hugging Face Downloads

The Models tab can download GGUF models from Hugging Face with the `hf` CLI
and deliver them to providers:

- **Ollama providers:** the router uploads the file as a blob and creates the
  model, so remote Ollama servers need no internet access or shared disk.
  Name the model yourself or let the router derive one, such as
  `qwen3-8b:q4_k_m`.
- **llama.cpp providers:** mount `LLAMA_ROUTER_HF_MODELS_DIR` into the server's
  `--models-dir` (router mode). The router then only refreshes the provider so
  the new model is discovered.

Worker providers are not push targets, since the worker link buffers whole
request bodies; download on the worker itself instead.

Use **List files** to preview what a pattern selects, then click a file to pick
one quant. Split GGUF files work through a shared volume but cannot be pushed
to Ollama. Set `HF_TOKEN` for gated or private repositories and `HF_ENDPOINT`
for a mirror; see `.env.example` for the full list of variables the CLI reads.

## Debugging from inside the container

The image includes basic network tools so you can check reachability from the
router's own point of view, which is often different from your laptop's:

```bash
docker compose exec llama-router ping -c 3 192.168.1.100
docker compose exec llama-router host gpu-server.local
docker compose exec llama-router dig +short gpu-server.local
docker compose exec llama-router nc -zv 192.168.1.100 11434
docker compose exec llama-router curl -s http://192.168.1.100:11434/api/version
docker compose exec llama-router wget -qO- http://192.168.1.100:8080/health
docker compose exec llama-router ip route
docker compose exec llama-router traceroute 192.168.1.100
```

## Routing

Every request is routed from an in-memory routing table. Building the table reads the database once, combines it with live measurements, and ranks each model's providers separately for every proxied endpoint and every routing mode. A request only looks up its model and endpoint, adds the number of requests already in flight on each candidate, and takes the best one. It does no database queries on the way to the backend. The only hashing is the session fingerprint described under **Session affinity**, which takes tens of microseconds. Request logging happens after the response is sent.

### Routing modes

Each API key picks a mode. Requests without a key use **balanced**.

| Mode | Picks the provider that |
|------|-------------------------|
| `balanced` | finishes a typical response soonest: startup or time to first token, plus network round trip, plus 256 tokens at the provider's speed. For embeddings, images and audio, the measured time those calls take |
| `latency` | starts answering soonest: a model already loaded wins, then the lowest startup time and round trip |
| `throughput` | generates the most tokens per second. For embeddings, images and audio, completes calls fastest |
| `cheapest` | charges least for a typical request (see [Cost](#cost)); local providers with no price set cost nothing |
| `chaos` | is chosen at random, with a random model |

Pins on the key override the mode for the pinned models.

### Session affinity

Coding agents such as Claude Code, Cursor, Continue, aider and Codex send a conversation as a series of requests. Each one repeats the same long system prompt and history. llama.cpp and Ollama only reuse their prompt cache when the next request reaches the same backend. Without affinity, two equally good providers can take turns serving one conversation, and each re-reads the whole prompt every time.

With session affinity, the router remembers which provider served each conversation. Later requests in that conversation go to the same provider, even when another provider is idle. A session moves only when its provider stops being eligible: it goes offline, its circuit opens, or it no longer serves the model. It then stays on its new provider. Pins and chaos mode take precedence over affinity.

The router identifies a conversation by the first of these it finds:

1. An `X-Session-ID`, `Session-ID` or `session_id` request header.
2. `prompt_cache_key`, `metadata.session_id`, `user` or `metadata.user_id` in the request body.
3. A fingerprint of how the conversation starts. For chat, that is the system prompt and the first user message. For `/v1/responses`, it is the instructions and the first input item. For `/v1/completions` and `/api/generate`, it is the system prompt and the first 2 KB of a prompt at least that long. Shorter prompts, such as autocomplete requests, get no session.

Sessions are kept separately per API key and per requested model. Affinity applies to chat, completions, generate and responses. Embeddings, images and audio are routed per request.

Bindings live in memory only. Nothing is written to the database per request, and a restart forgets every binding. A session is forgotten after `SESSION_AFFINITY_TTL_SECONDS` without a request, or right after a request in it fails upstream.

Affinity is on by default. Turn it off with `LLAMA_ROUTER_SESSION_AFFINITY_ENABLED=false`. Each API key can follow that default, or force affinity on or off. Choose this when you create the key.

`/api/routing-table` reports the number of live sessions, sessions per provider, and hit, miss and rebind counts. The **Route Explorer** takes an optional session ID and shows where that session is bound and whether the binding is kept:

```bash
curl 'http://localhost/api/routing/explain?model=qwen2.5-coder:32b&session_id=my-session'
```

### Local and cloud providers

Every provider is **local** (this machine, the LAN, a tailnet) or **cloud**. The router works it out from the first address: private, loopback and Tailscale addresses, hostnames without a dot, and names under `.local`, `.lan`, `.home`, `.internal`, `.home.arpa` and `.ts.net` are local; everything else, and every cloud preset, is cloud. A home server reached through a public name is guessed wrong; change it on the provider's page.

Each API key allows **local or cloud** (the default), **local only** or **cloud only**. A local-only key's prompts never reach a cloud provider: fallbacks, chaos mode and session affinity all stay within the key's providers, and a pin to a provider of the other kind is ignored (the Route Explorer says so). A key limited to local is routed among local providers even when a cloud one would score better.

A client can narrow a key that allows both, for one request, with a header:

```bash
curl http://localhost:8080/v1/chat/completions -H 'API-KEY: lr_...' \
  -H 'X-Llama-Router-Locality: local' -d '{"model": "qwen3:8b", "messages": [...]}'
```

The header never widens or flips a key's setting. The request log records whether each request was served locally or in the cloud and can be filtered on it.

### Endpoints

Each provider speaks one protocol: Ollama or OpenAI-compatible. To use both APIs of one server, add it twice. Routing always knows the endpoint. OpenAI-style routes such as `/v1/chat/completions` go to any provider, since Ollama serves them too. Ollama routes (`/api/chat`, `/api/generate`, `/api/embed`, `/api/embeddings`) go to Ollama providers and, translated, to OpenAI-compatible ones, so Ollama clients can use llama.cpp, vLLM and cloud services. Translation carries messages, images, tools and tool calls, `format`, the common sampling options and token counts; `suffix` and `raw` are refused with 400, and Ollama-only options such as `top_k` and `num_ctx` are dropped. Set `LLAMA_ROUTER_OLLAMA_API_TRANSLATION=false` to keep Ollama routes on Ollama providers. For each provider the table records whether its last probe found the endpoint, and a provider whose probe found it missing is used only when nothing else can serve the request.

Measurements are kept per kind of endpoint: text generation, embeddings, images, audio and decisions. A provider that is fast at chat but slow at embeddings is ranked accordingly on each. Failures on one kind count against the model only on that kind, and against the provider as a whole.

Candidates fall into three tiers, and a request goes to the best tier that has anyone in it:

1. Healthy providers.
2. Providers with an open circuit, because the model or the whole provider has been failing.
3. Providers whose last probe found the endpoint missing.

### Provider health

Every provider has a health score from 0 to 100. It drops with the share of recent requests that failed for reasons on the provider's side, such as errors, timeouts, 401, 404 and 429. Client mistakes such as HTTP 400 don't count. It also drops when a whole health check fails, meaning no address answered, so a provider that keeps dropping out is penalized for a while after it comes back. One dead address among live ones doesn't count. Failures fade with a two-minute half-life.

The score's failure rate adds a penalty to every routing mode. A provider failing at least half of its recent requests across all models, over at least five, has its circuit opened. It is then used only if no healthier provider can serve the request, and is retried as its failures age out.

### Health-check latency

Every health check times each address. The router keeps a running tally per address: checks, failures, consecutive failures, last, min, max and mean, and the median and 95th percentile of the last 120 checks, about an hour at the default interval. Routing uses a moving average of successful checks on live addresses. The tallies appear on the **API Keys** tab under **Provider health** and in `/api/routing-table`.

### What goes into the table

| Input | Source |
|-------|--------|
| Which providers serve which models, and under which backend name | Model discovery |
| Online or offline, and which addresses answer | Health checks |
| Startup time and tokens per second | Benchmarks, replaced by measured speed once real requests report it |
| Model already loaded | Ollama `/api/ps`, or a successful request in the last four minutes |
| Network round trip | Timed health-check probes, best live address |
| Time to first token | Measured on streamed requests |
| Service time | Measured end to end on embedding, image and audio requests |
| Recent failures | Per model and kind of endpoint, and for the provider as a whole. See **Provider health** |
| Endpoint support | The last probe of the API the request would reach |
| Model larger than GPU memory | Provider GPU RAM and model size, as a small penalty |

### When the table is rebuilt

The table is rebuilt whenever one of its inputs changes: models are discovered or deleted, a provider changes status, an address goes up or down, a benchmark is added or removed, a fallback or provider is edited, a probe runs, the set of loaded models changes, or new round-trip and request measurements arrive. Changes within a quarter of a second are batched into one rebuild, and rebuilds run at most once a second. Measurement-only rebuilds reuse the last database read.

Resolved API keys are cached for five minutes. Creating or deleting a key, editing its pins, or changing the authentication policy takes effect on the next request.

Inspect the current table, with each candidate's inputs and scores, or force a rebuild:

```bash
curl http://localhost/api/routing-table
curl -X POST http://localhost/api/routing-table/rebuild
```

The **API Keys** tab shows each model's first choice per mode for `/v1/chat/completions`, and each provider's health and latency tally.

### Explaining a route

The **Route Explorer** on the **API Keys** tab shows where a request would go right now, and why, without sending it. Pick a model, an endpoint, and a service level: a routing mode, or an API key to apply that key's mode, fallback setting and pins. It walks the fallback chain and lists every provider that has the model, with its score, the live queue, the measurements behind the score, and why any provider was left out. A provider can be left out because it is offline, doesn't serve that API, is failing, is missing the endpoint, or because another provider is pinned. The explanation uses the same table and code path as real requests.

The same is available as JSON:

```bash
curl 'http://localhost/api/routing/explain?model=llama3.2:latest&endpoint=/api/chat&mode=latency'
curl 'http://localhost/api/routing/explain?model=llama3.2:latest&key_id=1'
```

`endpoint` defaults to `/v1/chat/completions`. `mode` is `balanced`, `latency`, `throughput` or `chaos`, and defaults to `balanced`. Add `allow_fallback=false` to skip fallbacks. Add `session_affinity=on` or `off` to override the default, and `session_id` to explain one session. `key_id` is admin-only.

## Benchmarks

Benchmarks measure each model's cold-start time and generation speed on each provider and API. Routing uses them until live requests provide better numbers. The figures come from the backends' own counters, not from timing on the router. For Ollama, that's load and generation durations. For llama.cpp, it's the server's per-request timings. A run on a model that was already loaded can't measure a cold start, so it records the speed only. Routing then keeps the last real startup measurement.

### Auto-benchmark

The router benchmarks models on its own while their provider is idle. It's on by default, and you can control it from the **Auto-benchmark** card on the **Benchmarks** tab or with these settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_BENCHMARK_ENABLED` | `true` | Run benchmarks automatically |
| `AUTO_BENCHMARK_IDLE_SECONDS` | `300` | How long a provider must go without a request before it is benchmarked |
| `AUTO_BENCHMARK_MAX_AGE_DAYS` | `7` | Benchmarks older than this are run again |
| `AUTO_BENCHMARK_CHECK_SECONDS` | `60` | How often to look for idle providers with work |
| `AUTO_BENCHMARK_CONCURRENCY` | `1` | Providers benchmarked at once. Each provider runs one benchmark at a time |

Values saved from the dashboard override the environment.

How it stays out of the way:

- **Idle only.** A provider is benchmarked only after five minutes, by default, without a request. That matches Ollama's default `keep_alive`, so models have usually unloaded by then. The benchmark measures a real cold start and doesn't push out a model someone is using.
- **Traffic wins.** A request to a provider stops its running benchmark immediately. The benchmark runs again the next time the provider is idle.
- **Order.** Models that have never been benchmarked go first, then the oldest past the maximum age.
- **Failures.** A benchmark that fails waits a day before it's retried.
- **Decision check.** After each benchmark, the model answers one fixed decision question. That tells routing whether the provider returns logprobs (see [Decision models](#decision-models)) and seeds its decision latency.

Automatic results show an **auto** badge on the Benchmarks tab, and their log entries use the `auto-benchmark` endpoint.

```bash
curl http://localhost/api/auto-benchmark          # settings, running, due, recent results
curl -X POST http://localhost/api/auto-benchmark \
  -H 'Content-Type: application/json' \
  -d '{"enabled": true, "idle_minutes": 10, "max_age_days": 3}'
curl -X POST http://localhost/api/auto-benchmark/check   # look for idle providers now
```

### Load testing

These benchmarks are quick probes, one short request each, not load tests. To see how a provider or the router holds up under concurrent traffic, point a load-testing tool such as [GuideLLM](https://github.com/vllm-project/guidellm) or `vllm bench serve` at the router's OpenAI-compatible API on port 8080. Run it at a quiet time, because it saturates the GPUs it reaches.

## Decision models

The router answers decision requests in the shape of TypeSafe's [Jev](https://en.wikipedia.org/wiki/Jev_(AI_model)) API on port 8080. A request carries a `state`, which is text, a JSON object or a list of strings, and named, typed questions. Each answer is a probability distribution, not generated text.

| Type | The question carries | The answer carries |
|------|----------------------|--------------------|
| `noul` | `instructions`, a statement, and optionally `negative`, what the false case looks like | `noul`, the probability the statement is true |
| `choice` | `instructions` and `criteria`, an object of option keys and descriptions | `choice`, `probabilities` and `confidence` |
| `score` | `instructions` and `criteria`, a list of rubric levels | `score` (the expected level), `probabilities`, `legend` and `confidence` |

```bash
curl http://localhost:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "llama3.1:8b",
  "state": "Our API has returned 500 on every request for 20 minutes.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this",
      "criteria": {"billing": "Payments", "technical": "Bugs or outages", "sales": "Pricing"}},
    "is_urgent": {"type": "noul", "instructions": "The message conveys urgency"}
  }
}'
```

`/v1/decisions` is the same API under a vendor-neutral name. Both routes are also served without `/v1`, and on the Ollama API port (11434) as well as 8080. Routing, API key policies, pins and fallbacks apply as for chat.

How a request is answered:

- **Passed through.** If the provider's last probe found the same route upstream, for example LiteLLM in front of Jev, the request goes there unchanged apart from the model name.
- **Emulated.** Otherwise each question becomes a one-token chat completion with logprobs, all sent to one provider at once. The answer is read from the probabilities of the option labels. llama.cpp, vLLM and most OpenAI-compatible servers support this.
- **No logprobs.** A provider that answers without logprobs isn't offered for decisions for an hour, and the request is routed once more.
- **Workers.** A hub sends decisions to its workers the same way. A worker that relays its own router answers them there; an older worker that doesn't relay decision routes gets them emulated. See [docs/workers.md](docs/workers.md#decision-requests).

Local chat models are not trained for calibrated answers. In testing, choice questions came out accurate and well calibrated. Yes/no questions leaned toward yes, and score questions drifted toward the middle level. Calibrations correct this per model and question wording, with a temperature and an offset per answer:

```bash
# Given directly
curl -X POST http://localhost/api/decisions/calibrations -H 'Content-Type: application/json' \
  -d '{"model": "llama3.1:8b", "question": {"type": "noul", "instructions": "The message conveys urgency"},
       "offsets": {"true": -1.2, "false": 1.2}}'
# Fit from labelled examples: the uncalibrated probabilities and the right answer
curl -X POST http://localhost/api/decisions/calibrations -H 'Content-Type: application/json' \
  -d '{"model": "llama3.1:8b", "question": {...}, "examples": [{"probabilities": {"true": 0.9, "false": 0.1}, "answer": "false"}]}'
curl http://localhost/api/decisions/calibrations
curl -X DELETE http://localhost/api/decisions/calibrations/1
```

The design, measurements and plan are in [docs/decision-models-plan.md](docs/decision-models-plan.md).

## Model Fallbacks

You can configure fallback models so that when a requested model is unavailable (no provider has it or all providers with it are offline), the router transparently tries an alternative model.

Fallbacks cascade: if model A falls back to model B, and model B falls back to model C, a request for model A will try A → B → C in order.

Configure fallbacks through:
- The **Models** tab on the dashboard (click the arrow icon next to any model, or use the Fallbacks section)
- The REST API:

```bash
# Set a fallback
curl -X POST http://localhost/api/fallbacks \
  -H 'Content-Type: application/json' \
  -d '{"model": "llama3:70b", "fallback": "llama3:8b"}'

# List all fallbacks
curl http://localhost/api/fallbacks

# Remove a fallback
curl -X DELETE http://localhost/api/fallbacks/llama3:70b
```

## Cost

The request log records each request's tokens in and out, the API key that made it, and its cost when the provider's price for the model is known. Tokens come from the response's `usage` (or Ollama's counts); a streamed OpenAI-style response carries usage only when the client sends `stream_options: {"include_usage": true}`.

**Prices** are USD per million tokens, set per provider on its page: a model name, a pattern such as `gpt-4o*`, or `*` for every model. A price set by hand wins over one from the provider's own model list (OpenRouter lists one for every model). Local providers cost nothing unless priced; a cloud model with no price has an unknown cost.

**Cheapest** routing ranks candidates by the price of a typical request (1,000 tokens in, 256 out), then by the balanced score. A cloud model with an unknown price comes last.

**Daily spend caps.** An API key can have a cap in USD per UTC day. Once its logged spend reaches the cap, the key is limited to local providers until the next day; a cloud-only key is refused with HTTP 429. The key list shows each key's spend today.

## Model Aliases

The same model often has different names at different providers: `claude-sonnet-4-5` at Anthropic, `anthropic/claude-sonnet-4-5` at OpenRouter, `llama3.2:latest` and `llama3.2` across Ollama and llama.cpp. An alias joins them. A request for any name in the group can go to any provider that serves one of them, and each provider is sent its own name for the model. Aliases chain, so `a → b → c` is one group.

Add them in **Model Aliases** on the Models tab; **Suggest aliases** proposes `vendor/model` and `model:latest` names whose bare form another provider serves. Or use the API:

```bash
curl -X POST http://localhost/api/model-aliases \
  -H 'Content-Type: application/json' \
  -d '{"alias": "anthropic/claude-sonnet-4-5", "canonical": "claude-sonnet-4-5"}'
```

Aliases combine with API key locality: a local-only key asking for `vendor/model` reaches a local provider that serves `model`.

## API Key Model Pinning

API keys can optionally pin specific models to specific providers. When a pinned model is requested with that key, llama-router always chooses the configured provider for that model (if available), bypassing latency/throughput scoring.

You can configure pins from the **API Keys** tab in the dashboard, or with the REST API:

```bash
# Pin a model to a provider for API key 1
curl -X POST http://localhost/api/keys/1/pins \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"llama3.2:latest","provider_id":2}'

# List pins for API key 1
curl http://localhost/api/keys/1/pins

# Remove a pin
curl -X DELETE http://localhost/api/keys/1/pins/llama3.2:latest
```

## Development

```bash
pip install -e '.[dev]'
pytest -v
black src/ tests/
```

Or with Docker:

```bash
docker compose -f docker-compose.dev.yml run test
```

Tests use a temporary SQLite file. To run the database-backed tests against PostgreSQL, MySQL or MariaDB instead, point them at a throwaway database. It is emptied before every test:

```bash
LLAMA_ROUTER_TEST_DATABASE_URL=mysql+asyncmy://root:pw@127.0.0.1:3306/lr pytest tests/test_database.py tests/test_routing_table.py
```

A few dashboard tests drive the app through a synchronous test client and fail against a server database with "attached to a different loop"; that is the test client, not the database.
