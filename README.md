# llama-router

An LLM router that aggregates multiple Ollama and llama.cpp backends behind unified APIs.

## Features

- **Unified Ollama API** — Ollama-compatible API on port 11434 that routes to the best available backend
- **Unified llama.cpp / OpenAI API** — OpenAI-compatible API on port 8080 for llama.cpp server backends
- **Web Dashboard** — Bootstrap-based management UI on port 80 with provider status, model inventory, benchmarks, request log, and cache management
- **Auto-Discovery** — Queries backends for models and capabilities on registration
- **Health Checking** — Periodic monitoring of backend availability, including per-address liveness for multi-address providers
- **Benchmarking** — Measure startup time and tokens/sec per model per provider
- **Smart Routing** — Routes to the least-busy, most-capable provider based on active requests and benchmark data
- **Model Fallbacks** — Configure cascading fallback models so requests are transparently rerouted when a model is unavailable
- **API Key Routing Policies** — Per-key latency/throughput/chaos routing, fallback control, and optional model-to-provider pinning
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

### Connecting routers: hubs and workers

One llama-router can serve as a **hub** for others. A **worker** is a llama-router that opens a single outbound WebSocket to the hub and serves the inference jobs the hub sends down it. The worker needs no open ports, port forwarding, or VPN, so it works from a home network, behind NAT, or while the machine is on another tailnet or VPN. Nothing on the worker's network becomes reachable from the hub.

**On the hub:** open the dashboard **Workers** tab, enter a name, and click **Create worker token**. The token is shown once. The worker also appears on the **Providers** tab and is routed to like any other provider once it connects.

**On the worker:** open its **Workers** tab, paste the hub URL and token under **Serve a hub from this router**, and save. Or set the environment variables:

```bash
LLAMA_ROUTER_WORKER_HUB_URL=https://llm.example.org
LLAMA_ROUTER_WORKER_TOKEN=lrw_...
```

The hub listens at `/worker/ws` on the Ollama API port (11434). A hub URL of `https://host`, `wss://host`, or `http://host:11434` all work. Use TLS when workers connect over the internet; a reverse proxy in front of port 11434 is the easiest way.

**What a worker serves.** Jobs run against the worker's own Ollama and llama.cpp APIs, so the worker shares whatever providers it has. Only chat, generate, completions, embeddings, responses, images, audio, and model listing are allowed. Model pulls and deletes from the hub are refused unless `WORKER_ALLOW_ADMIN=true`. To point a worker straight at a local Ollama instead of its own router, set `WORKER_OLLAMA_URL`.

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
| `LLAMACPP_HOST` | `0.0.0.0` | llama.cpp API bind address |
| `LLAMACPP_PORT` | `8080` | llama.cpp API port |
| `HEALTH_CHECK_INTERVAL_SECONDS` | `30` | Seconds between health checks |
| `WORKER_HUB_ENABLED` | `true` | Accept worker connections at `/worker/ws` on the Ollama API port |
| `WORKER_HUB_URL` | *(empty)* | Hub to connect to as a worker. With `WORKER_TOKEN`, overrides the dashboard setting |
| `WORKER_TOKEN` | *(empty)* | Worker token issued by the hub |
| `WORKER_OLLAMA_URL` | *(this router)* | Ollama API the worker relays jobs to; `none` disables Ollama jobs |
| `WORKER_LLAMACPP_URL` | *(this router)* | llama.cpp API the worker relays jobs to; `none` disables llama.cpp jobs |
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

## Health Check

All three API ports expose a `GET /health` endpoint that returns JSON:

```json
{"status": "ok", "version": "0.7.8", "providers": 3, "providers_online": 2}
```

The dashboard port (80) includes provider counts; the Ollama (11434) and llama.cpp (8080) ports return a minimal response. The Docker Compose file includes a health check configuration using this endpoint.

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
