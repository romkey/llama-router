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
- **OCI Registry Cache** — Built-in pull-through cache for Ollama model downloads, serving cached layers at LAN speed

## Quick Start

```bash
docker compose up -d
```

Open http://localhost to access the dashboard and add your backends.

## Configuration

All settings are configured via environment variables with the prefix `LLAMA_ROUTER_`.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_PATH` | `llama_router.db` | Path to SQLite database |
| `DASHBOARD_HOST` | `0.0.0.0` | Dashboard bind address |
| `DASHBOARD_PORT` | `80` | Dashboard port |
| `API_HOST` | `0.0.0.0` | Ollama API bind address |
| `API_PORT` | `11434` | Ollama API port |
| `LLAMACPP_HOST` | `0.0.0.0` | llama.cpp API bind address |
| `LLAMACPP_PORT` | `8080` | llama.cpp API port |
| `HEALTH_CHECK_INTERVAL_SECONDS` | `30` | Seconds between health checks |
| `TZ` | (system) | Timezone for dashboard timestamps (e.g. `America/New_York`) |

### OCI Registry Cache

The cache acts as a pull-through proxy for the Ollama model registry (`registry.ollama.ai`). When enabled, model pulls are routed through the cache so that large blob layers are stored locally on first download. Subsequent pulls of the same model — on any provider — are served from the local cache at LAN speed instead of re-downloading from the internet.

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_ENABLED` | `true` | Enable the pull-through cache |
| `CACHE_DIR` | `./model_cache` | Directory to store cached blobs and manifests |
| `CACHE_HOST` | `0.0.0.0` | Cache registry bind address |
| `CACHE_PORT` | `9200` | Cache registry port |
| `CACHE_MANIFEST_TTL_HOURS` | `240` | Hours before re-fetching a cached manifest (default 10 days) |

#### How the cache works

1. You pull a model through the dashboard or the Ollama API (e.g. `llama3:8b`).
2. llama-router rewrites the pull request so Ollama fetches layers from `http://<cache-host>:9200/...` instead of `registry.ollama.ai`.
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
