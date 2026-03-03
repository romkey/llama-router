# llama-router

An Ollama router that aggregates multiple Ollama backends behind a single API.

## Features

- **Unified API** — Ollama-compatible API on port 11434 that routes to the best available backend
- **Web Dashboard** — Bootstrap-based management UI on port 80
- **Auto-Discovery** — Queries backends for models and capabilities on registration
- **Health Checking** — Periodic monitoring of backend availability
- **Benchmarking** — Measure startup time and tokens/sec per model per provider
- **Smart Routing** — Routes to the least-busy, most-capable provider

## Quick Start

```bash
docker compose up -d
```

Open http://localhost to access the dashboard and add your Ollama backends.

## Configuration

Environment variables (prefix `LLAMA_ROUTER_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_PATH` | `llama_router.db` | Path to SQLite database |
| `DASHBOARD_HOST` | `0.0.0.0` | Dashboard bind address |
| `DASHBOARD_PORT` | `80` | Dashboard port |
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_PORT` | `11434` | Ollama API port |
| `HEALTH_CHECK_INTERVAL_SECONDS` | `30` | Seconds between health checks |

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
