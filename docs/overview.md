# llama-router overview

**What:** an LLM router that puts many Ollama and OpenAI-compatible backends
(and cloud APIs) behind one Ollama API, one OpenAI API and a web dashboard.
**Runs as:** one process with four HTTP servers: dashboard :80, Ollama API
:11434, OpenAI API :8080, Ollama registry cache :9200.
**Talks to:** LLM backends over HTTP; SQLite by default (PostgreSQL or MySQL
optional); other llama-routers over WebSocket worker links; Hugging Face;
optionally Sentry and an OpenID Connect provider.

## Files

- `src/llama_router/__main__.py`: command line, logging, starts the servers, SIGTERM.
- `src/llama_router/config.py`: `Settings` from `LLAMA_ROUTER_*` variables.
- `src/llama_router/api/`: Ollama API, decisions API, worker hub socket.
- `src/llama_router/llamacpp_api/`: OpenAI-compatible API.
- `src/llama_router/dashboard/`, `templates/`: dashboard, sign-in, OIDC; `GET /healthz`.
- `src/llama_router/metrics.py`: Prometheus text format for `GET /metrics`.
- `src/llama_router/registry_cache/`: pull-through cache for Ollama pulls.
- `src/llama_router/worker_link/`: hub and worker links.
- `src/llama_router/provider_manager.py`, `probe.py`: discovery, probing, health.
- `src/llama_router/router.py`, `routing_table.py`, `routing_stats.py`, `affinity.py`: routing.
- `src/llama_router/model_listing.py`, `dashboard/model_groups.py`: model groups in listings and their API.
- `src/llama_router/dashboard/management_keys.py`: privilege-scoped management keys for the dashboard API.
- `src/llama_router/dashboard/chat.py`, `templates/chat.html`: Chat tab for trying a model on a provider.
- `src/llama_router/*_client.py`, `translate.py`: backend clients, Ollama/OpenAI translation.
- `src/llama_router/decision*.py`: decision models and per-key policies.
- `src/llama_router/database.py`, `db_*.py`, `alembic/`: storage, migrations (run at startup).
- `tests/`: pytest; mostly one file per feature.
- `scripts/decision_spike/`: one-off experiments for decision models.
- `Dockerfile`: `dev` and `production` stages.
- `docker-compose.yaml` (dev), `.test`, `.lint`, `.prod`.
- `docs/workers.md`: worker link protocol. `docs/*-plan.md`: design notes.

## Conventions

- All configuration from environment variables; see docs/configuration.md.
- Async throughout (FastAPI, httpx, SQLAlchemy asyncio).
- Routing never touches the database on the request path; tables are rebuilt
  in the background.
- Rules: https://github.com/romkey/python-ai-rules/blob/main/RULES.md;
  deviations in AGENTS.md.
