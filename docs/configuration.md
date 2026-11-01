# Configuration

llama-router reads its configuration from environment variables once at
startup (`src/llama_router/config.py`). Every variable is optional. In Docker,
put them in `.env` next to the compose file; [`.env.example`](../.env.example)
lists them all.

Most variables take the prefix `LLAMA_ROUTER_`, which the tables below leave
out: `DASHBOARD_PORT` means `LLAMA_ROUTER_DASHBOARD_PORT`. Booleans accept
`true`/`false` (also `1`/`0`, `yes`/`no`). Settings saved from the dashboard
(auto-benchmark, worker link) override the environment where noted.

## Database

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATABASE_PATH` | `llama_router.db` (Docker: `/app/data/llama_router.db`) | SQLite file used when `DATABASE_URL` is empty |
| `DATABASE_URL` | *(empty)* | Async SQLAlchemy URL: `sqlite+aiosqlite:///…`, `postgresql+asyncpg://…` or `mysql+asyncmy://…`. Overrides `DATABASE_PATH`. The Docker image includes the PostgreSQL and MySQL drivers; a pip or uv install needs the `postgres` or `mysql` extra; Alembic migrations use the matching sync drivers (`psycopg`, `pymysql`) automatically |

SQLite is the default and fits a single router. To move to PostgreSQL, see
"Migrating data from SQLite to PostgreSQL" in the README.

## Servers

| Variable | Default | Meaning |
|----------|---------|---------|
| `DASHBOARD_HOST` | `0.0.0.0` | Dashboard bind address |
| `DASHBOARD_PORT` | `80` | Dashboard port. Also serves `GET /healthz` (liveness) and `GET /health` (provider summary) |
| `API_HOST` | `0.0.0.0` | Ollama API bind address |
| `API_PORT` | `11434` | Ollama API port; also accepts worker links at `/worker/ws` |
| `LLAMACPP_HOST` | `0.0.0.0` | OpenAI-compatible API bind address (the name predates other backends) |
| `LLAMACPP_PORT` | `8080` | OpenAI-compatible API port |
| `METRICS_ENABLED` | `true` | Serve Prometheus metrics at `GET /metrics` on the dashboard port; see [metrics.md](metrics.md) |
| `METRICS_TOKEN` | *(empty)* | Bearer token scrapers send to `/metrics`. Empty: the endpoint needs a dashboard sign-in, like the rest of the dashboard |
| `API_ON_DASHBOARD_PORT` | `false` | Also serve the Ollama API, the OpenAI API, decisions and worker links on the dashboard port. The API ports keep listening either way. Require API keys before exposing it |

## Dashboard sign-in

| Variable | Default | Meaning |
|----------|---------|---------|
| `SESSION_SECRET` | *(empty)* | Secret for signing session cookies. Empty: a random one is generated and kept in the database |
| `DASHBOARD_COOKIE_SECURE` | `false` | Send session cookies over HTTPS only. Set `true` behind TLS |
| `MAX_API_KEYS_PER_USER` | `20` | Most API keys one user may issue themselves on My API keys (0: no limit). Keys an admin issues don't count |
| `MAX_CHAT_CONVERSATIONS_PER_USER` | `500` | Most chat conversations one user may keep; past it they must delete one to start another (0: no limit) |
| `OIDC_ISSUER` | *(empty)* | OpenID Connect issuer URL. Empty turns single sign-on off |
| `OIDC_CLIENT_ID` | *(empty)* | Client ID |
| `OIDC_CLIENT_SECRET` | *(empty)* | Client secret |
| `OIDC_SCOPES` | `openid profile email groups` | Requested scopes |
| `OIDC_GROUPS_CLAIM` | `groups` | Claim holding the user's groups; dots reach nested claims (Keycloak: `realm_access.roles`) |
| `OIDC_ADMIN_GROUP` | *(empty)* | Members are dashboard administrators |
| `OIDC_ALLOWED_GROUP` | *(empty)* | If set, only members (and admins) may sign in |
| `OIDC_REDIRECT_BASE_URL` | *(empty)* | External dashboard URL, e.g. `https://llm.example.org`; the callback is `{base}/login/oidc/callback`. Empty uses the request's host and `X-Forwarded-*` headers |
| `OIDC_DISPLAY_NAME` | `single sign-on` | Button text: "Sign in with …" |
| `OIDC_ONLY` | `false` | Hide the password form and refuse password sign-in |
| `OIDC_LOGOUT_AT_IDP` | `false` | Also end the session at the identity provider on logout |
| `OIDC_ALLOW_INSECURE` | `false` | Allow an `http://` issuer, for local testing only |

## Providers and routing

| Variable | Default | Meaning |
|----------|---------|---------|
| `HEALTH_CHECK_INTERVAL_SECONDS` | `30` | Seconds between provider health checks |
| `QUIET_HEALTH_CHECKS` | `false` | Leave successful `GET /healthz` and `GET /health` requests out of the access log. Failed ones are still logged |
| `TRUSTED_PROXIES` | (empty) | Reverse proxies whose `X-Forwarded-For` header is believed, as IPs or CIDRs separated by commas (e.g. `127.0.0.1,10.0.0.0/8`). Empty ignores the header: the TCP peer is the client in the request log and for anonymous session affinity |
| `REQUEST_LOG_RETENTION_DAYS` | `30` | Delete request log rows older than this, checked hourly; `0` keeps them forever |
| `CIRCUIT_TIMEOUT_MAX_TOKENS` | `4096` | A read timeout on a request asking for more output tokens than this (`max_tokens`, `num_predict`...) is blamed on the request and does not count toward the provider's circuit breaker |
| `AUTO_PROBE_ENABLED` | `true` | Probe a provider's endpoints on add, edit, address change and recovery. Probes never run inference |
| `PROBE_MAX_AGE_HOURS` | `24` | Probe again when the last probe is older than this; `0` turns periodic re-probing off |
| `OLLAMA_API_TRANSLATION` | `true` | Serve Ollama API requests from OpenAI-compatible providers too, translating both ways |
| `SESSION_AFFINITY_ENABLED` | `true` | Keep each conversation on one provider. An API key can override this |
| `SESSION_AFFINITY_TTL_SECONDS` | `1800` | A session forgets its provider after this long without a request |
| `SESSION_AFFINITY_MAX_SESSIONS` | `1000` | Most sessions remembered at once; the least recently used are dropped first |
| `SESSION_AFFINITY_MAX_SESSIONS_PER_KEY` | `200` | Most sessions one API key (or one client address, for requests without a key) may hold; its least recently used go first |
| `DECISION_POLICY_TIMEOUT_SECONDS` | `3.0` | Longest an API key's decision policy may take. After that a policy with gates refuses the request (503); a tier-only policy routes it as asked |
| `DECISION_POLICY_FAIL_OPEN` | `false` | When a policy with gates cannot be decided (no provider, error, timeout), let the request through unchecked instead of refusing it |
| `DECISION_MAX_QUESTIONS` | `8` | Most questions in one decision request (each is one backend call); a policy may have one gate fewer |
| `DECISION_MAX_STATE_CHARS` | `32000` | Longest decision request state, in characters |
| `DECISION_MAX_CONCURRENCY` | `4` | Questions of one decision request asked of the backend at once |

## Benchmarks

The dashboard's **Auto-benchmark** card overrides the `AUTO_BENCHMARK_*`
values once saved.

| Variable | Default | Meaning |
|----------|---------|---------|
| `BENCHMARK_PROMPT` | `Write a short sentence about the weather.` | Prompt used for benchmarks |
| `AUTO_BENCHMARK_ENABLED` | `true` | Benchmark models automatically while their provider is idle |
| `AUTO_BENCHMARK_IDLE_SECONDS` | `300` | How long a provider must go without a request before it is benchmarked |
| `AUTO_BENCHMARK_MAX_AGE_DAYS` | `7` | Benchmarks older than this are run again |
| `AUTO_BENCHMARK_CHECK_SECONDS` | `60` | How often to look for idle providers with work |
| `AUTO_BENCHMARK_CONCURRENCY` | `1` | Providers benchmarked at once; each runs one benchmark at a time |

## Ollama registry cache

| Variable | Default | Meaning |
|----------|---------|---------|
| `CACHE_ENABLED` | `true` | Run the pull-through cache for Ollama model downloads |
| `CACHE_DIR` | `./model_cache` (Docker: `/app/data/model_cache`) | Where cached blobs and manifests are stored |
| `CACHE_HOST` | `0.0.0.0` | Cache bind address |
| `CACHE_PORT` | `9200` | Cache port |
| `CACHE_EXTERNAL_HOST` | *(empty)* | Host or IP Ollama backends use to reach the cache, e.g. `192.168.1.50`. Empty means `127.0.0.1`, which only works when Ollama runs on the same machine |
| `CACHE_MANIFEST_TTL_HOURS` | `240` | Hours before a cached manifest is fetched again |
| `CACHE_MAX_CONCURRENT_BLOBS` | `4` | Parallel blob downloads when pre-caching a model from the dashboard |

## Hub and worker links

See [workers.md](workers.md). Setting `WORKER_HUB_URL` and `WORKER_TOKEN`
together pins the link and locks the dashboard form.

| Variable | Default | Meaning |
|----------|---------|---------|
| `WORKER_HUB_ENABLED` | `true` | Accept worker connections at `/worker/ws` on the Ollama API port |
| `WORKER_HUB_URL` | *(empty)* | Hub to connect to as a worker (`https://`, `wss://` or a bare host) |
| `WORKER_TOKEN` | *(empty)* | Worker token issued by the hub's Workers tab |
| `WORKER_OLLAMA_URL` | *(this router)* | Ollama API the worker relays jobs to; `none` disables Ollama jobs |
| `WORKER_LLAMACPP_URL` | *(this router)* | OpenAI-style API the worker relays jobs to; `none` disables those jobs |
| `WORKER_UPSTREAM_API_KEY` | *(empty)* | `API-KEY` the worker sends to its upstream when that requires one |
| `WORKER_MAX_CONCURRENCY` | `4` | Jobs the worker runs at once; the hub queues the rest |
| `WORKER_ALLOW_ADMIN` | `false` | Let the hub pull and delete models on this worker |
| `WORKER_EXPOSE` | *(dashboard; `combined`)* | `combined`: the hub sees this worker as one provider. `individual`: each shared provider on its own |

## Cloud providers

At startup the router creates a provider for each preset whose key is set, or
updates the key of the one it created before.

| Variable | Meaning |
|----------|---------|
| `PROVIDER_<ID>_API_KEY` | API key for a preset. `<ID>` is one of `OPENAI`, `OPENROUTER`, `ANTHROPIC`, `PERPLEXITY`, `GROQ`, `TOGETHER`, `FIREWORKS`, `DEEPSEEK`, `MISTRAL`, `GEMINI`, `XAI`, `CEREBRAS`, `DEEPINFRA`, `OLLAMA_CLOUD`, `AZURE_OPENAI`, `TYPESAFE` |
| `PROVIDER_AZURE_OPENAI_RESOURCE` | Azure OpenAI resource name, required with its key |

## Hugging Face downloads

| Variable | Default | Meaning |
|----------|---------|---------|
| `HF_MODELS_DIR` | `./hf_models` (Docker: `/app/data/hf_models`) | Where downloads are saved. Mount it into a llama.cpp server's `--models-dir` to share models |

The `hf` CLI reads these unprefixed variables itself; the Models tab shows
which are set. In Docker, `HF_HOME` defaults to `/app/data/hf`. Leave unused
ones unset (an empty `HF_ENDPOINT` breaks the client).

`HF_TOKEN`, `HF_ENDPOINT`, `HF_HOME`, `HF_HUB_CACHE`, `HF_XET_CACHE`,
`HF_DEBUG` (logs requests as curl commands, which can include your token),
`HF_HUB_VERBOSITY`, `HF_HUB_DOWNLOAD_TIMEOUT`, `HF_HUB_ETAG_TIMEOUT`,
`HF_HUB_OFFLINE`, `HF_HUB_DISABLE_TELEMETRY`, `HF_HUB_DISABLE_XET`,
`HF_XET_HIGH_PERFORMANCE`, `HF_HUB_DISABLE_SYMLINKS`, `HTTP_PROXY`,
`HTTPS_PROXY`, `NO_PROXY`, `REQUESTS_CA_BUNDLE`, `SSL_CERT_FILE`, and the
deprecated `HUGGING_FACE_HUB_TOKEN` and `HUGGINGFACE_HUB_CACHE`.

## Sentry

| Variable | Default | Meaning |
|----------|---------|---------|
| `SENTRY_DSN` | *(empty)* | Sentry DSN. Empty disables Sentry |
| `SENTRY_ENVIRONMENT` | `production` | Sentry environment name |
| `SENTRY_TRACES_SAMPLE_RATE` | `0.0` | Performance transaction sampling rate, `0.0` to `1.0` |
| `SENTRY_PROFILES_SAMPLE_RATE` | `0.0` | Profiling sampling rate, `0.0` to `1.0` |
| `SENTRY_SEND_DEFAULT_PII` | `false` | Include user and IP headers where available |

## Other

| Variable | Default | Meaning |
|----------|---------|---------|
| `TZ` | system | Timezone for dashboard timestamps, e.g. `America/New_York` |
| `APP_VERSION` | `latest` | Image tag `docker-compose.prod.yaml` runs |
| `LLAMA_ROUTER_TEST_DATABASE_URL` | *(empty)* | Tests only: run database tests against this server instead of a temporary SQLite file. It is emptied before every test |
