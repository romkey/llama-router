# Prometheus metrics

`GET /metrics` on the dashboard port (80 by default) serves the Prometheus
text format. Every value comes from state the router already keeps, so a
scrape costs about what the dashboard's status call costs.

## Access

Set `LLAMA_ROUTER_METRICS_TOKEN` and have the scraper send it as a bearer
token:

```yaml
scrape_configs:
  - job_name: llama-router
    static_configs:
      - targets: ["llm.example.org:80"]
    authorization:
      credentials: "the token"
```

Without a token the endpoint is protected like the rest of the dashboard:
open until the first dashboard user exists, then only with a dashboard
session. `LLAMA_ROUTER_METRICS_ENABLED=false` turns the endpoint off (404).

## Metrics

All names start with `llama_router_`. Provider metrics carry the labels
`provider`, `provider_id`, `protocol` (`ollama` or `openai`) and `locality`
(`local` or `cloud`).

| Metric | Type | Labels | Meaning |
|--------|------|--------|---------|
| `info` | gauge | `version` | Always 1 |
| `providers` | gauge | `status` | Providers in each status (idle, busy, offline, unknown) |
| `models` | gauge | | Distinct model names across online providers |
| `provider_up` | gauge | provider | 1 when not offline |
| `provider_active_requests` | gauge | provider | Requests in flight |
| `provider_models` | gauge | provider | Models the provider offers |
| `provider_health_score` | gauge | provider | 0 (failing everything) to 100 (healthy) |
| `provider_request_failure_rate` | gauge | provider | Share of recent requests failing for provider-wide reasons (0..1) |
| `provider_circuit_open` | gauge | provider | 1 while the provider is skipped for failing too often |
| `provider_rate_limited_seconds` | gauge | provider | Seconds left of a cooldown after HTTP 429 |
| `address_up` | gauge | provider, `url` | 1 when the address answered its last health check |
| `address_rtt_milliseconds` | gauge | provider, `url` | Moving-average health-check round trip |
| `address_health_checks_total` | counter | provider, `url` | Health checks run |
| `address_health_check_failures_total` | counter | provider, `url` | Health checks that failed |
| `benchmark_tokens_per_second` | gauge | provider, `model`, `benchmark_protocol` | Latest benchmark speed |
| `benchmark_startup_milliseconds` | gauge | provider, `model`, `benchmark_protocol` | Latest benchmark cold start |
| `requests_total` | counter | provider, `model`, `endpoint_class`, `outcome` | Proxied requests since start-up (`ok` or `error`). Requests the client got wrong are not counted against the provider |
| `model_tokens_per_second` | gauge | provider, `model`, `endpoint_class` | Moving-average speed from real requests |
| `model_time_to_first_token_milliseconds` | gauge | provider, `model`, `endpoint_class` | Moving-average time to first token (streamed requests) |
| `model_request_duration_milliseconds` | gauge | provider, `model`, `endpoint_class` | Moving-average time to the last byte of successful requests |
| `worker_connected` | gauge | `worker`, `provider_id` | 1 while the worker's socket is open (hub side) |
| `worker_in_flight` | gauge | `worker`, `provider_id` | Requests in flight over the worker's socket |
| `worker_jobs_total` | counter | `worker`, `provider_id` | Requests sent over the worker's socket |
| `worker_link_connected` | gauge | `hub` | 1 while this router is connected to its hub as a worker |
| `worker_link_jobs_total` | counter | | Jobs served for the hub since the link started |
| `routing_table_version` | gauge | | Rebuilds of the routing table |
| `routing_table_built_timestamp_seconds` | gauge | | When the table was last rebuilt |
| `routing_table_build_milliseconds` | gauge | | How long the last rebuild took |
| `session_affinity_sessions` | gauge | | Conversations currently bound to a provider |
| `auto_benchmark_runs_total` | counter | `outcome` | Automatic benchmarks since start-up: completed, failed, interrupted |

`endpoint_class` is `generate`, `embed`, `image`, `audio` or `decide`.
Counters reset when the process restarts; use `rate()` and `increase()`
as usual.

Per-request cost and token counts are in the request log (dashboard **Log**
tab), not here.
