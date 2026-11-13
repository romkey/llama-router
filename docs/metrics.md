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
it needs a dashboard session. `LLAMA_ROUTER_METRICS_ENABLED=false` turns the endpoint off (404).

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
| `provider_probe_failure_rate` | gauge | provider | Share of recent health checks the provider failed (0..1) |
| `provider_circuit_open` | gauge | provider | 1 while the provider is skipped for failing too often |
| `provider_rate_limited_seconds` | gauge | provider | Seconds left of a cooldown after HTTP 429 |
| `provider_max_concurrent` | gauge | provider | Concurrent-request limit (only providers with one) |
| `provider_queued_requests` | gauge | provider, `level` | Requests waiting for a slot on a provider with a limit, by queue level (1 latency, 2 balanced/chaos/no key, 3 throughput/cheapest) |
| `user_tokens_today` | gauge | `user`, `user_id` | Tokens a dashboard user's keys and chats used this UTC day |
| `user_spend_today_usd` | gauge | `user`, `user_id` | USD a dashboard user spent this UTC day |
| `user_daily_token_limit` | gauge | `user`, `user_id` | The user's daily token limit (users with one) |
| `user_requests_per_minute_limit` | gauge | `user`, `user_id` | The user's requests-per-minute limit (users with one) |
| `user_daily_spend_cap_usd` | gauge | `user`, `user_id` | The user's daily spend cap (users with one) |
| `quota_refusals_total` | counter | `limit` | Requests refused (429) by a quota: `key_tokens`, `user_tokens`, `key_rpm`, `user_rpm`, `key_spend`, `user_spend` |
| `upstream_queued_requests` | gauge | `level` | Requests waiting for a global upstream slot, by queue level |
| `provider_queue_timeouts_total` | counter | provider | Requests refused (429) after waiting too long for a slot |
| `provider_queue_wait_seconds` | histogram | provider | Time requests waited for a slot on a provider with a limit |
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
| `worker_jobs_total` | counter | `worker`, `provider_id` | Requests sent to the worker since start-up, across reconnects |
| `request_log_dropped_total` | counter | | Request log rows dropped since start-up because the log writer fell 10,000 rows behind the database |
| `worker_link_connected` | gauge | `hub` | 1 while this router is connected to its hub as a worker |
| `worker_link_jobs_total` | counter | | Jobs served for the hub since the link started |
| `routing_table_version` | gauge | | Rebuilds of the routing table |
| `routing_table_built_timestamp_seconds` | gauge | | When the table was last rebuilt |
| `routing_table_build_milliseconds` | gauge | | How long the last rebuild took |
| `session_affinity_sessions` | gauge | | Conversations currently bound to a provider |
| `session_affinity_overflows_total` | counter | | Sessions moved off an eligible provider because waiting in its queue cost more than the prompt cache it held |
| `event_loop_lag_seconds` | histogram | | How late the event loop woke a task sleeping a fixed half second: how long any request may wait behind others in this process. A p99 above 50 ms sustained is the signal to revisit running one process (see "One router process" in configuration.md) |
| `model_group_available` | gauge | `group` | 1 while at least one member of the model group can be served |
| `model_group_routes_total` | counter | provider, `group`, `model` | Requests for a model group since start-up, by the member model and provider chosen |
| `auto_benchmark_runs_total` | counter | `outcome` | Automatic benchmarks since start-up: completed, failed, interrupted |

`endpoint_class` is `generate`, `embed`, `image`, `audio` or `decide`.
Counters reset when the process restarts; use `rate()` and `increase()`
as usual.

Per-request cost and token counts are in the request log (dashboard **Log**
tab), not here.
