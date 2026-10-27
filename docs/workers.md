# Hub and worker links

A worker llama-router keeps one outbound WebSocket open to a hub. The hub
treats the worker as an ordinary provider. Its client uses an `httpx` transport
that relays each HTTP request over the socket, so routing, streaming, model
discovery, health checks and benchmarks need no special cases.

Like every provider, a worker speaks one protocol. The hub uses the worker's
`ollama` target whenever the worker offers it: a worker that relays its own
router serves every OpenAI-style route on its Ollama port too. Only a worker
that offers just the `llamacpp` target becomes an OpenAI-protocol provider. The
target names on the wire are unchanged, so hubs and workers of different
versions still connect.

```
client ──HTTP──▶ hub router ──WebSocket (outbound from worker)──▶ worker ──HTTP──▶ local Ollama / llama.cpp
```

## Setup

1. On the hub, **Workers** tab: create a worker. This creates a provider named
   after it with the address `worker://<name>` and shows a token once.
2. On the worker, **Workers** tab: enter the hub URL and token. Or set
   `LLAMA_ROUTER_WORKER_HUB_URL` and `LLAMA_ROUTER_WORKER_TOKEN`.
3. The worker connects, the hub discovers its models, and the provider goes
   idle. Rotating, disabling or deleting the worker on the hub disconnects it.

## Wire protocol (version 1)

Endpoint: `GET /worker/ws` on the hub's Ollama API port, upgraded to a
WebSocket. With `LLAMA_ROUTER_API_ON_DASHBOARD_PORT=true` it is also on the
dashboard port. The worker authenticates with `Authorization: Bearer <token>`. A
bad token is refused before the upgrade.

Control messages are JSON text frames with a `type` field. Body bytes travel
in binary frames laid out as one byte of kind, a four-byte big-endian request
id, then payload. Request ids let many exchanges share the socket.

| Direction | Frame | Fields |
|-----------|-------|--------|
| worker → hub | `hello` | `version`, `protocols` (`ollama`, `llamacpp`), `max_concurrency`, `software_version` |
| hub → worker | `welcome` | `version`, `worker_name`, `provider_id` |
| hub → worker | `request` | `id`, `target`, `method`, `path` (with query), `headers` |
| hub → worker | binary 1 / 2 | request body chunk / request body end |
| worker → hub | `response` | `id`, `status`, `headers` |
| worker → hub | binary 3 / 4 | response body chunk / response end |
| worker → hub | `error` | `id`, `kind` (`connect`, `timeout`, `upstream`, `internal`), `message` |
| hub → worker | `cancel` | `id` |

Every request ends with a binary request-end frame, even with no body. Every
exchange ends with a response-end frame or an `error`. The hub sends `cancel`
when its caller gives up, for example when an HTTP client disconnects mid
stream, and the worker aborts the upstream request.

Hop-by-hop headers, `Authorization`, `API-KEY` and cookies are stripped in both
directions. The worker adds its own upstream `API-KEY` when configured.

Close codes:

| Code | Meaning | Worker behaviour |
|------|---------|------------------|
| 4401 | Bad token | Retry after five minutes |
| 4403 | Token revoked, rotated, or worker disabled | Retry after five minutes |
| 4409 | Replaced by a newer connection with the same token | Retry after 30 seconds |
| 4400 | Protocol error | Retry after five minutes |

Anything else is a transient failure, retried with exponential backoff from one
second up to one minute, with jitter. Both sides send WebSocket pings every 20
seconds.

## Decision requests

Decision requests (`/v1/systemone`, `/v1/decisions` and their aliases without
`/v1`) reach a worker provider in one of two shapes, like any other provider:

- **Emulated.** One-token `POST /v1/chat/completions` calls with `logprobs` and
  `top_logprobs`. The worker relays bodies and responses byte for byte, so the
  logprobs come back as the upstream sent them. Every worker version serves
  this.
- **Passed through.** The request as it came, to the same route on the worker's
  upstream. Workers relay the four decision routes from this version on. A
  worker that relays its own router reaches them on that router's Ollama port,
  which serves them as well, so the worker's router answers the request with
  its own providers and calibrations.

Which shape the hub uses comes from the hub's probe of the worker, which runs
through the socket: a route is passed through only when it answers there. The
hub probes a worker each time it connects with a software version the hub has
not seen from it since the hub started, so an upgraded worker is picked up
without waiting for the probe to go stale.

A worker refuses a path outside its allowlist with 403 and an error starting
`worker does not allow`. Every worker version words it the same way, and the
hub's probe reads it as a missing route rather than one that needs a key. An
older worker therefore gets decisions emulated. If a probe stored before the
hub knew this still claims a decision route, the refused pass-through is
answered by emulation instead and the worker is probed again.

Errors map as for any provider: an upstream HTTP error comes back as its
status (`httpx.HTTPStatusError`), and a lost socket, refused upstream
connection or timeout as an `httpx.HTTPError`, which the decision endpoint
answers with 502.

## Limits

- The worker serves a fixed allowlist of inference, decision and listing
  endpoints.
  `POST /api/pull` and `DELETE /api/delete` need `LLAMA_ROUTER_WORKER_ALLOW_ADMIN=true`.
- Request bodies are capped at 64 MB on the worker. Frames are at most 4 MB.
- The hub respects the worker's announced `max_concurrency`. Extra requests wait
  for a free slot up to the client's pool timeout.
