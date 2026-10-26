# Hub and worker links

A worker llama-router keeps one outbound WebSocket open to a hub. The hub
treats the worker as an ordinary provider. Its Ollama and llama.cpp clients use
an `httpx` transport that relays each HTTP request over the socket, so routing,
streaming, model discovery, health checks and benchmarks need no special cases.

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
WebSocket. The worker authenticates with `Authorization: Bearer <token>`. A
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

## Limits

- The worker serves a fixed allowlist of inference and listing endpoints.
  `POST /api/pull` and `DELETE /api/delete` need `LLAMA_ROUTER_WORKER_ALLOW_ADMIN=true`.
- Request bodies are capped at 64 MB on the worker. Frames are at most 4 MB.
- The hub respects the worker's announced `max_concurrency`. Extra requests wait
  for a free slot up to the client's pool timeout.
