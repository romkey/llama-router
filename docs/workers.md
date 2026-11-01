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

## Choosing which providers a hub may use

By default a worker serves every provider its router has. The worker's
**Workers** tab, under *Serve a hub from this router*, lists the providers in
three groups, **Local**, **Cloud** and **Remotes** (workers that dial in to
this router), each with *all* and *none* quick picks, plus *All* and *None*
for the lot. Untick a provider to keep it from the hub: jobs from the hub
never route to it and the hub's model listing leaves its models out. Use it
to keep a metered cloud service, or a remote someone else lends you, from
being spent by the hub.

The worker adds `X-Llama-Router-Via: hub` to every job it relays. Its own
router reads the header and limits routing and `/api/tags` / `/v1/models` to
the shared providers (`Provider.hub_shared`); any other upstream ignores it.
The router believes the header only alongside `X-Llama-Router-Relay`, a
secret made fresh each time the router starts that only its own worker link
knows, and sent only to an upstream on this machine (`127.0.0.1`, `::1`,
`localhost`). The router strips all three headers from every request, so a
client cannot send them to steer its routing. A worker whose upstream is
another machine sends only `X-Llama-Router-Via`, which that upstream ignores.
The choice is saved per provider, so a new provider is shared until it is
unticked.

## Sharing providers one by one

By default the hub sees a worker as one provider, `<worker>`, with every
model the worker shares, and the worker's router picks which of its
providers serves each job. A worker can instead share each provider on its
own: on the worker's **Workers** tab, set *The hub sees* to *each provider
ticked below on its own*, or set `LLAMA_ROUTER_WORKER_EXPOSE=individual`.

The hub then shows one provider per shared provider, named
`<worker>/<provider>`, with the address `worker://<worker>/<id>`. Each has
its own models, status, health checks, benchmarks and routing scores, and
the hub's request log records which one served each request. A cloud
provider on the worker is marked cloud on the hub, so API keys limited to
local providers stay off it. The worker's own provider row stays, with no
models: it holds the socket and its concurrency limit, which all the
worker's providers share.

How it works:

- The worker says `expose: "individual"` in `hello`, then sends a `catalog`
  frame listing the providers it shares, and sends it again within 15
  seconds of any change (a provider ticked or unticked, added, removed or
  renamed). A remote that itself shares individually is left out of the
  list; its providers are listed instead.
- The hub creates a provider for each new entry and discovers its models
  through the socket. An entry that disappears goes offline on the hub with
  the reason *no longer shared by the worker* and keeps its history; it
  comes back if it is shared again. A renamed provider is renamed on the
  hub. Deleting the worker on the hub deletes them all.
- Each job for one of these providers carries `provider` (the provider's id
  on the worker) in its `request` frame. The worker passes it to its router
  as `X-Llama-Router-Provider` beside `X-Llama-Router-Via: hub`, and the
  router serves the job from that provider only, with no fallback to
  another. Model listings with the header list only that provider's models.
  The worker's upstream API key still applies: a model it pins to another
  provider is not served.
- Limits on the hub: a worker may share up to
  `LLAMA_ROUTER_WORKER_MAX_SHARED_PROVIDERS` (256) providers, counting those
  it stopped sharing that are not deleted yet; beyond that the longest
  unshared are deleted at once. Unshared providers are deleted after
  `LLAMA_ROUTER_WORKER_UNSHARED_PURGE_HOURS` (24). Catalogs are applied at
  most every `LLAMA_ROUTER_WORKER_CATALOG_MIN_INTERVAL_SECONDS` (5) per
  worker, the latest one winning.
- Switching back to one provider reconnects the link. The hub takes the
  individual providers offline and the worker's row gets its models back.

## Router IDs and loops

Every router makes a random ID (a UUID) on first start and keeps it in its
database. The **Workers** tab shows it. A router started from a copy of
another's data volume has the same ID, so give it a new one with *New ID*:
the link to its hub and every link from its workers reconnect so the other
side learns it.

Routers that link to each other can send a request round in a circle, for
example two routers that are each other's hub. Three things stop that:

1. **Handshake.** The worker sends its ID in `hello` and the hub sends its
   own in `welcome`. Either side closes the link with 4508 if the other has
   its own ID, and the worker waits five minutes before trying again.
2. **Remotes that lead back.** A router remembers the ID of each worker that
   dials in. When it serves a hub as a worker, it never offers that hub a
   worker whose ID is the hub's: routing and model listings for the hub
   leave it out, whatever its *share with hub* box says. Two routers that
   are each other's hub therefore see only each other's own providers.
3. **Path header.** Each request a router sends to a local provider (any
   provider that is not cloud, since it may be another llama-router)
   carries `X-Llama-Router-Path`: the IDs of the routers it has passed
   through, this one last. A worker relays the header unchanged. A router
   that receives a request whose path holds its own ID, or eight IDs or
   more, answers `508 Loop Detected` without routing it. Entries that are
   not router IDs are dropped, and a header over 512 characters is refused
   the same way. The caller sees
   the 508 as an upstream error from that provider. This catches loops
   through three or more routers and loops through plain HTTP providers that
   point at another router.

Older routers neither send IDs nor add to the path, so the checks only work
between routers of this version or later. Mixed versions still connect.

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
| worker → hub | `hello` | `version`, `protocols` (`ollama`, `llamacpp`), `max_concurrency`, `software_version`, `router_id`, `expose` (`combined`, `individual`) |
| worker → hub | `catalog` | `providers`: `[{id, name, locality}]`, sent only with `expose: individual` |
| hub → worker | `welcome` | `version`, `worker_name`, `provider_id`, `router_id` |
| hub → worker | `request` | `id`, `target`, `method`, `path` (with query), `headers`, and `provider` for a job for one provider shared individually |
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
| 4508 | Hub and worker are the same router (same router ID) | Retry after five minutes |

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
