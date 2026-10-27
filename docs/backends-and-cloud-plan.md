# Plan: any backend, local or cloud

## Where we are

A provider today has a `provider_type` of `ollama`, `llamacpp` or `both`,
and that one field does three jobs at once:

1. **Which wire protocol the router speaks to it.** `ollama` means the Ollama
   native API (`/api/*`). `llamacpp` means, in practice, the OpenAI-style
   `/v1/*` API: `LlamaCppClient` calls nothing but `/v1/chat/completions`,
   `/v1/models`, `/v1/embeddings` and friends, plus `/health` and `/props`
   for liveness and the loaded model. vLLM, LiteLLM, Lemonade, LM Studio,
   SGLang and every cloud service already work as "llamacpp" providers
   because of this; the name is just wrong.
2. **How many upstreams an address has.** `both` gives each
   `ProviderAddress` a second URL (`llamacpp_url`), and every client,
   discovery, benchmark and probe path has an `if supports_ollama` /
   `if supports_llamacpp` fork.
3. **Which backend it is.** The dashboard labels, the benchmark API picker
   and the cold-start logic all read the type as "it's Ollama" or "it's
   llama.cpp".

The probe (`probe.py`) already does the honest version of job 3: it sniffs
`ollama`, `llama.cpp`, `llama-router`, `litellm`, `vllm` and
`openai-compatible` from responses and records per-endpoint support. The
routing table already consults that record (`endpoint_support`). So most of
the machinery for a protocol-and-capabilities model exists; the data model
and the UI are what still think in backend names.

## Does the backend name matter?

For routing, no. What the router actually needs to know about an upstream:

| need                                    | source                                   |
|-----------------------------------------|------------------------------------------|
| which protocol to speak                 | `protocol` (ollama / openai)             |
| which endpoints work                    | probe, per endpoint                      |
| which models it serves                  | `/api/tags` or `/v1/models` (or a preset list) |
| which models are loaded now             | `/api/ps`, `/props`, `/slots`; or "all" for cloud |
| can it pull / delete models             | probe of `/api/pull`, `/api/delete`      |
| is it on the LAN or on the internet     | `locality` (local / cloud)               |
| does a request cost money               | `locality`, later a price table          |
| how to check liveness cheaply           | one GET chosen by the probe              |
| how to authenticate                     | key + header, from the preset or the form |

None of those is "is it vLLM". The name is a *user-agent string*: useful for
a badge in the dashboard, a link to the right docs, and a short quirks table
(LM Studio answers unknown routes with 200; llama.cpp router mode lists
models at `/models` and needs `/models/load`; Ollama's hot-model list is
`/api/ps`; cloud services are always hot and never cold-start). It should be
**detected, displayed and advisory**, never typed in by the user and never
read in the routing hot path.

So: keep a `flavor` string that the probe fills in, drop the idea that the
user declares what the backend is, and drive every behavioural decision from
`protocol`, `locality` and probed capabilities.

## Target data model

### providers

| column        | type   | notes                                                            |
|---------------|--------|------------------------------------------------------------------|
| `protocol`    | text   | `ollama` or `openai`. Replaces `provider_type`. One per provider. |
| `locality`    | text   | `local` or `cloud`. Set from the preset, else guessed from the host, always editable. |
| `flavor`      | text   | Detected by the probe (`ollama`, `llama.cpp`, `vllm`, `litellm`, `lemonade`, `lm-studio`, `sglang`, `openai`, `openrouter`, `anthropic`, ...). Null until probed. |
| `preset`      | text   | Cloud preset id this provider was created from, if any. Null for hand-entered. |
| `model_filter`| text   | Optional allow-list (glob or comma list) applied at discovery. Cloud catalogs are huge. |
| `api_key_enc`, `api_key_header` | as today |                                                  |
| `extra_headers` | json | Static headers some services need (e.g. `anthropic-version`, OpenRouter's `HTTP-Referer`). |
| `capabilities`, `probed_at` | as today | The probe result stays the source of truth for endpoint support. |

Dropped: `provider_type`. `supports_ollama` / `supports_llamacpp` become
`protocol == "ollama"` / `protocol == "openai"` for one release, then go.

### provider_addresses

Dropped: `llamacpp_url`. One URL per address. A provider never has two
upstreams.

### api_keys

| column     | type | notes                                                                   |
|------------|------|-------------------------------------------------------------------------|
| `locality` | text | `local`, `cloud` or null (don't care). Null is the default and today's behaviour. |

### benchmarks

`protocol` values `llamacpp` are rewritten to `openai`. Nothing else changes.

### Migration 009

Phase 1 implements only the `protocol` change, the `both` split and the
removal of `llamacpp_url`. The other new columns arrive with the phases that
use them.

- Add `protocol`, `locality`, `flavor`, `preset`, `model_filter`,
  `extra_headers` to `providers`; add `locality` to `api_keys`.
- `provider_type = 'ollama'` → `protocol = 'ollama'`;
  `'llamacpp'` → `'openai'`.
- `provider_type = 'both'` → **split into two providers.** The original row
  keeps its name and becomes `protocol = 'ollama'` with its addresses'
  `url`; a new row `"<name> (llama.cpp)"` gets `protocol = 'openai'`, the
  same machine/GPU metadata, the same encrypted key, and one address per
  original address using `llamacpp_url or url`. Models, benchmarks and
  pins that referenced the old provider are reassigned by the `source`
  recorded in `model.details` (`_in_ollama` / `_in_llamacpp`); benchmarks
  follow their `protocol` column. Request logs keep their provider id.
- `locality` for existing rows: `cloud` if any address host is a public
  name or IP, else `local`. (Loopback, RFC 1918, link-local, `.local`,
  `.lan`, `.home.arpa`, Tailscale 100.64/10 and `*.ts.net` count as local.)
- Drop `provider_addresses.llamacpp_url` and `providers.provider_type`.
- Rewrite `benchmarks.protocol`.

The SQLite→Postgres copy tool needs the new columns; the alembic script
must run under SQLite batch mode.

### Workers

Decided in phase 1: a worker stays **one** provider. The hub keys the
connection, its transports and the `workers` row by one provider id, and a
worker that relays its own router reaches every route through its Ollama
target, because the router's Ollama port also serves `/v1/*`. So a worker's
protocol is `ollama` whenever it offers that target, and `openai` only when
it offers just the `llamacpp` one. The wire target names are unchanged. The
one configuration this loses is a worker pointed at a separate real Ollama
and a separate real llama.cpp; give that machine two workers instead.

## Probe everywhere

The probe moves from "optional button" to the thing that configures a
provider.

- **Add Provider is probe-first.** The dialog asks for a URL and, optionally,
  a key. Probing fills in protocol, flavor, locality guess, model count and
  the auth-required flag, then shows them for confirmation. The protocol
  select stays, pre-filled, for the cases the probe cannot tell apart (a
  server that speaks both; pick the one you want, or add it twice).
- **Probe on save and on every address change**, in the background, as
  `probe_after_add` does now but always.
- **Re-probe periodically.** The health-check loop re-probes a provider
  whose `probed_at` is older than `probe_max_age_hours` (default 24) or
  whose last health check failed and then recovered (the server may have
  been upgraded). Probes are cheap and never run inference.
- **Pick the liveness endpoint from the probe.** Instead of hard-coding
  `/api/version` and `/health`, store `capabilities.liveness` as the first
  2xx GET among `/health`, `/api/version`, `/v1/models`, `/models`,
  `/health/liveliness`, and use that in `_probe_address_once`. Cloud
  services get `/v1/models` (or the preset's choice).
- **Pick the hot-model source from the probe.** `/api/ps` → Ollama style;
  `/props` or `/slots` → llama.cpp style; neither and `locality = cloud` →
  every model is always hot; neither and local → unknown (today's
  behaviour).
- **Gate features on probed support, not flavor.** Pull/push/delete buttons
  appear when `/api/pull` / `/api/blobs` / `/api/delete` probed as
  available; the `/models/load` router-mode path when it probed available;
  logprobs for decision models when a trial `/v1/chat/completions` with
  `logprobs` is accepted (that one is in the decision-models plan).
- **Extend detection** in `_detect`:
  - Lemonade: `/api/v1/health` and `/api/v1/models` (Lemonade also mounts
    `/v1/*`; add `/api/v1/*` to the catalog).
  - LM Studio: `/api/v0/models`; also its "unexpected endpoint" 200s.
  - SGLang: `/get_server_info`, `/health_generate`.
  - TGI: `/info`. KoboldCpp: `/api/v1/model`. LocalAI: `/readyz`.
  - OpenAI: `/v1/models` entries `owned_by: openai|system`.
  - OpenRouter: `/api/v1/models` entries with `pricing`.
  - Anthropic: `/v1/messages` answering 401 with an `anthropic` style error
    body, or 400 with `anthropic-version` complaint.
  - Hostname match for the known cloud services, which is cheaper and more
    reliable than response sniffing; do it first.
- **Probe through the key.** Cloud probes must send the key; the dialog
  already passes it. Treat `401/403` on everything as "key rejected" and say
  so in the dialog rather than listing every endpoint as needs-key.

## Cloud presets

A static table in `cloud_presets.py`, rendered in the Add Provider dialog as
a grid of buttons above the URL field. Choosing one fills URL, protocol,
header, extra headers, locality, flavor, docs link and a static model list
where the service has no listing endpoint. The user adds a name and a key
and clicks Probe (or just Save; save probes anyway).

| id            | base URL                                      | protocol | auth header / extras                       | models from          |
|---------------|-----------------------------------------------|----------|--------------------------------------------|----------------------|
| `openai`      | `https://api.openai.com/v1`                   | openai   | Bearer                                     | `/v1/models`         |
| `openrouter`  | `https://openrouter.ai/api/v1`                | openai   | Bearer; optional `HTTP-Referer`, `X-Title` | `/models` (filter!)  |
| `anthropic`   | `https://api.anthropic.com/v1`                | openai   | Bearer (OpenAI-compat layer); `anthropic-version` for `/v1/models` | `/v1/models` |
| `perplexity`  | `https://api.perplexity.ai`                   | openai   | Bearer                                     | preset list (no listing endpoint) |
| `groq`        | `https://api.groq.com/openai/v1`              | openai   | Bearer                                     | `/models`            |
| `together`    | `https://api.together.xyz/v1`                 | openai   | Bearer                                     | `/models`            |
| `fireworks`   | `https://api.fireworks.ai/inference/v1`       | openai   | Bearer                                     | `/models`            |
| `deepseek`    | `https://api.deepseek.com`                    | openai   | Bearer                                     | `/models`            |
| `mistral`     | `https://api.mistral.ai/v1`                   | openai   | Bearer                                     | `/models`            |
| `gemini`      | `https://generativelanguage.googleapis.com/v1beta/openai` | openai | Bearer                          | `/models`            |
| `xai`         | `https://api.x.ai/v1`                         | openai   | Bearer                                     | `/models`            |
| `cerebras`    | `https://api.cerebras.ai/v1`                  | openai   | Bearer                                     | `/models`            |
| `deepinfra`   | `https://api.deepinfra.com/v1/openai`         | openai   | Bearer                                     | `/models`            |
| `ollama-cloud`| `https://ollama.com`                          | ollama   | Bearer                                     | `/api/tags`          |

Fourteen; the four asked for plus ten. Ollama Cloud is worth including
because it is the one cloud service that speaks the Ollama protocol, so
Ollama-API clients get a cloud fallback without translation.

Left out on purpose: Azure OpenAI (deployment name and `api-version` in the
URL; a separate form later), AWS Bedrock and Vertex (SigV4 / OAuth, not a
bearer key; point people at LiteLLM, which the router already detects),
Cohere (own protocol; has an OpenAI-compat endpoint but little reason to
pick it over the others), Hugging Face Inference (routing semantics are its
own thing).

Preset entries need a `verified` date and a test that fetches each
`/models` with a key from the environment when one is present, so the URLs
do not rot silently.

Presets also get an environment-variable path for docker-compose users:
`LLAMA_ROUTER_PROVIDER_<PRESET>_API_KEY=...` creates or updates that preset's
provider at startup, so a compose file can bring up a router with OpenAI
and OpenRouter attached and no clicking.

## API keys: local, cloud or don't care

- `api_keys.locality` as above; the key dialog gets a three-way radio next
  to the routing mode. The list shows it as a badge.
- `RoutingPreferences.locality: str | None` carried from `auth.py`.
- `routing_table._serves` takes the preference and drops providers whose
  `locality` does not match. That covers primary routing, fallback,
  chaos and pins (a pin to the wrong locality is a routing error, reported
  in `explain`, not silently honoured).
- Session affinity: a bound provider is rechecked against the key's
  locality; a mismatch (the key was edited) drops the binding.
- `explain` gets a step "locality: key requires local; 2 cloud providers
  skipped".
- Request log rows record the serving provider's locality so the log and
  the dashboard can show, and filter on, "prompts that left the building".
- A `X-Llama-Router-Locality` request header may narrow but never widen the
  key's setting, for a client that wants to force local for one request.

## Cloud-aware operation

Things that are wrong for a cloud provider under today's rules:

- **Benchmarks cost money.** Auto-benchmark skips `locality = cloud`
  providers by default (a per-provider override exists for people who want
  numbers). Manual benchmarks warn once. `startup_time_ms` is always null
  for cloud; the routing table treats cloud models as always loaded.
- **429 is not offline.** A 429 or a `Retry-After` from any provider marks
  it busy for that many seconds (default 10) in `routing_stats`, without
  counting as a failure or taking the provider offline. Local servers send
  it too (llama.cpp with full slots).
- **Model catalogs are huge.** `model_filter` at discovery, default empty
  for presets that list under 50 models and a sensible starter list for
  OpenRouter. The routing table only grows by what passes the filter.
- **Static model lists.** When a provider cannot list models (Perplexity)
  the preset's list is used; the provider page lets the user add names by
  hand for any provider, stored as models with `details._manual`.
- **Health checks are a GET**, never inference, which is already true; the
  liveness endpoint chosen by the probe keeps it so.
- **Timeouts.** Connect timeout stays short; read timeout for cloud
  streaming goes to 120 s (configurable per provider).
- **Model name aliases.** The same model is `claude-sonnet-4-5` at
  Anthropic and `anthropic/claude-sonnet-4-5` at OpenRouter. The existing
  cache-prefix stripping is a one-off alias rule; generalise it into an
  `model_aliases` table (`alias → canonical`) so a request for one name can
  route to either provider and the fallback chain can say "same model,
  other provider". Phase 5; the plumbing (`raw_name` vs `name`) exists.
- **Cost.** Phase 5: a per-provider price table (preset defaults, editable)
  and `usage` captured from responses into the request log; a `cheapest`
  routing mode; a per-key daily spend cap that flips the key to local-only
  when hit. Worth doing after locality, because locality is the 80% case
  ("never send this key's prompts to a cloud service").

## Protocol translation

Today the Ollama API port only routes to `protocol = ollama` providers and
the `/v1` port routes to anything (Ollama serves `/v1` too). That means an
Ollama-API client cannot reach a cloud provider except Ollama Cloud.
Translating `/api/chat` ↔ `/v1/chat/completions` (messages, tool calls,
streaming chunks, `options` → sampling parameters, `format` → response
format) would open every provider to Ollama clients. It is a well-bounded
piece of work and a separate PR; the plan here only makes sure the data
model does not block it (protocol is per provider, the router already
knows the request's protocol).

Anthropic's native Messages API as a third **upstream** protocol is not
needed while their OpenAI-compat layer exists. Serving a Messages API
**downstream** (so Claude Code can point at the router) is a different
feature, tracked separately.

## Phases

1. **Data model** (migration 009, models.py, database.py, provider_manager,
   routing_table, router, worker split, templates, tests). Remove `both`
   and `llamacpp_url`; rename `llamacpp` → `openai` in code and UI. No
   behaviour change for existing single-protocol providers. Largest diff,
   least visible.
2. **Probe everywhere**: probe-first Add dialog, probe on save and on a
   schedule, liveness and hot-model source from the probe, extended
   detection, feature gating on probed support.
3. **Cloud presets** and the preset grid in Add Provider; `extra_headers`;
   static model lists; env-var provisioning.
4. **Locality**: provider and key columns, routing filter, explain, request
   log flag, header override.
5. **Cloud operation**: auto-benchmark policy, 429 busy handling,
   `model_filter`, manual models, timeouts.
6. **Later, separately**: aliases, cost and spend caps, Ollama↔OpenAI
   translation, Azure form.

Phases 1 and 2 can go in one PR if the diff stays reviewable; 3 and 4 are
independent of each other and of 5 once 1 has landed.

## Open questions

- Split or merge for migrated `both` providers: the split above keeps every
  routing feature working unchanged. The alternative, keeping one provider
  and dropping its Ollama half, loses pull/push for those hosts. Split is
  the recommendation.
- Name for the `/v1` listener and its `LLAMACPP_HOST` / `LLAMACPP_PORT`
  settings. Keep the env names for compatibility, relabel the UI "OpenAI
  API".
- Default `locality` for a hand-entered public hostname that is actually a
  home server behind a reverse proxy: the guess says cloud, the user wants
  local. The dialog shows the guess with a toggle; good enough.
