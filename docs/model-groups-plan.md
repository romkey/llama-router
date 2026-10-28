# Plan: model groups

Status: implemented in three PRs (storage and routing; listings, request
log and metrics; dashboard, API and docs). Deviations from the plan:
the request log gained a `routed_model` column, group endpoints take the
group's id rather than its name, and the registry cache needed nothing
because it never consults routing.

## What a model group is

A named set of models that clients can ask for as if it were one model.
`vision` might hold `llava:13b`, `qwen2.5-vl` and `gpt-4o`; `coding` might
hold `qwen2.5-coder:32b` on one provider and `deepseek-coder` anywhere. A
request for `model: "coding"` is routed to the best member that is online
right now, scored the same way a single model's providers are scored today.

Each member is either

- a **model name**: every online provider that serves it (through aliases
  too), or
- a **provider + model**: that model on that one provider only.

Groups differ from the two name-mapping features we already have:

| feature  | meaning                                   | request path cost |
|----------|-------------------------------------------|-------------------|
| alias    | the same model under another name         | none (table key)  |
| fallback | try this *other* model when mine is down  | none (chain walk) |
| group    | any of these models is acceptable; pick   | none (table key)  |

A group is a routing-time choice among different models, so it is a new
concept, not a special case of either.

## Design constraints

- **No extra latency.** A group name must be exactly as cheap to route as a
  model name: one dictionary lookup into the precomputed table, then the
  existing queue-depth pick. Nothing on the request path reads the
  database, walks members or resolves names.
- **Session affinity still holds, and gets stronger.** A conversation that
  asked for `coding` must stay on the same provider *and the same member
  model*. Today a binding is only a provider id; with groups a provider can
  serve several members, and moving a conversation between models would
  lose the prompt cache and change the model mid-conversation.
- **Everything else keeps working by name.** Fallback chains, pins,
  decision-policy target models, locality and hub scoping, pricing and the
  spend cap all key on the requested name. Groups must slot in as names.

## Design

### Storage

New Alembic migration `019_model_groups.py`:

```
model_groups
  id            integer pk
  name          varchar(255) unique
  description   text null
  created_at    timestamp default now

model_group_members
  id            integer pk
  group_id      fk model_groups.id on delete cascade
  model_name    varchar(255) not null
  provider_id   fk providers.id on delete cascade, null = any provider
  position      integer not null         -- display and (later) preference order
  unique (group_id, model_name, provider_id)
```

Pydantic `ModelGroup` and `ModelGroupMember` in `models.py`.

`database.py`: `CHANGE_GROUPS = "groups"` added to `ROUTING_CHANGES`, and
`@_notifies(CHANGE_GROUPS)` on the writers so the router rebuilds, as it
does for aliases. Methods: `list_model_groups()` (one pass, members
included), `create_model_group`, `update_model_group` (description),
`delete_model_group`, `add_group_member`, `remove_group_member`. Reads use
the SQLAlchemy session like `list_all_models`; writes use `_execute` with
the SQLite/PostgreSQL/MySQL upsert variants the alias code already shows.

Validation (in the dashboard API, mirroring aliases):

- name required, at most 255 characters, not equal to an existing alias,
  canonical alias target or another group; refusing a name that a provider
  currently serves as a model too (it would shadow it: see below);
- a member's `model_name` may not be a group (no nesting in this phase);
- `provider_id`, when given, must exist.

### Routing table

`RoutingInputs.groups: dict[str, tuple[ModelGroupMember, ...]]`, loaded in
`load_inputs`.

In `build_table`, after the per-model `grouped` buckets are filled and
before ranking, add a bucket per `(group_name, path)`:

1. For each member, take `grouped.get((member.model_name, path), ())`.
   Member names go through `alias_groups` the same way model names do, so
   a member written as `anthropic/claude-x` reaches the `claude-x`
   providers as well.
2. Keep only candidates whose `provider_id` matches when the member names
   a provider.
3. Deduplicate on `(provider_id, candidate.model)`, not on provider alone
   as the single-model buckets do: one provider may legitimately offer two
   members.
4. Put the merged list into `grouped[(group_name, path)]`.

Ranking, tiers, `ranked_by_locality` and `ranked_for_hub` then run
unchanged over the new buckets. Candidates keep their own `model`,
`backend_model`, price and scores, so `cheapest` compares the members'
real prices and `latency` compares their real cold starts. The best
member on the best provider wins; a hot small model beats a cold large one
in latency mode, which is the point of a group.

`RoutingTable` gains `groups: dict[str, tuple[str, ...]]` (name to member
display strings) for the dashboard and `explain`, and
`backend_names` needs nothing: route results already carry the candidate's
backend name.

Name collision: if a group name equals a model some provider serves, the
group's bucket replaces the model's and the build records a reason
(`"group X shadows a model of the same name"`) shown on the dashboard. The
API refuses to create such a group, so this only happens when a provider
grows the model later.

Build cost grows by the number of members times endpoints times modes,
all in memory; well under a millisecond for realistic configurations.
`build_ms` on the dashboard tells us if that assumption is wrong.

### Request path

`Router.route_with` needs no structural change: a group name is a chain
entry like any other, and `table.ranked_for(group, path, mode, ...)`
returns member candidates. The returned `RouteResult` is
`(provider, resolved_model=group_name, backend_model, model_key=member)`,
the same shape aliases produce, so the handlers' existing code sets
`body["model"]` to the backend name and keys stats by the member model.

Add `RouteResult.group: str | None` so logs, the request log and `explain`
can say "coding → qwen2.5-coder:32b on gpu-box" without re-deriving it.

Fallbacks, pins and decision policies work by name:

- a model may fall back to a group and a group may fall back to a model
  (`fallback_chains` is name-based);
- a pin on a group name pins the provider; among that provider's members
  the best-scored one is used (`_pinned` picks the first matching candidate
  in bucket order today; change it to prefer the lowest score for the mode);
- a decision policy's level-to-model map may name a group.

### Session affinity

`SessionAffinity` entries become `(provider_id, model, seen)` where `model`
is the member model the request actually went to (`candidate.model`). The
key itself is unchanged: `f"{key_id}:{requested_model}:{digest}"` already
scopes by the requested name, so a `coding` conversation and a direct
`qwen2.5-coder:32b` conversation are different sessions.

- `get`/`peek` return a small `Binding(provider_id, model)`; `bind` takes
  both.
- In `route_with`, a sticky binding matches a candidate when
  `c.provider_id == binding.provider_id and c.model == binding.model`.
  The candidate must still be in the eligible ranked set (best tier, not
  rate limited), exactly as today; otherwise the normal pick is made and
  the session is rebound, counted as a rebind.
- For single-model requests the model part is constant, so behaviour is
  unchanged.
- `stats()`, `forget_providers_not_in`, the TTL sweep and the LRU cap are
  untouched. `_explain_session` reports the bound model as well.

One extra string comparison per sticky lookup; no new allocations on the
hit path.

### Model listing

Clients discover models through `GET /api/tags` and `GET /v1/models`.
Groups must appear there or nobody will ask for them. `list_all_models`
gets `include_groups: bool = True` and appends one row per group with
`size: None`, `digest: None` and `details: {"group": True, "members":
[...]}`; `/v1/models/{name}` finds them the same way. A hub sees a group
only when at least one member is on a provider shared with it.

### Request log and metrics

`request_logger` already takes `model` (requested) and `routed_model`;
the handlers pass the group as the former and the member as the latter,
which the request-log table shows today as "model → routed". No schema
change. Prometheus counters stay keyed by the routed model; add a
`llama_router_group_routes_total{group, model, provider}` counter so group
behaviour is visible without reading logs.

### Dashboard and API

A "Model Groups" card next to "Model Aliases" in `dashboard.html`:

- create a group (name, description);
- per group: members as chips (`model` or `model @ provider`), an add-member
  form with a model name input and an optional provider dropdown, remove
  member, delete group;
- per group: first choice per routing mode from `table.first_choices`,
  and a count of members that currently have no candidate (grey chip).

REST, in `dashboard/routes.py`:

```
GET    /api/model-groups                       groups with members and first choices
POST   /api/model-groups                       {name, description}
PATCH  /api/model-groups/{name}                {description}
DELETE /api/model-groups/{name}
POST   /api/model-groups/{name}/members        {model_name, provider_id?}
DELETE /api/model-groups/{name}/members/{id}
```

`explain` adds `"group": {"name", "members": [...]}` to a step whose model
is a group, and the candidate rows already carry `backend_model`, so the
explain page shows which member each row is.

### Docs

- `docs/overview.md`: one line in the files list if a module is added
  (`model_groups.py` only if the table code outgrows `routing_table.py`;
  it should not).
- README: a "Model groups" section with the `vision`/`coding` example and
  the alias/fallback/group comparison table above.
- `docs/configuration.md`: nothing; there are no new settings.

## Phases

Each phase is a mergeable PR with tests; the first three are the feature,
the rest make it visible.

1. **Schema and storage.** Migration, models, `Database` methods, change
   notification. Tests in `test_database.py` and `test_db_migrate.py`
   (SQLite and PostgreSQL via the existing fixtures).
2. **Routing table.** Group buckets in `build_table`, `RouteResult.group`,
   pin-on-group preference, shadowing reason. New `tests/test_model_groups.py`:
   cross-model ranking per mode, provider-scoped member, alias member,
   locality and hub scoping, empty group routes nowhere, fallback into and
   out of a group, pin on a group, cheapest picks the cheaper member.
3. **Session affinity.** `(provider, model)` bindings; tests in
   `test_session_affinity.py`: a `coding` session stays on its member
   while a faster member appears; moves when the member's provider goes
   offline; a direct-model session is unaffected; explain reports the
   bound model.
4. **Listing, logging, metrics.** `/api/tags`, `/v1/models`, request log
   fields, the new counter. Tests in `test_llm_routes.py` and
   `test_metrics.py`.
5. **Dashboard and API.** Card, endpoints, explain output. Tests in
   `test_dashboard_routes.py` and `test_routing_explain.py`.
6. **Docs.**

New modules and test files must pass `mypy --strict` (do not add to the
override list in `pyproject.toml`); `test_routing_table.py` and
`test_session_affinity.py` are on the grandfathered list and may stay
there, but new tests for this feature go in new, strict files.

## Later, if wanted

- **Ordered groups.** A `strategy` column: `score` (this plan) or
  `ordered`, where the lowest `position` that has a healthy candidate is
  used and score only breaks ties within it. "Prefer the local coder, use
  the cloud one when it is down" is a common wish and cheap to add once
  buckets exist: it is one extra sort key at build time.
- **Group default routing mode.** A group that is always routed
  `cheapest` whatever the key's mode. Key mode would still win when set
  explicitly.
- **Nested groups.** Members that are groups, flattened at build time with
  the same loop guard `alias_groups` uses. Not needed for the use cases
  named so far.
- **Capability-derived groups.** Auto-populate `vision` from the
  capability probes (`005_provider_capabilities`) instead of by hand.

## Open questions

- Should a group be visible to a hub when only some members are shared?
  This plan says yes (the hub routes only to the shared ones). The
  alternative is to hide groups from hubs entirely.
- `ollama pull vision` through the registry cache makes no sense; the
  cache should 404 for a group name rather than try upstream. Confirm
  when implementing phase 4.
