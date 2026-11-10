# Plan: editing API keys, and limiting them to models and groups

Status: reviewed. Two PRs: (1) edit an existing key and regenerate its
secret, (2) choose which models and groups a key may use, when creating
or editing it.

## Where things stand

- **Admins** (API Keys tab, `/api/keys/*`) can create a key with label,
  routing mode, fallback, session affinity, locality and daily spend cap.
  Afterwards only locality and spend cap can change (one endpoint each),
  plus pins and the decision policy in their own modals. Label, routing
  mode, fallback and affinity are fixed until the key is deleted.
- **Users** (My Keys, `/api/my-keys`) can issue, list and revoke their own
  keys. Locality and spend cap come from the admin's self-service defaults;
  nothing can be changed after issuing.
- Resolved key settings are cached in `auth._AuthCache`, cleared on
  `CHANGE_API_KEYS`, so any repository write decorated with
  `@_notifies(CHANGE_API_KEYS)` takes effect on the next request.
- There is no notion of which models a key may use: every key reaches
  every model, alias and group the router serves.

## PR 1: edit an API key

Editing never changes the secret. Everything else that was chosen at
creation can be changed. Regenerating the secret is a separate action
(below), not part of the edit form.

### Repository

`ApiKeyRepository.update_api_key(key_id, *, owner_user_id=None, **fields)`:
one `UPDATE api_keys SET ... WHERE id = ?` over the fields given (label,
routing_mode, allow_fallback, session_affinity, locality,
daily_spend_cap_usd), with `AND owner_user_id = ?` when an owner is
given so a user can only touch their own keys. Returns whether a row
matched. `@_notifies(CHANGE_API_KEYS)`.

`get_api_key(key_id)` returning one row in the `list_api_keys` shape, for
the edit form and the 404 check (replacing `_api_key_exists`, which lists
every key to find one).

`set_api_key_spend_cap` and `set_api_key_locality` become thin wrappers
over `update_api_key`, or are removed once nothing calls them.

### Validation

One shared parser, `form_values.key_settings(body, *, allowed)`, that
reads only the fields present in the body (PATCH semantics: absent means
unchanged), validates each with the existing helpers (`_key_label`,
`_tristate`, `_key_locality`, `_usd`, routing mode in `ROUTING_MODES`),
and rejects fields not in *allowed* with 400. The create endpoints use it
too, so create and edit can't drift apart.

### Endpoints

- `PATCH /api/keys/{key_id}` (admin, management keys with the keys
  privilege): any of label, routing_mode, allow_fallback,
  session_affinity, locality, daily_spend_cap_usd. Works on self-issued
  keys too, as the module docstring already promises. Returns the updated
  key.
- `PATCH /api/my-keys/{key_id}` (signed-in owner): label, routing_mode,
  allow_fallback, session_affinity only. Locality and spend cap stay under
  admin control; sending them is a 400 so the client learns rather than
  being silently ignored. 404 for someone else's key.
- `GET /api/keys/{key_id}` and `GET /api/my-keys/{key_id}` for the form.
- `POST /api/keys/{key_id}/regenerate` (admin) and
  `POST /api/my-keys/{key_id}/regenerate` (owner): issue a new secret for
  the same key. New prefix and hash, same id, so settings, pins, decision
  policy, allowed models and usage history stay. The old secret stops
  working at once. The response carries the new key, shown once like a
  new key. Repository: `regenerate_api_key(key_id, key_prefix, key_hash,
  *, owner_user_id=None) -> bool`, notifying `CHANGE_API_KEYS` so the auth
  cache drops the old secret. The PBKDF2 hash runs off the event loop as
  on create.
- Keep `POST /api/keys/{id}/spend-cap` and `/locality` for existing
  management-API callers; implement them through `update_api_key`. Mark
  them deprecated in docs/api (removal is a later release).

### Dashboard

- API Keys tab: an **Edit** button per row opening a modal with the same
  fields as the create form, prefilled, locality and spend cap included.
  The inline locality and cap controls in the table stay editable too.
- My Keys: an **Edit** button per row with label, routing priority,
  fallback and affinity; locality and cap shown read-only with "set by an
  administrator".
- Both pages: a **Regenerate secret** button per row, with a confirmation
  ("the current key stops working immediately"), then the same show-once
  modal used for a new key.
- Template JS reuses the create form's markup via a small helper so create
  and edit render identical fields.

### Tests

- Repository: partial update touches only given columns; owner filter;
  missing key returns False; change notification fires (auth cache
  cleared, so a request right after the edit sees the new routing mode).
- Admin PATCH: each field, invalid values (400), unknown key (404).
- My-keys PATCH: own key ok; another user's key 404; locality/cap 400;
  management key 403 (no owner).
- Deprecated endpoints still work.
- Regenerate: the old secret is refused and the new one accepted right
  away (auth cache cleared); id, settings and pins unchanged; a user can't
  regenerate another user's key (404).

### Docs

docs/api (or wherever the dashboard API is listed), README section on
keys, CHANGELOG.

## PR 2: limit a key to chosen models and groups

### Meaning

A key is either **unrestricted** (today's behaviour, the default, and
what every existing key stays) or **restricted** to a list of entries,
each one of

- a **model name**: that model on any provider, under any alias of it;
- a **model group**: the group's name. Allowing a group does not allow
  its members by their own names, only through the group.

A restricted key with an empty list can use nothing (the restriction is an
explicit flag, so deleting the last allowed group never silently widens a
key to everything).

Rules on the request path, for a restricted key:

1. The requested name must be allowed (directly, through an alias of an
   allowed model, or as an allowed group); otherwise **403** "this API key
   may not use model X". Not 404: the client is authenticated and should
   know why.
2. Fallback: candidates in the fallback chain that aren't allowed are
   skipped, so a fallback never takes a key somewhere it wasn't allowed.
3. Pins only apply to allowed names (they already sit under rule 1).
4. Decision models (`decision_policy`) choose a model and then route it;
   the chosen model goes through the same check, and a disallowed choice
   falls through to the next answer or the request's own model as an
   invalid choice does today.
5. Listings (`/api/tags`, `/v1/models`, `/v1/models/{name}`, `/api/show`)
   show only what the key may use. A disallowed name looks absent (404) in
   `show`/`models/{name}`, matching the listing.

Locality, spend cap and the hub/worker rules still apply on top; the
allow-list only narrows.

### Storage (Alembic migration 033)

- `api_keys.restrict_models` boolean, default false.
- New table `api_key_allowed_models`:
  `api_key_id` (FK api_keys, ON DELETE CASCADE), `model_name` nullable,
  `group_id` (FK model_groups, ON DELETE CASCADE) nullable, a check that
  exactly one is set, unique on (api_key_id, model_name) and
  (api_key_id, group_id).

Groups are stored by id so renaming a group keeps the key's access;
models by name because a model has no identity across providers. A
deleted group drops out of the key's list (the key loses that access, it
never gains any).

Repository: `get_api_key_access(key_id) -> {restricted, models, groups}`
and `set_api_key_access(key_id, restricted, models, group_ids)` (replace
the whole list in one transaction), notifying `CHANGE_API_KEYS`.
`list_api_keys` gains `restrict_models` and an allowed count for the table
view. Group deletion must also notify `CHANGE_API_KEYS` (or the auth cache
must also clear on the model-groups change reason) so cached keys don't
keep a deleted group.

### Request path, with no extra latency

Per the routing convention, nothing on the request path reads the
database beyond the key lookup that already happens.

- `auth._resolve_key` loads the access list alongside the pins (already a
  per-key query on the slow, cached path) and puts it on
  `RoutingPreferences` as `allowed_models: frozenset[str] | None` and
  `allowed_groups: frozenset[int] | None` (None: unrestricted). Group ids
  are turned into names against the current routing table at check time,
  since the table already holds groups by name; storing ids in prefs keeps
  a rename from needing a cache flush.
- `RoutingTable` gets `allows(prefs, name) -> bool`: true if unrestricted;
  true if `name` is an allowed group's name; true if any name in `name`'s
  alias group (`alias_groups`, already computed for the table) is an
  allowed model. Pure dict/set lookups.
- `proxy.py` (and the OpenAI decisions endpoint) call the check before
  routing and raise 403. `Router.route_with` skips disallowed candidates
  in the chain walk, so fallbacks and decision choices obey it everywhere,
  including `explain`.
- `model_listing.listed_models` takes the prefs and filters rows with the
  same `allows`.

### Endpoints

- Admin create (`POST /api/keys/generate`) and PATCH (from PR 1) accept
  `allowed: null | {models: [names], groups: [ids]}`. `null` means
  unrestricted. Group ids must exist (400 otherwise); model names are
  validated like pin names (printable, at most 255), but needn't be served
  right now, since a provider may be offline.
- `GET /api/keys/{id}` returns it.
- My Keys: users may also restrict their own keys, on create and edit.
  This only ever narrows what the key can do, so it needs no admin
  default. There is no admin-set ceiling for self-issued keys for now.
- A picker endpoint: `GET /api/keys/allowable` returning served model names
  and groups (id, name), so both pages fill the picker from one place. It
  must be reachable by signed-in users, not only admins, since My Keys
  uses it.

### Dashboard

On the create form and in the edit modal, on both pages:

- "Models this key can use": **All models** (default) / **Only these**.
- With "Only these": a searchable multi-select with two sections, Groups
  and Models, filled from the picker endpoint, plus a free-text add for a
  model name not currently served.
- A note under the picker: "Allowing a group lets this key use the group's
  name. It does not allow the group's member models by their own names;
  add those separately if the key should use them directly."
- The keys table shows "all models" or "3 models, 1 group" with a
  tooltip listing them.

### Tests

- Migration up/down on SQLite; the CI database matrix covers PostgreSQL
  and MySQL.
- Repository: set/replace/clear; cascade on key delete and group delete;
  check constraint.
- Routing: allowed model, allowed via alias, allowed group, member of an
  allowed group requested directly (403), fallback to a disallowed model
  skipped, pin to a disallowed model unreachable, decision choice of a
  disallowed model ignored, empty restricted list blocks everything,
  unrestricted key unchanged.
- Listings filtered for a restricted key; unchanged for none or an
  unrestricted key.
- Endpoints: create and edit with `allowed`, bad group id, user restricting
  their own key, user unable to edit another's.
- Auth cache: editing the list or deleting a group takes effect on the
  next request.

### Docs

README/docs on keys, docs/configuration.md unchanged (no new settings),
CHANGELOG.

## Decisions from review

1. **Key rotation:** editing never changes the secret; a separate
   Regenerate secret action does (PR 1).
2. **Admin ceiling for self-issued keys:** not now.
3. **Locality and spend cap** are editable in the Edit modal, and the
   inline controls on the API Keys tab stay.
4. **A disallowed model returns 403.**
5. **Allowing a group does not allow its members by name.** The picker
   says so in a note.
