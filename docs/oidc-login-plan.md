# Plan: OpenID Connect login for the dashboard, with an admin group

Status: implemented. Written 2026-10-27 against 0.19.0.

Changes made while building it:

- `joserfc` instead of `authlib`. Authlib 1.9 deprecates its JOSE module in
  favor of joserfc, and the code exchange and PKCE are a few lines of `httpx`.
- When no accounts exist and an admin group is configured, a first sign-in
  by someone outside the group is refused (`oidc_first_admin`) instead of
  creating a non-admin account that would lock the dashboard.
- The password form is hidden while no accounts exist, since only SSO can
  sign anyone in then.
- `auth_source` is `VARCHAR(16)`, since MySQL rejects defaults on TEXT.

## Goal

Let people sign in to the dashboard through an OpenID Connect (OIDC) provider
(Authentik, Keycloak, Okta, Entra ID, Google, dex, ...) instead of, or in
addition to, a local username and password. Membership in a configured IdP
group makes the user a dashboard administrator; everyone else who is allowed
in is a read-only viewer. Local accounts keep working unchanged.

## What exists today (and stays)

- `dashboard_users` table: `id, username, password_hash, is_admin, created_at`.
- `dashboard/auth_core.py`: scrypt password hashing, `sign_session` /
  `unsign_session` (itsdangerous, 14-day `lr_session` cookie carrying only the
  user id), `normalize_username`, the non-admin forbidden path list.
- `dashboard/middleware.py`: no users at all → open "bootstrap" mode; otherwise
  cookie → `DashboardUser(id, username, is_admin)` on `request.state`; viewers
  get GET only and no sensitive paths. `/login*` and `/health` pass through.
- `dashboard/routes.py`: `GET/POST /login`, `POST /logout`, `/users` CRUD,
  `_safe_next_url`, `template_auth_ctx`.
- Session secret from `LLAMA_ROUTER_SESSION_SECRET` or `app_settings`.

OIDC only needs to produce a `dashboard_users` row and set the same cookie.
The middleware, RBAC and templates do not need to know how the user logged in.

## Design decisions

1. **Authorization Code flow with PKCE, confidential client.** Standard
   `openid` scope plus `profile email` and a groups scope. No implicit flow,
   no device flow.
2. **Library: `authlib`.** Use `authlib.integrations.httpx_client.AsyncOAuth2Client`
   for the code exchange and `authlib.jose` for ID token verification against
   the provider's JWKS. Do **not** use Authlib's Starlette integration: it
   wants Starlette's `SessionMiddleware`, and the project already has its own
   signed-cookie scheme. Flow state (state, nonce, PKCE verifier, `next` URL)
   goes in a separate short-lived signed cookie, `lr_oidc` (10 minutes,
   itsdangerous, salt `lr-oidc-v1`).
3. **Configuration by environment variables**, like the worker hub and
   Sentry. The dashboard shows whether OIDC is on but does not edit it. Moving
   it into `app_settings` with the encrypted-secret helper is a possible later
   step, not part of this.
4. **Identity is `(issuer, subject)`.** Never match an OIDC login to an
   existing account by email or username: an IdP-asserted email is not proof
   of ownership of a local account. First OIDC login creates a new row.
   Username is derived from `preferred_username`, else the email local part,
   else `sub`, run through `normalize_username`, with a numeric suffix on
   collision.
5. **The IdP is authoritative for admin.** On every OIDC login, `is_admin`
   is recomputed from the groups claim and written to the row. Demotion takes
   effect at the user's next login (sessions are 14 days); see "Later" for
   tightening that. Admin edits of `is_admin` on an OIDC user in the Users
   page are allowed but will be overwritten at the next login, and the UI says
   so.
6. **Bootstrap rule is preserved.** With zero users, the dashboard stays open
   as today. The first OIDC login creates the first user. If an admin group is
   configured, that user is admin only if they are in it; if no admin group is
   configured, the first user becomes admin (matching current local behavior).
7. **Password login stays on by default.** `LLAMA_ROUTER_OIDC_ONLY=true`
   hides the password form and makes `POST /login` return 404. Keep at least
   one local admin as break-glass; the README already documents the
   "delete all users" recovery and that still works.

## Configuration

All prefixed `LLAMA_ROUTER_`, added to `config.py`:

| Setting | Default | Meaning |
|---|---|---|
| `OIDC_ISSUER` | `""` | Issuer URL. Empty disables OIDC entirely. Discovery is fetched from `{issuer}/.well-known/openid-configuration`. |
| `OIDC_CLIENT_ID` | `""` | Required when issuer is set. |
| `OIDC_CLIENT_SECRET` | `""` | Required when issuer is set. |
| `OIDC_SCOPES` | `"openid profile email groups"` | Space-separated. Providers that do not know `groups` ignore or reject it; document per-provider values. |
| `OIDC_GROUPS_CLAIM` | `"groups"` | Claim name holding the group list. Supports dotted paths for nested claims (`realm_access.roles` for Keycloak). |
| `OIDC_ADMIN_GROUP` | `""` | Members become admins. Empty: nobody is promoted by group. |
| `OIDC_ALLOWED_GROUP` | `""` | If set, only members may log in at all. Empty: anyone the IdP authenticates. |
| `OIDC_REDIRECT_BASE_URL` | `""` | External base URL of the dashboard, e.g. `https://llm.example.org`. The callback is `{base}/login/oidc/callback`. Falls back to the request's scheme and host, honoring `X-Forwarded-Proto` / `X-Forwarded-Host`. |
| `OIDC_DISPLAY_NAME` | `"single sign-on"` | Button label: "Sign in with {name}". |
| `OIDC_ONLY` | `false` | Disable password login. |
| `OIDC_LOGOUT_AT_IDP` | `false` | On logout, also redirect to the provider's `end_session_endpoint`. |
| `OIDC_ALLOW_INSECURE` | `false` | Permit `http://` issuer for local dev (dex in compose). |

Validation at startup in `__main__.py`: if issuer is set, client id and secret
must be present and issuer must be `https://` unless insecure is allowed; log
the resolved callback URL. Discovery itself is lazy (first login attempt), so a
down IdP does not block router startup.

Behind a proxy, `dashboard_cookie_secure` should be true; the README section
will say so.

## Data model

Alembic migration `009_dashboard_users_oidc`:

- `password_hash` → nullable (use `batch_alter_table` for SQLite).
- Add `auth_source TEXT NOT NULL DEFAULT 'local'` (`local` | `oidc`).
- Add `oidc_issuer TEXT NULL`, `oidc_subject TEXT NULL`,
  unique index `uq_dashboard_users_oidc (oidc_issuer, oidc_subject)`.
- Add `email TEXT NULL`, `display_name TEXT NULL`, `last_login_at TIMESTAMP NULL`.

`database.py` additions:

- `get_dashboard_user_by_oidc(issuer, subject)`
- `create_oidc_dashboard_user(username, issuer, subject, email, display_name, is_admin)`
- `update_oidc_dashboard_user(user_id, email, display_name, is_admin)` (also stamps `last_login_at`)
- `list_dashboard_users` / `get_dashboard_user_by_id` return the new columns.
- `verify_password` must treat `None` or `""` as "no password": today
  `None.split` would raise. `login_submit` must refuse password login for
  rows with `auth_source != 'local'`.

`DashboardUser` dataclass gains `auth_source: str` so templates can branch.

## New module: `dashboard/oidc.py`

Pure-ish, no FastAPI imports beyond `Request` for URL building, so it tests
without the app:

- `OIDCConfig` built from `settings`, with `enabled` property.
- `ProviderMetadata` cache: discovery document + JWKS, fetched with `httpx`,
  cached in memory for one hour; JWKS refetched once on an unknown `kid`.
- `begin_login(next_url) -> (authorize_url, state_cookie_value)`: generates
  `state`, `nonce`, PKCE verifier/challenge (S256), signs them with the
  dashboard session secret.
- `complete_login(query, state_cookie_value) -> Claims`: checks state, exchanges
  the code, verifies the ID token (signature via JWKS, allowed algs RS256 /
  ES256 / PS256, `iss` equals configured issuer, `aud` contains client id,
  `exp`/`iat` with 60 s skew, `nonce` matches). If the groups claim is absent
  from the ID token, calls `userinfo_endpoint` with the access token and
  merges (Okta, Authentik and Entra often put groups only in userinfo).
- `resolve_roles(claims) -> (allowed: bool, is_admin: bool)` from
  `OIDC_ALLOWED_GROUP` / `OIDC_ADMIN_GROUP`; groups compared as exact strings.
- `derive_username(claims) -> str` per decision 4.

Errors are a small `OIDCError(code)` hierarchy mapped to
`/login?error=oidc_<code>`: `state`, `denied` (user cancelled at IdP),
`not_member`, `token`, `provider` (discovery or network failure).

## Routes (`dashboard/routes.py`)

- `GET /login`: render the SSO button when OIDC is enabled; hide the password
  form when `OIDC_ONLY`. New error strings in `login.html`.
- `GET /login/oidc?next=...`: `begin_login`, set `lr_oidc` cookie
  (httponly, samesite=lax, secure per settings, 600 s), 302 to the IdP.
- `GET /login/oidc/callback`: read and clear `lr_oidc`, `complete_login`,
  `resolve_roles`, upsert the user, `invalidate_dashboard_user_count_cache()`,
  set `lr_session` exactly as `login_submit` does, 303 to the `next` carried
  in the state cookie (through `_safe_next_url`). Any `OIDCError` → redirect
  to `/login?error=...`, nothing logged above INFO except `provider` errors.
- `POST /logout`: unchanged unless `OIDC_LOGOUT_AT_IDP`, in which case
  redirect to `end_session_endpoint?post_logout_redirect_uri=...&client_id=...`.
- `POST /login` returns 404 when `OIDC_ONLY`.

The middleware's existing `/login` prefix pass-through already covers the new
paths. SameSite=lax is fine: the callback is a top-level GET navigation from
the IdP, so the `lr_oidc` cookie is sent.

## Users page

- New "Source" column: Local / SSO, plus email and last login.
- Password reset form hidden for SSO users.
- Admin checkbox on SSO users shows a note "managed by IdP group, reset at next
  sign-in"; the current `count_dashboard_admins` last-admin guard still applies
  to deletes.
- Add-user form unchanged (local accounts only).
- `template_auth_ctx` passes `oidc_enabled` so `base.html` can show the
  source badge next to the username if wanted.

## Testing

Unit (no app):
- `resolve_roles` with admin group, allowed group, both, neither, missing
  claim, nested claim path, non-list claim value.
- `derive_username` normalization and collision suffixing.
- State cookie round trip, tampering, expiry.

Integration (reuse `dash_client` fixture, `pytest-httpx` for the IdP):
- Mock discovery, token and JWKS endpoints; sign ID tokens with a test RSA key
  from `authlib.jose`.
- Full happy path: `GET /login/oidc` sets cookie and redirects; callback with
  correct state creates a user, sets `lr_session`, lands on `next`.
- Admin group member becomes admin; non-member is viewer; demotion on second
  login; `OIDC_ALLOWED_GROUP` refusal.
- Wrong state, wrong nonce, bad signature, wrong `iss`, wrong `aud`, expired
  token each produce the right `error=` and no session.
- Groups only in userinfo.
- Bootstrap: zero users, first OIDC login, with and without admin group.
- `OIDC_ONLY` hides the form and 404s `POST /login`.
- Password login refused for an SSO row.
- Migration test in `test_db_migrate.py` for `009`.

Manual: add a `dex` service to `docker-compose.dev.yml` with a static user
and a `groups` claim so the flow can be clicked through locally.

## Documentation

README section "Single sign-on (OIDC)": the settings table above, the
callback URL to register, group claim notes per provider (Authentik: add
`groups` scope; Keycloak: add a Group Membership mapper, or use
`realm_access.roles`; Okta: groups claim filter; Entra: group IDs not names,
and the 200-group overage case means "use userinfo" does not help, so
recommend app roles via `roles` claim; Google: no groups, so admin group
cannot be used and the first-user rule applies), and the break-glass note.

## Work breakdown

1. Config, startup validation, `dashboard/oidc.py` with discovery, JWKS,
   token verification, role resolution. Unit tests. (`authlib` added to
   dependencies.)
2. Migration 009, `Database` methods, `DashboardUser.auth_source`,
   `verify_password` guard, password-login refusal for SSO rows.
3. Login page and the three OIDC routes, logout option, error strings.
4. Users page changes.
5. README, compose dev IdP, release notes.

Each step is independently mergeable and the feature is inert until
`OIDC_ISSUER` is set. One PR per step or one PR for 1–3 and another for 4–5.

## Later, not in this plan

- Re-check group membership mid-session (shorter OIDC sessions, or store the
  refresh token and re-validate every N hours).
- Admin-driven linking of an existing local account to an OIDC identity.
- OIDC settings editable in the dashboard, secret stored with
  `provider_secrets`.
- Optional `OIDC_ADMIN_EMAILS` for providers without groups.
- Multiple providers.

## Open questions for you

- Confirm `authlib` is acceptable as a new dependency (alternative: `httpx`
  plus `joserfc` and hand-rolled discovery, roughly 150 more lines).
- Env-only configuration for the first version, or dashboard-editable from
  the start?
- Should `OIDC_ONLY` exist in the first version, or defer it?
