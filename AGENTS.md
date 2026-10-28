# Instructions for AI assistants

Follow https://github.com/romkey/python-ai-rules/blob/main/RULES.md.
Follows python-ai-rules v0.0.0. When re-scanning, apply the entries in
its UPGRADING.md newer than that, then update this line.
With Claude Code, the `python-rules` plugin loads them for you.

Read docs/overview.md first. `main` is protected: work on a branch and
open a pull request; CI and Lint must pass before it can merge.

<!-- Project-specific notes go below. -->

## Deliberate differences from the rules

- **Databases:** SQLite is the default (one router, a file on the data
  volume); PostgreSQL and MySQL are optional extras via
  `LLAMA_ROUTER_DATABASE_URL`. The owner agreed to keep this.
- **Compose service name** is `llama-router`, not `app`, so existing
  `docker compose exec llama-router ...` commands keep working.
- **Production image** keeps network debugging tools (ping, dig, curl,
  nc, traceroute...) for `docker compose exec`; see the README.
- **Environment variables** use the `LLAMA_ROUTER_` prefix (pydantic-settings
  in `config.py`).

## Not yet up to the rules

These were grandfathered when the rules were first applied. Shrink them;
don't grow them.

- `[tool.mypy.overrides]` in `pyproject.toml` lists modules that don't pass
  `mypy --strict` yet (42 modules, down from 86: the clients, `database.py`,
  `dashboard/routes.py`, `migrate_sqlite_to_postgres.py`, `scripts/` and most
  test files). Remove a module once it does.

## Project notes

- `.cursor/rules/project.mdc` describes the architecture in more depth.
- Alembic migrations live in `src/llama_router/alembic/` and run at startup.
