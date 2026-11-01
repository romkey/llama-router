# syntax=docker/dockerfile:1
#
# Stages:
#   dev         uv and every dependency group. docker-compose.yaml and
#               docker-compose.test.yaml mount the working tree over /app.
#   production  the app and its runtime dependencies only, as a non-root
#               user. Released to ghcr.io by .github/workflows/release.yaml.
#
# Both include every install extra (the PostgreSQL and MySQL drivers), so
# LLAMA_ROUTER_DATABASE_URL works with the released image as it is.
#
# The python tag matches .python-version and requires-python in
# pyproject.toml; change all three together.

FROM python:3.14-slim-trixie AS base
# No extra OS packages: the python image already has ca-certificates, and
# the production image deliberately carries no network tools (curl, wget,
# nc, dig, ping...) for an attacker to use. The dev stage adds them.

# Everything the app keeps (database, registry cache, Hugging Face cache and
# downloads) lives on the /app/data volume.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/venv \
    PATH=/venv/bin:$PATH \
    LLAMA_ROUTER_DATABASE_PATH=/app/data/llama_router.db \
    LLAMA_ROUTER_CACHE_DIR=/app/data/model_cache \
    LLAMA_ROUTER_HF_MODELS_DIR=/app/data/hf_models \
    HF_HOME=/app/data/hf \
    HF_HUB_DISABLE_UPDATE_CHECK=1
WORKDIR /app

FROM base AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
# Dependencies first, so a source change doesn't reinstall them.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --all-extras --no-install-project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --all-extras --no-editable

FROM build AS dev
# Network and process debugging tools for use with `docker compose exec`:
# ping, host, dig/nslookup, nc, curl, wget, ip/ss, traceroute, ps/top.
# Unpinned: Debian's stable repo only ships security fixes for these.
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        bind9-dnsutils \
        bind9-host \
        curl \
        iproute2 \
        iputils-ping \
        netcat-openbsd \
        procps \
        traceroute \
        wget \
 && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-extras
EXPOSE 80 8080 9200 11434
CMD ["python", "-m", "llama_router"]

FROM base AS production
# Numeric, so orchestrators can verify it isn't root. /app/data is created
# owned by it so a new named volume mounted there is writable.
RUN groupadd --system --gid 10001 app \
 && useradd --system --gid 10001 --uid 10001 --no-log-init --no-create-home app \
 && mkdir -p /app/data \
 && chown 10001:10001 /app/data
COPY --from=build /venv /venv
USER 10001:10001
EXPOSE 80 8080 9200 11434
CMD ["llama-router"]
