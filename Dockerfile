FROM python:3.14-slim AS base

WORKDIR /app

# Network and process debugging tools for use with `docker compose exec`:
# ping, host, dig/nslookup, nc, curl, wget, ip/ss, traceroute, ps/top.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bind9-dnsutils \
        bind9-host \
        ca-certificates \
        curl \
        iproute2 \
        iputils-ping \
        netcat-openbsd \
        procps \
        traceroute \
        wget \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

RUN python - <<'PY'
import tomllib
from pathlib import Path

deps = tomllib.loads(Path("pyproject.toml").read_text())["project"]["dependencies"]
Path("/tmp/requirements.txt").write_text("\n".join(deps) + "\n")
PY

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

COPY src/ src/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps .

EXPOSE 80 8080 9200 11434

CMD ["llama-router"]
