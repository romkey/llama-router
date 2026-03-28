FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bind9-host \
        curl \
        iputils-ping \
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
