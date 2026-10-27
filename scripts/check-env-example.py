#!/usr/bin/env python3
"""Pre-commit check: .env.example lists every variable the app reads.

    scripts/check-env-example.py [PROJECT_DIR]

Finds the environment variables read in src/ (env.get("X"), env["X"],
os.environ.get("X"), os.environ["X"], os.getenv("X")) and fails, naming
each one, if .env.example has no line for it. A commented-out line
("# X=...") counts, for variables that are optional or set elsewhere.

.env.example may list more than the app reads: variables for compose
services (POSTGRES_PASSWORD) or tools (UV_PUBLISH_TOKEN).
"""

import ast
import re
import sys
from pathlib import Path

NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
ENV_OBJECTS = {"env", "environ"}  # env (a Mapping parameter) or os.environ


def is_env(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id in ENV_OBJECTS) or (
        isinstance(node, ast.Attribute) and node.attr == "environ"
    )


def first_name(args: list[ast.expr]) -> str | None:
    if args and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str):
        return args[0].value if NAME.match(args[0].value) else None
    return None


def variables_read(source: str) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(source)):
        name = None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                node.func.attr == "get" and is_env(node.func.value)
            ) or node.func.attr == "getenv":
                name = first_name(node.args)
        elif isinstance(node, ast.Subscript) and is_env(node.value):
            name = first_name([node.slice])
        if name:
            names.add(name)
    return names


def listed(example: Path) -> set[str]:
    names = set()
    for line in example.read_text().splitlines():
        line = line.lstrip("# ").removeprefix("export ")
        if "=" in line:
            names.add(line.split("=", 1)[0].strip())
    return names


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else Path()
    example = root / ".env.example"
    read = set()
    for path in sorted((root / "src").rglob("*.py")):
        read |= variables_read(path.read_text())
    missing = sorted(read - (listed(example) if example.exists() else set()))
    for name in missing:
        print(f"{example}: no line for {name}, which the app reads")
    if missing:
        print("Add each with a dummy value and a comment; see docs/configuration.md.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
