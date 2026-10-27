#!/usr/bin/env python3
"""Pre-commit check: no value from a local .env file in a committed file.

    scripts/check-env-secrets.py FILE...

.env files are gitignored, so their values (passwords, tokens, database
URLs) exist only on your machine. This fails if one of them turns up in a
file being committed: pasted into a test, a README, a log excerpt or a
compose file. It names the file, line and variable, never the value.

.env.example holds dummy values; it is never treated as real. Add
"not-a-secret" to a line to allow it.
"""

import sys
from pathlib import Path

ALLOW = "not-a-secret"
# Real secrets shorter than this are too likely to match ordinary text.
MIN_SECRET_LENGTH = 8
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__"}


def parse_env(path: Path) -> dict[str, str]:
    """KEY=value pairs from a dotenv file, unquoted."""
    values = {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def env_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob(".env*")
        if path.is_file() and not SKIP_DIRS.intersection(path.parts)
    ]


def local_secrets(root: Path) -> dict[str, str]:
    """Maps each real value in local .env files to its variable's name."""
    files = env_files(root)
    dummies = {
        value
        for path in files
        if path.name.endswith(".example")
        for value in parse_env(path).values()
    }
    return {
        value: key
        for path in files
        if not path.name.endswith(".example")
        for key, value in parse_env(path).items()
        if len(value) >= MIN_SECRET_LENGTH and value not in dummies
    }


def main(argv: list[str]) -> int:
    secrets = local_secrets(Path.cwd())
    if not secrets:
        return 0
    failed = False
    for name in argv:
        path = Path(name)
        if path.name.startswith(".env"):
            continue
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, 1):
            if ALLOW in line:
                continue
            for value, key in secrets.items():
                if value in line:
                    print(f"{name}:{number}: contains the value of {key} from .env")
                    failed = True
    if failed:
        print("Remove it (and rotate it if it was ever pushed).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
