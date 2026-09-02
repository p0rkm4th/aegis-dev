#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ -x "$repo_root/.venv/bin/python" ]]; then
    PATH="$repo_root/.venv/bin:$PATH"
    export PATH
fi
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest
ruff check .
ruff format --check .
mypy src/aegis
python -c 'from aegis.migrations import validate_migrations; validate_migrations()'
python scripts/validate_state.py
