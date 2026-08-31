#!/usr/bin/env bash
set -euo pipefail

python -m pytest
ruff check .
ruff format --check .
mypy src/aegis
python -c 'from aegis.migrations import validate_migrations; validate_migrations()'
