#!/usr/bin/env bash
set -euo pipefail

install_dir="$(mktemp -d)"
trap 'rm -rf "$install_dir"' EXIT

python -m pip install --no-deps --target "$install_dir" .
PYTHONPATH="$install_dir" python -c 'from aegis.migrations import validate_migrations; migrations = validate_migrations(); assert migrations and migrations[-1] == "014_security_lab_findings.sql"'
