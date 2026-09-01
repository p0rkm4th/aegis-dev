#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ -x "$repo_root/.venv/bin/python" ]]; then
    PATH="$repo_root/.venv/bin:$PATH"
    export PATH
fi

install_dir="$(mktemp -d)"
prefix_dir="$(mktemp -d)"
run_dir="$(mktemp -d)"
trap 'rm -rf "$install_dir" "$prefix_dir" "$run_dir"' EXIT

python -m pip install --no-deps --target "$install_dir" .
PYTHONPATH="$install_dir" python -c 'from aegis.migrations import validate_migrations; migrations = validate_migrations(); assert migrations and migrations[-1] == "014_security_lab_findings.sql"'

python -m pip install --no-deps --prefix "$prefix_dir" .
site_dir="$(find "$prefix_dir" -type d -path '*/site-packages' -print -quit)"
test -x "$prefix_dir/bin/aegis"
PYTHONPATH="$site_dir" "$prefix_dir/bin/aegis" --version >/dev/null
(cd "$run_dir" && PYTHONPATH="$site_dir" "$prefix_dir/bin/aegis" --init >/dev/null)
test "$(stat -c '%a' "$run_dir/.env")" = "600"
