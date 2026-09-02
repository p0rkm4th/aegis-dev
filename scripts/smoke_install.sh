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
dist_dir="$(mktemp -d)"
sdist_prefix_dir="$(mktemp -d)"
trap 'rm -rf "$install_dir" "$prefix_dir" "$run_dir" "$dist_dir" "$sdist_prefix_dir"' EXIT

python -m build --wheel --sdist --outdir "$dist_dir" --no-isolation
wheel="$(find "$dist_dir" -maxdepth 1 -type f -name '*.whl' -print -quit)"
sdist="$(find "$dist_dir" -maxdepth 1 -type f -name '*.tar.gz' -print -quit)"
test -n "$wheel"
test -n "$sdist"
python -m pip install --no-deps --target "$install_dir" "$wheel"
PYTHONPATH="$install_dir" python -c 'from aegis.migrations import validate_migrations; migrations = validate_migrations(); assert migrations and migrations[-1] == "014_security_lab_findings.sql"'

python -m pip install --no-deps --prefix "$prefix_dir" "$wheel"
site_dir="$(find "$prefix_dir" -type d -path '*/site-packages' -print -quit)"
test -x "$prefix_dir/bin/aegis"
PYTHONPATH="$site_dir" "$prefix_dir/bin/aegis" --version >/dev/null
(cd "$run_dir" && PYTHONPATH="$site_dir" "$prefix_dir/bin/aegis" --init >/dev/null)
test "$(stat -c '%a' "$run_dir/.env")" = "600"

python -m pip install --no-deps --prefix "$sdist_prefix_dir" "$sdist"
sdist_site_dir="$(find "$sdist_prefix_dir" -type d -path '*/site-packages' -print -quit)"
test -x "$sdist_prefix_dir/bin/aegis"
PYTHONPATH="$sdist_site_dir" "$sdist_prefix_dir/bin/aegis" --help >/dev/null
PYTHONPATH="$sdist_site_dir" python -c 'from aegis.migrations import validate_migrations; assert validate_migrations()[-1] == "014_security_lab_findings.sql"'
