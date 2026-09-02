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
python -m json.tool evaluation/semantic_dev.json >/dev/null
python -m json.tool evaluation/semantic_heldout.json >/dev/null
python -m json.tool evaluation/reports/qwen3-8b-semantic-153.json >/dev/null
python scripts/audit_semantic_corpus.py evaluation/semantic_dev.json evaluation/semantic_heldout.json >/dev/null
python scripts/evaluate_boundary.py --help >/dev/null
python scripts/validate_state.py
