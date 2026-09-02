"""Safe release/source identity helpers for operator diagnostics."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_SHA = re.compile(r"^[0-9a-f]{7,40}$")


def valid_sha(value: object) -> bool:
    """Accept the short or full lowercase Git forms used in release records."""

    return isinstance(value, str) and bool(_SHA.fullmatch(value))


def runtime_release_sha(module_file: str) -> str | None:
    """Derive an installed release fingerprint without reading secrets or Git state."""

    configured = os.environ.get("AEGIS_RELEASE_SHA")
    if valid_sha(configured):
        return configured
    path = Path(module_file).resolve()
    for parent in path.parents:
        if valid_sha(parent.name):
            return parent.name
    return None


def validate_state_pointers(state: dict[str, Any]) -> list[str]:
    """Validate repository/evidence pointers without claiming live capability.

    An older live-green release is valid while newer source is in development,
    so this intentionally checks consistency, not ancestry. Git-aware ancestry
    checks belong in the repository validation script.
    """

    errors: list[str] = []
    required = (
        "repository_head_sha",
        "deterministic_green_sha",
        "last_pushed_sha",
        "hosted_ci_green_sha",
        "installed_release_sha",
        "running_release_sha",
        "live_green_sha",
    )
    for key in required:
        value = state.get(key)
        if value is not None and not valid_sha(value):
            errors.append(f"{key} must be a 7-40 character lowercase Git SHA or null")
    if state.get("running_release_sha") and not state.get("installed_release_sha"):
        errors.append("running_release_sha requires installed_release_sha")
    if state.get("live_green_sha") and not state.get("deterministic_green_sha"):
        errors.append("live_green_sha requires deterministic_green_sha")
    return errors
