"""Deterministic migration manifest validation."""

from __future__ import annotations

import re
from pathlib import Path

_MIGRATION_NAME = re.compile(r"^(\d{3})_[a-z0-9_]+\.sql$")


def validate_migrations(directory: str | Path = "migrations") -> tuple[str, ...]:
    files = sorted(Path(directory).glob("*.sql"))
    versions: list[int] = []
    names: list[str] = []
    for path in files:
        match = _MIGRATION_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid migration filename: {path.name}")
        if not path.read_text().strip():
            raise ValueError(f"empty migration: {path.name}")
        versions.append(int(match.group(1)))
        names.append(path.name)
    if versions != list(range(1, len(versions) + 1)):
        raise ValueError("migration versions must be unique and contiguous from 001")
    return tuple(names)
