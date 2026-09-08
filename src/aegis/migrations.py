"""Deterministic migration manifest validation."""

from __future__ import annotations

import re
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

_MIGRATION_NAME = re.compile(r"^(\d{3})_[a-z0-9_]+\.sql$")


def _validate_names(entries: list[tuple[str, str]]) -> tuple[str, ...]:
    versions: list[int] = []
    names: list[str] = []
    for name, content in entries:
        match = _MIGRATION_NAME.fullmatch(name)
        if match is None:
            raise ValueError(f"invalid migration filename: {name}")
        if not content.strip():
            raise ValueError(f"empty migration: {name}")
        versions.append(int(match.group(1)))
        names.append(name)
    if versions != list(range(1, len(versions) + 1)):
        raise ValueError("migration versions must be unique and contiguous from 001")
    return tuple(names)


def validate_migrations(directory: str | Path = "migrations") -> tuple[str, ...]:
    root = Path(directory)
    return _validate_names(
        [(path.name, path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.sql"))]
    )


def validate_packaged_migrations() -> tuple[str, ...]:
    """Validate the migration inventory shipped inside an installed wheel."""

    root: Traversable = files("aegis").joinpath("migrations")
    if not root.is_dir():
        # Source checkouts keep migrations at the repository root; installed
        # wheels expose the same directory as package data.
        return validate_migrations("migrations")
    entries = [
        (entry.name, entry.read_text(encoding="utf-8"))
        for entry in sorted(root.iterdir(), key=lambda item: item.name)
        if entry.name.endswith(".sql")
    ]
    return _validate_names(entries)
