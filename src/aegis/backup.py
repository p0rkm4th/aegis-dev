"""Small SQLite backup/restore primitives for local canonical-state rehearsal."""

from __future__ import annotations

import sqlite3


def backup_sqlite(source_path: str, destination_path: str) -> None:
    source = sqlite3.connect(source_path)
    destination = sqlite3.connect(destination_path)
    try:
        source.backup(destination)
        result = destination.execute("PRAGMA integrity_check").fetchone()
        if result != ("ok",):
            raise ValueError("SQLite backup failed integrity check")
    finally:
        destination.close()
        source.close()


def restore_sqlite(backup_path: str, destination_path: str) -> None:
    backup_sqlite(backup_path, destination_path)
