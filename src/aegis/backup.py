"""Small SQLite backup/restore primitives for local canonical-state rehearsal."""

from __future__ import annotations

import os
import sqlite3
import subprocess
from pathlib import Path
from urllib.parse import quote, urlparse


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


def _postgres_environment(database_url: str) -> tuple[str, dict[str, str]]:
    parsed = urlparse(database_url)
    if parsed.scheme not in {"postgresql", "postgres"} or not parsed.hostname:
        raise ValueError("database URL must be a PostgreSQL URL")
    if parsed.username is None or parsed.path in {None, "", "/"}:
        raise ValueError("database URL requires a user and database")
    safe_url = f"{parsed.scheme}://{quote(parsed.username, safe='')}@{parsed.hostname}"
    if parsed.port is not None:
        safe_url += f":{parsed.port}"
    safe_url += parsed.path
    environment = os.environ.copy()
    if parsed.password is not None:
        environment["PGPASSWORD"] = parsed.password
    return safe_url, environment


def backup_postgres(database_url: str, destination_path: str) -> None:
    """Create a custom-format backup using a client compatible with the server."""
    safe_url, environment = _postgres_environment(database_url)
    subprocess.run(
        ["pg_dump", "--format=custom", "--file", str(Path(destination_path)), "--dbname", safe_url],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )


def restore_postgres(database_url: str, backup_path: str) -> None:
    """Restore an explicit backup using a client compatible with the server."""
    safe_url, environment = _postgres_environment(database_url)
    subprocess.run(
        [
            "pg_restore",
            "--clean",
            "--if-exists",
            "--no-owner",
            "--dbname",
            safe_url,
            str(Path(backup_path)),
        ],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
