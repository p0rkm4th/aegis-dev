from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest

from aegis.developer_jobs import PostgresDeveloperJobStore


class _Result:
    def __init__(self, row: Any = None, rows: Any = ()) -> None:
        self.row = row
        self.rows = list(rows)

    def fetchone(self) -> Any:
        return self.row

    def fetchall(self) -> list[Any]:
        return self.rows


class _Connection:
    def __init__(self) -> None:
        self.job_id = uuid4()
        self.state = "running"
        self.timestamp = datetime(2026, 9, 8, tzinfo=timezone.utc)
        self.closed = False
        self.committed = False

    def _row(self) -> tuple[Any, ...]:
        return (
            self.job_id,
            "aegis",
            "modify",
            "Fix the bug",
            self.state,
            {},
            None,
            self.timestamp,
            self.timestamp,
        )

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> _Result:
        if sql.startswith("INSERT INTO developer_jobs"):
            self.job_id = params[0]
            self.state = params[5]
            return _Result(self._row())
        if sql.startswith("UPDATE developer_jobs SET state = %s"):
            self.state = params[0]
            return _Result(self._row())
        if sql.startswith("UPDATE developer_jobs SET state = 'interrupted'"):
            self.state = "interrupted"
            return _Result()
        if sql.startswith("SELECT id, project_id"):
            return _Result(rows=(self._row(),))
        raise AssertionError(f"unexpected SQL: {sql}")

    def commit(self) -> None:
        self.committed = True

    def close(self) -> None:
        self.closed = True


def test_developer_job_store_persists_lifecycle_and_reconciles_running_jobs() -> None:
    connection = _Connection()
    store = PostgresDeveloperJobStore(lambda: connection, lambda _: None)

    created = store.create("alice", "aegis", "modify", "Fix the bug", "queued")
    assert created["state"] == "queued"
    updated = store.update("alice", UUID(created["job_id"]), "running")
    assert updated["state"] == "running"
    jobs = store.reconcile_running("alice")
    assert jobs[0]["state"] == "interrupted"
    assert connection.committed and connection.closed


def test_developer_job_store_rejects_unknown_job_state() -> None:
    def no_database() -> None:
        pytest.fail("database must not be touched")

    store = PostgresDeveloperJobStore(no_database, lambda _: None)
    with pytest.raises(ValueError):
        store.create("alice", "aegis", "modify", "Fix", "unknown")
