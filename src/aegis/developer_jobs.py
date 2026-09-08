"""Principal-scoped durable lifecycle for bounded Developer jobs."""

from __future__ import annotations

from builtins import list as builtin_list
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb

JOB_STATES = frozenset(
    {"queued", "running", "approval_required", "completed", "failed", "interrupted"}
)


class PostgresDeveloperJobStore:
    def __init__(self, connection_factory: Any, migrate: Any) -> None:
        self.connection_factory = connection_factory
        self.migrate = migrate

    def _connection(self) -> Any:
        connection = self.connection_factory()
        self.migrate(connection)
        return connection

    def create(
        self,
        principal_id: str,
        project_id: str,
        kind: str,
        objective: str,
        state: str,
    ) -> dict[str, Any]:
        if kind not in {"inspect", "modify"} or state not in JOB_STATES:
            raise ValueError("invalid developer job")
        job_id = uuid4()
        connection = self._connection()
        try:
            row = connection.execute(
                "INSERT INTO developer_jobs "
                "(id, principal_id, project_id, kind, objective, state) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "RETURNING id, project_id, kind, objective, state, result, error, "
                "created_at, updated_at",
                (job_id, principal_id, project_id, kind, objective[:2_000], state),
            ).fetchone()
            connection.commit()
            assert row is not None
            return self._row(row)
        finally:
            connection.close()

    def update(
        self,
        principal_id: str,
        job_id: UUID,
        state: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        if state not in JOB_STATES:
            raise ValueError("invalid developer job state")
        connection = self._connection()
        try:
            row = connection.execute(
                "UPDATE developer_jobs SET state = %s, result = %s, error = %s, "
                "updated_at = now() WHERE id = %s AND principal_id = %s "
                "RETURNING id, project_id, kind, objective, state, result, error, "
                "created_at, updated_at",
                (state, Jsonb(result or {}), error, job_id, principal_id),
            ).fetchone()
            if row is None:
                raise PermissionError("developer job is not owned by principal")
            connection.commit()
            return self._row(row)
        finally:
            connection.close()

    def reconcile_running(self, principal_id: str) -> builtin_list[dict[str, Any]]:
        connection = self._connection()
        try:
            connection.execute(
                "UPDATE developer_jobs SET state = 'interrupted', "
                "error = 'owner service restarted while job was running', updated_at = now() "
                "WHERE principal_id = %s AND state IN ('queued', 'running')",
                (principal_id,),
            )
            rows = connection.execute(
                "SELECT id, project_id, kind, objective, state, result, error, "
                "created_at, updated_at "
                "FROM developer_jobs WHERE principal_id = %s ORDER BY updated_at DESC LIMIT 30",
                (principal_id,),
            ).fetchall()
            connection.commit()
            return [self._row(row) for row in rows]
        finally:
            connection.close()

    @staticmethod
    def _row(row: tuple[Any, ...]) -> dict[str, Any]:
        return {
            "job_id": str(row[0]),
            "project_id": row[1],
            "kind": row[2],
            "objective": row[3],
            "state": row[4],
            "result": row[5] or {},
            "error": row[6],
            "created_at": _timestamp(row[7]),
            "updated_at": _timestamp(row[8]),
        }


def _timestamp(value: datetime) -> str:
    return value.isoformat()
