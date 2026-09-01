"""Canonical Tasks Pack state with Space membership enforced below the model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol
from uuid import UUID, uuid4

from .contracts import Principal


class TaskStatus(StrEnum):
    OPEN = "open"
    COMPLETED = "completed"


@dataclass(frozen=True)
class Task:
    task_id: UUID
    space_id: str
    title: str
    created_by: str
    assignee_id: str | None = None
    due_at: datetime | None = None
    status: TaskStatus = TaskStatus.OPEN


class TaskConnection(Protocol):
    def execute(self, query: str, params: tuple[object, ...] = ()) -> Any: ...

    def commit(self) -> None: ...


class PostgresTaskStore:
    """Persist Tasks and require current Space membership for every operation."""

    def __init__(self, connection: TaskConnection) -> None:
        self.connection = connection

    def create(
        self,
        principal: Principal,
        title: str,
        due_at: datetime | None = None,
        assignee_id: str | None = None,
    ) -> Task:
        space_id = self._space_for(principal)
        if not title.strip():
            raise ValueError("task title is required")
        if assignee_id is not None:
            self._require_member(space_id, assignee_id)
        task = Task(uuid4(), space_id, title, principal.id, assignee_id, due_at)
        self._write(task)
        return task

    def complete(self, principal: Principal, task_id: UUID) -> Task:
        task = self.get(principal, task_id)
        if task is None:
            raise KeyError("task is unavailable")
        completed = Task(
            task.task_id,
            task.space_id,
            task.title,
            task.created_by,
            task.assignee_id,
            task.due_at,
            TaskStatus.COMPLETED,
        )
        self._write(completed)
        return completed

    def get(self, principal: Principal, task_id: UUID) -> Task | None:
        row = self.connection.execute(
            "SELECT id, space_id, title, created_by, assignee_id, due_at, status "
            "FROM tasks WHERE id = %s",
            (str(task_id),),
        ).fetchone()
        if row is None:
            return None
        self._require_member(str(row[1]), principal.id)
        return self._from_row(row)

    def list(self, principal: Principal) -> tuple[Task, ...]:
        space_id = self._space_for(principal)
        rows = self.connection.execute(
            "SELECT id, space_id, title, created_by, assignee_id, due_at, status "
            "FROM tasks WHERE space_id = %s ORDER BY created_at, id",
            (space_id,),
        ).fetchall()
        return tuple(self._from_row(row) for row in rows)

    def _write(self, task: Task) -> None:
        self.connection.execute(
            "INSERT INTO tasks "
            "(id, space_id, title, created_by, assignee_id, due_at, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, "
            "assignee_id = EXCLUDED.assignee_id, due_at = EXCLUDED.due_at, "
            "status = EXCLUDED.status, updated_at = now()",
            (
                str(task.task_id),
                task.space_id,
                task.title,
                task.created_by,
                task.assignee_id,
                task.due_at,
                task.status.value,
            ),
        )
        self.connection.commit()

    def _space_for(self, principal: Principal) -> str:
        if not principal.space_ids:
            raise PermissionError("task operation requires an explicit Space")
        space_id = principal.space_ids[0]
        self._require_member(space_id, principal.id)
        return space_id

    def _require_member(self, space_id: str, principal_id: str) -> None:
        row = self.connection.execute(
            "SELECT 1 FROM space_memberships "
            "WHERE principal_id = %s AND space_id = %s AND active = TRUE",
            (principal_id, space_id),
        ).fetchone()
        if row is None:
            raise PermissionError("principal is not an active Space member")

    @staticmethod
    def _from_row(row: tuple[object, ...]) -> Task:
        return Task(
            UUID(str(row[0])),
            str(row[1]),
            str(row[2]),
            str(row[3]),
            str(row[4]) if row[4] is not None else None,
            row[5],  # type: ignore[arg-type]
            TaskStatus(str(row[6])),
        )
