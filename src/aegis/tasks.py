"""Canonical Tasks Pack state with Space membership enforced below the model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import StrEnum
from typing import Any, Protocol
from uuid import UUID, uuid4

from .contracts import (
    ExecutionRequest,
    IntentFrame,
    ObjectiveState,
    Observation,
    Principal,
    Result,
    VerificationContract,
    VerificationResult,
)
from .utterance import is_mutation_request


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
    idempotency_key: str = ""


def _task_projection(task: Task) -> dict[str, object]:
    """Expose canonical task fields without inventing a second task store."""

    projection: dict[str, object] = {
        "task_id": str(task.task_id),
        "title": task.title,
        "status": task.status.value,
    }
    if task.due_at is not None:
        projection["due_at"] = task.due_at.isoformat()
    return projection


def _aware_datetime(value: datetime) -> datetime:
    """Normalize legacy naive deadlines before relative-date comparisons."""

    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def requested_task_due_at(utterance: str, now: datetime | None = None) -> str | None:
    """Translate only explicit bounded relative task dates into canonical time."""

    current = now or datetime.now(timezone.utc)
    text = utterance.casefold().strip()
    if re.search(r"\btomorrow[.!?]?$", text):
        return (current + timedelta(days=1)).isoformat()
    if re.search(r"\bnext\s+week[.!?]?$", text):
        return (current + timedelta(days=7)).isoformat()
    return None


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
        idempotency_key: str | None = None,
    ) -> Task:
        space_id = self._space_for(principal)
        if not title.strip():
            raise ValueError("task title is required")
        if assignee_id is not None:
            self._require_member(space_id, assignee_id)
        key = idempotency_key or f"task:{uuid4()}"
        existing = self._get_by_idempotency_key(key)
        if existing is not None:
            self._require_member(existing.space_id, principal.id)
            if (
                existing.title != title
                or existing.space_id != space_id
                or existing.assignee_id != assignee_id
                or existing.due_at != due_at
            ):
                raise ValueError("idempotency key is already bound to different task arguments")
            return existing
        task = Task(
            uuid4(), space_id, title, principal.id, assignee_id, due_at, TaskStatus.OPEN, key
        )
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
            task.idempotency_key,
        )
        self._write(completed)
        return completed

    def get(self, principal: Principal, task_id: UUID) -> Task | None:
        row = self.connection.execute(
            "SELECT id, space_id, title, created_by, assignee_id, due_at, status, idempotency_key "
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
            "SELECT id, space_id, title, created_by, assignee_id, due_at, status, idempotency_key "
            "FROM tasks WHERE space_id = %s ORDER BY created_at, id",
            (space_id,),
        ).fetchall()
        return tuple(self._from_row(row) for row in rows)

    def _write(self, task: Task) -> None:
        self.connection.execute(
            "INSERT INTO tasks "
            "(id, space_id, title, created_by, assignee_id, due_at, status, idempotency_key) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, "
            "assignee_id = EXCLUDED.assignee_id, due_at = EXCLUDED.due_at, "
            "status = EXCLUDED.status, idempotency_key = EXCLUDED.idempotency_key, "
            "updated_at = now()",
            (
                str(task.task_id),
                task.space_id,
                task.title,
                task.created_by,
                task.assignee_id,
                task.due_at,
                task.status.value,
                task.idempotency_key,
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
            str(row[7]),
        )

    def _get_by_idempotency_key(self, key: str) -> Task | None:
        row = self.connection.execute(
            "SELECT id, space_id, title, created_by, assignee_id, due_at, status, idempotency_key "
            "FROM tasks WHERE idempotency_key = %s",
            (key,),
        ).fetchone()
        return self._from_row(row) if row is not None else None


class PostgresTaskExecutor:
    """Adapt the Tasks Pack mutation to the generic Executor port."""

    def __init__(self, store: PostgresTaskStore, principal: Principal) -> None:
        self.store = store
        self.principal = principal

    def execute(self, request: ExecutionRequest) -> Observation:
        if request.action.action_id not in {"tasks.create", "tasks.complete"}:
            return Observation(
                execution_id=request.action_id,
                evidence={"unknown_action": request.action.action_id},
                command_succeeded=False,
            )
        title = request.action.arguments.get("title")
        if not isinstance(title, str) or not title.strip():
            return Observation(
                execution_id=request.action_id,
                evidence={"invalid_title": True},
                command_succeeded=False,
            )
        if request.action.action_id == "tasks.create":
            due_at_value = request.action.arguments.get("due_at")
            if due_at_value is not None and not isinstance(due_at_value, str):
                return Observation(
                    execution_id=request.action_id,
                    evidence={"collection": "tasks", "invalid_due_at": True},
                    command_succeeded=False,
                )
            due_at = None
            if due_at_value is not None:
                try:
                    due_at = datetime.fromisoformat(due_at_value)
                except ValueError:
                    return Observation(
                        execution_id=request.action_id,
                        evidence={"collection": "tasks", "invalid_due_at": True},
                        command_succeeded=False,
                    )
            if due_at is None:
                task = self.store.create(
                    self.principal,
                    title,
                    idempotency_key=request.idempotency_key,
                )
            else:
                task = self.store.create(
                    self.principal,
                    title,
                    due_at=due_at,
                    idempotency_key=request.idempotency_key,
                )
        else:
            matches = tuple(
                task
                for task in self.store.list(self.principal)
                if task.title.casefold().strip().rstrip(".!?")
                == title.casefold().strip().rstrip(".!?")
            )
            if len(matches) != 1:
                return Observation(
                    execution_id=request.action_id,
                    evidence={
                        "collection": "tasks",
                        "title": title,
                        "task_unavailable": len(matches) == 0,
                        "ambiguous_task_title": len(matches) > 1,
                    },
                    command_succeeded=False,
                )
            task = (
                matches[0]
                if matches[0].status is TaskStatus.COMPLETED
                else self.store.complete(self.principal, matches[0].task_id)
            )
        evidence = {
            "collection": "tasks",
            "task_id": str(task.task_id),
            "title": task.title,
            "status": task.status.value,
            "idempotency_key": task.idempotency_key,
        }
        if task.due_at is not None:
            evidence["due_at"] = task.due_at.isoformat()
        return Observation(
            execution_id=request.action_id,
            evidence=evidence,
            command_succeeded=True,
        )


class TaskCompletionFastPath:
    """Resolve completion titles before model/executor dispatch can guess."""

    @staticmethod
    def resolve(intent: IntentFrame, title: str, tasks: tuple[Task, ...]) -> Result | None:
        normalized = title.casefold().strip().rstrip(".!?")
        matches = tuple(
            task for task in tasks if task.title.casefold().strip().rstrip(".!?") == normalized
        )
        if len(matches) == 1:
            return None
        if len(matches) == 0:
            message = (
                f"I couldn't find one task named '{title}'. "
                "Ask to complete a task that appears in your task list."
            )
        else:
            message = (
                f"I found multiple tasks named '{title}'. "
                "Please include more detail so I complete only the intended task."
            )
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=message,
            correlation_id=intent.correlation_id,
        )


class PostgresTaskVerifier:
    """Verify task creation by independently reading canonical PostgreSQL state."""

    def __init__(self, store: PostgresTaskStore, principal: Principal) -> None:
        self.store = store
        self.principal = principal

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        if contract.kind != "readback" or not observation.command_succeeded:
            return VerificationResult(
                verified=False, evidence=observation.evidence, reason="task execution failed"
            )
        task_id = observation.evidence.get("task_id")
        if not isinstance(task_id, str):
            return VerificationResult(
                verified=False, evidence=observation.evidence, reason="task identity missing"
            )
        try:
            task = self.store.get(self.principal, UUID(task_id))
        except (PermissionError, ValueError):
            task = None
        expected_status = observation.evidence.get("status", TaskStatus.OPEN.value)
        verified = (
            task is not None
            and task.title == observation.evidence.get("title")
            and task.status.value == expected_status
            and (
                observation.evidence.get("due_at") is None
                or (
                    task.due_at is not None
                    and task.due_at.isoformat() == observation.evidence["due_at"]
                )
            )
        )
        return VerificationResult(
            verified=verified,
            evidence={
                **observation.evidence,
                "canonical_status": task.status.value if task else None,
            },
            reason=(
                "canonical task readback verified" if verified else "canonical task readback failed"
            ),
        )


class PostgresTaskListExecutor:
    """Adapt the canonical Tasks read to the generic Executor port."""

    def __init__(self, store: PostgresTaskStore, principal: Principal) -> None:
        self.store = store
        self.principal = principal

    def execute(self, request: ExecutionRequest) -> Observation:
        if request.action.action_id != "tasks.list":
            return Observation(
                execution_id=request.action_id,
                evidence={"unknown_action": request.action.action_id},
                command_succeeded=False,
            )
        tasks = self.store.list(self.principal)
        return Observation(
            execution_id=request.action_id,
            evidence={
                "collection": "tasks",
                "tasks": [_task_projection(task) for task in tasks],
            },
            command_succeeded=True,
        )


class PostgresTaskListVerifier:
    """Independently compare a task list observation with canonical state."""

    def __init__(self, store: PostgresTaskStore, principal: Principal) -> None:
        self.store = store
        self.principal = principal

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        if contract.kind != "readback" or not observation.command_succeeded:
            return VerificationResult(
                verified=False, evidence=observation.evidence, reason="task list read failed"
            )
        expected = observation.evidence.get("tasks")
        actual = [_task_projection(task) for task in self.store.list(self.principal)]
        verified = expected == actual
        return VerificationResult(
            verified=verified,
            evidence={**observation.evidence, "canonical_tasks": actual},
            reason="canonical task list verified" if verified else "canonical task list changed",
        )


class TaskReadFastPath:
    """Deterministic task reads over the membership-checked canonical store."""

    _TRIGGERS = ("task", "tasks", "to-do", "todo", "what do i need to do")
    _READ_PREFIXES = ("what", "show", "list", "which", "see", "display")

    def __init__(self, store: PostgresTaskStore) -> None:
        self.store = store

    @classmethod
    def matches(cls, utterance: str) -> bool:
        text = utterance.casefold()
        if is_mutation_request(text):
            return False
        if not any(trigger in text for trigger in cls._TRIGGERS):
            return False
        # A domain noun alone is not evidence of a read. Keep this fast path
        # high-confidence and let bounded cognition resolve unfamiliar language.
        return text.startswith(cls._READ_PREFIXES) or text in {"task", "tasks", "todo", "to-do"}

    def resolve(self, intent: IntentFrame) -> Result | None:
        if not self.matches(intent.utterance):
            return None
        all_tasks = self.store.list(intent.principal)
        text = intent.utterance.casefold()
        if any(term in text for term in ("completed", "finished")):
            tasks = tuple(task for task in all_tasks if task.status is TaskStatus.COMPLETED)
            status_filter = "completed"
        elif any(term in text for term in ("open", "pending", "unfinished")):
            tasks = tuple(task for task in all_tasks if task.status is TaskStatus.OPEN)
            status_filter = "open"
        else:
            tasks = all_tasks
            status_filter = "all"
        now = datetime.now(timezone.utc)
        due_start: date | None = None
        due_end: date | None = None
        if "due tomorrow" in text:
            due_start = (now + timedelta(days=1)).date()
            due_end = due_start + timedelta(days=1)
            due_filter = "tomorrow"
        elif "due next week" in text:
            due_start = (now + timedelta(days=7)).date()
            due_end = due_start + timedelta(days=7)
            due_filter = "next_week"
        else:
            due_filter = "all"
        if due_start is not None and due_end is not None:
            tasks = tuple(
                task
                for task in tasks
                if task.due_at is not None
                and due_start <= _aware_datetime(task.due_at).date() < due_end
            )
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.COMPLETED,
            message="Canonical task list read",
            evidence={
                "collection": "tasks",
                "status_filter": status_filter,
                "due_filter": due_filter,
                "canonical_tasks": [_task_projection(task) for task in tasks],
            },
            correlation_id=intent.correlation_id,
        )


class TaskIntentClarificationFastPath:
    """Clarify vague task verbs before a read path can hide the ambiguity."""

    _UNRESOLVED = ("take care of", "handle", "deal with", "look after")

    @classmethod
    def resolve(cls, intent: IntentFrame) -> Result | None:
        text = intent.utterance.casefold()
        if is_mutation_request(text) or "task" not in text:
            return None
        if not any(term in text for term in cls._UNRESOLVED):
            return None
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=(
                "Do you want me to complete that task, or just show its current status? "
                "Please say 'complete the task' or 'show my tasks.'"
            ),
            correlation_id=intent.correlation_id,
        )
