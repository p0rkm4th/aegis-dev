"""Durable, bounded scheduling state for Luna engineering sprints.

This is development control-plane state, not AEGIS Objective or canonical
owner state.  It deliberately contains no authority, execution, or completion
semantics for product requests.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any


class SprintStateError(ValueError):
    """The durable development sprint state is invalid or corrupt."""


class TaskStatus(StrEnum):
    READY = "READY"
    ACTIVE = "ACTIVE"
    WAITING_DEPENDENCY = "WAITING_DEPENDENCY"
    WAITING_HUMAN = "WAITING_HUMAN"
    WAITING_EXTERNAL = "WAITING_EXTERNAL"
    COMPLETE = "COMPLETE"
    FAILED_BOUNDED = "FAILED_BOUNDED"
    CANCELLED = "CANCELLED"


class SprintStatus(StrEnum):
    READY = "READY"
    ACTIVE = "ACTIVE"
    WAITING_HUMAN = "WAITING_HUMAN"
    WAITING_EXTERNAL = "WAITING_EXTERNAL"
    COMPLETE = "COMPLETE"
    FAILED_BOUNDED = "FAILED_BOUNDED"
    CANCELLED = "CANCELLED"


TERMINAL_TASKS = {
    TaskStatus.COMPLETE,
    TaskStatus.FAILED_BOUNDED,
    TaskStatus.CANCELLED,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RepoState:
    branch: str
    starting_sha: str
    observed_head_sha: str
    last_green_sha: str = ""
    installed_sha: str = ""


@dataclass
class LaneState:
    lane_id: str
    priority: int
    status: str = "READY"
    blocker: str = ""
    evidence_needed: str = ""


@dataclass
class SprintTask:
    task_id: str
    lane_id: str
    title: str
    priority: int = 50
    dependencies: tuple[str, ...] = ()
    status: TaskStatus = TaskStatus.READY
    attempts: int = 0
    max_attempts: int = 3
    identical_failure_repeats: int = 0
    max_identical_failure_repeats: int = 2
    failure_fingerprint: str = ""
    evidence: tuple[str, ...] = ()
    changed_paths: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    blocker: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


@dataclass
class SprintState:
    schema_version: int
    sprint_id: str
    objective: str
    status: SprintStatus
    repo: RepoState
    lanes: dict[str, LaneState]
    tasks: dict[str, SprintTask]
    current_task: str | None = None
    replan_count: int = 0
    max_replans: int = 5
    last_progress_at: str = field(default_factory=_now)
    lessons: tuple[str, ...] = ()
    hosted_validation: dict[str, str] = field(default_factory=dict)
    next_ready_tasks: tuple[str, ...] = ()
    stop_reason: str = ""

    def validate(self) -> None:
        if self.schema_version != 1:
            raise SprintStateError("unsupported Luna sprint schema version")
        if not self.sprint_id.strip() or not self.objective.strip():
            raise SprintStateError("sprint_id and objective are required")
        if not self.repo.branch.strip() or not self.repo.observed_head_sha.strip():
            raise SprintStateError("repo branch and observed head SHA are required")
        if self.replan_count < 0 or self.max_replans < 0:
            raise SprintStateError("replan counts must be non-negative")
        task_ids = set(self.tasks)
        if any(task_id != task.task_id for task_id, task in self.tasks.items()):
            raise SprintStateError("task map key does not match task_id")
        for lane_id, lane in self.lanes.items():
            if lane_id != lane.lane_id:
                raise SprintStateError("lane map key does not match lane_id")
            if not 0 <= lane.priority <= 100:
                raise SprintStateError("lane priority must be in [0,100]")
        for task in self.tasks.values():
            if task.lane_id not in self.lanes:
                raise SprintStateError(f"task {task.task_id} references unknown lane")
            if not 0 <= task.priority <= 100:
                raise SprintStateError(f"task {task.task_id} priority must be in [0,100]")
            if not 0 <= task.attempts <= task.max_attempts:
                raise SprintStateError(f"task {task.task_id} has invalid attempt budget")
            if task.max_identical_failure_repeats < 1:
                raise SprintStateError("identical failure budget must be positive")
            if len(set(task.dependencies)) != len(task.dependencies):
                raise SprintStateError(f"task {task.task_id} repeats a dependency")
            if task.task_id in task.dependencies:
                raise SprintStateError(f"task {task.task_id} depends on itself")
            missing = set(task.dependencies) - task_ids
            if missing:
                raise SprintStateError(
                    f"task {task.task_id} has unknown dependencies: {sorted(missing)}"
                )
        _reject_cycles(self.tasks)
        if self.current_task is not None and self.current_task not in task_ids:
            raise SprintStateError("current_task does not exist")

    def ready_tasks(self) -> tuple[SprintTask, ...]:
        self._refresh_dependency_states()
        ready = [
            task
            for task in self.tasks.values()
            if task.status is TaskStatus.READY and _dependencies_complete(task, self.tasks)
        ]
        ready.sort(
            key=lambda task: (
                -self.lanes[task.lane_id].priority,
                -self._task_priority(task),
                task.task_id,
            )
        )
        self.next_ready_tasks = tuple(task.task_id for task in ready)
        return tuple(ready)

    def select_next_task(self) -> SprintTask | None:
        self.validate()
        if self.current_task:
            current = self.tasks[self.current_task]
            if current.status is TaskStatus.ACTIVE:
                return current
        ready = self.ready_tasks()
        if not ready:
            self._refresh_sprint_status()
            return None
        selected = ready[0]
        selected.status = TaskStatus.ACTIVE
        selected.updated_at = _now()
        self.current_task = selected.task_id
        self.status = SprintStatus.ACTIVE
        self.last_progress_at = selected.updated_at
        return selected

    def mark_waiting_human(self, task_id: str, blocker: str, evidence_needed: str) -> None:
        task = self._task(task_id)
        if not blocker.strip() or not evidence_needed.strip():
            raise SprintStateError("human blocker and evidence needed are required")
        task.status = TaskStatus.WAITING_HUMAN
        task.blocker = blocker.strip()
        task.evidence = (*task.evidence, evidence_needed.strip())
        task.updated_at = _now()
        if self.current_task == task_id:
            self.current_task = None
        self._refresh_dependency_states()
        self._refresh_sprint_status()

    def mark_waiting_external(self, task_id: str, blocker: str, evidence_needed: str) -> None:
        task = self._task(task_id)
        if not blocker.strip() or not evidence_needed.strip():
            raise SprintStateError("external blocker and evidence needed are required")
        task.status = TaskStatus.WAITING_EXTERNAL
        task.blocker = blocker.strip()
        task.evidence = (*task.evidence, evidence_needed.strip())
        task.updated_at = _now()
        if self.current_task == task_id:
            self.current_task = None
        self._refresh_sprint_status()

    def record_failure(
        self,
        task_id: str,
        failure_fingerprint: str,
        evidence: str,
        *,
        hypothesis_changed: bool,
    ) -> None:
        task = self._task(task_id)
        if task.status not in {TaskStatus.ACTIVE, TaskStatus.READY}:
            raise SprintStateError("only runnable tasks can record failure")
        if not failure_fingerprint.strip() or not evidence.strip():
            raise SprintStateError("failure fingerprint and new evidence are required")
        if not hypothesis_changed and task.failure_fingerprint == failure_fingerprint:
            task.identical_failure_repeats += 1
        else:
            task.identical_failure_repeats = 0
        task.failure_fingerprint = failure_fingerprint.strip()
        task.evidence = (*task.evidence, evidence.strip())
        task.attempts += 1
        task.updated_at = _now()
        if (
            task.attempts >= task.max_attempts
            or task.identical_failure_repeats >= task.max_identical_failure_repeats
        ):
            task.status = TaskStatus.FAILED_BOUNDED
        else:
            task.status = TaskStatus.READY
        if self.current_task == task_id:
            self.current_task = None
        self._refresh_dependency_states()
        self._refresh_sprint_status()

    def mark_complete(
        self,
        task_id: str,
        *,
        changed_paths: tuple[str, ...] = (),
        tests: tuple[str, ...] = (),
        evidence: tuple[str, ...] = (),
    ) -> None:
        task = self._task(task_id)
        if task.status is not TaskStatus.ACTIVE:
            raise SprintStateError("only active tasks can complete")
        task.status = TaskStatus.COMPLETE
        task.changed_paths = changed_paths
        task.tests = tests
        task.evidence = (*task.evidence, *evidence)
        task.updated_at = _now()
        if self.current_task == task_id:
            self.current_task = None
        self.last_progress_at = task.updated_at
        self._refresh_sprint_status()

    def resume(self, observed_head_sha: str) -> SprintTask | None:
        if not observed_head_sha.strip():
            raise SprintStateError("observed head SHA is required for resume")
        self.repo.observed_head_sha = observed_head_sha.strip()
        self.last_progress_at = _now()
        return self.select_next_task()

    def _task(self, task_id: str) -> SprintTask:
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            raise SprintStateError(f"unknown task: {task_id}") from exc

    def _task_priority(self, task: SprintTask) -> int:
        return task.priority

    def _refresh_dependency_states(self) -> None:
        for task in self.tasks.values():
            if task.status is TaskStatus.WAITING_DEPENDENCY:
                if any(
                    self.tasks[dep].status in {TaskStatus.FAILED_BOUNDED, TaskStatus.CANCELLED}
                    for dep in task.dependencies
                ):
                    continue
                if _dependencies_complete(task, self.tasks):
                    task.status = TaskStatus.READY
            elif task.status is TaskStatus.READY and any(
                self.tasks[dep].status in {TaskStatus.FAILED_BOUNDED, TaskStatus.CANCELLED}
                for dep in task.dependencies
            ):
                task.status = TaskStatus.WAITING_DEPENDENCY
                task.blocker = "dependency failed or was cancelled"

    def _refresh_sprint_status(self) -> None:
        if self.tasks and all(task.status in TERMINAL_TASKS for task in self.tasks.values()):
            self.status = (
                SprintStatus.FAILED_BOUNDED
                if any(task.status is TaskStatus.FAILED_BOUNDED for task in self.tasks.values())
                else SprintStatus.COMPLETE
            )
        elif any(task.status is TaskStatus.READY for task in self.tasks.values()):
            self.status = SprintStatus.READY
        elif any(task.status is TaskStatus.WAITING_HUMAN for task in self.tasks.values()):
            self.status = SprintStatus.WAITING_HUMAN
        elif any(task.status is TaskStatus.WAITING_EXTERNAL for task in self.tasks.values()):
            self.status = SprintStatus.WAITING_EXTERNAL


def _dependencies_complete(task: SprintTask, tasks: dict[str, SprintTask]) -> bool:
    return all(tasks[dependency].status is TaskStatus.COMPLETE for dependency in task.dependencies)


def _reject_cycles(tasks: dict[str, SprintTask]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise SprintStateError("sprint task graph contains a cycle")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in tasks[task_id].dependencies:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in tasks:
        visit(task_id)


def _from_json(data: dict[str, Any]) -> SprintState:
    try:
        repo = RepoState(**data["repo"])
        lanes = {key: LaneState(**value) for key, value in data["lanes"].items()}
        tasks = {
            key: SprintTask(
                **{
                    **value,
                    "status": TaskStatus(value["status"]),
                    "dependencies": tuple(value.get("dependencies", ())),
                    "evidence": tuple(value.get("evidence", ())),
                    "changed_paths": tuple(value.get("changed_paths", ())),
                    "tests": tuple(value.get("tests", ())),
                }
            )
            for key, value in data["tasks"].items()
        }
        state = SprintState(
            schema_version=int(data["schema_version"]),
            sprint_id=str(data["sprint_id"]),
            objective=str(data["objective"]),
            status=SprintStatus(data["status"]),
            repo=repo,
            lanes=lanes,
            tasks=tasks,
            current_task=data.get("current_task"),
            replan_count=int(data.get("replan_count", 0)),
            max_replans=int(data.get("max_replans", 5)),
            last_progress_at=str(data.get("last_progress_at", _now())),
            lessons=tuple(data.get("lessons", ())),
            hosted_validation=dict(data.get("hosted_validation", {})),
            next_ready_tasks=tuple(data.get("next_ready_tasks", ())),
            stop_reason=str(data.get("stop_reason", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SprintStateError("corrupt Luna sprint state") from exc
    state.validate()
    return state


def load(path: Path) -> SprintState:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SprintStateError(f"cannot load Luna sprint state: {path}") from exc
    if not isinstance(data, dict):
        raise SprintStateError("Luna sprint state must be a JSON object")
    return _from_json(data)


def save(path: Path, state: SprintState) -> None:
    state.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(state), indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
