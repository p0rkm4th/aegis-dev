"""Durable-ready repository boundary with an in-memory implementation for Phase 1."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from .contracts import Objective, Result


class ObjectiveStore(Protocol):
    def save_objective(self, objective: Objective) -> None: ...
    def get_objective(self, objective_id: UUID) -> Objective | None: ...
    def save_result(self, key: str, result: Result) -> None: ...
    def get_result(self, key: str) -> Result | None: ...


class InMemoryObjectiveStore:
    def __init__(self) -> None:
        self.objectives: dict[UUID, Objective] = {}
        self.results: dict[str, Result] = {}

    def save_objective(self, objective: Objective) -> None:
        self.objectives[objective.id] = objective

    def get_objective(self, objective_id: UUID) -> Objective | None:
        return self.objectives.get(objective_id)

    def save_result(self, key: str, result: Result) -> None:
        self.results[key] = result

    def get_result(self, key: str) -> Result | None:
        return self.results.get(key)
