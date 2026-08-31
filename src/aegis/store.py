"""Durable-ready repository boundary with an in-memory implementation for Phase 1."""

from __future__ import annotations

import sqlite3
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


class SqliteObjectiveStore:
    """Transactional rehearsal store; the same port will back PostgreSQL in Phase 3."""

    def __init__(self, path: str) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS objectives (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.connection.commit()

    def save_objective(self, objective: Objective) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO objectives (id, payload) VALUES (?, ?)",
            (str(objective.id), objective.model_dump_json()),
        )
        self.connection.commit()

    def get_objective(self, objective_id: UUID) -> Objective | None:
        row = self.connection.execute(
            "SELECT payload FROM objectives WHERE id = ?", (str(objective_id),)
        ).fetchone()
        return Objective.model_validate_json(row[0]) if row else None

    def save_result(self, key: str, result: Result) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO results (key, payload) VALUES (?, ?)",
            (key, result.model_dump_json()),
        )
        self.connection.commit()

    def get_result(self, key: str) -> Result | None:
        row = self.connection.execute(
            "SELECT payload FROM results WHERE key = ?", (key,)
        ).fetchone()
        return Result.model_validate_json(row[0]) if row else None

    def close(self) -> None:
        self.connection.close()
