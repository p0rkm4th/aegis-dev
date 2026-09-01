"""Durable-ready repository boundary with an in-memory implementation for Phase 1."""

from __future__ import annotations

import json
import sqlite3
from typing import Protocol
from uuid import UUID

from .contracts import Objective, ObjectiveState, Result


class ObjectiveStore(Protocol):
    def save_objective(self, objective: Objective) -> None: ...
    def get_objective(self, objective_id: UUID) -> Objective | None: ...
    def save_result(self, key: str, result: Result) -> None: ...
    def get_result(self, key: str) -> Result | None: ...


class PostgresCursor(Protocol):
    def fetchone(self) -> tuple[object, ...] | None: ...


class PostgresConnection(Protocol):
    def execute(self, query: str, params: tuple[object, ...] = ()) -> PostgresCursor: ...

    def commit(self) -> None: ...


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


class PostgresObjectiveStore:
    """PostgreSQL implementation of the canonical ObjectiveStore port."""

    def __init__(self, connection: PostgresConnection) -> None:
        self.connection = connection

    def save_objective(self, objective: Objective) -> None:
        self.connection.execute(
            """INSERT INTO objectives (id, principal_id, vault_id, space_id, state, payload)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET state = EXCLUDED.state,
                   payload = EXCLUDED.payload, updated_at = now()""",
            (
                str(objective.id),
                objective.intent.principal.id,
                objective.intent.principal.vault_id,
                objective.intent.principal.space_ids[0]
                if objective.intent.principal.space_ids
                else None,
                objective.state.value,
                objective.model_dump_json(),
            ),
        )
        self.connection.commit()

    def get_objective(self, objective_id: UUID) -> Objective | None:
        cursor = self.connection.execute(
            "SELECT payload FROM objectives WHERE id = %s", (str(objective_id),)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        payload = row[0]
        return (
            Objective.model_validate(payload)
            if isinstance(payload, dict)
            else Objective.model_validate_json(str(payload))
        )

    def save_result(self, key: str, result: Result) -> None:
        self.connection.execute(
            """INSERT INTO results
               (id, objective_id, idempotency_key, state, evidence, message)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (idempotency_key) DO UPDATE SET state = EXCLUDED.state,
                   evidence = EXCLUDED.evidence, message = EXCLUDED.message""",
            (
                str(result.correlation_id),
                str(result.objective_id),
                key,
                result.state.value,
                json.dumps(result.evidence, sort_keys=True),
                result.message,
            ),
        )
        self.connection.commit()

    def get_result(self, key: str) -> Result | None:
        cursor = self.connection.execute(
            "SELECT id, objective_id, state, evidence, message "
            "FROM results WHERE idempotency_key = %s",
            (key,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return Result(
            objective_id=UUID(str(row[1])),
            state=ObjectiveState(str(row[2])),
            evidence=row[3] if isinstance(row[3], dict) else json.loads(str(row[3])),
            message=str(row[4]),
            correlation_id=UUID(str(row[0])),
        )
