"""Durable-ready repository boundary with an in-memory implementation for Phase 1."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Protocol, cast
from uuid import UUID

from .contracts import ExecutionRequest, Objective, ObjectiveState, Observation, Principal, Result


class ObjectiveStore(Protocol):
    def save_objective(self, objective: Objective) -> None: ...
    def get_objective(self, objective_id: UUID) -> Objective | None: ...
    def save_result(self, key: str, result: Result) -> None: ...
    def get_result(self, key: str) -> Result | None: ...
    def save_action(self, request: ExecutionRequest, state: ObjectiveState) -> None: ...
    def get_action(self, key: str) -> ExecutionRequest | None: ...
    def update_action_state(self, key: str, state: ObjectiveState) -> None: ...
    def save_observation(self, key: str, observation: Observation) -> None: ...
    def get_observation(self, key: str) -> Observation | None: ...


class PostgresCursor(Protocol):
    def fetchone(self) -> tuple[object, ...] | None: ...


class PostgresConnection(Protocol):
    def execute(self, query: str, params: tuple[object, ...] = ()) -> PostgresCursor: ...

    def commit(self) -> None: ...


class InMemoryObjectiveStore:
    def __init__(self) -> None:
        self.objectives: dict[UUID, Objective] = {}
        self.results: dict[str, Result] = {}
        self.actions: dict[str, tuple[ExecutionRequest, ObjectiveState]] = {}
        self.observations: dict[str, Observation] = {}

    def save_objective(self, objective: Objective) -> None:
        self.objectives[objective.id] = objective

    def get_objective(self, objective_id: UUID) -> Objective | None:
        return self.objectives.get(objective_id)

    def save_result(self, key: str, result: Result) -> None:
        self.results[key] = result

    def get_result(self, key: str) -> Result | None:
        return self.results.get(key)

    def save_action(self, request: ExecutionRequest, state: ObjectiveState) -> None:
        self.actions.setdefault(request.idempotency_key, (request, state))

    def get_action(self, key: str) -> ExecutionRequest | None:
        row = self.actions.get(key)
        return row[0] if row else None

    def update_action_state(self, key: str, state: ObjectiveState) -> None:
        row = self.actions.get(key)
        if row:
            self.actions[key] = (row[0], state)

    def save_observation(self, key: str, observation: Observation) -> None:
        self.observations[key] = observation

    def get_observation(self, key: str) -> Observation | None:
        return self.observations.get(key)


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
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS actions "
            "(id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL, "
            "state TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS observations "
            "(id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL, payload TEXT NOT NULL)"
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

    def save_action(self, request: ExecutionRequest, state: ObjectiveState) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO actions "
            "(id, idempotency_key, state, payload) VALUES (?, ?, ?, ?)",
            (
                str(request.action_id),
                request.idempotency_key,
                state.value,
                request.model_dump_json(),
            ),
        )
        self.connection.commit()

    def get_action(self, key: str) -> ExecutionRequest | None:
        row = self.connection.execute(
            "SELECT payload FROM actions WHERE idempotency_key = ?", (key,)
        ).fetchone()
        return ExecutionRequest.model_validate_json(row[0]) if row else None

    def update_action_state(self, key: str, state: ObjectiveState) -> None:
        self.connection.execute(
            "UPDATE actions SET state = ? WHERE idempotency_key = ?", (state.value, key)
        )
        self.connection.commit()

    def save_observation(self, key: str, observation: Observation) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO observations (id, idempotency_key, payload) VALUES (?, ?, ?)",
            (str(observation.execution_id), key, observation.model_dump_json()),
        )
        self.connection.commit()

    def get_observation(self, key: str) -> Observation | None:
        row = self.connection.execute(
            "SELECT payload FROM observations WHERE idempotency_key = ?", (key,)
        ).fetchone()
        return Observation.model_validate_json(row[0]) if row else None

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
               ON CONFLICT (id) DO UPDATE SET objective_id = EXCLUDED.objective_id,
                   idempotency_key = EXCLUDED.idempotency_key,
                   state = EXCLUDED.state,
                   evidence = EXCLUDED.evidence, message = EXCLUDED.message""",
            (
                str(result.correlation_id),
                str(result.objective_id),
                key,
                result.state.value,
                json.dumps({**result.evidence, "retryable": result.retryable}, sort_keys=True),
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
        evidence = row[3] if isinstance(row[3], dict) else json.loads(str(row[3]))
        return Result(
            objective_id=UUID(str(row[1])),
            state=ObjectiveState(str(row[2])),
            evidence=evidence,
            message=str(row[4]),
            correlation_id=UUID(str(row[0])),
            retryable=bool(evidence.get("retryable")),
        )

    def get_request_status(
        self, correlation_id: UUID, principal: Principal
    ) -> tuple[UUID, ObjectiveState, str | None, bool] | None:
        cursor = self.connection.execute(
            """SELECT o.id, COALESCE(r.state, o.state), r.message,
                      COALESCE((r.evidence->>'retryable')::boolean, FALSE)
               FROM objectives o
               LEFT JOIN results r ON r.objective_id = o.id AND r.id = %s
               WHERE o.principal_id = %s
                 AND o.vault_id = %s
                 AND o.payload->>'correlation_id' = %s
                 AND (o.space_id IS NULL OR EXISTS (
                     SELECT 1 FROM space_memberships sm
                     WHERE sm.principal_id = %s AND sm.space_id = o.space_id AND sm.active = TRUE
                 ))
               ORDER BY (r.id IS NOT NULL) DESC, o.updated_at DESC
               LIMIT 1""",
            (
                str(correlation_id),
                principal.id,
                principal.vault_id,
                str(correlation_id),
                principal.id,
            ),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return (
            UUID(str(row[0])),
            ObjectiveState(str(row[1])),
            str(row[2]) if row[2] else None,
            bool(row[3]),
        )

    def save_action(self, request: ExecutionRequest, state: ObjectiveState) -> None:
        self.connection.execute(
            """INSERT INTO actions
               (id, objective_id, idempotency_key, capability, state, payload)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (idempotency_key) DO NOTHING""",
            (
                str(request.action_id),
                str(request.objective_id),
                request.idempotency_key,
                request.action.capability,
                state.value,
                request.model_dump_json(),
            ),
        )
        self.connection.commit()

    def get_action(self, key: str) -> ExecutionRequest | None:
        cursor = self.connection.execute(
            "SELECT payload FROM actions WHERE idempotency_key = %s", (key,)
        )
        row = cursor.fetchone()
        return (
            ExecutionRequest.model_validate(row[0])
            if row and isinstance(row[0], dict)
            else (ExecutionRequest.model_validate_json(str(row[0])) if row else None)
        )

    def update_action_state(self, key: str, state: ObjectiveState) -> None:
        self.connection.execute(
            "UPDATE actions SET state = %s WHERE idempotency_key = %s", (state.value, key)
        )
        self.connection.commit()

    def save_observation(self, key: str, observation: Observation) -> None:
        self.connection.execute(
            """INSERT INTO observations (id, action_id, command_succeeded, evidence)
               SELECT %s, id, %s, %s FROM actions WHERE idempotency_key = %s
               ON CONFLICT (id) DO UPDATE SET command_succeeded = EXCLUDED.command_succeeded,
                   evidence = EXCLUDED.evidence, observed_at = now()""",
            (
                str(observation.execution_id),
                observation.command_succeeded,
                json.dumps(observation.evidence, sort_keys=True),
                key,
            ),
        )
        self.connection.commit()

    def get_observation(self, key: str) -> Observation | None:
        cursor = self.connection.execute(
            """SELECT o.id, o.command_succeeded, o.evidence, o.observed_at
               FROM observations o JOIN actions a ON a.id = o.action_id
               WHERE a.idempotency_key = %s""",
            (key,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        evidence = row[2] if isinstance(row[2], dict) else json.loads(str(row[2]))
        return Observation(
            execution_id=UUID(str(row[0])),
            evidence=evidence,
            command_succeeded=bool(row[1]),
            observed_at=cast(datetime, row[3]),
        )
