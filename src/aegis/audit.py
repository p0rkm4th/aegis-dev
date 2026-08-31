"""Tamper-evident semantic audit records without raw secrets or transcripts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4


class AuditError(ValueError):
    """An audit event is invalid or the chain has been altered."""


_FORBIDDEN_KEYS = {"password", "token", "secret", "api_key", "access_token", "private_key"}


def _reject_sensitive(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise AuditError(f"sensitive audit field rejected: {key}")
            _reject_sensitive(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_sensitive(child)


@dataclass(frozen=True)
class AuditEvent:
    event_id: UUID
    event_type: str
    principal_id: str
    objective_id: UUID | None
    action_id: UUID | None
    payload: dict[str, Any]
    previous_hash: str
    event_hash: str


class AuditLog:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def append(
        self,
        event_type: str,
        principal_id: str,
        payload: dict[str, Any],
        objective_id: UUID | None = None,
        action_id: UUID | None = None,
    ) -> AuditEvent:
        if not event_type or not principal_id:
            raise AuditError("audit identity and event type are required")
        _reject_sensitive(payload)
        previous = self.events[-1].event_hash if self.events else "GENESIS"
        event_id = uuid4()
        body = {
            "event_id": str(event_id),
            "event_type": event_type,
            "principal_id": principal_id,
            "objective_id": str(objective_id) if objective_id else None,
            "action_id": str(action_id) if action_id else None,
            "payload": payload,
            "previous_hash": previous,
        }
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()
        event_hash = hashlib.sha256(encoded).hexdigest()
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            principal_id=principal_id,
            objective_id=objective_id,
            action_id=action_id,
            payload=payload,
            previous_hash=previous,
            event_hash=event_hash,
        )
        self.events.append(event)
        return event

    def verify(self) -> bool:
        previous = "GENESIS"
        for event in self.events:
            if event.previous_hash != previous:
                return False
            body = {
                "event_id": str(event.event_id),
                "event_type": event.event_type,
                "principal_id": event.principal_id,
                "objective_id": str(event.objective_id) if event.objective_id else None,
                "action_id": str(event.action_id) if event.action_id else None,
                "payload": event.payload,
                "previous_hash": event.previous_hash,
            }
            encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()
            if hashlib.sha256(encoded).hexdigest() != event.event_hash:
                return False
            previous = event.event_hash
        return True


class SqliteAuditLog(AuditLog):
    """Persistent audit log rehearsal for the PostgreSQL audit table."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                principal_id TEXT NOT NULL,
                objective_id TEXT,
                action_id TEXT,
                payload TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL
            )"""
        )
        rows = self.connection.execute(
            """SELECT event_id, event_type, principal_id, objective_id, action_id,
                      payload, previous_hash, event_hash
               FROM audit_events ORDER BY rowid"""
        ).fetchall()
        self.events.extend(
            AuditEvent(
                event_id=UUID(row[0]),
                event_type=row[1],
                principal_id=row[2],
                objective_id=UUID(row[3]) if row[3] else None,
                action_id=UUID(row[4]) if row[4] else None,
                payload=json.loads(row[5]),
                previous_hash=row[6],
                event_hash=row[7],
            )
            for row in rows
        )

    def append(
        self,
        event_type: str,
        principal_id: str,
        payload: dict[str, Any],
        objective_id: UUID | None = None,
        action_id: UUID | None = None,
    ) -> AuditEvent:
        event = super().append(event_type, principal_id, payload, objective_id, action_id)
        self.connection.execute(
            """INSERT INTO audit_events
               (event_id, event_type, principal_id, objective_id, action_id,
                payload, previous_hash, event_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(event.event_id),
                event.event_type,
                event.principal_id,
                str(event.objective_id) if event.objective_id else None,
                str(event.action_id) if event.action_id else None,
                json.dumps(event.payload, sort_keys=True),
                event.previous_hash,
                event.event_hash,
            ),
        )
        self.connection.commit()
        return event

    def close(self) -> None:
        self.connection.close()
