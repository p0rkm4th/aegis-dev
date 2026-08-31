"""Ambient notification and background-work contracts.

Ambient surfaces can propose and deliver context, but they do not grant action
authority. Any proposed mutation still returns to the Core policy/execution
pipeline.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from threading import Lock
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import Field

from .contracts import ActionSpec, StrictModel


class Notification(StrictModel):
    notification_id: UUID = Field(default_factory=uuid4)
    recipient_ids: tuple[str, ...] = Field(min_length=1)
    text: str = Field(min_length=1)
    correlation_id: UUID


class BackgroundTask(StrictModel):
    task_id: UUID = Field(default_factory=uuid4)
    run_at: datetime
    task_type: str = Field(min_length=1)
    payload: dict[str, Any] = {}
    idempotency_key: str = Field(min_length=1)


class AmbientSuggestion(StrictModel):
    suggestion_id: UUID = Field(default_factory=uuid4)
    reason: str = Field(min_length=1)
    text: str = Field(min_length=1)
    proposed_action: ActionSpec | None = None


class AmbientPlatform(Protocol):
    def deliver_notification(self, notification: Notification) -> None: ...

    def schedule_background(self, task: BackgroundTask) -> None: ...

    def cancel_background(self, task: BackgroundTask) -> None: ...


class AmbientState(Protocol):
    def claim(self, key: str) -> bool: ...

    def release(self, key: str) -> None: ...


class InMemoryAmbientState:
    """Replaceable idempotency state for tests and single-process operation."""

    def __init__(self) -> None:
        self._claimed: set[str] = set()
        self._lock = Lock()

    def claim(self, key: str) -> bool:
        with self._lock:
            if key in self._claimed:
                return False
            self._claimed.add(key)
            return True

    def release(self, key: str) -> None:
        with self._lock:
            self._claimed.discard(key)


class SqliteAmbientState:
    """Restart-safe idempotency claims for ambient delivery and scheduling."""

    def __init__(self, path: str) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS ambient_claims (claim_key TEXT PRIMARY KEY NOT NULL)"
        )
        self.connection.commit()

    def claim(self, key: str) -> bool:
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO ambient_claims (claim_key) VALUES (?)", (key,)
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def release(self, key: str) -> None:
        self.connection.execute("DELETE FROM ambient_claims WHERE claim_key = ?", (key,))
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


class AmbientGateway(Protocol):
    """Narrow seam implemented by the OpenClaw Gateway/RPC adapter."""

    def notify(self, params: dict[str, Any]) -> None: ...

    def schedule(self, params: dict[str, Any]) -> None: ...

    def cancel(self, params: dict[str, Any]) -> None: ...


class OpenClawAmbientPlatform:
    """Map ambient contracts to OpenClaw without owning scheduling or delivery."""

    def __init__(self, gateway: AmbientGateway) -> None:
        self.gateway = gateway

    def deliver_notification(self, notification: Notification) -> None:
        self.gateway.notify(notification.model_dump(mode="json"))

    def schedule_background(self, task: BackgroundTask) -> None:
        self.gateway.schedule(task.model_dump(mode="json"))

    def cancel_background(self, task: BackgroundTask) -> None:
        self.gateway.cancel({"task_id": str(task.task_id), "idempotency_key": task.idempotency_key})


class AmbientPolicy(Protocol):
    def allow_notification(self, notification: Notification) -> bool: ...

    def allow_background_task(self, task: BackgroundTask) -> bool: ...


class AmbientService:
    """Thin platform adapter; policy remains a required external boundary."""

    def __init__(
        self,
        platform: AmbientPlatform,
        policy: AmbientPolicy,
        state: AmbientState | None = None,
    ) -> None:
        self.platform = platform
        self.policy = policy
        self.state = state or InMemoryAmbientState()

    def deliver(self, notification: Notification) -> bool:
        if not self.policy.allow_notification(notification):
            raise PermissionError("ambient notification denied by policy")
        if not self.state.claim(f"notification:{notification.notification_id}"):
            return False
        try:
            self.platform.deliver_notification(notification)
        except Exception:
            self.state.release(f"notification:{notification.notification_id}")
            raise
        return True

    def schedule(self, task: BackgroundTask) -> bool:
        if not self.policy.allow_background_task(task):
            raise PermissionError("ambient background task denied by policy")
        if not self.state.claim(f"background:{task.idempotency_key}"):
            return False
        try:
            self.platform.schedule_background(task)
        except Exception:
            self.state.release(f"background:{task.idempotency_key}")
            raise
        return True

    def cancel(self, task: BackgroundTask) -> None:
        self.platform.cancel_background(task)

    @staticmethod
    def propose(
        reason: str, text: str, proposed_action: ActionSpec | None = None
    ) -> AmbientSuggestion:
        """Create a suggestion only; it never executes or authorizes its action."""
        return AmbientSuggestion(reason=reason, text=text, proposed_action=proposed_action)
