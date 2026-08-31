"""Ambient notification and background-work contracts.

Ambient surfaces can propose and deliver context, but they do not grant action
authority. Any proposed mutation still returns to the Core policy/execution
pipeline.
"""

from __future__ import annotations

from datetime import datetime
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


class AmbientPolicy(Protocol):
    def allow_notification(self, notification: Notification) -> bool: ...

    def allow_background_task(self, task: BackgroundTask) -> bool: ...


class AmbientService:
    """Thin platform adapter; policy remains a required external boundary."""

    def __init__(self, platform: AmbientPlatform, policy: AmbientPolicy) -> None:
        self.platform = platform
        self.policy = policy

    def deliver(self, notification: Notification) -> None:
        if not self.policy.allow_notification(notification):
            raise PermissionError("ambient notification denied by policy")
        self.platform.deliver_notification(notification)

    def schedule(self, task: BackgroundTask) -> None:
        if not self.policy.allow_background_task(task):
            raise PermissionError("ambient background task denied by policy")
        self.platform.schedule_background(task)

    @staticmethod
    def propose(
        reason: str, text: str, proposed_action: ActionSpec | None = None
    ) -> AmbientSuggestion:
        """Create a suggestion only; it never executes or authorizes its action."""
        return AmbientSuggestion(reason=reason, text=text, proposed_action=proposed_action)
