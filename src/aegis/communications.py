"""Provider-neutral bounded communications reads and outbound message contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


@dataclass(frozen=True)
class Message:
    message_id: str
    sender: str
    subject: str
    body: str
    source: str = "communication"


class CommunicationProvider(Protocol):
    def list_messages(self) -> tuple[Message, ...]: ...


class SendStatus(StrEnum):
    DRAFTED = "DRAFTED"
    SEND_ATTEMPTED = "SEND_ATTEMPTED"
    PROVIDER_ACCEPTED = "PROVIDER_ACCEPTED"
    DELIVERED = "DELIVERED"


@dataclass(frozen=True)
class OutboundMessage:
    target: str
    body: str
    channel: str = "default"
    account: str | None = None


@dataclass(frozen=True)
class SendResult:
    status: SendStatus
    provider_message_id: str | None = None
    detail: str = ""


class CommunicationSendProvider(Protocol):
    """Provider boundary; acceptance is not delivery proof."""

    def send(self, message: OutboundMessage, idempotency_key: str) -> SendResult: ...


@dataclass
class FixtureCommunicationSendProvider:
    """Deterministic provider contract for tests and non-live owner acceptance."""

    sent: list[tuple[OutboundMessage, str]]

    def __init__(self) -> None:
        self.sent = []

    def send(self, message: OutboundMessage, idempotency_key: str) -> SendResult:
        if not any(key == idempotency_key for _, key in self.sent):
            self.sent.append((message, idempotency_key))
        return SendResult(
            status=SendStatus.PROVIDER_ACCEPTED,
            provider_message_id=f"fixture:{idempotency_key}",
            detail="fixture provider accepted the outbound message",
        )


@dataclass(frozen=True)
class FixtureCommunicationProvider:
    messages: tuple[Message, ...] = ()

    def list_messages(self) -> tuple[Message, ...]:
        return self.messages


def communications_evidence(messages: tuple[Message, ...]) -> dict[str, object]:
    return {
        "source": "authorized_communications_fixture",
        "messages": [
            {
                "message_id": message.message_id,
                "sender": message.sender,
                "subject": message.subject,
                "body": message.body[:20_000],
                "source": message.source,
            }
            for message in messages[:50]
        ],
    }
