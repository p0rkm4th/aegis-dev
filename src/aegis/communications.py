"""Provider-neutral bounded communications reads; sending is intentionally separate."""

from __future__ import annotations

from dataclasses import dataclass
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
