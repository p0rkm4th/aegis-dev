"""Provider-neutral bounded communications reads and outbound message contracts."""

from __future__ import annotations

import json
import os
import subprocess
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


def configured_communication_targets() -> frozenset[tuple[str, str, str | None]] | None:
    """Load the optional owner-approved target boundary.

    An absent setting preserves the explicit-grounding contract. When present,
    every outbound target must match one exact ``target``/``channel``/``account``
    tuple; malformed configuration fails closed rather than widening authority.
    """

    raw = os.environ.get("AEGIS_APPROVED_COMMUNICATION_TARGETS")
    if raw is None:
        return None
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("approved communication targets must be valid JSON") from exc
    if not isinstance(values, list) or not values or len(values) > 20:
        raise ValueError("approved communication targets must contain 1-20 entries")
    targets: set[tuple[str, str, str | None]] = set()
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("approved communication target must be an object")
        target = value.get("target")
        channel = value.get("channel", "default")
        account = value.get("account")
        if (
            not isinstance(target, str)
            or not target.strip()
            or not isinstance(channel, str)
            or not channel.strip()
            or (account is not None and not isinstance(account, str))
        ):
            raise ValueError("approved communication target fields are invalid")
        targets.add((target.strip(), channel.strip(), account))
    return frozenset(targets)


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


class OpenClawMessageCommand(Protocol):
    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]: ...


class OpenClawCliCommunicationSendProvider:
    """Adapt OpenClaw's explicit ``message send`` command to the Core port.

    OpenClaw owns channel transport and returns provider evidence. AEGIS owns
    the grounded target, authorization, and the distinction between provider
    acceptance and delivery. The command is argv-based: no shell or inherited
    command string is involved.
    """

    def __init__(
        self,
        executable: str = "openclaw",
        runner: OpenClawMessageCommand | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.executable = executable
        self.runner = runner or self._run
        self.timeout_seconds = timeout_seconds
        self._accepted: dict[str, SendResult] = {}

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env={},
        )

    def send(self, message: OutboundMessage, idempotency_key: str) -> SendResult:
        if cached := self._accepted.get(idempotency_key):
            return cached
        args = [
            self.executable,
            "message",
            "send",
            "--channel",
            message.channel,
            "--target",
            message.target,
            "--message",
            message.body,
            "--json",
        ]
        if message.account:
            args.extend(("--account", message.account))
        try:
            completed = self.runner(args)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return SendResult(
                status=SendStatus.SEND_ATTEMPTED,
                detail=f"OpenClaw message send unavailable: {type(exc).__name__}",
            )
        if completed.returncode != 0:
            return SendResult(
                status=SendStatus.SEND_ATTEMPTED,
                detail="OpenClaw message send was rejected by the provider",
            )
        try:
            payload = json.loads(completed.stdout)
        except (TypeError, json.JSONDecodeError):
            return SendResult(
                status=SendStatus.SEND_ATTEMPTED,
                detail="OpenClaw returned no structured message acceptance",
            )
        if not isinstance(payload, dict):
            return SendResult(
                status=SendStatus.SEND_ATTEMPTED,
                detail="OpenClaw returned an invalid message result",
            )
        provider_message_id = payload.get("messageId") or payload.get("message_id")
        if not isinstance(provider_message_id, str) or not provider_message_id.strip():
            return SendResult(
                status=SendStatus.SEND_ATTEMPTED,
                detail="OpenClaw did not return a provider message id",
            )
        result = SendResult(
            status=SendStatus.PROVIDER_ACCEPTED,
            provider_message_id=provider_message_id,
            detail="OpenClaw accepted the outbound message; delivery is not independently proven",
        )
        self._accepted[idempotency_key] = result
        return result


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
