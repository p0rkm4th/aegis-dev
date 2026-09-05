from uuid import uuid4

import pytest

from aegis.communications import (
    FixtureCommunicationProvider,
    FixtureCommunicationSendProvider,
    Message,
    OpenClawCliCommunicationSendProvider,
    OutboundMessage,
    SendStatus,
    communications_evidence,
    configured_communication_targets,
)
from aegis.contracts import ActionSpec, ExecutionRequest, Principal
from aegis.reference_packs import CommunicationsSendExecutor, reference_bundles
from aegis.reference_runtime import default_runtime_registry


def test_approved_communication_targets_are_exact_and_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv(
        "AEGIS_APPROVED_COMMUNICATION_TARGETS",
        '[{"target":"sms:+15555550123","channel":"sms","account":"personal"}]',
    )
    assert configured_communication_targets() == frozenset(
        {("sms:+15555550123", "sms", "personal")}
    )
    monkeypatch.setenv("AEGIS_APPROVED_COMMUNICATION_TARGETS", "not-json")
    with pytest.raises(ValueError):
        configured_communication_targets()


def test_send_executor_rejects_target_outside_approved_boundary() -> None:
    provider = FixtureCommunicationSendProvider()
    executor = CommunicationsSendExecutor(
        provider,
        frozenset({("sms:+15555550123", "sms", "personal")}),
    )
    action = {
        "target": "sms:+15555550124",
        "body": "Milk",
        "channel": "sms",
        "account": "personal",
    }
    observation = executor.execute(
        ExecutionRequest(
            objective_id=uuid4(),
            action_id=uuid4(),
            action=ActionSpec(
                action_id="communications.messages.send",
                capability="communications.messages.send",
                arguments=action,
            ),
            idempotency_key="blocked-target",
        )
    )
    assert observation.command_succeeded is False
    assert observation.evidence["communication_send"] == "target_not_approved"
    assert provider.sent == []


def test_fixture_communications_are_bounded_and_read_only() -> None:
    evidence = communications_evidence(
        FixtureCommunicationProvider(
            (Message("m-1", "Ada", "Hello", "Read this."),)
        ).list_messages()
    )
    assert evidence["source"] == "authorized_communications_fixture"
    assert evidence["messages"][0]["sender"] == "Ada"


def test_communications_pack_uses_generic_verified_read_runtime() -> None:
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "communications.messages.list"
    )
    runtime = default_runtime_registry(lambda: None).resolve(
        card, None, Principal(id="alice", vault_id="vault")
    )
    observation = runtime.executor.execute(
        ExecutionRequest(
            objective_id=uuid4(),
            action_id=uuid4(),
            action=card.action,
            idempotency_key="messages-1",
        )
    )
    assert observation.command_succeeded is True
    assert runtime.verifier.verify(observation, card.action.verification).verified is True


def test_fixture_send_is_idempotent_and_reports_acceptance_not_delivery() -> None:
    provider = FixtureCommunicationSendProvider()
    message = OutboundMessage(target="scotty", body="Milk", channel="sms")
    first = provider.send(message, "send-1")
    second = provider.send(message, "send-1")
    assert first.status is SendStatus.PROVIDER_ACCEPTED
    assert second.provider_message_id == first.provider_message_id
    assert len(provider.sent) == 1


def test_openclaw_cli_provider_adapts_explicit_message_send_without_claiming_delivery() -> None:
    calls: list[list[str]] = []

    def run(args: list[str]):
        import subprocess

        calls.append(args)
        return subprocess.CompletedProcess(args, 0, '{"messageId":"openclaw-42"}\n', "")

    provider = OpenClawCliCommunicationSendProvider(executable="/usr/bin/openclaw", runner=run)
    message = OutboundMessage(
        target="sms:+15555550123", body="Milk", channel="sms", account="personal"
    )
    result = provider.send(message, "send-42")
    replay = provider.send(message, "send-42")

    assert result.status is SendStatus.PROVIDER_ACCEPTED
    assert result.provider_message_id == "openclaw-42"
    assert "delivery is not independently proven" in result.detail
    assert replay == result
    assert calls == [
        [
            "/usr/bin/openclaw",
            "message",
            "send",
            "--channel",
            "sms",
            "--target",
            "sms:+15555550123",
            "--message",
            "Milk",
            "--json",
            "--account",
            "personal",
        ]
    ]


def test_openclaw_cli_provider_downgrades_missing_acceptance() -> None:
    def run(args: list[str]):
        import subprocess

        return subprocess.CompletedProcess(args, 0, "{}", "")

    provider = OpenClawCliCommunicationSendProvider(runner=run)
    result = provider.send(OutboundMessage(target="scotty", body="Milk"), "send-43")

    assert result.status is SendStatus.SEND_ATTEMPTED
    assert result.provider_message_id is None


def test_communications_send_pack_uses_explicit_provider_contract() -> None:
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "communications.messages.send"
    )
    runtime = default_runtime_registry(lambda: None).resolve(
        card, None, Principal(id="alice", vault_id="vault")
    )
    action = card.action.model_copy(
        update={
            "arguments": {
                "target": "scotty",
                "body": "Milk",
                "channel": "sms",
                "account": "household",
            }
        }
    )
    observation = runtime.executor.execute(
        ExecutionRequest(
            objective_id=uuid4(), action_id=uuid4(), action=action, idempotency_key="send-1"
        )
    )
    result = runtime.verifier.verify(observation, card.action.verification)
    assert result.verified is True
    assert result.evidence["communication_send_status"] == "PROVIDER_ACCEPTED"
    assert result.evidence["independent_delivery"] is False
