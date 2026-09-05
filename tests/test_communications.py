from uuid import uuid4

from aegis.communications import (
    FixtureCommunicationProvider,
    FixtureCommunicationSendProvider,
    Message,
    OutboundMessage,
    SendStatus,
    communications_evidence,
)
from aegis.contracts import ExecutionRequest, Principal
from aegis.reference_packs import reference_bundles
from aegis.reference_runtime import default_runtime_registry


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
