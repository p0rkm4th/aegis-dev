from uuid import uuid4

from aegis.communications import (
    FixtureCommunicationProvider,
    Message,
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
