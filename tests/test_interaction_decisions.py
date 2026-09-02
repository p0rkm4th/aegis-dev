from uuid import uuid4

from aegis.contracts import (
    ActionCard,
    ActionSpec,
    Context,
    Decision,
    DecisionKind,
    IntentFrame,
    Principal,
)
from aegis.interaction_decisions import resolve_fallback_decision


def test_action_resolution_uses_the_supplied_bounded_working_set() -> None:
    card = ActionCard(
        action=ActionSpec(action_id="weather.note.write", capability="weather.write"),
        summary="Record a weather note",
        relevance=1,
    )
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance="save a weather note",
        correlation_id=uuid4(),
    )
    result = resolve_fallback_decision(
        Decision(
            kind=DecisionKind.ACTION,
            action_ref=card.action.action_id,
            action=card.action,
        ),
        intent,
        Context(),
        (card,),
    )

    assert result == card
