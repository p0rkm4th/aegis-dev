from types import SimpleNamespace
from uuid import uuid4

import pytest

from aegis.contracts import (
    ActionCard,
    ActionSpec,
    Context,
    Decision,
    DecisionKind,
    IntentFrame,
    ModelResponse,
    ObjectiveState,
    Principal,
)
from aegis.interaction_cognition import decide_fallback
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


def _plan_review_dependencies(responses: list[dict[str, object]]) -> SimpleNamespace:
    class Provider:
        def __init__(self) -> None:
            self.requests: list[object] = []

        def decide(self, request: object) -> ModelResponse:
            self.requests.append(request)
            return ModelResponse(raw=responses.pop(0))

    provider = Provider()
    dependencies = SimpleNamespace(
        model_provider=lambda: provider,
        reuse_classification_action_reference=True,
        decision_rewriter=None,
    )
    dependencies.provider = provider
    return dependencies


def _plan_probe_inputs() -> tuple[IntentFrame, tuple[ActionCard, ...]]:
    principal = Principal(id="alice", vault_id="alice-vault")
    intent = IntentFrame(
        principal=principal,
        utterance="create a task and a chore",
        correlation_id=uuid4(),
    )
    cards = tuple(
        ActionCard(
            action=ActionSpec(
                action_id=action_id,
                capability=action_id,
                required_permissions=("tasks.write",),
            ),
            summary=action_id,
            relevance=1,
            argument_keys=("title",),
        )
        for action_id in ("tasks.create", "tasks.chores.create")
    )
    return intent, cards


def test_plan_coverage_review_blocks_incomplete_proposal_before_execution() -> None:
    dependencies = _plan_review_dependencies(
        [
            {"kind": "ANSWER", "semantic_mode": "ACTION", "answer": "state change"},
            {
                "kind": "PLAN",
                "semantic_mode": "ACTION",
                "plan": {
                    "steps": [
                        {"action_ref": "tasks.create", "arguments": {"title": "task"}},
                        {
                            "action_ref": "tasks.chores.create",
                            "arguments": {"title": "chore"},
                        },
                    ]
                },
            },
            {
                "kind": "ANSWER",
                "plan_complete": False,
                "answer": "incomplete",
            },
        ]
    )
    intent, cards = _plan_probe_inputs()

    result = decide_fallback(dependencies, intent, cards, Context())

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "No action was executed" in result.message
    assert dependencies.provider.requests[2].plan_coverage_review is True


def test_plan_coverage_review_allows_only_explicit_complete_review() -> None:
    dependencies = _plan_review_dependencies(
        [
            {"kind": "ANSWER", "semantic_mode": "ACTION", "answer": "state change"},
            {
                "kind": "PLAN",
                "semantic_mode": "ACTION",
                "plan": {
                    "steps": [
                        {"action_ref": "tasks.create", "arguments": {"title": "task"}},
                        {
                            "action_ref": "tasks.chores.create",
                            "arguments": {"title": "chore"},
                        },
                    ]
                },
            },
            {"kind": "ANSWER", "plan_complete": True, "answer": "complete"},
        ]
    )
    intent, cards = _plan_probe_inputs()

    result = decide_fallback(dependencies, intent, cards, Context())

    assert isinstance(result, Decision)
    assert result.kind is DecisionKind.PLAN


def test_decoder_rejects_plan_coverage_outside_review_mode() -> None:
    from aegis.decoding import InvalidDecision, StrictDecisionDecoder

    response = SimpleNamespace(raw={"kind": "ANSWER", "answer": "complete", "plan_complete": True})

    with pytest.raises(InvalidDecision, match="only valid during a coverage review"):
        StrictDecisionDecoder().decode(response, (), allow_argument_proposals=False)
