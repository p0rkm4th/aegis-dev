from uuid import uuid4

from aegis.contracts import (
    ActionCard,
    ActionSpec,
    Context,
    Decision,
    DecisionKind,
    IntentFrame,
    ModelRequest,
    ModelResponse,
    Principal,
    ProposedPlan,
    ProposedPlanStep,
    VerificationContract,
)
from aegis.decoding import StrictDecisionDecoder
from aegis.interaction_cognition import _scope_plan_by_capability
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


def test_scoped_plan_decomposition_collects_independent_candidate_actions() -> None:
    cards = tuple(
        ActionCard(
            action=ActionSpec(
                action_id=action_id,
                capability=f"{action_id}.write",
                required_permissions=(f"{action_id}.write",),
                verification=VerificationContract(kind="readback"),
            ),
            summary=action_id,
            relevance=1,
            argument_keys=("title",),
        )
        for action_id in ("tasks.create", "tasks.events.create", "tasks.chores.create")
    )

    class Provider:
        def decide(self, request: ModelRequest) -> ModelResponse:
            card = request.action_cards[0]
            return ModelResponse(
                raw={
                    "kind": "ACTION",
                    "action_ref": card.action.action_id,
                    "action_arguments": {"title": card.action.action_id},
                    "semantic_mode": "ACTION",
                }
            )

    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance="do three separate things",
        correlation_id=uuid4(),
    )
    proposal = Decision(
        kind=DecisionKind.PLAN,
        semantic_mode="ACTION",
        plan=ProposedPlan(
            steps=(
                ProposedPlanStep(action_ref="tasks.create", arguments={"title": "task"}),
                ProposedPlanStep(
                    action_ref="tasks.chores.create", arguments={"title": "chore"}, depends_on=(0,)
                ),
            )
        ),
    )

    result = _scope_plan_by_capability(
        Provider(), StrictDecisionDecoder(), intent, cards, Context(), proposal
    )

    assert result.plan is not None
    assert [step.action_ref for step in result.plan.steps] == [
        "tasks.create",
        "tasks.chores.create",
        "tasks.events.create",
    ]
    assert result.plan.steps[0].depends_on == ()
    assert result.plan.steps[1].depends_on == (0,)
    assert result.plan.steps[2].depends_on == (1,)
