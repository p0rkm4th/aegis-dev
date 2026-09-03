from types import SimpleNamespace
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
    Result,
    VerificationContract,
)
from aegis.decoding import StrictDecisionDecoder
from aegis.interaction_cognition import _scope_plan_by_capability, decide_fallback
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


def test_fresh_source_request_fails_truthfully_without_research_provider() -> None:
    class Provider:
        def decide(self, _request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                raw={
                    "kind": "ANSWER",
                    "answer": "The latest answer would need verification.",
                    "semantic_mode": "GENERATION",
                    "knowledge_source": "external_evidence",
                }
            )

    result = decide_fallback(
        SimpleNamespace(
            model_provider=lambda: Provider(),
            reuse_classification_action_reference=True,
            decision_rewriter=None,
            research_answer=None,
        ),
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What is the latest release?",
        ),
        (),
        Context(),
    )

    assert result is not None
    assert result.state.value == "failed"
    assert "couldn't verify current" in result.message
    assert result.evidence["authoritative"] is False


def test_fresh_source_request_uses_answer_only_research_callback() -> None:
    calls: list[tuple[str, str]] = []

    class Provider:
        def decide(self, _request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                raw={
                    "kind": "ANSWER",
                    "answer": "unused fallback",
                    "semantic_mode": "GENERATION",
                    "knowledge_source": "external_evidence",
                }
            )

    def research_answer(intent: IntentFrame, _context: Context, source_kind: str) -> Result:
        calls.append((intent.utterance, source_kind))
        return Result(
            objective_id=uuid4(),
            state="completed",
            message="verified from bounded evidence",
            evidence={"source_kind": source_kind, "authoritative": False},
            correlation_id=intent.correlation_id,
        )

    result = decide_fallback(
        SimpleNamespace(
            model_provider=lambda: Provider(),
            reuse_classification_action_reference=True,
            decision_rewriter=None,
            research_answer=research_answer,
        ),
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What changed in the latest release?",
        ),
        (),
        Context(values={"private": "never sent to search"}),
    )

    assert isinstance(result, Result)
    assert result.message == "verified from bounded evidence"
    assert calls == [("What changed in the latest release?", "external_evidence")]


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
        utterance="do one thing and another thing and one more thing",
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


def test_scoped_plan_does_not_copy_optional_arguments_into_existing_steps() -> None:
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
            argument_keys=("title", "due_at"),
        )
        for action_id in ("tasks.create", "tasks.chores.create", "tasks.events.create")
    )

    class Provider:
        def decide(self, request: ModelRequest) -> ModelResponse:
            card = request.action_cards[0]
            return ModelResponse(
                raw={
                    "kind": "ACTION",
                    "action_ref": card.action.action_id,
                    "action_arguments": {
                        "title": card.action.action_id,
                        "due_at": "next Friday",
                    },
                    "semantic_mode": "ACTION",
                }
            )

    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance="do one thing and another thing and one more thing",
        correlation_id=uuid4(),
    )
    proposal = Decision(
        kind=DecisionKind.PLAN,
        semantic_mode="ACTION",
        plan=ProposedPlan(
            steps=(
                ProposedPlanStep(action_ref="tasks.create", arguments={"title": "task"}),
                ProposedPlanStep(action_ref="tasks.chores.create", arguments={"title": "chore"}),
            )
        ),
    )

    result = _scope_plan_by_capability(
        Provider(), StrictDecisionDecoder(), intent, cards, Context(), proposal
    )

    assert result.plan is not None
    assert "due_at" not in result.plan.steps[0].arguments
    assert "due_at" not in result.plan.steps[1].arguments


def test_scoped_plan_decomposition_fails_closed_when_no_capability_is_selected() -> None:
    card = ActionCard(
        action=ActionSpec(action_id="tasks.create", capability="tasks.write"),
        summary="Create a task",
        relevance=1,
        argument_keys=("title",),
    )
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance="do one thing and another thing and one more thing",
        correlation_id=uuid4(),
    )
    proposal = Decision(
        kind=DecisionKind.PLAN,
        semantic_mode="ACTION",
        plan=ProposedPlan(
            steps=(
                ProposedPlanStep(action_ref="tasks.create", arguments={"title": "task"}),
                ProposedPlanStep(action_ref="tasks.create", arguments={"title": "other"}),
            )
        ),
    )

    class Provider:
        def decide(self, _request: ModelRequest) -> ModelResponse:
            return ModelResponse(raw={"kind": "ANSWER", "answer": "not this capability"})

    other = card.model_copy(
        update={"action": ActionSpec(action_id="chores.create", capability="chores.write")}
    )
    third = card.model_copy(
        update={"action": ActionSpec(action_id="events.create", capability="events.write")}
    )
    result = _scope_plan_by_capability(
        Provider(), StrictDecisionDecoder(), intent, (card, other, third), Context(), proposal
    )

    assert result.kind is DecisionKind.CLARIFY
    assert result.clarification is not None
