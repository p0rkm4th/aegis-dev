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
    ObjectiveRequirementProposal,
    ObjectiveSpecProposal,
    ObjectiveState,
    Principal,
    ProposalFailureEvidence,
    ProposalFailureKind,
    ProposedPlan,
    ProposedPlanStep,
    RequestedEffect,
    Result,
    StructuralAnchor,
    StructuralCoverageSignal,
    VerificationContract,
)
from aegis.decoding import StrictDecisionDecoder
from aegis.interaction_cognition import (
    _repair_clarification,
    _scope_plan_by_capability,
    _unresolved_investigation_result,
    decide_fallback,
)
from aegis.interaction_decisions import resolve_fallback_decision


def test_clarification_proposal_gets_bounded_repair_before_returning_blocked() -> None:
    card = ActionCard(
        action=ActionSpec(
            action_id="tasks.complete",
            capability="tasks.complete",
            required_permissions=("tasks.write",),
        ),
        summary="Complete a named task",
        relevance=1,
        argument_keys=("title",),
    )
    calls: list[ModelRequest] = []

    class Provider:
        def decide(self, request: ModelRequest) -> ModelResponse:
            calls.append(request)
            if request.proposal_repair_only:
                return ModelResponse(
                    raw={
                        "kind": "ACTION",
                        "semantic_mode": "ACTION",
                        "knowledge_source": "general_model_knowledge",
                        "action_ref": "tasks.complete",
                        "action_arguments": {"title": "Replace porch bulb"},
                    }
                )
            if request.classification_only:
                return ModelResponse(
                    raw={
                        "semantic_mode": "ACTION",
                        "knowledge_source": "general_model_knowledge",
                    }
                )
            return ModelResponse(
                raw={
                    "kind": "CLARIFY",
                    "semantic_mode": "CLARIFY",
                    "knowledge_source": "general_model_knowledge",
                    "clarification": "Which task?",
                }
            )

    result = decide_fallback(
        SimpleNamespace(
            model_provider=lambda: Provider(),
            reuse_classification_action_reference=False,
            decision_rewriter=None,
            research_answer=None,
        ),
        IntentFrame(
            principal=Principal(id="alice", vault_id="v"),
            utterance="finish that bulb thing",
        ),
        (card,),
        Context(values={"referents": {"those": {"candidates": [{"task_id": "t1"}]}}}),
    )
    assert isinstance(result, Decision)
    assert result.kind is DecisionKind.ACTION
    assert result.action is not None
    assert result.action.arguments["title"] == "Replace porch bulb"
    assert sum(request.proposal_repair_only for request in calls) == 1


def test_optional_repair_provider_is_used_only_for_bounded_repair() -> None:
    card = ActionCard(
        action=ActionSpec(
            action_id="tasks.complete",
            capability="tasks.complete",
            required_permissions=("tasks.write",),
        ),
        summary="Complete a named task",
        relevance=1,
        argument_keys=("title",),
    )
    resident_calls: list[ModelRequest] = []
    repair_calls: list[ModelRequest] = []

    class Resident:
        recovery_events: list[dict[str, object]] = []
        request_mode_counts: dict[str, int] = {}

        def decide(self, request: ModelRequest) -> ModelResponse:
            resident_calls.append(request)
            return ModelResponse(
                raw=(
                    {"semantic_mode": "ACTION", "knowledge_source": "general_model_knowledge"}
                    if request.classification_only
                    else {
                        "kind": "CLARIFY",
                        "semantic_mode": "CLARIFY",
                        "knowledge_source": "general_model_knowledge",
                        "clarification": "Which task?",
                    }
                )
            )

    class Repair:
        recovery_events: list[dict[str, object]] = []
        request_mode_counts: dict[str, int] = {}

        def decide(self, request: ModelRequest) -> ModelResponse:
            repair_calls.append(request)
            return ModelResponse(
                raw={
                    "kind": "ACTION",
                    "semantic_mode": "ACTION",
                    "knowledge_source": "general_model_knowledge",
                    "action_ref": "tasks.complete",
                    "action_arguments": {"title": "Replace porch bulb"},
                }
            )

    resident = Resident()
    repair = Repair()
    result = decide_fallback(
        SimpleNamespace(
            model_provider=lambda: resident,
            repair_model_provider=lambda: repair,
            reuse_classification_action_reference=False,
            decision_rewriter=None,
            research_answer=None,
        ),
        IntentFrame(
            principal=Principal(id="alice", vault_id="v"),
            utterance="finish that bulb thing",
        ),
        (card,),
        Context(values={"referents": {"those": {"candidates": [{"task_id": "t1"}]}}}),
    )
    assert isinstance(result, Decision)
    assert result.kind is DecisionKind.ACTION
    assert len(repair_calls) == 1
    assert all(not request.proposal_repair_only for request in resident_calls)


def test_missing_action_argument_uses_bounded_repair_after_focused_retry() -> None:
    card = ActionCard(
        action=ActionSpec(
            action_id="tasks.create",
            capability="tasks.create",
            required_permissions=("tasks.write",),
        ),
        summary="Create a named task",
        relevance=1,
        argument_keys=("title",),
    )
    calls: list[ModelRequest] = []

    class Provider:
        def decide(self, request: ModelRequest) -> ModelResponse:
            calls.append(request)
            if request.classification_only:
                return ModelResponse(raw={"semantic_mode": "ACTION"})
            if request.proposal_repair_only:
                return ModelResponse(
                    raw={
                        "kind": "ACTION",
                        "semantic_mode": "ACTION",
                        "knowledge_source": "general_model_knowledge",
                        "action_ref": "tasks.create",
                        "action_arguments": {"title": "buy stamps"},
                    }
                )
            if request.action_cards == (card,):
                return ModelResponse(raw={"kind": "ACTION"})
            return ModelResponse(
                raw={
                    "kind": "ACTION",
                    "semantic_mode": "ACTION",
                    "knowledge_source": "general_model_knowledge",
                    "action_ref": "tasks.create",
                    "action_arguments": {},
                }
            )

    result = decide_fallback(
        SimpleNamespace(
            model_provider=lambda: Provider(),
            reuse_classification_action_reference=False,
            decision_rewriter=None,
            research_answer=None,
        ),
        IntentFrame(
            principal=Principal(id="alice", vault_id="v"), utterance="add buy stamps to my tasks"
        ),
        (card,),
        Context(),
    )

    assert isinstance(result, Decision)
    assert result.action is not None
    assert result.action.arguments == {"title": "buy stamps"}
    assert sum(request.proposal_repair_only for request in calls) == 1


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


def test_ordinary_model_answer_cannot_self_label_as_external_evidence() -> None:
    result = resolve_fallback_decision(
        Decision(
            kind=DecisionKind.ANSWER,
            answer="A stable answer",
            semantic_mode="GENERATION",
            knowledge_source="external_evidence",
        ),
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Explain mutexes",
        ),
        Context(),
        (),
    )

    assert isinstance(result, Result)
    assert result.evidence["source_kind"] == "general_model_knowledge"
    assert result.evidence["authoritative"] is False


def test_unknown_consequential_clarification_preserves_open_objective() -> None:
    result = resolve_fallback_decision(
        Decision(
            kind=DecisionKind.CLARIFY,
            clarification="Could you clarify whether this is a task, chore, or event?",
            semantic_mode="CLARIFY",
        ),
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Spin up a Palworld server for the family on easy mode.",
        ),
        Context(),
        (),
    )
    assert isinstance(result, Result)
    assert result.state is ObjectiveState.BLOCKED
    assert result.evidence["objective_open"] is True
    assert result.evidence["resolution"] == "UNSUPPORTED"
    assert "remains open" in result.message


def test_unknown_consequential_clarification_hands_off_to_investigator() -> None:
    calls = []

    def investigate(intent, context, effects):
        calls.append((intent, context, effects))
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message="capability inventory",
            evidence={"authoritative": False, "objective_open": True},
            correlation_id=intent.correlation_id,
        )

    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance="Set up a local Kubernetes cluster.",
    )
    result = resolve_fallback_decision(
        Decision(
            kind=DecisionKind.CLARIFY,
            clarification="Could you clarify whether this is a task, chore, or event?",
            semantic_mode="CLARIFY",
        ),
        intent,
        Context(),
        (),
        investigate,
    )

    assert isinstance(result, Result)
    assert result.message == "capability inventory"
    assert len(calls) == 1
    assert calls[0][2][0].normalized_effect == intent.utterance
    assert calls[0][2][0].source_spans == ((0, len(intent.utterance)),)


def test_known_domain_clarification_is_not_reclassified_as_unknown() -> None:
    result = resolve_fallback_decision(
        Decision(
            kind=DecisionKind.CLARIFY,
            clarification="Which task should I update?",
            semantic_mode="CLARIFY",
        ),
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Set up a task to inspect the server.",
        ),
        Context(),
        (),
    )
    assert isinstance(result, Result)
    assert result.message == "Which task should I update?"
    assert result.evidence == {}


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


def test_plan_fidelity_does_not_accept_matching_plan_that_omits_human_effect() -> None:
    class Provider:
        def decide(self, request: ModelRequest) -> ModelResponse:
            if request.classification_only:
                return ModelResponse(raw={"kind": "ANSWER", "semantic_mode": "ACTION"})
            if request.objective_effect_only:
                return ModelResponse(
                    raw={
                        "effects": [
                            {
                                "effect_text": effect,
                                "source_span": (start, start + len(effect)),
                                "action_ref": action_id,
                                "arguments": {},
                            }
                            for action_id, effect, start in (
                                ("first", "do first", 0),
                                ("second", "second", 10),
                                ("third", "third", 22),
                            )
                        ]
                    }
                )
            return ModelResponse(
                raw=Decision(
                    kind=DecisionKind.PLAN,
                    semantic_mode="ACTION",
                    objective_spec=ObjectiveSpecProposal(
                        requirements=tuple(
                            ObjectiveRequirementProposal(action_ref=action_id)
                            for action_id in ("first", "second")
                        )
                    ),
                    plan=ProposedPlan(
                        steps=tuple(
                            ProposedPlanStep(action_ref=action_id)
                            for action_id in ("first", "second")
                        )
                    ),
                ).model_dump(mode="json")
            )

    cards = tuple(
        ActionCard(
            action=ActionSpec(action_id=action_id, capability=action_id),
            summary=action_id,
            relevance=1,
        )
        for action_id in ("first", "second", "third")
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
            utterance="do first, second, and third",
        ),
        cards,
        Context(),
    )

    assert isinstance(result, Decision)
    assert result.kind is DecisionKind.CLARIFY


def test_structural_repair_rejects_prettier_plan_with_stale_effect_coverage() -> None:
    utterance = "do A, B, and C"
    signal = StructuralCoverageSignal(
        anchors=tuple(
            StructuralAnchor(source_span=span, kind="clause") for span in ((0, 4), (6, 7), (13, 14))
        )
    )
    calls: list[ModelRequest] = []

    class Provider:
        def decide(self, request: ModelRequest) -> ModelResponse:
            calls.append(request)
            if request.classification_only:
                return ModelResponse(raw={"semantic_mode": "ACTION"})
            if request.objective_effect_only:
                # The proposed repair still claims only A+C. The structural
                # validator must reject it, regardless of any plan wording.
                return ModelResponse(
                    raw={
                        "effects": [
                            {
                                "effect_text": "do A",
                                "source_span": (0, 4),
                                "action_ref": "A",
                                "arguments": {},
                            },
                            {
                                "effect_text": "C",
                                "source_span": (13, 14),
                                "action_ref": "C",
                                "arguments": {},
                            },
                        ]
                    }
                )
            return ModelResponse(
                raw=Decision(
                    kind=DecisionKind.PLAN,
                    semantic_mode="ACTION",
                    objective_spec=ObjectiveSpecProposal(
                        requirements=tuple(
                            ObjectiveRequirementProposal(action_ref=ref) for ref in ("A", "C")
                        )
                    ),
                    plan=ProposedPlan(
                        steps=tuple(ProposedPlanStep(action_ref=ref) for ref in ("A", "C"))
                    ),
                ).model_dump(mode="json")
            )

    cards = tuple(
        ActionCard(action=ActionSpec(action_id=ref, capability=ref), summary=ref, relevance=1)
        for ref in ("A", "B", "C")
    )
    result = decide_fallback(
        SimpleNamespace(
            model_provider=lambda: Provider(),
            reuse_classification_action_reference=True,
            decision_rewriter=None,
            research_answer=None,
            structural_parser=lambda _utterance: signal,
        ),
        IntentFrame(principal=Principal(id="alice", vault_id="v"), utterance=utterance),
        cards,
        Context(),
    )

    assert isinstance(result, Decision)
    assert result.kind is DecisionKind.CLARIFY
    assert (
        sum(request.proposal_repair_only and request.objective_effect_only for request in calls)
        == 1
    )
    repair_request = next(
        request
        for request in calls
        if request.proposal_repair_only and request.objective_effect_only
    )
    assert repair_request.proposal_failure is not None
    assert "anchor_count=3" in repair_request.proposal_failure.detail
    assert "effect_count=2" in repair_request.proposal_failure.detail
    assert "unmatched_anchors=[(6, 7)]" in repair_request.proposal_failure.detail
    assert repair_request.proposal_failure.related_source_spans == ((6, 7),)


def test_structural_repair_reenters_fidelity_with_complete_effects() -> None:
    utterance = "do A, B, and C"
    signal = StructuralCoverageSignal(
        anchors=tuple(
            StructuralAnchor(source_span=span, kind="clause") for span in ((0, 4), (6, 7), (13, 14))
        )
    )
    repair_calls = 0

    class Provider:
        def decide(self, request: ModelRequest) -> ModelResponse:
            nonlocal repair_calls
            if request.classification_only:
                return ModelResponse(raw={"semantic_mode": "ACTION"})
            if request.objective_effect_only and request.proposal_repair_only:
                repair_calls += 1
                return ModelResponse(
                    raw={
                        "effects": [
                            {
                                "effect_text": "do A",
                                "source_span": (0, 4),
                                "action_ref": "A",
                                "arguments": {},
                            },
                            {
                                "effect_text": "B",
                                "source_span": (6, 7),
                                "action_ref": "B",
                                "arguments": {},
                            },
                            {
                                "effect_text": "C",
                                "source_span": (13, 14),
                                "action_ref": "C",
                                "arguments": {},
                            },
                        ]
                    }
                )
            if request.proposal_repair_only:
                return ModelResponse(
                    raw=Decision(
                        kind=DecisionKind.PLAN,
                        semantic_mode="ACTION",
                        objective_spec=ObjectiveSpecProposal(
                            requirements=tuple(
                                ObjectiveRequirementProposal(action_ref=ref)
                                for ref in ("A", "B", "C")
                            )
                        ),
                        plan=ProposedPlan(
                            steps=tuple(ProposedPlanStep(action_ref=ref) for ref in ("A", "B", "C"))
                        ),
                    ).model_dump(mode="json")
                )
            if request.objective_effect_only:
                return ModelResponse(
                    raw={
                        "effects": [
                            {
                                "effect_text": "do A",
                                "source_span": (0, 4),
                                "action_ref": "A",
                                "arguments": {},
                            },
                            {
                                "effect_text": "C",
                                "source_span": (13, 14),
                                "action_ref": "C",
                                "arguments": {},
                            },
                        ]
                    }
                )
            if request.objective_interpretation_only:
                return ModelResponse(
                    raw={
                        "requirements": [
                            {"action_ref": ref, "arguments": {}} for ref in ("A", "B", "C")
                        ]
                    }
                )
            return ModelResponse(
                raw=Decision(
                    kind=DecisionKind.PLAN,
                    semantic_mode="ACTION",
                    objective_spec=ObjectiveSpecProposal(
                        requirements=tuple(
                            ObjectiveRequirementProposal(action_ref=ref) for ref in ("A", "C")
                        )
                    ),
                    plan=ProposedPlan(
                        steps=tuple(ProposedPlanStep(action_ref=ref) for ref in ("A", "C"))
                    ),
                ).model_dump(mode="json")
            )

    cards = tuple(
        ActionCard(action=ActionSpec(action_id=ref, capability=ref), summary=ref, relevance=1)
        for ref in ("A", "B", "C")
    )
    result = decide_fallback(
        SimpleNamespace(
            model_provider=lambda: Provider(),
            reuse_classification_action_reference=True,
            decision_rewriter=None,
            research_answer=None,
            structural_parser=lambda _utterance: signal,
        ),
        IntentFrame(principal=Principal(id="alice", vault_id="v"), utterance=utterance),
        cards,
        Context(),
    )

    assert isinstance(result, Decision)
    assert result.kind is DecisionKind.PLAN
    assert result.plan is not None
    assert tuple(step.action_ref for step in result.plan.steps) == ("A", "B", "C")
    assert repair_calls == 1


def test_effects_use_separate_capability_mapping_pass_even_with_volunteered_refs() -> None:
    utterance = "do A and B"
    signal = StructuralCoverageSignal(
        anchors=tuple(
            StructuralAnchor(source_span=span, kind="clause") for span in ((0, 4), (9, 10))
        )
    )
    mapping_calls = 0

    class Provider:
        def decide(self, request: ModelRequest) -> ModelResponse:
            nonlocal mapping_calls
            if request.classification_only:
                return ModelResponse(raw={"semantic_mode": "ACTION"})
            if request.objective_effect_only:
                return ModelResponse(
                    raw={
                        "effects": [
                            {
                                "effect_text": "do A",
                                "source_span": (0, 4),
                                "action_ref": "B",
                                "arguments": {},
                            },
                            {
                                "effect_text": "B",
                                "source_span": (9, 10),
                                "action_ref": "A",
                                "arguments": {},
                            },
                        ]
                    }
                )
            if request.objective_interpretation_only:
                mapping_calls += 1
                assert request.working_set.context.values["grounded_requested_effects"]
                return ModelResponse(
                    raw={
                        "requirements": [
                            {"action_ref": "A", "arguments": {}},
                            {"action_ref": "B", "arguments": {}},
                        ]
                    }
                )
            return ModelResponse(
                raw=Decision(
                    kind=DecisionKind.PLAN,
                    semantic_mode="ACTION",
                    objective_spec=ObjectiveSpecProposal(
                        requirements=tuple(
                            ObjectiveRequirementProposal(action_ref=ref) for ref in ("A", "B")
                        )
                    ),
                    plan=ProposedPlan(
                        steps=tuple(ProposedPlanStep(action_ref=ref) for ref in ("A", "B"))
                    ),
                ).model_dump(mode="json")
            )

    cards = tuple(
        ActionCard(action=ActionSpec(action_id=ref, capability=ref), summary=ref, relevance=1)
        for ref in ("A", "B")
    )
    result = decide_fallback(
        SimpleNamespace(
            model_provider=lambda: Provider(),
            reuse_classification_action_reference=True,
            decision_rewriter=None,
            research_answer=None,
            structural_parser=lambda _utterance: signal,
        ),
        IntentFrame(principal=Principal(id="alice", vault_id="v"), utterance=utterance),
        cards,
        Context(),
    )

    assert isinstance(result, Decision)
    assert result.kind is DecisionKind.PLAN
    assert mapping_calls == 1


def test_unbound_effect_mapping_receives_bounded_candidate_repair() -> None:
    utterance = "do A and B"
    signal = StructuralCoverageSignal(
        anchors=tuple(
            StructuralAnchor(source_span=span, kind="clause") for span in ((0, 4), (9, 10))
        )
    )
    repair_calls = 0

    class Provider:
        recovery_events: list[dict[str, object]] = []

        def decide(self, request: ModelRequest) -> ModelResponse:
            nonlocal repair_calls
            if request.classification_only:
                return ModelResponse(raw={"semantic_mode": "ACTION"})
            if request.objective_effect_only:
                return ModelResponse(
                    raw={
                        "effects": [
                            {"effect_text": "do A", "source_span": (0, 4)},
                            {"effect_text": "B", "source_span": (9, 10)},
                        ]
                    }
                )
            if request.proposal_repair_only and request.objective_interpretation_only:
                repair_calls += 1
                return ModelResponse(
                    raw={
                        "requirements": [
                            {"action_ref": "A", "arguments": {}},
                            {"action_ref": "B", "arguments": {}},
                        ]
                    }
                )
            if request.objective_interpretation_only:
                return ModelResponse(raw={"requirements": [{"action_ref": "A"}]})
            return ModelResponse(
                raw=Decision(
                    kind=DecisionKind.PLAN,
                    semantic_mode="ACTION",
                    objective_spec=ObjectiveSpecProposal(
                        requirements=tuple(
                            ObjectiveRequirementProposal(action_ref=ref) for ref in ("A", "B")
                        )
                    ),
                    plan=ProposedPlan(
                        steps=tuple(ProposedPlanStep(action_ref=ref) for ref in ("A", "B"))
                    ),
                ).model_dump(mode="json")
            )

    cards = tuple(
        ActionCard(action=ActionSpec(action_id=ref, capability=ref), summary=ref, relevance=1)
        for ref in ("A", "B")
    )
    result = decide_fallback(
        SimpleNamespace(
            model_provider=lambda: Provider(),
            reuse_classification_action_reference=True,
            decision_rewriter=None,
            research_answer=None,
            structural_parser=lambda _utterance: signal,
        ),
        IntentFrame(principal=Principal(id="alice", vault_id="v"), utterance=utterance),
        cards,
        Context(),
    )

    assert isinstance(result, Decision)
    assert result.kind is DecisionKind.PLAN
    assert repair_calls == 1


def test_compound_clarification_receives_one_bounded_plan_repair() -> None:
    utterance = "do A and B"
    signal = StructuralCoverageSignal(
        anchors=tuple(
            StructuralAnchor(source_span=span, kind="clause") for span in ((0, 4), (9, 10))
        )
    )
    repair_calls = 0
    repair_requests: list[ModelRequest] = []

    class Provider:
        def decide(self, request: ModelRequest) -> ModelResponse:
            nonlocal repair_calls
            if request.classification_only:
                return ModelResponse(raw={"semantic_mode": "ACTION"})
            if request.proposal_repair_only:
                repair_calls += 1
                repair_requests.append(request)
                return ModelResponse(
                    raw=Decision(
                        kind=DecisionKind.PLAN,
                        semantic_mode="ACTION",
                        objective_spec=ObjectiveSpecProposal(
                            requirements=tuple(
                                ObjectiveRequirementProposal(action_ref=ref) for ref in ("A", "B")
                            )
                        ),
                        plan=ProposedPlan(
                            steps=tuple(ProposedPlanStep(action_ref=ref) for ref in ("A", "B"))
                        ),
                    ).model_dump(mode="json")
                )
            if request.objective_effect_only:
                return ModelResponse(
                    raw={
                        "effects": [
                            {
                                "effect_text": "do A",
                                "source_span": (0, 4),
                                "action_ref": "A",
                                "arguments": {},
                            },
                            {
                                "effect_text": "B",
                                "source_span": (9, 10),
                                "action_ref": "B",
                                "arguments": {},
                            },
                        ]
                    }
                )
            if request.objective_interpretation_only:
                return ModelResponse(
                    raw={
                        "requirements": [{"action_ref": ref, "arguments": {}} for ref in ("A", "B")]
                    }
                )
            return ModelResponse(
                raw={"kind": "CLARIFY", "semantic_mode": "CLARIFY", "clarification": "unclear"}
            )

    cards = tuple(
        ActionCard(action=ActionSpec(action_id=ref, capability=ref), summary=ref, relevance=1)
        for ref in ("A", "B")
    )
    result = decide_fallback(
        SimpleNamespace(
            model_provider=lambda: Provider(),
            reuse_classification_action_reference=True,
            decision_rewriter=None,
            research_answer=None,
            structural_parser=lambda _utterance: signal,
        ),
        IntentFrame(principal=Principal(id="alice", vault_id="v"), utterance=utterance),
        cards,
        Context(),
    )

    assert isinstance(result, Decision)
    assert result.kind is DecisionKind.PLAN
    assert repair_calls == 1
    assert repair_requests[0].proposal_failure is not None
    assert "anchor_count=2" in repair_requests[0].proposal_failure.detail
    assert repair_requests[0].proposal_failure.related_source_spans == ((0, 4), (9, 10))


def test_compound_plan_repair_budget_is_one_before_generic_recovery() -> None:
    calls = 0

    class Provider:
        recovery_events: list[dict[str, object]] = []

        def decide(self, _request: ModelRequest) -> ModelResponse:
            nonlocal calls
            calls += 1
            return ModelResponse(
                raw=Decision(
                    kind=DecisionKind.CLARIFY,
                    semantic_mode="CLARIFY",
                    clarification="still unclear",
                ).model_dump(mode="json")
            )

    result = _repair_clarification(
        Provider(),
        IntentFrame(principal=Principal(id="alice", vault_id="v"), utterance="do A and B"),
        Context(),
        (),
        Decision(kind=DecisionKind.CLARIFY, semantic_mode="CLARIFY", clarification="unclear"),
        "multiple changes",
        ProposalFailureEvidence(kind=ProposalFailureKind.UNACCOUNTED_STRUCTURAL_ANCHOR),
        plans_only=True,
        max_attempts=1,
    )

    assert result.kind is DecisionKind.CLARIFY
    assert calls == 1
    assert len(Provider.recovery_events) == 1


def test_unsupported_effect_remains_open_with_truthful_capability_evidence() -> None:
    utterance = "do the unavailable thing"
    signal = StructuralCoverageSignal(
        anchors=(StructuralAnchor(source_span=(0, len(utterance)), kind="clause"),)
    )

    class Provider:
        def decide(self, request: ModelRequest) -> ModelResponse:
            if request.classification_only:
                return ModelResponse(raw={"semantic_mode": "ACTION"})
            if request.objective_effect_only:
                return ModelResponse(
                    raw={
                        "effects": [
                            {
                                "effect_text": utterance,
                                "source_span": (0, len(utterance)),
                                "action_ref": "unavailable.thing",
                                "arguments": {},
                            }
                        ]
                    }
                )
            return ModelResponse(
                raw=Decision(
                    kind=DecisionKind.PLAN,
                    semantic_mode="ACTION",
                    objective_spec=ObjectiveSpecProposal(
                        requirements=(ObjectiveRequirementProposal(action_ref="available"),)
                    ),
                    plan=ProposedPlan(
                        steps=(
                            ProposedPlanStep(action_ref="available"),
                            ProposedPlanStep(action_ref="available-2"),
                        )
                    ),
                ).model_dump(mode="json")
            )

    result = decide_fallback(
        SimpleNamespace(
            model_provider=lambda: Provider(),
            reuse_classification_action_reference=True,
            decision_rewriter=None,
            research_answer=None,
            structural_parser=lambda _utterance: signal,
        ),
        IntentFrame(principal=Principal(id="alice", vault_id="v"), utterance=utterance),
        (
            ActionCard(
                action=ActionSpec(action_id="available", capability="available"),
                summary="Available action",
                relevance=1,
            ),
            ActionCard(
                action=ActionSpec(action_id="available-2", capability="available-2"),
                summary="Second available action",
                relevance=1,
            ),
        ),
        Context(),
    )

    assert isinstance(result, Result)
    assert result.state is ObjectiveState.BLOCKED
    assert result.evidence["objective_open"] is True
    requirement = result.evidence["unsatisfied_requirements"][0]
    assert requirement["resolution"] == "UNSUPPORTED"
    assert requirement["normalized_effect"] == utterance


def test_unresolved_investigation_requires_explicit_non_authoritative_result() -> None:
    intent = IntentFrame(principal=Principal(id="alice", vault_id="v"), utterance="set up X")
    effect = RequestedEffect(source_spans=((0, 9),), normalized_effect="set up X")

    for evidence in ({}, {"authoritative": True}):
        result = _unresolved_investigation_result(
            SimpleNamespace(
                unresolved_requirement_investigator=lambda *_args, evidence=evidence: Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.BLOCKED,
                    message="investigation",
                    evidence=evidence,
                    correlation_id=intent.correlation_id,
                )
            ),
            intent,
            Context(),
            (effect,),
        )
        assert result is None

    result = _unresolved_investigation_result(
        SimpleNamespace(
            unresolved_requirement_investigator=lambda *_args: Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message="investigation",
                evidence={"authoritative": False},
                correlation_id=intent.correlation_id,
            )
        ),
        intent,
        Context(),
        (effect,),
    )
    assert result is not None
    assert result.evidence["objective_open"] is True
    assert result.evidence["unsatisfied_requirements"][0]["resolution"] == "UNSUPPORTED"


def test_plan_fidelity_provider_failure_fails_closed() -> None:
    class Provider:
        def decide(self, request: ModelRequest) -> ModelResponse:
            if request.classification_only:
                return ModelResponse(raw={"kind": "ANSWER", "semantic_mode": "ACTION"})
            if request.objective_effect_only:
                raise TimeoutError("fidelity timeout")
            return ModelResponse(
                raw=Decision(
                    kind=DecisionKind.PLAN,
                    semantic_mode="ACTION",
                    objective_spec=ObjectiveSpecProposal(
                        requirements=(
                            ObjectiveRequirementProposal(action_ref="first"),
                            ObjectiveRequirementProposal(action_ref="second"),
                        )
                    ),
                    plan=ProposedPlan(
                        steps=(
                            ProposedPlanStep(action_ref="first"),
                            ProposedPlanStep(action_ref="second"),
                        )
                    ),
                ).model_dump(mode="json")
            )

    cards = tuple(
        ActionCard(
            action=ActionSpec(action_id=action_id, capability=action_id),
            summary=action_id,
            relevance=1,
        )
        for action_id in ("first", "second")
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
            utterance="do both",
        ),
        cards,
        Context(),
    )

    assert isinstance(result, Result)
    assert result.state is ObjectiveState.FAILED
    assert result.retryable is True


def test_plan_without_primary_objective_spec_cannot_use_effects_as_objective() -> None:
    class Provider:
        def decide(self, request: ModelRequest) -> ModelResponse:
            if request.classification_only:
                return ModelResponse(raw={"kind": "ANSWER", "semantic_mode": "ACTION"})
            if request.objective_effect_only:
                return ModelResponse(
                    raw={
                        "effects": [
                            {
                                "effect_text": "do first and second",
                                "source_span": (0, 19),
                                "action_ref": "first",
                                "arguments": {},
                            }
                        ]
                    }
                )
            return ModelResponse(
                raw=Decision(
                    kind=DecisionKind.PLAN,
                    semantic_mode="ACTION",
                    plan=ProposedPlan(
                        steps=(
                            ProposedPlanStep(action_ref="first"),
                            ProposedPlanStep(action_ref="second"),
                        )
                    ),
                ).model_dump(mode="json")
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
            utterance="do first and second",
        ),
        tuple(
            ActionCard(
                action=ActionSpec(action_id=action_id, capability=action_id),
                summary=action_id,
                relevance=1,
            )
            for action_id in ("first", "second")
        ),
        Context(),
    )

    assert isinstance(result, Decision)
    assert result.kind is DecisionKind.CLARIFY


def test_compound_action_cannot_be_reduced_to_one_consequential_action() -> None:
    class Provider:
        def decide(self, request: ModelRequest) -> ModelResponse:
            if request.classification_only:
                return ModelResponse(raw={"kind": "ANSWER", "semantic_mode": "ACTION"})
            return ModelResponse(
                raw=Decision(
                    kind=DecisionKind.ACTION,
                    semantic_mode="ACTION",
                    action_ref="first",
                    action_arguments={},
                ).model_dump(mode="json")
            )

    cards = tuple(
        ActionCard(
            action=ActionSpec(
                action_id=action_id,
                capability=action_id,
                required_permissions=(f"{action_id}.write",),
            ),
            summary=action_id,
            relevance=1,
        )
        for action_id in ("first", "second")
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
            utterance="add a task and send a message",
        ),
        cards,
        Context(),
    )

    assert isinstance(result, Decision)
    assert result.kind is DecisionKind.CLARIFY


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
