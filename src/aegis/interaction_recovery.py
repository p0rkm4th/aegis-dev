"""Bounded recovery for malformed cognition proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .contracts import (
    ActionCard,
    ActionSpec,
    ClarificationAmbiguityType,
    ClarificationRecoveryOutcome,
    ClarificationRecoveryProposal,
    Context,
    Decision,
    DecisionKind,
    IntentFrame,
    ModelRequest,
    ObjectiveState,
    Result,
    WorkingSet,
)
from .decoding import InvalidDecision, StrictDecisionDecoder
from .interaction_context import grounded_context_answer
from .utterance import is_mutation_request, is_question_request


@dataclass(frozen=True)
class ClarificationRecoveryEvaluationCase:
    """Development-only case for the deterministic recovery safety boundary."""

    name: str
    proposal: ClarificationRecoveryProposal
    cards: tuple[ActionCard, ...]
    context: Context
    expected_resolved: bool


@dataclass(frozen=True)
class ClarificationRecoveryEvaluationMetrics:
    """Core safety metrics; provider latency/model quality stay with the caller."""

    cases: int
    expected_resolutions: int
    accepted_resolutions: int
    unsafe_acceptances: int
    expected_rejections: int
    rejected_cases: int


def validate_clarification_recovery(
    proposal: ClarificationRecoveryProposal,
    cards: tuple[ActionCard, ...],
    context: Context,
) -> bool:
    """Validate a recovery proposal without turning it into executable authority."""

    if proposal.outcome is not ClarificationRecoveryOutcome.RESOLVED:
        return False
    card = next((card for card in cards if card.action.action_id == proposal.action_ref), None)
    if card is None:
        return False
    if not set(proposal.arguments).issubset(card.argument_keys):
        return False
    if proposal.referent_ref is None:
        return True
    referents = context.values.get("referents")
    those = referents.get("those") if isinstance(referents, dict) else None
    candidates = those.get("candidates") if isinstance(those, dict) else None
    if not isinstance(candidates, list):
        return False
    matches = []
    label_matches = []
    proposed_title = proposal.arguments.get("title")
    for candidate in candidates:
        if isinstance(candidate, dict):
            identity = candidate.get("id") or candidate.get("task_id") or candidate.get("chore_id")
            if identity == proposal.referent_ref:
                matches.append(candidate)
            if isinstance(proposed_title, str) and candidate.get("title") == proposed_title:
                label_matches.append(candidate)
        elif candidate == proposal.referent_ref:
            matches.append(candidate)
    return len(matches) == 1 and (not proposed_title or len(label_matches) == 1)


def development_clarification_recovery_cases() -> tuple[ClarificationRecoveryEvaluationCase, ...]:
    """Return varied, execution-free recovery cases for the bounded spike."""

    card = ActionCard(
        action=ActionSpec(action_id="example.complete", capability="example.complete"),
        summary="Complete a named task",
        relevance=1,
        argument_keys=("title",),
    )

    def context(*items: dict[str, str]) -> Context:
        return Context(
            values={
                "referents": {"those": {"fact_key": "canonical_tasks", "candidates": list(items)}}
            },
            sources=("authorized_canonical_result",),
        )

    def proposal(
        outcome: ClarificationRecoveryOutcome,
        *,
        referent_ref: str | None = "task-1",
        title: str | None = "Replace porch bulb",
        ambiguity_type: ClarificationAmbiguityType = ClarificationAmbiguityType.REFERENT,
    ) -> ClarificationRecoveryProposal:
        return ClarificationRecoveryProposal(
            outcome=outcome,
            ambiguity_type=ambiguity_type,
            action_ref="example.complete",
            referent_ref=referent_ref,
            arguments={"title": title} if title is not None else {},
        )

    return (
        ClarificationRecoveryEvaluationCase(
            "unique-referent-pronoun-typo",
            proposal(ClarificationRecoveryOutcome.RESOLVED),
            (card,),
            context({"task_id": "task-1", "title": "Replace porch bulb"}),
            True,
        ),
        ClarificationRecoveryEvaluationCase(
            "duplicate-partial-name",
            proposal(ClarificationRecoveryOutcome.RESOLVED, title="Call dentist"),
            (card,),
            context(
                {"task_id": "task-1", "title": "Call dentist"},
                {"task_id": "task-2", "title": "Call dentist"},
            ),
            False,
        ),
        ClarificationRecoveryEvaluationCase(
            "stale-referent",
            proposal(ClarificationRecoveryOutcome.RESOLVED, referent_ref="gone"),
            (card,),
            context({"task_id": "task-1", "title": "Replace porch bulb"}),
            False,
        ),
        ClarificationRecoveryEvaluationCase(
            "missing-argument",
            proposal(ClarificationRecoveryOutcome.RESOLVED).model_copy(
                update={"arguments": {"task_id": "task-1"}}
            ),
            (card,),
            context({"task_id": "task-1", "title": "Replace porch bulb"}),
            False,
        ),
        ClarificationRecoveryEvaluationCase(
            "unsupported-capability",
            proposal(
                ClarificationRecoveryOutcome.UNSUPPORTED,
                ambiguity_type=ClarificationAmbiguityType.CAPABILITY,
            ),
            (),
            Context(values={}, sources=()),
            False,
        ),
        ClarificationRecoveryEvaluationCase(
            "ambiguous-read-write",
            proposal(ClarificationRecoveryOutcome.NEED_USER),
            (card,),
            context({"task_id": "task-1", "title": "Replace porch bulb"}),
            False,
        ),
    )


def evaluate_clarification_recovery_cases(
    cases: tuple[ClarificationRecoveryEvaluationCase, ...] | None = None,
) -> ClarificationRecoveryEvaluationMetrics:
    """Measure Core recovery safety without invoking a provider or Kernel."""

    cases = development_clarification_recovery_cases() if cases is None else cases
    accepted = sum(
        validate_clarification_recovery(case.proposal, case.cards, case.context) for case in cases
    )
    expected = sum(case.expected_resolved for case in cases)
    unsafe = sum(
        int(
            validate_clarification_recovery(case.proposal, case.cards, case.context)
            and not case.expected_resolved
        )
        for case in cases
    )
    return ClarificationRecoveryEvaluationMetrics(
        cases=len(cases),
        expected_resolutions=expected,
        accepted_resolutions=accepted,
        unsafe_acceptances=unsafe,
        expected_rejections=len(cases) - expected,
        rejected_cases=len(cases) - accepted,
    )


def request_clarification_recovery(
    provider: Any,
    intent: IntentFrame,
    context: Context,
    cards: tuple[ActionCard, ...],
    reason: str,
) -> ClarificationRecoveryProposal | None:
    """Run the isolated recovery mode and return only its validated-shaped proposal."""

    if not context.values.get("referents") and not context.values.get("canonical_facts"):
        return None
    response = provider.decide(
        ModelRequest(
            working_set=WorkingSet(intent=intent, context=context),
            action_cards=cards,
            clarification_recovery_only=True,
            clarification_reason=reason,
        )
    )
    try:
        proposal = ClarificationRecoveryProposal.model_validate(response.raw)
    except Exception:
        return None
    return proposal if validate_clarification_recovery(proposal, cards, context) else None


def recover_invalid_model_decision(
    dependencies: Any,
    intent: IntentFrame,
    context: Context,
    focused_raw: dict[str, Any] | None,
    error: InvalidDecision,
) -> Decision | Result:
    """Recover only to grounded answers or a truthful retryable failure.

    Recovery has no action-capable path.  A malformed proposal can therefore
    never become execution merely because a best-effort retry succeeds.
    """
    if focused_raw is not None:
        grounded = grounded_context_answer(context, focused_raw)
        if grounded is not None:
            return grounded
    if is_question_request(intent.utterance) and not is_mutation_request(intent.utterance):
        try:
            provider_factory = dependencies.model_provider
            if provider_factory is None:
                raise RuntimeError("model provider unavailable")
            recovered = StrictDecisionDecoder().decode(
                provider_factory().decide(
                    ModelRequest(
                        working_set=WorkingSet(intent=intent, context=context), action_cards=()
                    )
                ),
                (),
                allow_argument_proposals=False,
            )
            if recovered.kind is DecisionKind.ANSWER:
                return recovered
        except Exception:
            # Recovery is deliberately best-effort and answer-only; retain the
            # original bounded failure if it cannot produce a valid answer.
            pass
    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.BLOCKED,
        message="I could not safely interpret that request. Please rephrase it.",
        evidence={
            "provenance": "model_boundary",
            "authoritative": False,
            "failure_class": "invalid_model_decision",
            "failure_reason": str(error),
        },
        correlation_id=intent.correlation_id,
        retryable=True,
    )
