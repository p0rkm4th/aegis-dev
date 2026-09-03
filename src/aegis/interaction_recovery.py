"""Bounded recovery for malformed cognition proposals."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from .contracts import (
    ActionCard,
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
