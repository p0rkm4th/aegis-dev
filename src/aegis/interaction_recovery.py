"""Bounded recovery for malformed cognition proposals."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from .contracts import (
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
