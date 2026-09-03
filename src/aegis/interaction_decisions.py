"""Convert bounded cognition decisions into client-neutral interaction outcomes."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from .contracts import (
    ActionCard,
    Context,
    Decision,
    DecisionKind,
    IntentFrame,
    ObjectiveState,
    Result,
)
from .interaction_context import authorized_context_evidence
from .utterance import has_multiple_question_clauses


def resolve_fallback_decision(
    decision: Decision,
    intent: IntentFrame,
    context: Context,
    cards: tuple[ActionCard, ...],
) -> ActionCard | Result:
    """Resolve a bounded proposal without granting it authority.

    The returned ActionCard still requires Pack grounding, authorization,
    execution, observation, verification, and Kernel completion below this
    boundary.  This module owns only the generic decision/result conversion.
    """

    if decision.kind is DecisionKind.ANSWER:
        evidence: dict[str, Any] = {
            "provenance": "model_generated",
            "authoritative": False,
            "answer_mode": decision.semantic_mode,
            "source_kind": decision.knowledge_source or "general_model_knowledge",
        }
        authorized_facts = authorized_context_evidence(context)
        if decision.context_focus is not None:
            focused = authorized_facts.get(decision.context_focus)
            authorized_facts = {decision.context_focus: focused} if focused is not None else {}
        if authorized_facts:
            evidence.update(authorized_facts)
            evidence["context_provenance"] = "authorized_working_set"
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.COMPLETED,
            message=decision.answer or "",
            evidence=evidence,
            correlation_id=intent.correlation_id,
        )

    if decision.kind is DecisionKind.CLARIFY:
        clarification = decision.clarification or "Please clarify your request."
        if has_multiple_question_clauses(intent.utterance):
            clarification = (
                "That request contains multiple independent questions. "
                "Please ask one at a time so I can answer each from "
                "authorized information."
            )
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=clarification,
            correlation_id=intent.correlation_id,
        )

    if decision.kind is DecisionKind.NEED_CONTEXT:
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message="I need more context to safely interpret that request.",
            correlation_id=intent.correlation_id,
        )

    if decision.kind is DecisionKind.ACTION and decision.action is not None:
        fallback_card = next(
            (
                candidate
                for candidate in cards
                if candidate.action.action_id == decision.action.action_id
            ),
            None,
        )
        if fallback_card is not None:
            # Preserve model-proposed arguments, but restore canonical card
            # metadata before Pack grounding and Kernel authority checks.
            return fallback_card.model_copy(update={"action": decision.action})
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message="I could not safely match that request to an available capability.",
            correlation_id=intent.correlation_id,
            retryable=True,
        )

    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.BLOCKED,
        message=decision.reason or str(decision.kind),
        correlation_id=intent.correlation_id,
    )
