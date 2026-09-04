"""Convert bounded cognition decisions into client-neutral interaction outcomes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from .contracts import (
    ActionCard,
    Context,
    Decision,
    DecisionKind,
    IntentFrame,
    ObjectiveState,
    RequestedEffect,
    Result,
)
from .interaction_context import authorized_context_evidence
from .utterance import has_multiple_question_clauses


def _unknown_consequential_objective(intent: IntentFrame, clarification: str) -> bool:
    """Recognize a domain-mismatch clarification without inventing an effect."""

    text = " ".join(intent.utterance.casefold().split())
    clarification_text = clarification.casefold()
    action_language = (
        "spin up" in text
        or "set up" in text
        or "deploy" in text
        or "install" in text
        or "configure" in text
        or "provision" in text
        or "launch" in text
    )
    known_domain = any(
        term in text
        for term in (
            "task",
            "todo",
            "chore",
            "event",
            "appointment",
            "grocery",
            "shopping list",
            "memory",
            "finance",
            "homelab",
            "network",
        )
    )
    domain_mismatch = any(term in clarification_text for term in ("task", "chore", "event"))
    return action_language and domain_mismatch and not known_domain


def resolve_fallback_decision(
    decision: Decision,
    intent: IntentFrame,
    context: Context,
    cards: tuple[ActionCard, ...],
    unresolved_requirement_investigator: Callable[..., Result | None] | None = None,
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
            # This is the ordinary model-answer path.  A model cannot label
            # its own output as externally sourced; research provenance is
            # attached only by the answer-only research callback.
            "source_kind": "general_model_knowledge",
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
        if _unknown_consequential_objective(intent, clarification):
            if unresolved_requirement_investigator is not None:
                effect = RequestedEffect(
                    source_spans=((0, len(intent.utterance)),),
                    normalized_effect=intent.utterance,
                )
                try:
                    investigated = unresolved_requirement_investigator(intent, context, (effect,))
                except Exception:
                    investigated = None
                if (
                    isinstance(investigated, Result)
                    and investigated.state in {ObjectiveState.BLOCKED, ObjectiveState.FAILED}
                    and investigated.evidence.get("authoritative") is False
                ):
                    return investigated
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message=(
                    "I could not map that consequential objective to an available capability. "
                    "It remains open; please clarify the capability or provide an approved "
                    "workflow."
                ),
                evidence={
                    "authoritative": False,
                    "provenance": "model_boundary",
                    "objective_open": True,
                    "capability_state": "unresolved",
                    "resolution": "UNSUPPORTED",
                },
                correlation_id=intent.correlation_id,
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
