"""Bounded, authorized conversational context for the interaction service."""

from __future__ import annotations

from typing import Any, Literal, cast
from uuid import UUID

from .contracts import Context, Decision, DecisionKind, Principal, Result

_MAX_CONTEXT_TURN_CHARS = 500
_MAX_CONTEXT_CANDIDATES = 10


def compact_context_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Keep only bounded canonical facts for a follow-up model working set."""

    compact: dict[str, Any] = {}
    for key in (
        "canonical_items",
        "canonical_tasks",
        "canonical_chores",
        "canonical_obligations",
        "title",
        "item",
    ):
        value = evidence.get(key)
        if isinstance(value, (list, tuple)):
            if key == "canonical_items":
                compact[key] = list(dict.fromkeys(str(item) for item in value))[:20]
            elif key == "canonical_tasks":
                tasks = list(value)
                dated_open = [
                    item
                    for item in tasks
                    if isinstance(item, dict)
                    and item.get("status") == "open"
                    and isinstance(item.get("due_at"), str)
                ]
                remaining = [item for item in tasks if item not in dated_open]
                compact[key] = (dated_open + remaining)[:20]
            else:
                compact[key] = list(value[:20])
        elif isinstance(value, str):
            compact[key] = value
    planning = evidence.get("planning")
    if isinstance(planning, dict):
        compact_planning: dict[str, Any] = {}
        for key in ("open_tasks", "open_obligations", "memories", "affordability"):
            value = planning.get(key)
            if isinstance(value, (list, tuple)):
                compact_planning[key] = list(value[:10])
            elif isinstance(value, dict):
                compact_planning[key] = value
        if compact_planning:
            compact["planning"] = compact_planning
    return compact


def authorized_context_evidence(context: Context) -> dict[str, Any]:
    """Carry only the authorized working-set facts needed by a later turn."""

    raw = context.values.get("canonical_facts")
    if not isinstance(raw, dict):
        return {}
    return compact_context_evidence(raw)


def grounded_context_answer(context: Context, raw: dict[str, Any]) -> Decision | None:
    """Recover an answer from one model-selected, authorized structured focus."""

    if raw.get("kind") != DecisionKind.ANSWER.value or raw.get("semantic_mode") != "READ":
        return None
    focus = raw.get("context_focus")
    facts = context.values.get("canonical_facts")
    if not isinstance(focus, str) or not isinstance(facts, dict):
        return None
    if focus not in {"canonical_items", "canonical_tasks", "canonical_obligations"}:
        return None
    value = facts.get(focus)
    if not isinstance(value, list) or not value:
        return None
    if focus == "canonical_items":
        answer = "Authorized groceries: " + ", ".join(str(item) for item in value)
    elif focus == "canonical_tasks":
        titles = [
            str(item.get("title"))
            for item in value
            if isinstance(item, dict) and isinstance(item.get("title"), str)
        ]
        if not titles:
            return None
        answer = "Authorized tasks: " + "; ".join(titles)
    else:
        titles = [
            str(item.get("title"))
            for item in value
            if isinstance(item, dict) and isinstance(item.get("title"), str)
        ]
        if not titles:
            return None
        answer = "Authorized obligations: " + "; ".join(titles)
    return Decision(
        kind=DecisionKind.ANSWER,
        answer=answer,
        semantic_mode="READ",
        context_focus=cast(
            Literal["canonical_items", "canonical_tasks", "canonical_obligations"], focus
        ),
    )


def with_continuation_context(result: Result, context: Context) -> Result:
    """Trace a follow-up to authorized context without changing its identity."""

    prior_objective_id = context.values.get("prior_objective_id")
    if not isinstance(prior_objective_id, str):
        return result
    evidence = dict(result.evidence)
    evidence["continuation_of_objective_id"] = prior_objective_id
    evidence["continuation_context"] = "authorized_prior_result"
    return result.model_copy(update={"evidence": evidence})


def context_from_prior_result(
    objective_store: Any, correlation_id: UUID | None, principal: Principal
) -> Context:
    """Resolve follow-up context only through the authorized canonical store."""

    if correlation_id is None:
        return Context()
    objective = objective_store.get_objective_by_correlation(correlation_id, principal)
    getter = getattr(objective_store, "get_result_for_correlation", None)
    if objective is None or not callable(getter):
        return Context()
    result = getter(correlation_id, principal)
    if result is None:
        return Context()
    evidence = compact_context_evidence(result.evidence)
    if not evidence:
        return Context()
    referents: dict[str, Any] = {}
    for fact_key in ("canonical_items", "canonical_tasks", "canonical_obligations"):
        candidates = evidence.get(fact_key)
        if isinstance(candidates, list) and candidates:
            referents["those"] = {
                "source": "canonical_facts",
                "fact_key": fact_key,
                "candidates": candidates[:_MAX_CONTEXT_CANDIDATES],
            }
            break
    return Context(
        values={
            "prior_correlation_id": str(correlation_id),
            "prior_objective_id": str(objective.id),
            "prior_state": result.state.value,
            "recent_turns": [
                {
                    "role": "user",
                    "utterance": objective.intent.utterance[:_MAX_CONTEXT_TURN_CHARS],
                    "correlation_id": str(correlation_id),
                }
            ],
            "referents": referents,
            "canonical_facts": evidence,
        },
        sources=("authorized_canonical_result",),
    )
