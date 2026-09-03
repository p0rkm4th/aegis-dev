"""Bounded, authorized conversational context for the interaction service."""

from __future__ import annotations

import re
from typing import Any, Literal, cast
from uuid import UUID

from .contracts import Context, Decision, DecisionKind, Principal, Result

_MAX_CONTEXT_TURN_CHARS = 500
_MAX_CONTEXT_CANDIDATES = 10
_ORDINALS = {"first": 0, "second": 1, "third": 2, "fourth": 3, "last": -1}


def compact_context_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Keep only bounded canonical facts for a follow-up model working set."""

    compact: dict[str, Any] = {}
    raw_steps = evidence.get("steps")
    if isinstance(raw_steps, (list, tuple)):
        compact["plan_steps"] = [
            {
                "index": step.get("index"),
                "state": step.get("state"),
            }
            for step in raw_steps[:5]
            if isinstance(step, dict)
        ]
    raw_plan_steps = evidence.get("plan_steps")
    if isinstance(raw_plan_steps, (list, tuple)):
        compact["plan_steps"] = [
            {
                "index": step.get("index"),
                "state": step.get("state"),
            }
            for step in raw_plan_steps[:5]
            if isinstance(step, dict)
        ]
    for key in ("task",):
        value = evidence.get(key)
        if isinstance(value, dict):
            compact[key] = dict(value)
    for key in (
        "canonical_items",
        "canonical_tasks",
        "canonical_chores",
        "chores",
        "events",
        "canonical_obligations",
        "memories",
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
            elif key == "chores":
                compact["canonical_chores"] = list(value[:20])
            else:
                compact[key] = list(value[:20])
        elif isinstance(value, str):
            compact[key] = value
    planning = evidence.get("planning")
    if isinstance(planning, dict):
        compact_planning: dict[str, Any] = {}
        for key in (
            "open_tasks",
            "open_chores",
            "open_obligations",
            "priority_candidates",
            "memories",
            "affordability",
        ):
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


def resolve_obvious_ordinal(
    utterance: str, context: Context, fact_key: str
) -> dict[str, Any] | None:
    """Resolve one ordinal only from the immediately authorized canonical list."""

    if context.sources != ("authorized_canonical_result",):
        return None
    match = re.search(r"\b(?:the\s+)?(first|second|third|fourth|last)\b", utterance.casefold())
    if match is None:
        return None
    referents = context.values.get("referents")
    if not isinstance(referents, dict):
        return None
    those = referents.get("those")
    if not isinstance(those, dict) or those.get("fact_key") != fact_key:
        return None
    candidates = those.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    index = _ORDINALS[match.group(1)]
    try:
        candidate = candidates[index]
    except IndexError:
        return None
    return candidate if isinstance(candidate, dict) else None


def resolve_obvious_ordinal_item(utterance: str, context: Context) -> str | None:
    """Resolve one grocery ordinal from the immediately authorized item list."""

    if context.sources != ("authorized_canonical_result",):
        return None
    match = re.search(r"\b(?:the\s+)?(first|second|third|fourth|last)\b", utterance.casefold())
    if match is None:
        return None
    referents = context.values.get("referents")
    if not isinstance(referents, dict):
        return None
    those = referents.get("those")
    if not isinstance(those, dict) or those.get("fact_key") != "canonical_items":
        return None
    candidates = those.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    index = _ORDINALS[match.group(1)]
    try:
        candidate = candidates[index]
    except IndexError:
        return None
    return candidate if isinstance(candidate, str) and candidate.strip() else None


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
    raw_evidence = result.evidence
    evidence = compact_context_evidence(raw_evidence)
    if not evidence:
        return Context()
    referents: dict[str, Any] = {}
    for fact_key in (
        "canonical_items",
        "canonical_tasks",
        "canonical_chores",
        "events",
        "canonical_obligations",
    ):
        # Ordinals refer to the list the owner just received.  Keep this
        # separate from compacted model evidence, which may use a priority
        # ordering for cognition.
        candidates = raw_evidence.get(fact_key)
        if fact_key == "canonical_chores" and not isinstance(candidates, list):
            candidates = raw_evidence.get("chores")
        if isinstance(candidates, list) and candidates:
            selected = candidates[:_MAX_CONTEXT_CANDIDATES]
            if fact_key == "canonical_tasks":
                compact_tasks = evidence.get("canonical_tasks")
                if isinstance(compact_tasks, list):
                    selected.extend(
                        item
                        for item in compact_tasks
                        if isinstance(item, dict)
                        and isinstance(item.get("due_at"), str)
                        and item not in selected
                    )
            referents["those"] = {
                "source": "canonical_facts",
                "fact_key": fact_key,
                "candidates": selected,
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
