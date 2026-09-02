"""Turn bounded owner feedback metadata into safe defect candidates."""

from __future__ import annotations

from typing import Any


def harvest_defect_candidates(feedback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select actionable feedback without inferring truth or replaying work."""

    candidates: list[dict[str, Any]] = []
    for item in feedback:
        outcome = item.get("outcome")
        reason = item.get("reason")
        if outcome == "helpful" and reason not in {"objective_failed", "incorrect"}:
            continue
        if outcome != "not_helpful" and reason not in {
            "objective_failed",
            "incorrect",
            "unclear",
        }:
            continue
        classification = (
            "objective_failure"
            if reason == "objective_failed"
            else "incorrect_result"
            if reason == "incorrect"
            else "unclear_result"
        )
        candidates.append(
            {
                "event_id": item.get("event_id"),
                "objective_id": item.get("objective_id"),
                "correlation_id": item.get("correlation_id"),
                "created_at": item.get("created_at"),
                "classification": classification,
                "result_state": item.get("result_state"),
                "retryable": item.get("retryable"),
                "reproduction_required": True,
                "replay_consequential_action": False,
                "next_steps": (
                    "inspect the correlated canonical Result and bounded evidence; "
                    "reproduce with a new correlation; classify the failure family; "
                    "add a deterministic regression before changing production behavior"
                ),
            }
        )
    return candidates
