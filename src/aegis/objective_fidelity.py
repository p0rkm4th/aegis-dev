"""Small Core-owned comparison seam for objective interpretation fidelity."""

from __future__ import annotations

import json

from pydantic import Field

from .contracts import (
    ObjectiveFidelityVerdict,
    ObjectiveRequirementProposal,
    ObjectiveSpecProposal,
    StrictModel,
)


class RequestedEffectProposal(StrictModel):
    """Development-only segmented effect; it is never executable or authoritative."""

    effect_text: str = Field(min_length=1, max_length=500)
    source_span: tuple[int, int]


def validate_effect_spans(utterance: str, effects: tuple[RequestedEffectProposal, ...]) -> bool:
    """Core-check that independently segmented effects are grounded in the utterance."""

    return all(
        0 <= start < end <= len(utterance)
        and utterance[start:end].strip() == effect.effect_text.strip()
        for effect in effects
        for start, end in (effect.source_span,)
    )


def _key(requirement: ObjectiveRequirementProposal) -> str:
    return json.dumps(
        {"action_ref": requirement.action_ref, "arguments": requirement.arguments},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def compare_objective_proposals(
    proposed: ObjectiveSpecProposal,
    independent: ObjectiveSpecProposal,
) -> ObjectiveFidelityVerdict:
    """Compare two bounded interpretations without plan or model completion input.

    The independent interpretation is still untrusted cognition.  This function
    only computes a deterministic verdict; it grants no authority and does not
    bind either proposal to an ActionCard.
    """

    left = [_key(requirement) for requirement in proposed.requirements]
    right = [_key(requirement) for requirement in independent.requirements]
    left_set = set(left)
    right_set = set(right)
    if len(left) != len(left_set) or len(right) != len(right_set):
        return ObjectiveFidelityVerdict.NEED_CLARIFICATION
    if left_set == right_set:
        return ObjectiveFidelityVerdict.COMPLETE
    if left_set - right_set and not right_set - left_set:
        return ObjectiveFidelityVerdict.MISSING_REQUIREMENT
    if right_set - left_set and not left_set - right_set:
        return ObjectiveFidelityVerdict.EXTRA_REQUIREMENT
    return ObjectiveFidelityVerdict.NEED_CLARIFICATION


def fidelity_message(verdict: ObjectiveFidelityVerdict) -> str:
    messages = {
        ObjectiveFidelityVerdict.MISSING_REQUIREMENT: (
            "I could not safely account for every requested change; "
            "please clarify the missing change."
        ),
        ObjectiveFidelityVerdict.EXTRA_REQUIREMENT: (
            "I found an additional change that was not clearly requested; "
            "please clarify the intended changes."
        ),
        ObjectiveFidelityVerdict.NEED_CLARIFICATION: (
            "I could not safely reconcile the requested changes; please clarify the objective."
        ),
    }
    return messages[verdict]
