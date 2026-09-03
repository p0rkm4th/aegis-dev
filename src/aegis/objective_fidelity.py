"""Small Core-owned comparison seam for objective interpretation fidelity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Literal

from pydantic import Field

from .contracts import (
    ObjectiveFidelityVerdict,
    ObjectiveRequirementProposal,
    ObjectiveSpecProposal,
    RequestedEffect,
    StrictModel,
    StructuralCoverageSignal,
)


class RequestedEffectProposal(StrictModel):
    """Untrusted utterance-grounded effect proposal; never executable."""

    effect_text: str = Field(min_length=1, max_length=500)
    source_span: tuple[int, int]
    action_ref: str | None = None
    arguments: dict[str, object] = {}
    polarity: Literal["ACTIVE", "NEGATED", "SUPERSEDED"] = "ACTIVE"


def materialize_requested_effects(
    utterance: str, effects: tuple[RequestedEffectProposal, ...]
) -> tuple[RequestedEffect, ...] | None:
    """Assign Core identities only after exact source grounding succeeds."""

    normalized = normalize_effect_spans(utterance, effects)
    if normalized is None or not validate_effect_spans(utterance, normalized):
        return None
    spans = [effect.source_span for effect in normalized]
    if len(spans) != len(set(spans)):
        return None
    return tuple(
        RequestedEffect(
            source_spans=(effect.source_span,),
            normalized_effect=effect.effect_text.strip(),
            polarity=effect.polarity,
        )
        for effect in normalized
    )


def validate_effect_spans(utterance: str, effects: tuple[RequestedEffectProposal, ...]) -> bool:
    """Core-check that independently segmented effects are grounded in the utterance."""

    return all(
        0 <= start < end <= len(utterance)
        and utterance[start:end].strip().casefold() == effect.effect_text.strip().casefold()
        for effect in effects
        for start, end in (effect.source_span,)
    )


def validate_structural_coverage(
    utterance: str,
    effects: tuple[RequestedEffect, ...],
    signal: StructuralCoverageSignal,
) -> bool:
    """Require one meaningful parser anchor for each requested effect.

    The signal is intentionally supplied by a separate structural component.
    This function neither assigns meaning nor maps actions.  Exact cardinality
    and one-to-one span correspondence prevent one full-utterance or duplicated
    span from claiming coverage of several structural units.
    """

    if not effects or len(effects) != len(signal.anchors):
        return False
    effect_spans = [span for effect in effects for span in effect.source_spans]
    if len(effect_spans) != len(set(effect_spans)):
        return False
    anchor_spans = [anchor.source_span for anchor in signal.anchors]
    if len(anchor_spans) != len(set(anchor_spans)):
        return False
    if any(not (0 <= start < end <= len(utterance)) for start, end in anchor_spans):
        return False
    if len(signal.anchors) > 1 and any(
        start == 0 and end == len(utterance) for start, end in effect_spans
    ):
        return False

    def overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
        return left[0] < right[1] and right[0] < left[1]

    # Every anchor must correspond to exactly one effect, and every effect to
    # exactly one anchor.  The parser's internal token/dependency details do
    # not cross this Core boundary.
    matches = [sum(overlaps(anchor, effect) for effect in effect_spans) for anchor in anchor_spans]
    reverse = [sum(overlaps(effect, anchor) for anchor in anchor_spans) for effect in effect_spans]
    return all(count == 1 for count in matches) and all(count == 1 for count in reverse)


def normalize_effect_spans(
    utterance: str, effects: tuple[RequestedEffectProposal, ...]
) -> tuple[RequestedEffectProposal, ...] | None:
    """Repair only malformed citations whose exact text has one safe location."""

    normalized: list[RequestedEffectProposal] = []
    for effect in effects:
        start, end = effect.source_span
        if (
            not (0 <= start < end <= len(utterance))
            or utterance[start:end].strip() != effect.effect_text.strip()
        ):
            matches: list[int] = []
            source = utterance.casefold()
            target = effect.effect_text.casefold()
            cursor = source.find(target)
            while cursor >= 0:
                matches.append(cursor)
                cursor = source.find(target, cursor + 1)
            if len(matches) != 1:
                return None
            start = matches[0]
            end = start + len(effect.effect_text)
        normalized.append(effect.model_copy(update={"source_span": (start, end)}))
    return tuple(normalized)


def effects_to_proposal(
    utterance: str, effects: tuple[RequestedEffectProposal, ...]
) -> ObjectiveSpecProposal | None:
    """Bind grounded effect segments to an untrusted requirement proposal."""

    # Establish the action-agnostic Core boundary before consulting any
    # proposed capability mapping.  The materialized value is intentionally
    # not used as executable state here; this helper remains a proposal path.
    materialized = materialize_requested_effects(utterance, effects)
    if materialized is None or any(effect.polarity != "ACTIVE" for effect in materialized):
        return None
    normalized_effects = normalize_effect_spans(utterance, effects)
    if (
        normalized_effects is None
        or not validate_effect_spans(utterance, normalized_effects)
        or any(effect.action_ref is None for effect in normalized_effects)
    ):
        return None
    return ObjectiveSpecProposal(
        requirements=tuple(
            ObjectiveRequirementProposal(
                action_ref=effect.action_ref or "", arguments=dict(effect.arguments)
            )
            for effect in normalized_effects
        )
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


@dataclass(frozen=True)
class FidelityEvaluationCase:
    """A bounded development case for measuring independent objective fidelity.

    This is evaluation data, not a runtime authorization contract.  The expected
    effects are supplied by the evaluator; the production path still derives its
    verdict from untrusted proposals and Core's deterministic comparison.
    """

    name: str
    expected: ObjectiveSpecProposal
    proposed: ObjectiveSpecProposal
    independent: ObjectiveSpecProposal
    ambiguous: bool = False
    unsupported: bool = False
    decoder_failure: bool = False


@dataclass(frozen=True)
class FidelityEvaluationMetrics:
    """Small, comparable metrics separating interpretation from Core safety."""

    cases: int
    requirement_recall: float
    extra_requirement_precision: float
    correlated_omission_false_acceptances: int
    ambiguity_detection: float
    unsupported_effect_detection: float
    false_clarifications: int
    decoder_failures: int
    core_false_acceptances: int


def development_fidelity_cases() -> tuple[FidelityEvaluationCase, ...]:
    """Return fresh, wording-independent cases for the development spike."""

    def proposal(*items: tuple[str, dict[str, object]]) -> ObjectiveSpecProposal:
        return ObjectiveSpecProposal(
            requirements=tuple(
                ObjectiveRequirementProposal(action_ref=action_ref, arguments=arguments)
                for action_ref, arguments in items
            )
        )

    return (
        FidelityEvaluationCase(
            "same-domain-two",
            proposal(("tasks.create", {"title": "A"}), ("tasks.create", {"title": "B"})),
            proposal(("tasks.create", {"title": "A"}), ("tasks.create", {"title": "B"})),
            proposal(("tasks.create", {"title": "B"}), ("tasks.create", {"title": "A"})),
        ),
        FidelityEvaluationCase(
            "cross-domain-three",
            proposal(
                ("tasks.create", {"title": "A"}),
                ("chores.create", {"title": "B"}),
                ("kitchen.groceries.add", {"item": "C"}),
            ),
            proposal(
                ("tasks.create", {"title": "A"}),
                ("chores.create", {"title": "B"}),
                ("kitchen.groceries.add", {"item": "C"}),
            ),
            proposal(
                ("tasks.create", {"title": "A"}),
                ("chores.create", {"title": "B"}),
                ("kitchen.groceries.add", {"item": "C"}),
            ),
        ),
        FidelityEvaluationCase(
            "correlated-middle-omission",
            proposal(
                ("tasks.create", {"title": "A"}),
                ("chores.create", {"title": "B"}),
                ("kitchen.groceries.add", {"item": "C"}),
            ),
            proposal(("tasks.create", {"title": "A"}), ("kitchen.groceries.add", {"item": "C"})),
            proposal(("tasks.create", {"title": "A"}), ("kitchen.groceries.add", {"item": "C"})),
        ),
        FidelityEvaluationCase(
            "helpful-extra",
            proposal(("tasks.create", {"title": "A"}), ("chores.create", {"title": "B"})),
            proposal(
                ("tasks.create", {"title": "A"}),
                ("chores.create", {"title": "B"}),
                ("tasks.create", {"title": "C"}),
            ),
            proposal(("tasks.create", {"title": "A"}), ("chores.create", {"title": "B"})),
        ),
        FidelityEvaluationCase(
            "ambiguous-effect",
            proposal(
                ("tasks.create", {"title": "the inspection"}),
                ("tasks.complete", {"title": "the inspection"}),
            ),
            proposal(
                ("tasks.create", {"title": "the inspection"}),
                ("tasks.complete", {"title": "the inspection"}),
            ),
            proposal(
                ("tasks.create", {"title": "the inspection"}),
                ("tasks.complete", {"title": "the other inspection"}),
            ),
            ambiguous=True,
        ),
        FidelityEvaluationCase(
            "unsupported-second-effect",
            proposal(("tasks.create", {"title": "A"}), ("unsupported.send", {"to": "B"})),
            proposal(("tasks.create", {"title": "A"})),
            proposal(("tasks.create", {"title": "A"}), ("unsupported.send", {"to": "B"})),
            unsupported=True,
        ),
        FidelityEvaluationCase(
            "temporal-single-effect",
            proposal(("tasks.create", {"title": "A", "due_at": "Friday"})),
            proposal(("tasks.create", {"title": "A", "due_at": "Friday"})),
            proposal(("tasks.create", {"title": "A", "due_at": "Friday"})),
        ),
        FidelityEvaluationCase(
            "correction-like",
            proposal(("tasks.create", {"title": "B"})),
            proposal(("tasks.create", {"title": "B"})),
            proposal(("tasks.create", {"title": "A"})),
            ambiguous=True,
        ),
    )


def _keys(specification: ObjectiveSpecProposal) -> set[str]:
    return {_key(requirement) for requirement in specification.requirements}


def evaluate_fidelity_cases(
    cases: Iterable[FidelityEvaluationCase],
) -> FidelityEvaluationMetrics:
    """Evaluate a fresh bounded corpus without invoking a model or a plan.

    ``core_false_acceptances`` is intentionally an architectural metric: an
    omission accepted by deterministic comparison is a safety failure even when
    both cognition proposals agree.  Model-call and latency metrics belong to the
    caller because this pure evaluator must remain deterministic.
    """

    cases = tuple(cases)
    if not cases:
        raise ValueError("fidelity evaluation requires at least one case")
    recall_values: list[float] = []
    precision_values: list[float] = []
    ambiguity_hits = 0
    unsupported_hits = 0
    false_clarifications = 0
    correlated_omissions = 0
    core_false_acceptances = 0
    decoder_failures = 0
    for case in cases:
        expected = _keys(case.expected)
        proposed = _keys(case.proposed)
        independent = _keys(case.independent)
        recall_values.append(len(proposed & expected) / len(expected))
        precision_values.append(len(proposed & expected) / len(proposed) if proposed else 0.0)
        verdict = compare_objective_proposals(case.proposed, case.independent)
        if case.ambiguous and verdict is ObjectiveFidelityVerdict.NEED_CLARIFICATION:
            ambiguity_hits += 1
        if case.unsupported and verdict is not ObjectiveFidelityVerdict.COMPLETE:
            unsupported_hits += 1
        if (
            not case.ambiguous
            and not case.unsupported
            and verdict is not ObjectiveFidelityVerdict.COMPLETE
        ):
            false_clarifications += 1
        omission = expected - proposed
        if omission and proposed == independent:
            correlated_omissions += 1
            if verdict is ObjectiveFidelityVerdict.COMPLETE:
                core_false_acceptances += 1
        decoder_failures += int(case.decoder_failure)
    return FidelityEvaluationMetrics(
        cases=len(cases),
        requirement_recall=sum(recall_values) / len(cases),
        extra_requirement_precision=sum(precision_values) / len(cases),
        correlated_omission_false_acceptances=correlated_omissions,
        ambiguity_detection=ambiguity_hits / sum(case.ambiguous for case in cases)
        if any(case.ambiguous for case in cases)
        else 1.0,
        unsupported_effect_detection=unsupported_hits / sum(case.unsupported for case in cases)
        if any(case.unsupported for case in cases)
        else 1.0,
        false_clarifications=false_clarifications,
        decoder_failures=decoder_failures,
        core_false_acceptances=core_false_acceptances,
    )
