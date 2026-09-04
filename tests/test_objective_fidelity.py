from aegis.contracts import (
    ObjectiveFidelityVerdict,
    ObjectiveRequirementProposal,
    ObjectiveSpecProposal,
    StructuralAnchor,
    StructuralCoverageSignal,
)
from aegis.objective_fidelity import (
    FidelityEvaluationCase,
    RequestedEffectProposal,
    compare_objective_proposals,
    effects_to_proposal,
    evaluate_fidelity_cases,
    materialize_requested_effects,
    validate_effect_spans,
    validate_structural_coverage,
)


def spec(*requirements: tuple[str, dict[str, object]]) -> ObjectiveSpecProposal:
    return ObjectiveSpecProposal(
        requirements=tuple(
            ObjectiveRequirementProposal(action_ref=action_ref, arguments=arguments)
            for action_ref, arguments in requirements
        )
    )


def test_fidelity_accepts_equivalent_requirements_without_plan_input() -> None:
    assert (
        compare_objective_proposals(
            spec(("tasks.create", {"title": "A"}), ("chores.create", {"title": "B"})),
            spec(("chores.create", {"title": "B"}), ("tasks.create", {"title": "A"})),
        )
        is ObjectiveFidelityVerdict.COMPLETE
    )


def test_fidelity_rejects_correlated_omission_as_missing_requirement() -> None:
    assert (
        compare_objective_proposals(
            spec(("tasks.create", {"title": "A"}), ("chores.create", {"title": "B"})),
            spec(("tasks.create", {"title": "A"})),
        )
        is ObjectiveFidelityVerdict.MISSING_REQUIREMENT
    )


def test_fidelity_rejects_helpful_extra_as_extra_requirement() -> None:
    assert (
        compare_objective_proposals(
            spec(("tasks.create", {"title": "A"})),
            spec(("tasks.create", {"title": "A"}), ("chores.create", {"title": "B"})),
        )
        is ObjectiveFidelityVerdict.EXTRA_REQUIREMENT
    )


def test_fidelity_requires_clarification_when_both_interpretations_differ() -> None:
    assert (
        compare_objective_proposals(
            spec(("tasks.create", {"title": "A"})),
            spec(("chores.create", {"title": "B"})),
        )
        is ObjectiveFidelityVerdict.NEED_CLARIFICATION
    )


def test_segmented_effects_must_be_grounded_in_original_utterance() -> None:
    utterance = "add towels and schedule an inspection"
    assert validate_effect_spans(
        utterance,
        (
            RequestedEffectProposal(effect_text="add towels", source_span=(0, 10)),
            RequestedEffectProposal(effect_text="schedule an inspection", source_span=(15, 37)),
        ),
    )
    assert not validate_effect_spans(
        utterance,
        (RequestedEffectProposal(effect_text="delete the list", source_span=(0, 10)),),
    )


def test_effect_text_allows_only_unique_deterministic_span_repair() -> None:
    utterance = "Add rice to groceries and add milk to groceries"
    effects = (
        RequestedEffectProposal(
            effect_text="add rice to groceries",
            source_span=(-1, 0),
            action_ref="kitchen.groceries.add",
            arguments={"item": "rice"},
        ),
    )
    assert effects_to_proposal(utterance, effects) is not None
    duplicate = (
        RequestedEffectProposal(
            effect_text="groceries",
            source_span=(-1, 0),
            action_ref="kitchen.groceries.add",
            arguments={"item": "rice"},
        ),
    )
    assert effects_to_proposal(utterance, duplicate) is None


def test_requested_effect_is_action_agnostic_and_gets_core_identity() -> None:
    effects = materialize_requested_effects(
        "Add milk and eggs",
        (
            RequestedEffectProposal(effect_text="Add milk", source_span=(0, 8)),
            RequestedEffectProposal(effect_text="eggs", source_span=(13, 17)),
        ),
    )
    assert effects is not None
    assert len(effects) == 2
    assert all(effect.effect_id is not None for effect in effects)
    assert all(effect.resolution.value == "UNRESOLVED" for effect in effects)


def test_structural_coverage_rejects_correlated_middle_omission() -> None:
    utterance = "Add milk, schedule inspection, and clean the porch"
    effects = materialize_requested_effects(
        utterance,
        (
            RequestedEffectProposal(effect_text="Add milk", source_span=(0, 8)),
            RequestedEffectProposal(effect_text="clean the porch", source_span=(35, 50)),
        ),
    )
    assert effects is not None
    signal = StructuralCoverageSignal(
        anchors=(
            StructuralAnchor(source_span=(0, 8), kind="clause"),
            StructuralAnchor(source_span=(10, 29), kind="clause"),
            StructuralAnchor(source_span=(35, 50), kind="clause"),
        )
    )
    assert not validate_structural_coverage(utterance, effects, signal)


def test_structural_signal_preserves_negation_evidence() -> None:
    from aegis.structural import SpacyStructuralParser

    class Token:
        def __init__(self, idx: int, text: str, pos: str, dep: str) -> None:
            self.idx = idx
            self.text = text
            self.pos_ = pos
            self.dep_ = dep

    class Model:
        def __call__(self, _utterance: str) -> tuple[Token, ...]:
            return (
                Token(0, "Make", "VERB", "ROOT"),
                Token(5, "not", "PART", "neg"),
            )

    signal = SpacyStructuralParser(model=Model()).parse(
        "Make the inspection task, not the cleaning chore."
    )

    assert signal.negation_spans


def test_structural_coverage_rejects_full_span_and_duplicate_gaming() -> None:
    utterance = "Add milk and eggs"
    full = materialize_requested_effects(
        utterance,
        (RequestedEffectProposal(effect_text=utterance, source_span=(0, len(utterance))),),
    )
    assert full is not None
    signal = StructuralCoverageSignal(
        anchors=(
            StructuralAnchor(source_span=(0, 8), kind="clause"),
            StructuralAnchor(source_span=(13, 17), kind="clause"),
        )
    )
    assert not validate_structural_coverage(utterance, full, signal)
    duplicate = (
        RequestedEffectProposal(effect_text="Add milk", source_span=(0, 8)),
        RequestedEffectProposal(effect_text="Add milk", source_span=(0, 8)),
    )
    assert materialize_requested_effects(utterance, duplicate) is None


def test_negated_or_superseded_effect_cannot_become_an_action_requirement() -> None:
    utterance = "Add milk but do not add eggs"
    effects = (
        RequestedEffectProposal(effect_text="Add milk", source_span=(0, 8)),
        RequestedEffectProposal(
            effect_text="add eggs", source_span=(21, 29), polarity="NEGATED", action_ref="add"
        ),
    )
    materialized = materialize_requested_effects(utterance, effects)
    assert materialized is not None
    assert effects_to_proposal(utterance, effects) is None


def test_fidelity_development_metrics_expose_correlated_omission_safety() -> None:
    cases = (
        FidelityEvaluationCase(
            name="two-effects-complete",
            expected=spec(("tasks.create", {"title": "A"}), ("chores.create", {"title": "B"})),
            proposed=spec(("tasks.create", {"title": "A"}), ("chores.create", {"title": "B"})),
            independent=spec(("chores.create", {"title": "B"}), ("tasks.create", {"title": "A"})),
        ),
        FidelityEvaluationCase(
            name="correlated-middle-omission",
            expected=spec(
                ("tasks.create", {"title": "A"}),
                ("chores.create", {"title": "B"}),
                ("kitchen.groceries.add", {"item": "C"}),
            ),
            proposed=spec(
                ("tasks.create", {"title": "A"}), ("kitchen.groceries.add", {"item": "C"})
            ),
            independent=spec(
                ("tasks.create", {"title": "A"}), ("kitchen.groceries.add", {"item": "C"})
            ),
            structural_effect_count=3,
        ),
        FidelityEvaluationCase(
            name="ambiguous-effect",
            expected=spec(("tasks.create", {"title": "the inspection"})),
            proposed=spec(("tasks.create", {"title": "the inspection"})),
            independent=spec(("tasks.complete", {"title": "the inspection"})),
            ambiguous=True,
        ),
        FidelityEvaluationCase(
            name="unsupported-effect",
            expected=spec(("tasks.create", {"title": "A"}), ("unsupported.send", {"to": "B"})),
            proposed=spec(("tasks.create", {"title": "A"})),
            independent=spec(
                ("tasks.create", {"title": "A"}),
                ("unsupported.send", {"to": "B"}),
            ),
            unsupported=True,
        ),
    )
    metrics = evaluate_fidelity_cases(cases)
    assert metrics.cases == 4
    assert metrics.requirement_recall < 1
    assert metrics.correlated_omission_false_acceptances == 1
    # The structural signal catches the omission even when both cognition
    # proposals agree, so the Core safety metric is zero while model recall
    # remains imperfect and the correlated omission remains measurable.
    assert metrics.core_false_acceptances == 0
    assert metrics.ambiguity_detection == 1
    assert metrics.unsupported_effect_detection == 1
