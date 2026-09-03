from aegis.contracts import (
    ObjectiveFidelityVerdict,
    ObjectiveRequirementProposal,
    ObjectiveSpecProposal,
)
from aegis.objective_fidelity import (
    FidelityEvaluationCase,
    RequestedEffectProposal,
    compare_objective_proposals,
    effects_to_proposal,
    evaluate_fidelity_cases,
    validate_effect_spans,
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
            effect_text="Add rice to groceries",
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
    # The intentionally adversarial case documents why independent review alone
    # cannot be treated as a complete fidelity solution: correlated cognition can
    # omit the same effect twice.  Core's plan/objective comparison itself still
    # accepts the two matching untrusted proposals, so this remains a campaign
    # blocker until a selected independent-effect mechanism catches it.
    assert metrics.core_false_acceptances == 1
    assert metrics.ambiguity_detection == 1
    assert metrics.unsupported_effect_detection == 1
