from aegis.contracts import (
    ObjectiveFidelityVerdict,
    ObjectiveRequirementProposal,
    ObjectiveSpecProposal,
)
from aegis.objective_fidelity import compare_objective_proposals


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
