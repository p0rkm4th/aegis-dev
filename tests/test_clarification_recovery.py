from aegis.contracts import (
    ActionCard,
    ActionSpec,
    ClarificationAmbiguityType,
    ClarificationRecoveryOutcome,
    ClarificationRecoveryProposal,
    Context,
)
from aegis.interaction_recovery import validate_clarification_recovery


def card() -> ActionCard:
    return ActionCard(
        action=ActionSpec(action_id="tasks.complete", capability="tasks.complete"),
        summary="Complete a named task",
        relevance=1,
        argument_keys=("title",),
    )


def resolved(referent_ref: str | None = "task-1") -> ClarificationRecoveryProposal:
    return ClarificationRecoveryProposal(
        outcome=ClarificationRecoveryOutcome.RESOLVED,
        ambiguity_type=ClarificationAmbiguityType.REFERENT,
        action_ref="tasks.complete",
        referent_ref=referent_ref,
        arguments={"title": "Replace porch bulb"},
    )


def context(*candidates: dict[str, str]) -> Context:
    return Context(
        values={
            "referents": {"those": {"fact_key": "canonical_tasks", "candidates": list(candidates)}}
        },
        sources=("authorized_canonical_result",),
    )


def test_recovery_accepts_one_current_authorized_referent() -> None:
    assert validate_clarification_recovery(
        resolved(),
        (card(),),
        context({"task_id": "task-1", "title": "Replace porch bulb"}),
    )


def test_recovery_rejects_duplicate_referent_identity() -> None:
    assert not validate_clarification_recovery(
        resolved(),
        (card(),),
        context(
            {"task_id": "task-1", "title": "Call dentist"},
            {"task_id": "task-2", "title": "Call dentist"},
        ),
    )


def test_recovery_rejects_stale_or_unauthorized_referent() -> None:
    assert not validate_clarification_recovery(
        resolved("missing"),
        (card(),),
        context({"task_id": "task-1", "title": "Replace porch bulb"}),
    )


def test_recovery_rejects_nonresolved_proposal_and_undeclared_arguments() -> None:
    proposal = resolved()
    unsupported = proposal.model_copy(update={"outcome": ClarificationRecoveryOutcome.UNSUPPORTED})
    assert not validate_clarification_recovery(
        unsupported, (card(),), context({"task_id": "task-1", "title": "Replace porch bulb"})
    )
    invalid_args = proposal.model_copy(update={"arguments": {"task_id": "task-1"}})
    assert not validate_clarification_recovery(
        invalid_args, (card(),), context({"task_id": "task-1", "title": "Replace porch bulb"})
    )
