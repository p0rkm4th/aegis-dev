import pytest

from aegis.contracts import (
    ActionCard,
    ActionSpec,
    Context,
    IntentFrame,
    Principal,
    ProposedPlan,
    ProposedPlanStep,
    VerificationContract,
)
from aegis.planning import PlanProgressFastPath, PlanValidationError, materialize_proposed_plan


def card(action_id: str, *arguments: str) -> ActionCard:
    action = ActionSpec(
        action_id=action_id,
        capability=f"pack.{action_id}",
        required_permissions=(f"{action_id}.write",),
        verification=VerificationContract(kind="readback"),
    )
    return ActionCard(
        action=action,
        summary=action_id,
        relevance=1,
        argument_keys=arguments,
    )


def test_proposed_plan_materializes_only_candidate_authority():
    task = card("tasks.create", "title")
    chore = card("chores.create", "title")
    proposal = ProposedPlan(
        steps=(
            ProposedPlanStep(action_ref="tasks.create", arguments={"title": "Review backup"}),
            ProposedPlanStep(
                action_ref="chores.create",
                arguments={"title": "Clean kitchen"},
                depends_on=(0,),
            ),
        )
    )

    actions = materialize_proposed_plan(proposal, (task, chore))

    assert [action.action_id for action in actions] == ["tasks.create", "chores.create"]
    assert actions[0].capability == task.action.capability
    assert actions[0].required_permissions == task.action.required_permissions
    assert actions[1].arguments == {"title": "Clean kitchen"}


@pytest.mark.parametrize(
    ("proposal", "cards", "message"),
    [
        (
            ProposedPlan(steps=(ProposedPlanStep(action_ref="invented.create"),)),
            (card("tasks.create"),),
            "authorized candidate",
        ),
        (
            ProposedPlan(
                steps=(
                    ProposedPlanStep(
                        action_ref="tasks.create", arguments={"title": "x", "secret": True}
                    ),
                )
            ),
            (card("tasks.create", "title"),),
            "ActionCard contract",
        ),
        (
            ProposedPlan(
                steps=(
                    ProposedPlanStep(action_ref="tasks.create", depends_on=(1,)),
                    ProposedPlanStep(action_ref="chores.create"),
                )
            ),
            (card("tasks.create"), card("chores.create")),
            "earlier steps",
        ),
        (
            ProposedPlan(steps=(ProposedPlanStep(action_ref="tasks.create", depends_on=(0,)),)),
            (card("tasks.create"),),
            "earlier steps",
        ),
    ],
)
def test_proposed_plan_rejects_untrusted_shape(proposal, cards, message):
    with pytest.raises(PlanValidationError, match=message):
        materialize_proposed_plan(proposal, cards)


def test_proposed_plan_rejects_duplicate_candidate_ids():
    proposal = ProposedPlan(steps=(ProposedPlanStep(action_ref="tasks.create"),))
    duplicate = card("tasks.create")

    with pytest.raises(PlanValidationError, match="duplicate action IDs"):
        materialize_proposed_plan(proposal, (duplicate, duplicate))


def test_proposed_plan_is_bounded():
    with pytest.raises(ValueError):
        ProposedPlan(
            steps=tuple(ProposedPlanStep(action_ref=f"tasks.{index}") for index in range(6))
        )


def test_proposed_plan_does_not_copy_card_mutable_arguments():
    task = card("tasks.create", "title")
    proposal = ProposedPlan(
        steps=(ProposedPlanStep(action_ref="tasks.create", arguments={"title": "x"}),)
    )
    action = materialize_proposed_plan(proposal, (task,))[0]

    assert action.arguments == {"title": "x"}
    assert action.arguments is not task.action.arguments


def test_plan_progress_reads_only_authorized_persisted_step_state():
    result = PlanProgressFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="what's left?",
        ),
        Context(
            sources=("authorized_canonical_result",),
            values={
                "plan_steps": [
                    {"index": 0, "state": "completed"},
                    {"index": 1, "state": "blocked"},
                ]
            },
        ),
    )

    assert result is not None
    assert result.message == "1 of 2 plan steps are complete; 1 remain."
    assert result.evidence == {"plan_progress": {"completed": 1, "total": 2}}


def test_plan_progress_does_not_trust_unscoped_context():
    assert (
        PlanProgressFastPath.resolve(
            IntentFrame(
                principal=Principal(id="alice", vault_id="alice-vault"),
                utterance="what's left?",
            ),
            Context(values={"plan_steps": [{"index": 0, "state": "completed"}]}),
        )
        is None
    )
