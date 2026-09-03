from uuid import UUID, uuid4

import pytest

from aegis.contracts import (
    ActionCard,
    ActionSpec,
    Context,
    DecisionKind,
    IntentFrame,
    ModelResponse,
    ObjectiveRequirement,
    ObjectiveRequirementProposal,
    ObjectiveSpec,
    ObjectiveSpecProposal,
    ObjectiveState,
    Principal,
    ProposedPlan,
    ProposedPlanStep,
    VerificationContract,
)
from aegis.decoding import InvalidDecision, StrictDecisionDecoder
from aegis.planning import (
    MultiActionFastPath,
    PlanModificationFastPath,
    PlanProgressFastPath,
    PlanValidationError,
    materialize_proposed_plan,
    materialize_validated_plan,
    objective_requirements_satisfied,
)


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


def test_proposed_plan_rejects_temporal_argument_copied_between_steps():
    cards = (
        card("tasks.create", "title", "due_at"),
        card("tasks.events.create", "title", "starts_at"),
    )
    proposal = ProposedPlan(
        steps=(
            ProposedPlanStep(
                action_ref="tasks.create",
                arguments={"title": "check the latch", "due_at": "next Saturday"},
            ),
            ProposedPlanStep(
                action_ref="tasks.events.create",
                arguments={"title": "inspection", "starts_at": "next Saturday"},
            ),
        )
    )

    with pytest.raises(PlanValidationError, match="temporal argument"):
        materialize_proposed_plan(proposal, cards)


def test_overlapping_fast_paths_decline_three_capability_objectives():
    utterance = (
        "Put checking the basement window on my task list, schedule a plumber inspection "
        "for next Saturday, and add a chore to clear the mudroom bench."
    )

    assert MultiActionFastPath.task_chore_titles(utterance) is None
    assert MultiActionFastPath.task_event_details(utterance) is None


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


def test_validated_plan_requires_exact_one_to_one_requirement_coverage():
    cards = (card("tasks.create", "title"), card("chores.create", "title"))
    objective_id = UUID("00000000-0000-0000-0000-000000000001")
    objective = ObjectiveSpec(
        requirements=(
            ObjectiveRequirement(action_ref="tasks.create", arguments={"title": "a"}),
            ObjectiveRequirement(action_ref="chores.create", arguments={"title": "b"}),
        )
    )
    proposal = ProposedPlan(
        steps=(
            ProposedPlanStep(action_ref="tasks.create", arguments={"title": "a"}),
            ProposedPlanStep(action_ref="chores.create", arguments={"title": "b"}),
        )
    )

    validated = materialize_validated_plan(objective_id, objective, proposal, cards)

    assert [step.requirement_id for step in validated.steps] == [
        requirement.requirement_id for requirement in objective.requirements
    ]
    assert validated.steps[1].depends_on == ()
    assert validated.steps[0].step_id == UUID("e8ea9666-6e66-516e-8733-0b4848100342")


def test_strict_decoder_accepts_plan_with_core_bound_objective_proposal():
    action_card = card("tasks.create", "title")
    second_card = card("chores.create", "title")
    decision = StrictDecisionDecoder().decode(
        ModelResponse(
            raw={
                "kind": "PLAN",
                "semantic_mode": "ACTION",
                "objective_spec": ObjectiveSpecProposal(
                    requirements=(
                        ObjectiveRequirementProposal(
                            action_ref="tasks.create", arguments={"title": "a"}
                        ),
                        ObjectiveRequirementProposal(
                            action_ref="chores.create", arguments={"title": "b"}
                        ),
                    )
                ).model_dump(mode="json"),
                "plan": {
                    "steps": [
                        {"action_ref": "tasks.create", "arguments": {"title": "a"}},
                        {"action_ref": "chores.create", "arguments": {"title": "b"}},
                    ]
                },
            }
        ),
        (action_card, second_card),
        allow_argument_proposals=True,
    )

    assert decision.objective_spec is not None
    assert decision.plan is not None


@pytest.mark.parametrize("arguments", [{"title": "a"}, {"title": "b"}])
def test_validated_plan_rejects_missing_duplicate_or_extra_coverage(arguments):
    cards = (card("tasks.create", "title"), card("chores.create", "title"))
    objective = ObjectiveSpec(
        requirements=(
            ObjectiveRequirement(action_ref="tasks.create", arguments={"title": "a"}),
            ObjectiveRequirement(action_ref="chores.create", arguments={"title": "b"}),
        )
    )
    proposal = ProposedPlan(
        steps=(
            ProposedPlanStep(action_ref="tasks.create", arguments=arguments),
            ProposedPlanStep(action_ref="tasks.create", arguments=arguments),
        )
    )

    with pytest.raises(PlanValidationError, match="coverage"):
        materialize_validated_plan(uuid4(), objective, proposal, cards)


def test_validated_plan_translates_dependencies_to_stable_step_ids():
    cards = (card("tasks.create", "title"), card("chores.create", "title"))
    objective = ObjectiveSpec(
        requirements=(
            ObjectiveRequirement(action_ref="tasks.create", arguments={"title": "a"}),
            ObjectiveRequirement(action_ref="chores.create", arguments={"title": "b"}),
        )
    )
    proposal = ProposedPlan(
        steps=(
            ProposedPlanStep(action_ref="tasks.create", arguments={"title": "a"}),
            ProposedPlanStep(action_ref="chores.create", arguments={"title": "b"}, depends_on=(0,)),
        )
    )

    first = materialize_validated_plan(UUID(int=2), objective, proposal, cards)
    second = materialize_validated_plan(UUID(int=2), objective, proposal, cards)

    assert first == second
    assert second.steps[1].depends_on == (second.steps[0].step_id,)


def test_objective_completion_requires_every_requirement_id():
    objective = ObjectiveSpec(
        requirements=(
            ObjectiveRequirement(action_ref="tasks.create"),
            ObjectiveRequirement(action_ref="chores.create"),
        )
    )
    first_id = objective.requirements[0].requirement_id
    second_id = objective.requirements[1].requirement_id

    assert not objective_requirements_satisfied(objective, {first_id})
    assert objective_requirements_satisfied(objective, {first_id, second_id})


def test_plan_progress_reads_only_authorized_persisted_step_state():
    result = PlanProgressFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="what's left?",
        ),
        Context(
            sources=("authorized_canonical_result",),
            values={
                "canonical_facts": {
                    "plan_steps": [
                        {"index": 0, "state": "completed"},
                        {"index": 1, "state": "blocked"},
                    ]
                }
            },
        ),
    )

    assert result is not None
    assert result.message == "1 of 2 plan steps are complete; 1 remain."
    assert result.evidence == {
        "plan_progress": {"completed": 1, "total": 2},
        "plan_steps": [
            {"index": 0, "state": "completed"},
            {"index": 1, "state": "blocked"},
        ],
    }


def test_plan_progress_prefers_persisted_objective_requirements():
    result = PlanProgressFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="what remains on the objective?",
        ),
        Context(
            sources=("authorized_canonical_result",),
            values={
                "canonical_facts": {
                    "objective_requirements": [
                        {"requirement_id": "req-a", "state": "completed"},
                        {"requirement_id": "req-b", "state": "failed"},
                    ],
                    "plan_steps": [
                        {"index": 0, "state": "completed"},
                        {"index": 1, "state": "completed"},
                    ],
                }
            },
        ),
    )

    assert result is not None
    assert result.message == "1 of 2 requested changes are complete; 1 remain."
    assert result.evidence["progress_basis"] == "persisted_objective_requirements"


def test_plan_progress_does_not_treat_model_done_question_as_completion():
    result = PlanProgressFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="did you finish everything?",
        ),
        Context(
            sources=("authorized_canonical_result",),
            values={
                "canonical_facts": {
                    "objective_requirements": [
                        {"requirement_id": "req-a", "state": "completed"},
                        {"requirement_id": "req-b", "state": "failed"},
                    ]
                }
            },
        ),
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert "1 of 2 requested changes" in result.message


def test_plan_progress_accepts_what_remains_followup():
    result = PlanProgressFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What remains?",
        ),
        Context(
            sources=("authorized_canonical_result",),
            values={
                "canonical_facts": {
                    "plan_steps": [
                        {"index": 0, "state": "completed"},
                        {"index": 1, "state": "completed"},
                    ]
                }
            },
        ),
    )

    assert result is not None
    assert result.message == "All 2 plan steps are complete."


def test_plan_progress_accepts_outstanding_followup():
    result = PlanProgressFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="How much of that is still outstanding?",
        ),
        Context(
            sources=("authorized_canonical_result",),
            values={
                "canonical_facts": {
                    "plan_steps": [
                        {"index": 0, "state": "completed"},
                        {"index": 1, "state": "pending"},
                    ]
                }
            },
        ),
    )

    assert result is not None
    assert result.message == "1 of 2 plan steps are complete; 1 remain."


def test_plan_progress_accepts_still_left_followup():
    result = PlanProgressFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What is still left on that plan?",
        ),
        Context(
            sources=("authorized_canonical_result",),
            values={
                "canonical_facts": {
                    "plan_steps": [
                        {"index": 0, "state": "completed"},
                        {"index": 1, "state": "completed"},
                    ]
                }
            },
        ),
    )

    assert result is not None
    assert result.message == "All 2 plan steps are complete."


def test_plan_progress_yields_to_explicit_domain_followup():
    result = PlanProgressFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What chores still need attention?",
        ),
        Context(
            sources=("authorized_canonical_result",),
            values={"canonical_facts": {"plan_steps": [{"index": 0, "state": "completed"}]}},
        ),
    )

    assert result is None


def test_plan_modification_does_not_rewrite_verified_history():
    result = PlanModificationFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Skip the appointment part.",
        ),
        Context(
            sources=("authorized_canonical_result",),
            values={
                "canonical_facts": {
                    "plan_steps": [
                        {"index": 0, "state": "completed"},
                        {"index": 1, "state": "completed"},
                    ]
                }
            },
        ),
    )

    assert result is not None
    assert result.state.value == "blocked"
    assert "already verified" in result.message
    assert "rewrite" in result.message


def test_plan_progress_does_not_trust_unscoped_context():
    assert (
        PlanProgressFastPath.resolve(
            IntentFrame(
                principal=Principal(id="alice", vault_id="alice-vault"),
                utterance="what's left?",
            ),
            Context(
                values={"canonical_facts": {"plan_steps": [{"index": 0, "state": "completed"}]}}
            ),
        )
        is None
    )


def test_decoder_accepts_candidate_bound_plan_only_in_proposal_mode():
    cards = (card("tasks.create", "title"), card("chores.create", "title"))
    response = type(
        "Response",
        (),
        {
            "raw": {
                "kind": DecisionKind.PLAN.value,
                "semantic_mode": "ACTION",
                "plan": {
                    "steps": [
                        {"action_ref": "tasks.create", "arguments": {"title": "x"}},
                        {
                            "action_ref": "chores.create",
                            "arguments": {"title": "y"},
                            "depends_on": [0],
                        },
                    ]
                },
            }
        },
    )()

    decision = StrictDecisionDecoder().decode(response, cards, allow_argument_proposals=True)

    assert decision.kind is DecisionKind.PLAN
    assert decision.plan is not None
    with pytest.raises(InvalidDecision, match="proposal mode"):
        StrictDecisionDecoder().decode(response, cards)


def test_decoder_rejects_mixed_plan_and_answer():
    cards = (card("tasks.create", "title"), card("chores.create", "title"))
    response = type(
        "Response",
        (),
        {
            "raw": {
                "kind": DecisionKind.PLAN.value,
                "semantic_mode": "ACTION",
                "answer": "also here",
                "plan": {
                    "steps": [
                        {"action_ref": "tasks.create"},
                        {"action_ref": "chores.create"},
                    ]
                },
            }
        },
    )()

    with pytest.raises(InvalidDecision, match="another decision kind"):
        StrictDecisionDecoder().decode(response, cards, allow_argument_proposals=True)
