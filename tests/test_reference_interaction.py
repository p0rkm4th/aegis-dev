from datetime import datetime, timezone
from typing import cast
from uuid import uuid4

from aegis.contracts import (
    ActionCard,
    ActionSpec,
    Context,
    Decision,
    DecisionKind,
    IntentFrame,
    ObjectiveState,
    Principal,
    Result,
    VerificationContract,
)
from aegis.household import PostgresHouseholdStore
from aegis.interaction_context import (
    compact_context_evidence,
    resolve_obvious_ordinal,
    resolve_obvious_ordinal_item,
)
from aegis.personal import MemoryRecord, PersonalMemoryFastPath, PersonalState, Provenance
from aegis.reference_interaction import (
    ground_reference_action,
    reference_format_result,
    resolve_contextual_ordinal_read,
    resolve_contextual_remaining,
    rewrite_reference_decision,
)
from aegis.tasks import Task, requested_task_due_at


def _task_card(arguments: dict[str, object]) -> ActionCard:
    return ActionCard(
        action=ActionSpec(
            action_id="tasks.create",
            capability="tasks.write",
            arguments=arguments,
        ),
        summary="Create a task",
        relevance=1,
        argument_keys=("title", "due_at"),
    )


def test_task_completion_cannot_ground_a_chore_referent_from_wrong_context():
    card = ActionCard(
        action=ActionSpec(
            action_id="tasks.complete",
            capability="tasks.complete",
            arguments={"title": "Review backup architecture"},
        ),
        summary="Complete a task",
        relevance=1,
        argument_keys=("title",),
    )
    result = ground_reference_action(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which chore is second?",
        ),
        card,
        None,
        None,
        PersonalState(),
        None,
        None,
        None,
        None,
        Context(
            values={
                "referents": {
                    "those": {
                        "fact_key": "canonical_tasks",
                        "candidates": [{"title": "Review backup architecture", "status": "open"}],
                    }
                }
            },
            sources=("authorized_canonical_result",),
        ),
    )
    assert result.state is ObjectiveState.BLOCKED


def test_contextual_ordinal_read_stays_in_authorized_task_domain():
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [
                        {"title": "first task", "status": "open"},
                        {"title": "second task", "status": "open", "due_at": "2026-09-03"},
                    ],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )
    result = resolve_contextual_ordinal_read(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What about the second one?",
        ),
        context,
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.message == "Task: second task (open); due 2026-09-03"
    assert result.evidence["authorized_ordinal_referent"]["title"] == "second task"


def test_grocery_ordinal_preserves_collection_for_correction():
    context = Context(
        values={
            "referents": {"those": {"fact_key": "canonical_items", "candidates": ["rice", "beans"]}}
        },
        sources=("authorized_canonical_result",),
    )
    result = resolve_contextual_ordinal_read(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which grocery item is last?",
        ),
        context,
    )
    assert result is not None
    assert result.evidence["canonical_items"] == ["rice", "beans"]
    assert reference_format_result(result) == "Grocery item: beans"
    correction = resolve_contextual_ordinal_read(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Actually, I meant the first one.",
        ),
        context,
    )
    assert correction is not None
    assert correction.evidence["authorized_ordinal_item"] == "rice"


def test_contextual_remaining_returns_open_tasks_from_authorized_prior_list():
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [
                        {"title": "done task", "status": "completed"},
                        {"title": "remaining task", "status": "open"},
                    ],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )

    result = resolve_contextual_remaining(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What's left?",
        ),
        context,
    )

    assert result is not None
    assert result.evidence["canonical_tasks"] == [{"title": "remaining task", "status": "open"}]


def test_contextual_remaining_preserves_grocery_and_chore_collections():
    principal = Principal(id="alice", vault_id="alice-vault")
    cases = (
        ("canonical_items", ["milk", "eggs"], "canonical_items"),
        ("canonical_chores", [{"title": "wash dishes", "completed": False}], "chores"),
    )
    for fact_key, candidates, evidence_key in cases:
        result = resolve_contextual_remaining(
            IntentFrame(principal=principal, utterance="What remains?"),
            Context(
                values={"referents": {"those": {"fact_key": fact_key, "candidates": candidates}}},
                sources=("authorized_canonical_result",),
            ),
        )
        assert result is not None
        assert result.evidence[evidence_key] == candidates


def test_contextual_ordinal_read_stays_in_authorized_grocery_domain():
    context = Context(
        values={
            "referents": {"those": {"fact_key": "canonical_items", "candidates": ["rice", "beans"]}}
        },
        sources=("authorized_canonical_result",),
    )
    assert resolve_obvious_ordinal_item("What about the second one?", context) == "beans"
    result = resolve_contextual_ordinal_read(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What about the second one?",
        ),
        context,
    )

    assert result is not None
    assert result.message == "Grocery item: beans"
    assert result.evidence["authorized_ordinal_item"] == "beans"


def test_contextual_ordinal_read_stays_in_authorized_event_domain():
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "events",
                    "candidates": [
                        {"title": "Apartment inspection", "starts_at": "2026-09-03T10:00:00+00:00"},
                        {"title": "Dentist appointment", "starts_at": "2026-09-04T15:00:00+00:00"},
                    ],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )

    result = resolve_contextual_ordinal_read(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which appointment is first?",
        ),
        context,
    )

    assert result is not None
    assert result.message == "Event: Apartment inspection (open); starts 2026-09-03T10:00:00+00:00"


def test_reference_action_grounding_rejects_unrequested_model_deadline() -> None:
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance="add review the restore drill to my list",
    )

    result = ground_reference_action(
        intent,
        _task_card({"title": "review the restore drill", "due_at": "2099-01-01"}),
        task_store=object(),
        household_store=cast(PostgresHouseholdStore, object()),
        personal_state=PersonalState(),
        goal_task_title=None,
        goal_chore_title=None,
        memory_task_title=None,
        memory_chore_title=None,
    )

    assert isinstance(result, Result)
    assert result.state.value == "blocked"
    assert "deadline" in result.message.lower()


def test_reference_action_grounding_preserves_explicit_deadline() -> None:
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance="add review the restore drill to my list tomorrow",
    )

    result = ground_reference_action(
        intent,
        _task_card({"title": "review the restore drill"}),
        task_store=object(),
        household_store=cast(PostgresHouseholdStore, object()),
        personal_state=PersonalState(),
        goal_task_title=None,
        goal_chore_title=None,
        memory_task_title=None,
        memory_chore_title=None,
    )

    assert isinstance(result, ActionCard)
    assert result.action.arguments["title"] == "review the restore drill"
    assert isinstance(result.action.arguments["due_at"], str)
    assert datetime.fromisoformat(result.action.arguments["due_at"]).tzinfo == timezone.utc


def test_reference_action_grounding_blocks_unknown_completion_target() -> None:
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance="finish the nonexistent task zz-aegis-safety-probe-2026",
    )

    class TaskStore:
        def list(self, _principal: Principal) -> tuple[Task, ...]:
            return (Task(uuid4(), "apartment", "rotate the backups", "alice"),)

    card = ActionCard(
        action=ActionSpec(
            action_id="tasks.complete",
            capability="tasks.write",
            arguments={"title": "zz-aegis-safety-probe-2026"},
            verification=VerificationContract(kind="readback"),
        ),
        summary="Complete a task",
        relevance=1,
        argument_keys=("title",),
    )
    result = ground_reference_action(
        intent,
        card,
        task_store=TaskStore(),
        household_store=cast(PostgresHouseholdStore, object()),
        personal_state=PersonalState(),
        goal_task_title=None,
        goal_chore_title=None,
        memory_task_title=None,
        memory_chore_title=None,
    )

    assert isinstance(result, Result)
    assert result.state.value == "blocked"
    assert "couldn't find" in result.message.lower()


def test_obvious_ordinal_resolves_only_authorized_prior_canonical_tasks() -> None:
    first_id = str(uuid4())
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [
                        {"task_id": first_id, "title": "buy milk", "status": "open"},
                        {"title": "send invoice", "status": "open"},
                    ],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )

    assert resolve_obvious_ordinal("complete the first one", context, "canonical_tasks") == {
        "task_id": first_id,
        "title": "buy milk",
        "status": "open",
    }
    assert resolve_obvious_ordinal("complete the first one", Context(), "canonical_tasks") is None


def test_compact_task_context_prioritizes_deadlines_for_model_working_set() -> None:
    tasks = [
        {"task_id": "first", "title": "first task", "status": "open"},
        {
            "task_id": "second",
            "title": "second task",
            "status": "open",
            "due_at": "2026-09-02T12:00:00+00:00",
        },
    ]

    assert compact_context_evidence({"canonical_tasks": tasks})["canonical_tasks"] == [
        tasks[1],
        tasks[0],
    ]


def test_compact_planning_context_preserves_open_chores() -> None:
    chores = [{"chore_id": "chore-1", "title": "clean the kitchen"}]

    assert compact_context_evidence({"planning": {"open_chores": chores}})["planning"] == {
        "open_chores": chores
    }


def test_compact_planning_context_preserves_priority_candidates() -> None:
    candidates = ["task: review the backup", "chore: clean the kitchen"]

    assert compact_context_evidence({"planning": {"priority_candidates": candidates}})[
        "planning"
    ] == {"priority_candidates": candidates}


def test_reference_task_display_is_bounded_without_truncating_canonical_evidence() -> None:
    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="Canonical task list read",
        correlation_id=uuid4(),
        evidence={
            "canonical_tasks": [{"title": f"task {index}", "status": "open"} for index in range(22)]
        },
    )

    rendered = reference_format_result(result)

    assert "task 0 (open)" in rendered
    assert "task 19 (open)" in rendered
    assert "task 20 (open)" not in rendered
    assert "… and 2 more" in rendered
    assert len(result.evidence["canonical_tasks"]) == 22


def test_reference_chore_display_is_bounded_without_truncating_canonical_evidence() -> None:
    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="Shared household state read",
        correlation_id=uuid4(),
        evidence={
            "chores": [{"title": f"chore {index}", "assignee_id": "alice"} for index in range(22)]
        },
    )

    rendered = reference_format_result(result)

    assert "chore 0 (alice)" in rendered
    assert "chore 19 (alice)" in rendered
    assert "chore 20 (alice)" not in rendered
    assert "… and 2 more" in rendered
    assert len(result.evidence["chores"]) == 22


def test_reference_ordinal_domain_reads_do_not_render_as_mutations() -> None:
    for collection, message, title in (
        ("canonical_tasks", "Task: send the rent receipt (open)", "send the rent receipt"),
        ("canonical_chores", "Chore: clean the kitchen (open)", "clean the kitchen"),
        ("events", "Event: inspection; starts 2026-09-05T10:00:00+00:00", "inspection"),
    ):
        result = Result(
            objective_id=uuid4(),
            state=ObjectiveState.COMPLETED,
            message=message,
            correlation_id=uuid4(),
            evidence={
                "collection": collection,
                "authorized_ordinal_referent": {"title": title},
            },
        )

        rendered = reference_format_result(result)

        assert rendered == message
        assert not rendered.startswith("Done —")


def test_memory_read_fast_path_handles_ordinary_remember_language() -> None:
    memory = MemoryRecord(
        uuid4(),
        "The owner prefers a quiet dark interface.",
        datetime(2026, 9, 1, tzinfo=timezone.utc),
        Provenance.EXPLICIT_USER,
    )
    state = PersonalState(memories={memory.memory_id: memory})
    result = PersonalMemoryFastPath(state).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="what do you remember about me?",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["memories"][0]["content"] == memory.content


def test_memory_followup_reuses_only_authorized_prior_memory_projection() -> None:
    context = Context(
        values={
            "canonical_facts": {
                "memories": [
                    {
                        "content": "The owner prefers a quiet dark interface.",
                        "provenance": "explicit_user",
                    }
                ]
            }
        },
        sources=("authorized_canonical_result",),
    )
    result = PersonalMemoryFastPath(PersonalState()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="tell me more about that",
        ),
        context,
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["memories"][0]["content"] == (
        "The owner prefers a quiet dark interface."
    )


def test_memory_activity_query_returns_timestamped_memories_without_literal_match() -> None:
    memory = MemoryRecord(
        uuid4(),
        "Investigated the server backup failure.",
        datetime(2026, 9, 1, 20, 0, tzinfo=timezone.utc),
        Provenance.OBSERVED,
    )
    state = PersonalState(memories={memory.memory_id: memory})
    result = PersonalMemoryFastPath(
        state, now=datetime(2026, 9, 2, 14, 0, tzinfo=timezone.utc)
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="what did I work on yesterday?",
        )
    )

    assert result is not None
    assert result.evidence["memories"][0]["content"] == memory.content


def test_grounding_uses_current_canonical_task_for_authorized_ordinal() -> None:
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance="complete the first one",
    )
    card = ActionCard(
        action=ActionSpec(
            action_id="tasks.complete",
            capability="tasks.write",
            arguments={"title": "model guessed the wrong task"},
            verification=VerificationContract(kind="readback"),
        ),
        summary="Complete a task",
        relevance=1,
        argument_keys=("title",),
    )
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [{"title": "buy milk", "status": "open"}],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )

    class TaskStore:
        def list(self, _principal: Principal) -> tuple[Task, ...]:
            return (Task(uuid4(), "apartment", "buy milk", "alice"),)

    grounded = ground_reference_action(
        intent,
        card,
        task_store=TaskStore(),
        household_store=cast(PostgresHouseholdStore, object()),
        personal_state=PersonalState(),
        goal_task_title=None,
        goal_chore_title=None,
        memory_task_title=None,
        memory_chore_title=None,
        context=context,
    )

    assert isinstance(grounded, ActionCard)
    assert grounded.action.arguments["title"] == "buy milk"


def test_grounding_uses_authorized_task_id_when_titles_are_duplicated() -> None:
    first_id = uuid4()
    second_id = uuid4()
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance="complete the first one",
    )
    card = ActionCard(
        action=ActionSpec(
            action_id="tasks.complete",
            capability="tasks.write",
            arguments={"title": "model guessed the wrong task"},
            verification=VerificationContract(kind="readback"),
        ),
        summary="Complete a task",
        relevance=1,
        argument_keys=("title",),
    )
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [
                        {
                            "task_id": str(first_id),
                            "title": "review restore drill",
                            "status": "open",
                        },
                        {
                            "task_id": str(second_id),
                            "title": "review restore drill",
                            "status": "open",
                        },
                    ],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )

    class TaskStore:
        def list(self, _principal: Principal) -> tuple[Task, ...]:
            return (
                Task(first_id, "apartment", "review restore drill", "alice"),
                Task(second_id, "apartment", "review restore drill", "alice"),
            )

    grounded = ground_reference_action(
        intent,
        card,
        task_store=TaskStore(),
        household_store=cast(PostgresHouseholdStore, object()),
        personal_state=PersonalState(),
        goal_task_title=None,
        goal_chore_title=None,
        memory_task_title=None,
        memory_chore_title=None,
        context=context,
    )

    assert isinstance(grounded, ActionCard)
    assert grounded.action.arguments == {
        "title": "review restore drill",
        "task_id": str(first_id),
    }


def test_clarification_for_authorized_ordinal_becomes_bounded_completion_action() -> None:
    task_id = str(uuid4())
    card = ActionCard(
        action=ActionSpec(
            action_id="tasks.complete",
            capability="tasks.write",
            verification=VerificationContract(kind="readback"),
        ),
        summary="Complete a task",
        relevance=1,
        argument_keys=("title",),
    )
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [
                        {"task_id": str(uuid4()), "title": "buy eggs", "status": "open"},
                        {"task_id": task_id, "title": "buy milk", "status": "open"},
                    ],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )

    rewritten = rewrite_reference_decision(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="complete the second one",
        ),
        Decision(
            kind=DecisionKind.CLARIFY,
            clarification="Which task do you mean?",
        ),
        (card,),
        context,
    )

    assert isinstance(rewritten, Decision)
    assert rewritten.kind is DecisionKind.ACTION
    assert rewritten.action is not None
    assert rewritten.action.arguments == {"title": "buy milk", "task_id": task_id}


def test_clarification_for_explicit_task_destination_becomes_create_action() -> None:
    rewritten = rewrite_reference_decision(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Could you jot down replace the porch bulb on my to-do list for Friday?",
        ),
        Decision(kind=DecisionKind.CLARIFY, clarification="What do you mean?"),
        (_task_card({}),),
    )

    assert isinstance(rewritten, Decision)
    assert rewritten.kind is DecisionKind.ACTION
    assert rewritten.action is not None
    assert rewritten.action.arguments["title"] == "replace the porch bulb"
    assert (
        requested_task_due_at(
            "Could you jot down replace the porch bulb on my to-do list for Friday?",
            datetime(2026, 9, 2, tzinfo=timezone.utc),
        )
        == "2026-09-04T00:00:00+00:00"
    )
