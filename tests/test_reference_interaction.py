from datetime import datetime, timedelta, timezone
from typing import cast
from uuid import uuid4

from aegis.contracts import (
    ActionCard,
    ActionSpec,
    ArgumentProvenanceKind,
    Context,
    Decision,
    DecisionKind,
    IntentFrame,
    ObjectiveState,
    Principal,
    Result,
    StructuralAnchor,
    StructuralCoverageSignal,
    VerificationContract,
)
from aegis.household import GroceryReadFastPath, PostgresHouseholdStore
from aegis.interaction_cognition import _structural_write_failure
from aegis.interaction_context import (
    compact_context_evidence,
    resolve_obvious_ordinal,
    resolve_obvious_ordinal_item,
)
from aegis.personal import MemoryRecord, PersonalMemoryFastPath, PersonalState, Provenance
from aegis.planning import MultiActionFastPath
from aegis.reference_interaction import (
    ground_reference_action,
    reference_fallback_cards,
    reference_format_result,
    resolve_contextual_event_temporal_read,
    resolve_contextual_ordinal_read,
    resolve_contextual_recent_action_read,
    resolve_contextual_remaining,
    resolve_reference_fast_paths,
    resolve_reference_safety_fast_paths,
    rewrite_reference_decision,
)
from aegis.tasks import Task, requested_task_due_at


def test_memory_fast_path_yields_to_standalone_general_subject_questions() -> None:
    fast_path = PersonalMemoryFastPath(PersonalState())
    for utterance in (
        "Tell me about Cult of the Lamb, the game.",
        "Explain the fall of the Roman Republic.",
        "Tell me about photosynthesis.",
        "What is the Rust programming language?",
        "Tell me about The Left Hand of Darkness.",
    ):
        result = fast_path.resolve(
            IntentFrame(
                principal=Principal(id="alice", vault_id="alice-vault"),
                utterance=utterance,
            )
        )
        assert result is None, utterance


def test_grocery_read_fast_path_does_not_substitute_shopping_list_for_inventory() -> None:
    class GroceryStore:
        def list_groceries(self, _principal: object) -> tuple[str, ...]:
            return ("rice",)

    principal = Principal(id="alice", vault_id="alice-vault")
    for utterance in (
        "What groceries do we have?",
        "What's in the pantry?",
        "How much rice is left?",
    ):
        result = GroceryReadFastPath(cast(PostgresHouseholdStore, GroceryStore())).resolve(
            IntentFrame(principal=principal, utterance=utterance)
        )
        assert result is not None
        assert result.state is ObjectiveState.BLOCKED
        assert "not pantry" in result.message


def test_grocery_read_fast_path_preserves_shopping_list_scope() -> None:
    class GroceryStore:
        def list_groceries(self, _principal: object) -> tuple[str, ...]:
            return ("rice",)

    result = GroceryReadFastPath(cast(PostgresHouseholdStore, GroceryStore())).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What groceries do we need?",
        )
    )
    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["semantic_scope"] == "kitchen.shopping_list"
    assert result.evidence["canonical_items"] == ["rice"]


def test_grocery_read_fast_path_accepts_punctuated_topic_follow_up() -> None:
    class GroceryStore:
        def list_groceries(self, _principal: object) -> tuple[str, ...]:
            return ("rice",)

    result = GroceryReadFastPath(cast(PostgresHouseholdStore, GroceryStore())).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What about groceries?",
        )
    )
    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["canonical_items"] == ["rice"]


def test_grocery_read_fast_path_accepts_explicit_read_correction() -> None:
    class GroceryStore:
        def list_groceries(self, _principal: object) -> tuple[str, ...]:
            return ("rice",)

    result = GroceryReadFastPath(cast(PostgresHouseholdStore, GroceryStore())).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="No, I meant what groceries do we need?",
        )
    )
    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["canonical_items"] == ["rice"]


def test_grocery_read_fast_path_accepts_only_read_correction() -> None:
    class GroceryStore:
        def list_groceries(self, _principal: object) -> tuple[str, ...]:
            return ("rice",)

    result = GroceryReadFastPath(cast(PostgresHouseholdStore, GroceryStore())).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Actually, only show groceries on my list.",
        )
    )
    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["canonical_items"] == ["rice"]


def test_grocery_read_fast_path_does_not_claim_undated_temporal_scope() -> None:
    class GroceryStore:
        def list_groceries(self, _principal: object) -> tuple[str, ...]:
            return ("rice",)

    result = GroceryReadFastPath(cast(PostgresHouseholdStore, GroceryStore())).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What groceries do we need tomorrow?",
        )
    )
    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "not date-specific" in result.message


def test_memory_fast_path_keeps_explicit_memory_requests() -> None:
    result = PersonalMemoryFastPath(PersonalState()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What memories do I have about the apartment?",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED


def test_memory_fast_path_does_not_invent_context_for_vague_followup() -> None:
    result = PersonalMemoryFastPath(PersonalState()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Tell me about that.",
        )
    )

    assert result is None


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


def test_grocery_due_priority_followup_does_not_select_an_ordinal_item():
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_items",
                    "candidates": ["rice", "beans"],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )

    result = resolve_contextual_ordinal_read(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which one is due first?",
        ),
        context,
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "no canonical priority order" in result.message


def test_grocery_direct_due_priority_request_fails_closed():
    class GroceryStore:
        def list_groceries(self, _principal: object) -> tuple[str, ...]:
            return ("rice",)

    result = GroceryReadFastPath(cast(PostgresHouseholdStore, GroceryStore())).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which grocery is due first?",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "no canonical deadline" in result.message


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
    local_date = datetime.fromisoformat("2026-09-03T10:00:00+00:00").astimezone().date().isoformat()
    assert result.message.startswith(f"Event: Apartment inspection (open); starts {local_date}")


def test_ordinal_domain_read_preserves_collection_and_blocks_ambiguous_correction():
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_chores",
                    "candidates": [
                        {"chore_id": "one", "title": "wash dishes", "completed": False},
                        {"chore_id": "two", "title": "take out trash", "completed": False},
                    ],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )
    selected = resolve_contextual_ordinal_read(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which chore is last?",
        ),
        context,
    )

    assert selected is not None
    assert (
        selected.evidence["canonical_chores"] == context.values["referents"]["those"]["candidates"]
    )

    correction = resolve_contextual_ordinal_read(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="No, the other one.",
        ),
        context,
    )

    assert correction is not None
    assert correction.state is ObjectiveState.BLOCKED
    assert "choose an ordinal" in correction.message


def test_contextual_correction_reaches_domain_specific_referent_guard():
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "events",
                    "candidates": [
                        {"title": "first event", "starts_at": "2026-09-03T10:00:00+00:00"},
                        {"title": "second event", "starts_at": "2026-09-03T11:00:00+00:00"},
                    ],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )

    result = resolve_reference_safety_fast_paths(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="No, the other event",
        ),
        None,
        True,
        context,
    )

    assert result is None


def test_bare_ordinal_correction_reads_authorized_task_projection():
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [
                        {"title": "first task", "status": "open"},
                        {"title": "second task", "status": "open"},
                    ],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )

    result = resolve_contextual_ordinal_read(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="No, the second one.",
        ),
        context,
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["authorized_ordinal_referent"]["title"] == "second task"


def test_ordinal_priority_word_cannot_cross_from_tasks_into_chores():
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [
                        {"title": "check the gate", "status": "open", "due_at": "2026-09-05"}
                    ],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )

    result = resolve_contextual_ordinal_read(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which chore is due first?",
        ),
        context,
    )

    assert result is None


def test_event_temporal_follow_up_reuses_authorized_event_collection():
    from aegis.household import HouseholdEvent

    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "events",
                    "candidates": [{"title": "tomorrow event"}],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )
    result = resolve_contextual_event_temporal_read(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What about tomorrow?",
        ),
        context,
        {
            "space_id": "home",
            "obligations": (),
            "chores": (),
            "events": (HouseholdEvent("tomorrow", "tomorrow event", tomorrow),),
        },
    )

    assert result is not None
    assert result.evidence["date_filter"] == "tomorrow"
    assert result.evidence["events"][0]["title"] == "tomorrow event"


def test_event_temporal_correction_reuses_authorized_event_collection():
    from aegis.household import HouseholdEvent

    now = datetime.now(timezone.utc)
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "events",
                    "candidates": [{"title": "weekend event"}],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )
    result = resolve_contextual_event_temporal_read(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="No, this weekend.",
        ),
        context,
        {
            "space_id": "home",
            "obligations": (),
            "chores": (),
            "events": (HouseholdEvent("weekend", "weekend event", now),),
        },
    )

    assert result is not None
    assert result.evidence["date_filter"] == "this_weekend"
    assert result.evidence["events"][0]["title"] == "weekend event"


def test_bounded_task_event_plan_owns_its_dependent_reference():
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance=(
            "Add a task to check the water softener and schedule an appointment "
            "to check it tomorrow."
        ),
    )

    assert resolve_reference_safety_fast_paths(intent, None, True) is None


def test_model_enabled_structural_compound_reaches_bounded_cognition():
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance=(
            "Get the guest room ready for Friday: put checking the towels on my task list "
            "and schedule a bathroom inspection for Friday."
        ),
    )

    assert MultiActionFastPath.matches(intent.utterance)
    assert resolve_reference_safety_fast_paths(intent, None, True) is None


def test_compound_task_destination_retrieves_event_candidate():
    class Manager:
        def retrieve(self, domain, limit=5):
            assert domain == "tasks"
            assert limit == 10
            return tuple(_task_card({"title": str(index)}) for index in range(6))

    cards = reference_fallback_cards(
        Manager(),
        "Put checking towels on my task list and schedule a bathroom inspection Friday.",
    )

    assert len(cards) == 6


def test_explicit_grocery_topic_switch_does_not_reuse_task_fallback() -> None:
    calls: list[str] = []

    class Manager:
        def retrieve(self, domain, limit=5):
            calls.append(domain)
            return tuple(_task_card({"title": str(index)}) for index in range(limit))

    cards = reference_fallback_cards(
        Manager(),
        "What about groceries?",
        Context(
            values={"canonical_facts": {"canonical_tasks": [{"title": "old task"}]}},
            sources=("authorized_canonical_context",),
        ),
    )

    assert calls == ["kitchen"]
    assert len(cards) == 5


def test_grocery_context_does_not_turn_ambiguous_date_follow_up_into_mutation() -> None:
    result = resolve_reference_safety_fast_paths(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What about tomorrow?",
        ),
        None,
        True,
        Context(
            values={
                "referents": {"those": {"fact_key": "canonical_items", "candidates": ["rice"]}}
            },
            sources=("authorized_canonical_result",),
        ),
    )
    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "tasks, events, or groceries" in result.message


def test_structural_compound_single_action_is_blocked_before_execution():
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance="Put checking towels on my task list and schedule a bathroom inspection Friday.",
    )
    decision = Decision(
        kind=DecisionKind.ACTION,
        action=_task_card({"title": "checking towels"}).action,
        semantic_mode="ACTION",
    )

    result = rewrite_reference_decision(
        intent, decision, (_task_card({"title": "checking towels"}),)
    )

    assert isinstance(result, Result)
    assert result.state is ObjectiveState.BLOCKED
    assert "No action was executed" in result.message


def test_negated_structural_write_proposal_is_blocked_before_execution():
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance="Make the inspection task, not the cleaning chore.",
    )
    decision = Decision(
        kind=DecisionKind.ACTION,
        action=_task_card({"title": "inspection task"}).action,
        semantic_mode="ACTION",
    )

    class Dependencies:
        structural_parser = staticmethod(
            lambda _utterance: StructuralCoverageSignal(
                anchors=(StructuralAnchor(source_span=(0, 4), kind="predicate"),),
                negation_spans=((5, 8),),
            )
        )

    card = _task_card({}).model_copy(
        update={
            "action": decision.action.model_copy(update={"required_permissions": ("tasks.write",)})
        }
    )
    result = _structural_write_failure(Dependencies(), intent, decision, (card,))

    assert result is not None
    assert "negated" in result


def test_scalar_task_focus_allows_explicit_completion_followup():
    context = Context(
        values={
            "canonical_facts": {
                "task": {
                    "task_id": "11111111-1111-4111-8111-111111111111",
                    "title": "renew insurance",
                    "status": "open",
                }
            }
        },
        sources=("authorized_canonical_result",),
    )

    assert (
        resolve_reference_safety_fast_paths(
            IntentFrame(
                principal=Principal(id="alice", vault_id="alice-vault"),
                utterance="Complete that one.",
            ),
            None,
            True,
            context,
        )
        is None
    )


def test_compound_mutation_cannot_be_swallowed_by_contextual_read_fast_path():
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance=(
            "Get the guest room ready for Friday: add a task to check the linens "
            "and schedule an appointment to inspect the smoke alarm Friday."
        ),
    )
    # The recognized plan must be handed to the plan runner even when a prior
    # authorized task result is present; otherwise TaskReadFastPath wins.
    assert (
        resolve_reference_fast_paths(
            intent,
            object(),
            intent.principal,
            Context(
                values={
                    "canonical_facts": {"canonical_tasks": []},
                    "referents": {},
                },
                sources=("authorized_canonical_result",),
            ),
            None,
            lambda name: name,
            True,
        )
        is None
    )


def test_compound_cross_domain_read_does_not_claim_only_one_result() -> None:
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance="What tasks are still open and what groceries do we need?",
    )
    result = resolve_reference_fast_paths(
        intent, object(), intent.principal, Context(), None, lambda name: name, True
    )
    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "multiple independent reads" in result.message


def test_grocery_collection_has_no_task_priority_semantics() -> None:
    result = resolve_contextual_ordinal_read(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which one should I handle first?",
        ),
        Context(
            values={
                "referents": {
                    "those": {
                        "fact_key": "canonical_items",
                        "candidates": ["rice", "beans"],
                    }
                }
            },
            sources=("authorized_canonical_result",),
        ),
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "no canonical priority" in result.message


def test_non_question_correction_after_blocked_result_cannot_create_action() -> None:
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance="Actually, skip the chore part.",
    )
    result = resolve_reference_safety_fast_paths(
        intent,
        None,
        True,
        Context(
            values={"canonical_facts": {}, "referents": {}},
            sources=("authorized_prior_result",),
        ),
    )
    assert result is not None
    assert result.state is ObjectiveState.BLOCKED


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


def test_reference_event_grounding_blocks_missing_user_supplied_time() -> None:
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance="schedule the review",
    )
    card = ActionCard(
        action=ActionSpec(
            action_id="tasks.events.create",
            capability="tasks.events.create",
            arguments={"title": "review", "starts_at": "2026-09-04T00:00:00+00:00"},
            required_permissions=("tasks.write",),
            verification=VerificationContract(kind="readback"),
        ),
        summary="Schedule an event",
        relevance=1,
        argument_keys=("title", "starts_at"),
    )

    result = ground_reference_action(
        intent,
        card,
        task_store=object(),
        household_store=cast(PostgresHouseholdStore, object()),
        personal_state=PersonalState(),
        goal_task_title=None,
        goal_chore_title=None,
        memory_task_title=None,
        memory_chore_title=None,
    )

    assert isinstance(result, Result)
    assert result.state is ObjectiveState.BLOCKED
    assert "date and time" in result.message.lower()


def test_reference_event_grounding_rejects_model_invented_clock_time() -> None:
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance="schedule the review tomorrow",
    )
    card = ActionCard(
        action=ActionSpec(
            action_id="tasks.events.create",
            capability="tasks.events.create",
            arguments={"title": "review", "starts_at": "2026-09-05T15:00:00+00:00"},
            required_permissions=("tasks.write",),
            verification=VerificationContract(kind="readback"),
        ),
        summary="Schedule an event",
        relevance=1,
        argument_keys=("title", "starts_at"),
    )

    result = ground_reference_action(
        intent,
        card,
        task_store=object(),
        household_store=cast(PostgresHouseholdStore, object()),
        personal_state=PersonalState(),
        goal_task_title=None,
        goal_chore_title=None,
        memory_task_title=None,
        memory_chore_title=None,
    )

    assert isinstance(result, Result)
    assert result.state is ObjectiveState.BLOCKED
    assert result.evidence["failure"] == "CONSEQUENTIAL_ARGUMENT_PROVENANCE_UNAVAILABLE"


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


def test_recent_canonical_action_result_can_answer_follow_up_without_model() -> None:
    context = Context(
        values={
            "canonical_facts": {
                "collection": "tasks",
                "title": "check the backup checklist",
                "status": "open",
            }
        },
        sources=("authorized_canonical_result",),
    )
    result = resolve_contextual_recent_action_read(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What did you just add?",
        ),
        context,
    )
    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.message == "Task: check the backup checklist"
    assert result.evidence["referent"] == "prior_result"


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

    assert rendered.startswith("Tasks:\n• task 0 (open)\n• task 1 (open)")
    assert "task 0 (open)" in rendered
    assert "task 19 (open)" in rendered
    assert "task 20 (open)" not in rendered
    assert "… and 2 more" in rendered
    assert len(result.evidence["canonical_tasks"]) == 22


def test_reference_task_display_shortens_canonical_due_timestamp() -> None:
    due_at = "2026-09-04T00:08:21.956546+00:00"
    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="Canonical task list read",
        correlation_id=uuid4(),
        evidence={
            "canonical_tasks": [
                {"title": "replace the porch bulb", "status": "open", "due_at": due_at}
            ]
        },
    )

    rendered = reference_format_result(result)

    assert "replace the porch bulb (open) due " in rendered
    assert ".956546" not in rendered
    assert result.evidence["canonical_tasks"][0]["due_at"] == due_at


def test_contextual_ordinal_task_display_shortens_canonical_due_timestamp() -> None:
    due_at = "2026-09-04T00:08:21.956546+00:00"
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [{"title": "replace filter", "status": "open", "due_at": due_at}],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )

    result = resolve_contextual_ordinal_read(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which task is first?",
        ),
        context,
    )

    assert result is not None
    local_date = datetime.fromisoformat(due_at).astimezone().date().isoformat()
    assert result.message.startswith(f"Task: replace filter (open); due {local_date}")
    assert "+00:00" not in result.message
    assert ".956546" not in result.message
    assert due_at not in result.message


def test_contextual_ordinal_event_display_shortens_canonical_start_timestamp() -> None:
    starts_at = "2026-09-04T02:08:21.956546+00:00"
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "events",
                    "candidates": [
                        {"title": "inspection", "status": "open", "starts_at": starts_at}
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
    local_date = datetime.fromisoformat(starts_at).astimezone().date().isoformat()
    assert result.message.startswith(f"Event: inspection (open); starts {local_date}")
    assert "+00:00" not in result.message
    assert ".956546" not in result.message
    assert starts_at not in result.message


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


def test_reference_event_display_is_bounded_without_truncating_canonical_evidence() -> None:
    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="Canonical event list read",
        correlation_id=uuid4(),
        evidence={"events": [{"title": f"event {index}"} for index in range(22)]},
    )

    rendered = reference_format_result(result)

    assert "event 0" in rendered
    assert "event 19" in rendered
    assert "event 20" not in rendered
    assert "… and 2 more" in rendered
    assert len(result.evidence["events"]) == 22


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


def test_memory_read_fast_path_handles_plural_memory_list_request() -> None:
    memory = MemoryRecord(
        uuid4(),
        "The owner keeps the house interface calm and dark.",
        datetime(2026, 9, 1, tzinfo=timezone.utc),
        Provenance.EXPLICIT_USER,
    )
    state = PersonalState(memories={memory.memory_id: memory})
    result = PersonalMemoryFastPath(state).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What memories do I have?",
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


def test_memory_followup_clarifies_when_prior_list_has_multiple_referents() -> None:
    context = Context(
        values={
            "canonical_facts": {
                "memories": [
                    {
                        "content": "The owner prefers quiet mornings.",
                        "provenance": "explicit_user",
                    },
                    {
                        "content": "The owner is planning a pantry project.",
                        "provenance": "explicit_user",
                    },
                ]
            }
        },
        sources=("authorized_canonical_result",),
    )

    result = PersonalMemoryFastPath(PersonalState()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Tell me more about that.",
        ),
        context,
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "Which memory" in result.message


def test_memory_followup_does_not_cross_into_memory_without_memory_context() -> None:
    context = Context(
        values={"canonical_facts": {"canonical_items": ["rice"]}},
        sources=("authorized_canonical_result",),
    )

    result = PersonalMemoryFastPath(PersonalState()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Tell me more about that",
        ),
        context,
    )

    assert result is None


def test_general_continuation_wording_does_not_claim_canonical_network_as_memory() -> None:
    context = Context(
        values={"canonical_facts": {"network": {"devices": ["router"]}}},
        sources=("authorized_canonical_result",),
    )

    result = PersonalMemoryFastPath(PersonalState()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Tell me about my network",
        ),
        context,
    )

    assert result is None


def test_explicit_memory_topic_switch_survives_other_canonical_context() -> None:
    memory = MemoryRecord(
        uuid4(),
        "The owner prefers a calm workstation.",
        datetime(2026, 9, 1, tzinfo=timezone.utc),
        Provenance.EXPLICIT_USER,
    )
    context = Context(
        values={"canonical_facts": {"canonical_items": ["rice"]}},
        sources=("authorized_canonical_result",),
    )

    result = PersonalMemoryFastPath(PersonalState(memories={memory.memory_id: memory})).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What memories do I have?",
        ),
        context,
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["memories"][0]["content"] == memory.content


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
    assert (
        grounded.action.argument_provenance["title"].kind
        is ArgumentProvenanceKind.AUTHORIZED_CANONICAL_REFERENT
    )


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
    assert (
        grounded.action.argument_provenance["task_id"].kind
        is ArgumentProvenanceKind.AUTHORIZED_CANONICAL_REFERENT
    )


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
