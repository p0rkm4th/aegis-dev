"""First-party interaction composition for the reference Packs.

This module is deliberately outside the shared InteractionBoundary.  It owns
domain-specific fast-path grounding that the reference Tasks/Household/Personal
Packs currently need; the boundary receives only a typed card/result callback.
Pack-specific behavior therefore remains composition-owned and cannot become
the generic client/Core contract.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, cast
from uuid import uuid4

from .audit import PostgresAuditLog
from .contracts import (
    ActionCard,
    ActionSpec,
    Context,
    Decision,
    DecisionKind,
    IntentFrame,
    ObjectiveState,
    Principal,
    Result,
)
from .decoding import StrictDecisionDecoder
from .dispatch import ActionExecutorDispatch, ActionVerifierDispatch
from .embeddings import OllamaEmbeddingProvider, PostgresMemoryVectorIndex
from .finance import FinanceLedger, FinanceReadFastPath, PostgresFinanceSnapshotStore
from .household import (
    Chore,
    ChoreCompletionFastPath,
    GroceryReadFastPath,
    HouseholdObligation,
    HouseholdReadFastPath,
    PostgresHouseholdStore,
)
from .identity import PostgresSpacePolicy, Role
from .kernel import Kernel
from .pack_lifecycle import PackManager
from .pack_runtime import PackRuntimeRegistry
from .personal import PersonalMemoryFastPath, PersonalState, PostgresPersonalStateStore
from .planning import (
    CrossDomainPlanningFastPath,
    MultiActionFastPath,
    PersonalChoreComposer,
    PersonalMemoryChoreComposer,
    PersonalMemoryTaskComposer,
    PersonalTaskComposer,
)
from .projections import SharedObligation
from .store import PostgresObjectiveStore
from .tasks import (
    ContextualTaskPriorityFastPath,
    PostgresTaskStore,
    TaskCompletionFastPath,
    TaskIntentClarificationFastPath,
    TaskPriorityFastPath,
    TaskReadFastPath,
    ground_task_due_at,
    requested_task_due_at,
)
from .utterance import is_task_destination_request


def reference_fallback_cards(manager: PackManager, utterance: str) -> tuple[ActionCard, ...]:
    """Reduce legacy no-provider candidates for the reference Packs."""

    text = utterance.casefold()
    domain = next(
        (
            pack_id
            for marker, pack_id in (
                ("task", "tasks"),
                ("chore", "tasks"),
                ("event", "tasks"),
                ("grocery", "kitchen"),
                ("grocerie", "kitchen"),
                ("homelab", "homelab"),
                ("service", "homelab"),
                ("network", "network"),
            )
            if marker in text
        ),
        None,
    )
    if is_task_destination_request(utterance):
        return tuple(manager.retrieve("tasks"))[:10]
    cards = manager.retrieve(domain) if domain is not None else manager.enabled_cards()
    return tuple(cards)[:10]


def resolve_reference_fast_paths(
    intent: IntentFrame,
    connection: Any,
    principal: Principal,
    context: Context,
    recovered_plan_actions: tuple[ActionSpec, ...] | None,
    required: Callable[[str], str],
    model_enabled: bool,
) -> Result | None:
    """Resolve reference-Pack reads and personal grounding before cognition."""

    if recovered_plan_actions is not None:
        return None
    utterance = intent.utterance
    task_store = PostgresTaskStore(connection)
    household_store = PostgresHouseholdStore(connection)
    personal_state = PostgresPersonalStateStore(connection, principal.vault_id).load_for_principal(
        principal
    )
    composer_results = (
        PersonalTaskComposer.resolve(utterance, personal_state),
        PersonalChoreComposer.resolve(utterance, personal_state),
        PersonalMemoryTaskComposer.resolve(utterance, personal_state),
        PersonalMemoryChoreComposer.resolve(utterance, personal_state),
    )
    errors = tuple(error for _title, error in composer_results if error is not None)
    if errors:
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=errors[0],
            correlation_id=intent.correlation_id,
        )
    composed_title = next((title for title, _error in composer_results if title is not None), None)
    snapshot = household_store.read_snapshot(principal)
    if composed_title is None and HouseholdReadFastPath.matches(utterance):
        result = HouseholdReadFastPath(snapshot).resolve(intent)
        if result is not None:
            return result
    if composed_title is None:
        result = GroceryReadFastPath(household_store).resolve(intent)
        if result is not None:
            return result
        if not model_enabled:
            result = TaskIntentClarificationFastPath.resolve(intent)
            if result is not None:
                return result
        result = TaskReadFastPath(task_store).resolve(intent)
        if result is not None:
            return result
        result = ContextualTaskPriorityFastPath().resolve(intent, context)
        if result is not None:
            return result
        result = TaskPriorityFastPath(task_store).resolve(intent)
        if result is not None:
            return result
    semantic_enabled = os.environ.get("AEGIS_SEMANTIC_MEMORY", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    if semantic_enabled:
        embedding_provider = OllamaEmbeddingProvider(
            os.environ.get("AEGIS_EMBEDDING_MODEL", "nomic-embed-text"),
            required("AEGIS_OLLAMA_URL"),
        )
        vector_index = PostgresMemoryVectorIndex(connection)
        embeddings = embedding_provider.embed(
            tuple(memory.content for memory in personal_state.memories.values())
        )
        for memory, embedding in zip(personal_state.memories.values(), embeddings):
            vector_index.upsert(
                principal.vault_id, memory.memory_id, embedding, embedding_provider.model
            )
        connection.commit()
        memory_fast_path = PersonalMemoryFastPath(
            personal_state,
            embedding_provider=embedding_provider,
            vector_index=vector_index,
            vault_id=principal.vault_id,
        )
    else:
        memory_fast_path = PersonalMemoryFastPath(personal_state)
    if composed_title is None:
        return memory_fast_path.resolve(intent)
    return None


def run_reference_plan(
    intent: IntentFrame,
    connection: Any,
    principal: Principal,
    manager: PackManager,
    task_store: PostgresTaskStore,
    household_store: PostgresHouseholdStore,
    recovered_plan_actions: tuple[ActionSpec, ...] | None,
    context: Context,
    model: Any,
    runtime_registry: PackRuntimeRegistry | None,
) -> Result | None:
    """Build and execute the reference Pack's bounded multi-action plans."""

    plan_actions: tuple[ActionSpec, ...] | None
    if recovered_plan_actions is not None:
        plan_actions = recovered_plan_actions
    elif (plan_titles := MultiActionFastPath.task_chore_titles(intent.utterance)) is not None:
        task_card = next(
            card for card in manager.retrieve("tasks") if card.action.action_id == "tasks.create"
        )
        chore_card = next(
            card
            for card in manager.retrieve("tasks")
            if card.action.action_id == "tasks.chores.create"
        )
        plan_actions = (
            task_card.action.model_copy(update={"arguments": {"title": plan_titles[0]}}),
            chore_card.action.model_copy(update={"arguments": {"title": plan_titles[1]}}),
        )
    elif (event_details := MultiActionFastPath.task_event_details(intent.utterance)) is not None:
        task_card = next(
            card for card in manager.retrieve("tasks") if card.action.action_id == "tasks.create"
        )
        event_card = next(
            card
            for card in manager.retrieve("tasks")
            if card.action.action_id == "tasks.events.create"
        )
        plan_actions = (
            task_card.action.model_copy(update={"arguments": {"title": event_details[0]}}),
            event_card.action.model_copy(
                update={
                    "arguments": {
                        "title": event_details[1],
                        "starts_at": event_details[2],
                    }
                }
            ),
        )
    else:
        plan_actions = None

    if plan_actions is None:
        return None
    if runtime_registry is None:
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.FAILED,
            message="Pack runtime is unavailable; request can be retried",
            correlation_id=intent.correlation_id,
            retryable=True,
        )
    runtimes = {}
    for action in plan_actions:
        card = next(
            card for card in manager.retrieve("tasks") if card.action.action_id == action.action_id
        )
        runtimes[action.action_id] = runtime_registry.resolve(card, connection, principal)
    return Kernel(
        model,
        StrictDecisionDecoder(),
        PostgresSpacePolicy(
            connection,
            {"tasks.write": frozenset({Role.OWNER, Role.MEMBER})},
        ),
        ActionExecutorDispatch(
            {action_id: runtime.executor for action_id, runtime in runtimes.items()}
        ),
        ActionVerifierDispatch(
            {action_id: runtime.verifier for action_id, runtime in runtimes.items()}
        ),
        store=PostgresObjectiveStore(connection),
        audit=PostgresAuditLog(connection),
    ).run_sequence(intent, plan_actions, context=context)


def rewrite_reference_decision(
    intent: IntentFrame, decision: Decision, cards: tuple[ActionCard, ...]
) -> Decision | Result | None:
    """Correct a reference-Pack event proposal when the user named a task destination."""

    action = decision.action
    if action is None or action.action_id != "tasks.events.create":
        return None
    if not is_task_destination_request(intent.utterance):
        return None
    task_card = next((card for card in cards if card.action.action_id == "tasks.create"), None)
    if task_card is None:
        return None
    title = action.arguments.get("title")
    if not isinstance(title, str) or not title.strip():
        return Decision(
            kind=DecisionKind.CLARIFY,
            clarification="What should I add to your task list?",
        )
    arguments: dict[str, Any] = {"title": title}
    due_at = requested_task_due_at(intent.utterance)
    if due_at is not None:
        arguments["due_at"] = due_at
    return Decision(
        kind=DecisionKind.ACTION,
        action=task_card.action.model_copy(update={"arguments": arguments}),
        semantic_mode="ACTION",
    )


def resolve_reference_pre_model(
    intent: IntentFrame,
    connection: Any,
    principal: Principal,
    household_store: PostgresHouseholdStore,
) -> Result | None:
    """Resolve reference-Pack finance/planning fast paths before cognition."""

    utterance = intent.utterance
    if FinanceReadFastPath.needs_purchase_amount(utterance):
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=(
                "What purchase amount should I compare with your available balance and obligations?"
            ),
            correlation_id=intent.correlation_id,
        )

    task_store = PostgresTaskStore(connection)
    personal_state = PostgresPersonalStateStore(connection, principal.vault_id).load_for_principal(
        principal
    )
    household_snapshot = household_store.read_snapshot(principal)
    raw_obligations = cast(
        tuple[HouseholdObligation, ...], household_snapshot.get("obligations", ())
    )
    obligations = tuple(
        SharedObligation(item.title, item.amount) for item in raw_obligations if not item.settled
    )
    finance: dict[str, Any] | None = None
    if FinanceReadFastPath.matches(utterance):
        finance_result = FinanceReadFastPath(
            FinanceLedger(PostgresFinanceSnapshotStore(connection))
        ).resolve(intent, obligations)
        if finance_result is not None:
            finance = finance_result.evidence

    if CrossDomainPlanningFastPath.matches(utterance):
        planning_result = CrossDomainPlanningFastPath(
            personal_state,
            household_snapshot,
            task_store.list(principal),
            finance,
        ).resolve(intent)
        if planning_result is not None:
            return planning_result

    if finance is not None:
        finance_result = FinanceReadFastPath(
            FinanceLedger(PostgresFinanceSnapshotStore(connection))
        ).resolve(intent, obligations)
        if finance_result is not None:
            PostgresAuditLog(connection).append(
                "finance.affordability.read",
                principal.id,
                {
                    "purchase_cents": finance_result.evidence["purchase_cents"],
                    "shared_obligations_cents": finance_result.evidence["shared_obligations_cents"],
                    "affordable": finance_result.evidence["affordable"],
                },
            )
        return finance_result
    return None


def ground_reference_action(
    intent: IntentFrame,
    card: ActionCard,
    task_store: Any,
    household_store: PostgresHouseholdStore,
    personal_state: PersonalState,
    goal_task_title: str | None,
    goal_chore_title: str | None,
    memory_task_title: str | None,
    memory_chore_title: str | None,
) -> ActionCard | Result:
    """Apply reference-Pack grounding before generic Core execution.

    The returned ActionCard is still only a proposal.  The shared Kernel owns
    policy, execution, observation, verification, persistence, and completion.
    """

    principal = intent.principal
    if card.action.action_id == "tasks.complete":
        title = card.action.arguments.get("title")
        if not isinstance(title, str) or not title.strip():
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message=("Name the task to complete, for example: Complete the task buy cat food."),
                correlation_id=intent.correlation_id,
            )
        tasks = task_store.list(principal)
        canonical_title = TaskCompletionFastPath.canonical_title(title, tasks)
        if canonical_title is not None and canonical_title != title:
            card = card.model_copy(
                update={
                    "action": card.action.model_copy(
                        update={"arguments": {"title": canonical_title}}
                    )
                }
            )
            title = canonical_title
        completion_result = TaskCompletionFastPath.resolve(intent, title, tasks)
        if completion_result is not None:
            return completion_result

    if card.action.action_id == "tasks.chores.complete":
        title = card.action.arguments.get("title")
        if not isinstance(title, str) or not title.strip():
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message=(
                    "Name the chore to complete, for example: Complete the chore clean the kitchen."
                ),
                correlation_id=intent.correlation_id,
            )
        household_snapshot = household_store.read_snapshot(principal)
        completion_result = ChoreCompletionFastPath.resolve(
            intent,
            title,
            cast(tuple[Chore, ...], household_snapshot["chores"]),
        )
        if completion_result is not None:
            return completion_result

    if goal_task_title is not None and card.action.action_id == "tasks.create":
        card = card.model_copy(
            update={
                "action": card.action.model_copy(update={"arguments": {"title": goal_task_title}})
            }
        )
    elif goal_chore_title is not None and card.action.action_id == "tasks.chores.create":
        card = card.model_copy(
            update={
                "action": card.action.model_copy(update={"arguments": {"title": goal_chore_title}})
            }
        )
    elif memory_task_title is not None and card.action.action_id == "tasks.create":
        arguments: dict[str, Any] = {"title": memory_task_title}
        due_at = requested_task_due_at(intent.utterance)
        if due_at is not None:
            arguments["due_at"] = due_at
        card = card.model_copy(
            update={"action": card.action.model_copy(update={"arguments": arguments})}
        )
    elif memory_chore_title is not None and card.action.action_id == "tasks.chores.create":
        card = card.model_copy(
            update={
                "action": card.action.model_copy(
                    update={"arguments": {"title": memory_chore_title}}
                )
            }
        )

    if card.action.action_id == "tasks.create":
        proposed_due_at = card.action.arguments.get("due_at")
        if proposed_due_at is not None and not isinstance(proposed_due_at, str):
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message="I need a clear deadline before adding that task.",
                correlation_id=intent.correlation_id,
            )
        grounded, due_at = ground_task_due_at(intent.utterance, proposed_due_at)
        if not grounded:
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message=(
                    "What deadline should I use for that task? I won't infer one from context."
                ),
                correlation_id=intent.correlation_id,
            )
        arguments = dict(card.action.arguments)
        if due_at is None:
            arguments.pop("due_at", None)
        else:
            arguments["due_at"] = due_at
        card = card.model_copy(
            update={"action": card.action.model_copy(update={"arguments": arguments})}
        )

    return card


def ground_reference_action_runtime(
    intent: IntentFrame, card: ActionCard, connection: Any
) -> ActionCard | Result:
    """Load reference Pack state before applying its canonical grounding rules."""

    principal = intent.principal
    task_store = PostgresTaskStore(connection)
    household_store = PostgresHouseholdStore(connection)
    personal_state = PostgresPersonalStateStore(connection, principal.vault_id).load_for_principal(
        principal
    )
    composer_results = (
        PersonalTaskComposer.resolve(intent.utterance, personal_state),
        PersonalChoreComposer.resolve(intent.utterance, personal_state),
        PersonalMemoryTaskComposer.resolve(intent.utterance, personal_state),
        PersonalMemoryChoreComposer.resolve(intent.utterance, personal_state),
    )
    titles = tuple(title for title, _error in composer_results)
    return ground_reference_action(
        intent,
        card,
        task_store,
        household_store,
        personal_state,
        titles[0],
        titles[1],
        titles[2],
        titles[3],
    )
