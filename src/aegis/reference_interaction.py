"""First-party interaction composition for the reference Packs.

This module is deliberately outside the shared InteractionBoundary.  It owns
domain-specific fast-path grounding that the reference Tasks/Household/Personal
Packs currently need; the boundary receives only a typed card/result callback.
Pack-specific behavior therefore remains composition-owned and cannot become
the generic client/Core contract.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

from .audit import PostgresAuditLog
from .contracts import (
    ActionCard,
    ActionSpec,
    Context,
    IntentFrame,
    ObjectiveState,
    Principal,
    Result,
)
from .decoding import StrictDecisionDecoder
from .dispatch import ActionExecutorDispatch, ActionVerifierDispatch
from .finance import FinanceLedger, FinanceReadFastPath, PostgresFinanceSnapshotStore
from .household import (
    Chore,
    ChoreCompletionFastPath,
    HouseholdObligation,
    PostgresHouseholdStore,
)
from .identity import PostgresSpacePolicy, Role
from .kernel import Kernel
from .pack_lifecycle import PackManager
from .pack_runtime import PackRuntimeRegistry
from .personal import PersonalState, PostgresPersonalStateStore
from .planning import CrossDomainPlanningFastPath, MultiActionFastPath
from .projections import SharedObligation
from .store import PostgresObjectiveStore
from .tasks import (
    PostgresTaskStore,
    TaskCompletionFastPath,
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
