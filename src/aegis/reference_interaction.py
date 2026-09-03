"""First-party interaction composition for the reference Packs.

This module is deliberately outside the shared InteractionBoundary.  It owns
domain-specific fast-path grounding that the reference Tasks/Household/Personal
Packs currently need; the boundary receives only a typed card/result callback.
Pack-specific behavior therefore remains composition-owned and cannot become
the generic client/Core contract.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from uuid import UUID, uuid4

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
    ProposedPlan,
    ProposedPlanStep,
    Result,
)
from .decoding import StrictDecisionDecoder
from .dispatch import ActionExecutorDispatch, ActionVerifierDispatch
from .embeddings import OllamaEmbeddingProvider, PostgresMemoryVectorIndex
from .finance import FinanceLedger, FinanceReadFastPath, PostgresFinanceSnapshotStore
from .homelab import PostgresHomelabStore
from .household import (
    Chore,
    ChoreCompletionFastPath,
    ContextualChorePriorityFastPath,
    GroceryReadFastPath,
    HouseholdObligation,
    HouseholdReadFastPath,
    PostgresHouseholdStore,
)
from .identity import PostgresSpacePolicy, Role
from .interaction import InteractionInputError
from .interaction_context import resolve_obvious_ordinal, resolve_obvious_ordinal_item
from .kernel import Kernel
from .network import PostgresNetworkStore
from .pack_lifecycle import PackManager, PostgresPackStore
from .pack_runtime import PackRuntimeRegistry
from .personal import PersonalMemoryFastPath, PersonalState, PostgresPersonalStateStore
from .planning import (
    ContextualCrossDomainPriorityFastPath,
    ContextualMutationGuard,
    CrossDomainPlanningFastPath,
    DomainClarificationFastPath,
    MultiActionFastPath,
    PersonalChoreComposer,
    PersonalMemoryChoreComposer,
    PersonalMemoryTaskComposer,
    PersonalTaskComposer,
    PlanProgressFastPath,
)
from .projections import SharedObligation
from .reference_packs import reference_bundles
from .store import PostgresObjectiveStore
from .tasks import (
    ContextualTaskPriorityFastPath,
    ContextualTaskTemporalFastPath,
    PostgresTaskStore,
    TaskCompletionFastPath,
    TaskIntentClarificationFastPath,
    TaskPriorityFastPath,
    TaskReadFastPath,
    ground_task_due_at,
    requested_task_due_at,
)
from .utterance import is_mutation_request, is_task_destination_request, strip_context_reset


def reference_domain_and_action(utterance: str, manager: PackManager) -> tuple[str, ActionCard]:
    """Compatibility router for reference Packs when cognition is disabled.

    This legacy path is composition-owned. The generic CLI only supplies this
    callback to InteractionBoundary; it does not know reference action IDs or
    argument grammar.
    """

    text = utterance.lower()
    if "task" in text:
        domain = "tasks"
        if any(term in text for term in ("complete", "completed", "finish", "finished")) or (
            "mark" in text and "done" in text
        ):
            action_id = "tasks.complete"
        elif "event" in text:
            action_id = "tasks.events.create"
        elif "chore" in text:
            action_id = (
                "tasks.chores.list"
                if text.startswith(("what", "show", "list"))
                else "tasks.chores.create"
            )
        else:
            action_id = (
                "tasks.list" if text.startswith(("what", "show", "list")) else "tasks.create"
            )
    elif "chore" in text:
        domain = "tasks"
        if any(term in text for term in ("complete", "completed", "finish", "finished")) or (
            "mark" in text and "done" in text
        ):
            action_id = "tasks.chores.complete"
        else:
            action_id = (
                "tasks.chores.list"
                if text.startswith(("what", "show", "list"))
                else "tasks.chores.create"
            )
    elif "event" in text or "inspection" in text:
        domain = "tasks"
        action_id = "tasks.events.create"
    elif any(word in text for word in ("grocery", "groceries", "shopping list", "rice", "food")):
        domain = "kitchen"
        action_id = (
            "kitchen.groceries.list"
            if text.startswith(("what", "show", "list"))
            else "kitchen.groceries.add"
        )
    else:
        raise InteractionInputError(
            "alpha supports groceries and tasks; try one of the four demo requests"
        )
    cards = manager.retrieve(domain)
    card = next((item for item in cards if item.action.action_id == action_id), None)
    if card is None:
        card = manager.action_card(domain, action_id)
    if card is None:
        raise RuntimeError(f"enabled Pack did not provide ActionCard {action_id}")
    action = card.action
    if action_id == "kitchen.groceries.add":
        match = re.search(r"add\s+(.+?)\s+to\s+(?:the\s+)?grocer(?:y|ies)\b", text)
        if match is None:
            raise InteractionInputError(
                "tell AEGIS what to add, for example: Add rice to groceries."
            )
        action = action.model_copy(update={"arguments": {"item": match.group(1).strip()}})
    elif action_id == "tasks.create":
        match = re.search(
            r"(?:(?:(?:could|would|can)\s+you|(?:i\s+want|i\s+would\s+like|i'd\s+like))\s+to\s+)?"
            r"(?:put|place)\s+(?:a\s+)?task\s+on\s+(?:my|the)\s+"
            r"(?:task\s+)?list\s+(?:to\s+)?(.+)$",
            text,
        )
        if match is None:
            match = re.search(r"(?:create\s+)?(?:a\s+)?task\s+(?:to\s+)?(.+)$", text)
        if match is None:
            if not any(source in text for source in ("goal", "memory")) or not any(
                phrase in text for phrase in ("turn", "make", "add", "create")
            ):
                raise InteractionInputError(
                    "tell AEGIS the task, for example: Create a task to buy cat food."
                )
            action = action.model_copy(update={"arguments": {}})
        else:
            title = match.group(1).strip()
            due_at = requested_task_due_at(title)
            if due_at is not None:
                title = re.sub(r"\s+(?:tomorrow|next\s+week)[.!?]?$", "", title).rstrip()
            arguments: dict[str, Any] = {"title": title}
            if due_at is not None:
                arguments["due_at"] = due_at
            action = action.model_copy(update={"arguments": arguments})
    elif action_id == "tasks.complete":
        match = re.search(
            r"(?:complete|completed|finish|finished)\s+(?:the\s+)?task\s+"
            r"(?:called\s+|named\s+)?(.+)$",
            text,
        )
        if match is None:
            match = re.search(
                r"mark\s+(?:the\s+)?task\s+(.+?)\s+as\s+(?:done|complete|completed)[.!?]?$",
                text,
            )
        if match is None:
            raise InteractionInputError(
                "name the task to complete, for example: Complete the task buy cat food."
            )
        action = action.model_copy(update={"arguments": {"title": match.group(1).strip()}})
    elif action_id == "tasks.chores.create":
        match = re.search(r"(?:create|add)\s+(?:a\s+)?chore\s+(?:to\s+)?(.+)$", text)
        if match is None:
            if not any(source in text for source in ("goal", "memory")) or not any(
                phrase in text for phrase in ("turn", "make", "add", "create")
            ):
                raise InteractionInputError(
                    "tell AEGIS the chore, for example: Create a chore to clean the kitchen."
                )
            action = action.model_copy(update={"arguments": {}})
        else:
            action = action.model_copy(update={"arguments": {"title": match.group(1).strip()}})
    elif action_id == "tasks.chores.complete":
        match = re.search(
            r"(?:complete|completed|finish|finished)\s+(?:the\s+)?chore\s+"
            r"(?:called\s+|named\s+)?(.+)$",
            text,
        )
        if match is None:
            match = re.search(
                r"mark\s+(?:the\s+)?chore\s+(.+?)\s+as\s+(?:done|complete|completed)[.!?]?$",
                text,
            )
        if match is None:
            raise InteractionInputError(
                "name the chore to complete, for example: Complete the chore clean the kitchen."
            )
        action = action.model_copy(update={"arguments": {"title": match.group(1).strip()}})
    elif action_id == "tasks.events.create":
        match = re.search(r"(?:create|add)\s+(?:an?\s+)?event\s+(?:for\s+)?(.+)$", text)
        if match is None:
            raise InteractionInputError(
                "tell AEGIS the event, for example: Create an event for apartment inspection."
            )
        title = match.group(1).strip()
        if title.endswith(" tomorrow"):
            title = title.removesuffix(" tomorrow").strip()
            starts_at = datetime.now(timezone.utc) + timedelta(days=1)
        else:
            starts_at = datetime.now(timezone.utc)
        action = action.model_copy(
            update={"arguments": {"title": title, "starts_at": starts_at.isoformat()}}
        )
    return domain, ActionCard(action=action, summary=card.summary, relevance=card.relevance)


class _ReadOnlyHomelabRuntime:
    def restart(self, _service: Any) -> bool:
        return False

    def health(self, _service: Any) -> bool:
        return False


def reference_constellation_state(
    principal: Principal,
    connect: Callable[[str], Any],
    required: Callable[[str], str],
    apply_migrations: Callable[[Any], None],
    *,
    household_store_factory: Callable[[Any], PostgresHouseholdStore] = PostgresHouseholdStore,
    task_store_factory: Callable[[Any], PostgresTaskStore] = PostgresTaskStore,
    personal_store_factory: Callable[
        [Any, str], PostgresPersonalStateStore
    ] = PostgresPersonalStateStore,
    finance_store_factory: Callable[
        [Any], PostgresFinanceSnapshotStore
    ] = PostgresFinanceSnapshotStore,
    network_store_factory: Callable[[Any], PostgresNetworkStore] = PostgresNetworkStore,
    homelab_store_factory: Callable[[Any], PostgresHomelabStore] = PostgresHomelabStore,
    pack_store_factory: Callable[[Any], PostgresPackStore] = PostgresPackStore,
) -> dict[str, Any]:
    """Build the authorized reference-Pack projection for a client adapter.

    The browser receives this projection through a callback, but domain
    knowledge and canonical store reads remain in reference-Pack composition.
    Generic web transport only validates and presents the projection.
    """

    connection = connect(required("AEGIS_DATABASE_URL"))
    try:
        apply_migrations(connection)
        household = household_store_factory(connection).read_snapshot(principal)
        tasks = task_store_factory(connection).list(principal)
        groceries = cast(tuple[str, ...], household.get("groceries", ()))
        personal = personal_store_factory(connection, principal.vault_id).load_for_principal(
            principal
        )
        finance = finance_store_factory(connection).load(principal.id)
        network = network_store_factory(connection).load(principal)
        homelab = homelab_store_factory(connection).load(principal, _ReadOnlyHomelabRuntime())
        persisted = {
            bundle.manifest.pack_id: (bundle, status)
            for bundle, status, _grants in PackManager(
                store=pack_store_factory(connection)
            ).lifecycle_snapshot()
        }
        nodes: list[dict[str, Any]] = [
            {
                "id": "aegis",
                "label": "AEGIS",
                "detail": "central hub",
                "category": "core",
                "detail_view": "overview",
            },
        ]
        edges: list[dict[str, str]] = []
        available = {bundle.manifest.pack_id: bundle for bundle in reference_bundles()}
        available.update(
            {pack_id: item[0] for pack_id, item in persisted.items() if pack_id not in available}
        )
        for pack_id, bundle in sorted(available.items()):
            ui = bundle.manifest.ui
            label = ui.label if ui is not None else pack_id.replace("-", " ").title()
            status = persisted.get(pack_id, (None, "available"))[1]
            status_text = status.value if hasattr(status, "value") else str(status)
            node_id = f"pack-{pack_id}"
            detail = f"{ui.category if ui else 'domain'} · {status_text}"
            if pack_id == "tasks":
                detail += f" · {len(tasks)} tasks"
            elif pack_id == "kitchen":
                detail += f" · {len(groceries)} groceries"
            nodes.append(
                {
                    "id": node_id,
                    "label": label,
                    "detail": detail,
                    "category": ui.category if ui else "domain",
                    "detail_view": ui.detail_view if ui else None,
                }
            )
            edges.append({"source": "aegis", "target": node_id})
        domain_summaries = (
            (
                "personal",
                "Personal",
                f"{len(personal.projects)} projects · {len(personal.goals)} goals · "
                f"{len(personal.memories)} memories",
            ),
            (
                "household",
                "Household",
                f"{len(cast(tuple[object, ...], household.get('chores', ())))} chores · "
                f"{len(cast(tuple[object, ...], household.get('events', ())))} events",
            ),
            (
                "finance",
                "Finance",
                "private snapshot available" if finance is not None else "no private snapshot",
            ),
            (
                "homelab",
                "Infrastructure",
                f"{len(homelab.hosts)} hosts · {len(homelab.services)} services",
            ),
            (
                "network",
                "Network",
                f"{len(network.devices)} devices · {len(network.scopes)} authorized scopes",
            ),
        )
        pack_ids = set(available)
        for domain_id, label, detail in domain_summaries:
            if domain_id in pack_ids:
                continue
            node_id = f"domain-{domain_id}"
            nodes.append(
                {
                    "id": node_id,
                    "label": label,
                    "detail": detail,
                    "category": "domain",
                    "detail_view": "list",
                }
            )
            edges.append({"source": "aegis", "target": node_id})
        details: dict[str, Any] = {
            "domain-personal": {
                "projects": [
                    {"name": project.name, "created_at": project.created_at.isoformat()}
                    for project in personal.projects.values()
                ],
                "goals": [
                    {
                        "description": goal.description,
                        "project_id": str(goal.project_id) if goal.project_id else None,
                    }
                    for goal in personal.goals.values()
                ],
            },
            "domain-household": {
                "groceries": list(groceries),
                "chores": [
                    {
                        "title": chore.title,
                        "assignee_id": chore.assignee_id,
                        "completed": chore.completed,
                    }
                    for chore in cast(tuple[Any, ...], household.get("chores", ()))
                ],
                "events": [
                    {"title": event.title, "starts_at": event.starts_at.isoformat()}
                    for event in cast(tuple[Any, ...], household.get("events", ()))
                ],
            },
            "domain-finance": {"snapshot_available": finance is not None},
            "pack-homelab": {
                "hosts": [{"hostname": host.hostname} for host in homelab.hosts.values()],
                "services": [{"name": service.name} for service in homelab.services.values()],
            },
            "pack-network": {
                "devices": [
                    {
                        "name": device.hostname or device.address,
                        "address": device.address,
                    }
                    for device in network.devices.values()
                ],
                "authorized_scopes": [scope.scope_id for scope in network.scopes.values()],
            },
            "pack-tasks": {
                "tasks": [{"title": task.title, "status": task.status.value} for task in tasks]
            },
            "pack-kitchen": {"groceries": list(groceries)},
        }
        return {"nodes": nodes, "edges": edges, "details": details}
    finally:
        connection.close()


def reference_format_result(result: Any) -> str:
    """Render reference-Pack canonical evidence for human-facing clients."""

    if result.state.value != "completed":
        return f"Not completed — {result.message}"
    evidence = result.evidence
    if evidence.get("provenance") == "model_generated":
        return str(result.message)
    if evidence.get("priority_basis") and evidence.get("task") is not None:
        return str(result.message)
    if evidence.get("authorized_ordinal_item") is not None:
        return f"Grocery item: {evidence['authorized_ordinal_item']}"
    # Ordinal follow-ups are canonical reads.  Their evidence carries the
    # already-rendered domain label so it must not fall through to the generic
    # mutation-shaped title formatter ("Done — ...").
    if isinstance(evidence.get("authorized_ordinal_referent"), dict):
        return str(result.message)
    if evidence.get("canonical_items") is not None:
        items = evidence["canonical_items"]
        counts: dict[str, int] = {}
        order: list[str] = []
        for item in items:
            value = str(item)
            if value not in counts:
                order.append(value)
                counts[value] = 0
            counts[value] += 1
        listing = ", ".join(
            f"{item} (x{counts[item]})" if counts[item] > 1 else item for item in order
        )
        return "Groceries: " + (listing if listing else "(empty)")
    if evidence.get("canonical_tasks") is not None:
        tasks = evidence["canonical_tasks"]
        display_tasks = tasks[:20]
        listing = "; ".join(
            f"{item['title']} ({item['status']})"
            + (f" due {item['due_at']}" if item.get("due_at") else "")
            for item in display_tasks
        )
        if len(tasks) > len(display_tasks):
            listing += f"; … and {len(tasks) - len(display_tasks)} more"
        return "Tasks: " + (listing if tasks else "(empty)")
    if evidence.get("memories") is not None:
        memories = evidence["memories"]
        if not memories:
            return "Memories: (none found)"
        return "Memories: " + "; ".join(
            f"{item['content']} [{item['provenance']}]" for item in memories
        )
    if evidence.get("projects") is not None:
        projects = evidence["projects"]
        return "Projects: " + (
            "; ".join(item["name"] for item in projects) if projects else "(none)"
        )
    if evidence.get("goals") is not None:
        goals = evidence["goals"]
        return "Goals: " + (
            "; ".join(
                f"{item['description']}" + (f" [{item['project']}]" if item["project"] else "")
                for item in goals
            )
            if goals
            else "(none)"
        )
    if evidence.get("obligations") is not None:
        obligations = evidence["obligations"]
        outstanding = [item for item in obligations if not item["settled"]]
        return "Outstanding obligations: " + (
            "; ".join(f"{item['title']} ({item['responsible_id']})" for item in outstanding)
            if outstanding
            else "(none)"
        )
    if evidence.get("chores") is not None:
        chores = evidence["chores"]
        display_chores = chores[:20]
        listing = "; ".join(f"{item['title']} ({item['assignee_id']})" for item in display_chores)
        if len(chores) > len(display_chores):
            listing += f"; … and {len(chores) - len(display_chores)} more"
        return "Chores: " + (listing if chores else "(none)")
    if evidence.get("events") is not None:
        events = evidence["events"]
        return "Events: " + ("; ".join(item["title"] for item in events) if events else "(none)")
    if isinstance(evidence.get("planning"), dict):
        planning = evidence["planning"]
        summaries: list[str] = []
        affordability = planning.get("affordability")
        if isinstance(affordability, dict) and affordability.get("affordable") is not None:
            status = "yes" if affordability["affordable"] else "no"
            purchase = affordability.get("purchase_cents")
            obligations = affordability.get("shared_obligations_cents")
            if isinstance(purchase, int) and isinstance(obligations, int):
                summaries.append(
                    f"affordable: {status} (purchase ${purchase / 100:.2f}; "
                    f"shared obligations ${obligations / 100:.2f})"
                )
            else:
                summaries.append(f"affordable: {status}")
        open_tasks = planning.get("open_tasks")
        if isinstance(open_tasks, list):
            titles = [
                str(item["title"])
                for item in open_tasks
                if isinstance(item, dict) and isinstance(item.get("title"), str)
            ]
            summaries.append("open tasks: " + ("; ".join(titles) if titles else "(none)"))
        open_chores = planning.get("open_chores")
        if isinstance(open_chores, list):
            titles = [
                str(item["title"])
                for item in open_chores
                if isinstance(item, dict) and isinstance(item.get("title"), str)
            ]
            summaries.append("open chores: " + ("; ".join(titles) if titles else "(none)"))
        obligations = planning.get("open_obligations")
        if isinstance(obligations, list):
            titles = [
                str(item["title"])
                for item in obligations
                if isinstance(item, dict) and isinstance(item.get("title"), str)
            ]
            summaries.append("open obligations: " + ("; ".join(titles) if titles else "(none)"))
        memories = planning.get("memories")
        if isinstance(memories, list):
            contents = [
                str(item["content"])
                for item in memories
                if isinstance(item, dict) and isinstance(item.get("content"), str)
            ]
            summaries.append(
                "relevant memories: " + ("; ".join(contents) if contents else "(none)")
            )
        if summaries:
            return "Planning: " + "; ".join(summaries)
    if evidence.get("affordable") is not None:
        status = "yes" if evidence["affordable"] else "no"
        return (
            f"Affordable: {status} (purchase ${evidence['purchase_cents'] / 100:.2f}; "
            f"shared obligations ${evidence['shared_obligations_cents'] / 100:.2f})"
        )
    if evidence.get("collection") == "chores" and evidence.get("title"):
        if evidence.get("completed") is True:
            return f"Done — completed chore: {evidence['title']}"
        return f"Done — created chore: {evidence['title']}"
    if evidence.get("collection") == "events" and evidence.get("title"):
        return f"Done — created event: {evidence['title']}"
    if evidence.get("title"):
        if evidence.get("status") == "completed":
            return f"Done — completed task: {evidence['title']}"
        return f"Done — created task: {evidence['title']}"
    if evidence.get("item"):
        return f"Done — added {evidence['item']} to groceries"
    return f"Done — {result.message}"


def resolve_reference_safety_fast_paths(
    intent: IntentFrame,
    recovered_plan_actions: tuple[ActionSpec, ...] | None,
    model_enabled: bool,
    context: Context | None = None,
) -> Result | None:
    """Apply reference-Pack safety guards before generic cognition."""

    if recovered_plan_actions is None:
        result = MultiActionFastPath.resolve(intent)
        if result is not None:
            return result
    if not model_enabled:
        result = DomainClarificationFastPath.resolve(intent)
        if result is not None:
            return result
    if (
        context is not None
        and "complete" in intent.utterance.casefold()
        and (
            resolve_obvious_ordinal(intent.utterance, context, "canonical_tasks") is not None
            or resolve_obvious_ordinal(intent.utterance, context, "canonical_chores") is not None
        )
    ):
        return None
    return ContextualMutationGuard.resolve(intent)


def build_reference_fallback_context(
    context: Context,
    task_store: PostgresTaskStore,
    household_store: PostgresHouseholdStore,
    principal: Principal,
    utterance: str,
) -> Context:
    """Build the bounded authorized context used by reference-Pack fallback cognition."""

    values = dict(context.values)
    values.setdefault("as_of_date", datetime.now(timezone.utc).date().isoformat())
    facts = dict(values.get("canonical_facts", {}))
    if "canonical_items" in facts or "canonical_tasks" in facts:
        return Context(
            values=values,
            sources=tuple(dict.fromkeys((*context.sources, "authorized_canonical_context"))),
        )
    facts["canonical_items"] = list(
        dict.fromkeys(str(item) for item in household_store.list_groceries(principal))
    )[:20]
    read_snapshot = getattr(household_store, "read_snapshot", None)
    snapshot = read_snapshot(principal) if callable(read_snapshot) else {}
    if isinstance(snapshot, dict):
        chores = snapshot.get("chores", ())
        if isinstance(chores, (list, tuple)):
            facts["canonical_chores"] = [
                {"title": str(chore.title), "completed": bool(chore.completed)}
                for chore in chores
                if hasattr(chore, "title") and hasattr(chore, "completed")
            ][:20]
        obligations = snapshot.get("obligations", ())
        if isinstance(obligations, (list, tuple)):
            facts["canonical_obligations"] = [
                {
                    "title": str(obligation.title),
                    "settled": bool(obligation.settled),
                    "responsible_id": str(obligation.responsible_id),
                }
                for obligation in obligations
                if hasattr(obligation, "title")
                and hasattr(obligation, "settled")
                and hasattr(obligation, "responsible_id")
            ][:20]
    tasks = list(task_store.list(principal))
    lowered = utterance.casefold()
    if ("which" in lowered and "first" in lowered) or any(
        term in lowered for term in ("prioritize", "priority", "focus")
    ):
        tasks = [task for task in tasks if task.status.value == "open"]
        tasks.sort(
            key=lambda task: (
                task.due_at is None,
                task.due_at.isoformat() if task.due_at is not None else "",
            )
        )
    query_terms = {
        term
        for term in lowered.split()
        if len(term) >= 3 and term not in {"the", "task", "tasks", "please", "complete"}
    }
    ranked = sorted(
        enumerate(tasks),
        key=lambda item: (
            len(query_terms.intersection(set(item[1].title.casefold().split()))),
            -item[0],
        ),
        reverse=True,
    )
    selected = [
        task for _, task in ranked if query_terms.intersection(task.title.casefold().split())
    ] or tasks
    facts["canonical_tasks"] = [
        {
            "task_id": str(task.task_id),
            "title": task.title,
            "status": task.status.value,
            **({"due_at": task.due_at.isoformat()} if task.due_at is not None else {}),
        }
        for task in selected[:20]
    ]
    values["canonical_facts"] = facts
    return Context(
        values=values,
        sources=tuple(dict.fromkeys((*context.sources, "authorized_task_candidates"))),
    )


def build_reference_fallback_context_runtime(
    context: Context, connection: Any, principal: Principal, utterance: str
) -> Context:
    """Construct fallback context from reference stores in the composition layer."""

    return build_reference_fallback_context(
        context,
        PostgresTaskStore(connection),
        PostgresHouseholdStore(connection),
        principal,
        utterance,
    )


def reference_fallback_cards(
    manager: PackManager, utterance: str, context: Context | None = None
) -> tuple[ActionCard, ...]:
    """Reduce legacy no-provider candidates for the reference Packs."""

    facts = (context.values if context is not None else {}).get("canonical_facts", {})
    if isinstance(facts, dict) and context is not None:
        if (
            isinstance(facts.get("canonical_items"), list)
            and "authorized_canonical_context" in context.sources
        ):
            # A contextual grocery question is answered from the authorized
            # projection; do not offer a mutation card that could turn an
            # unresolved reference into a write.
            return ()
        if (
            isinstance(facts.get("canonical_tasks"), list)
            and "authorized_canonical_context" in context.sources
        ):
            return tuple(manager.retrieve("tasks"))[:10]

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
                ("shopping list", "kitchen"),
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
    utterance = strip_context_reset(intent.utterance)
    if utterance != intent.utterance:
        intent = intent.model_copy(update={"utterance": utterance})
    progress_result = PlanProgressFastPath.resolve(intent, context)
    if progress_result is not None:
        return progress_result
    task_store = PostgresTaskStore(connection)
    household_store = PostgresHouseholdStore(connection)
    personal_state = PostgresPersonalStateStore(connection, principal.vault_id).load_for_principal(
        principal
    )
    if FinanceReadFastPath.unsupported_balance_read(utterance):
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=(
                "I can assess a purchase amount against your available snapshot, "
                "but a general balance read is not available in this alpha."
            ),
            correlation_id=intent.correlation_id,
        )
    reminder_result = DomainClarificationFastPath.resolve_underspecified_reminder(intent)
    if reminder_result is not None:
        return reminder_result
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
    if composed_title is None:
        result = resolve_contextual_ordinal_read(intent, context)
        if result is not None:
            return result
    if composed_title is None:
        result = resolve_contextual_remaining(intent, context)
        if result is not None:
            return result
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
        result = ContextualTaskTemporalFastPath().resolve(intent, context, task_store)
        if result is not None:
            return result
        result = ContextualTaskPriorityFastPath().resolve(intent, context)
        if result is not None:
            return result
        result = ContextualChorePriorityFastPath.resolve(intent, context)
        if result is not None:
            return result
        result = ContextualCrossDomainPriorityFastPath().resolve(intent, context)
        if result is not None:
            return result
        result = resolve_contextual_ordinal_read(intent, context)
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
        return memory_fast_path.resolve(intent, context)
    return None


def resolve_contextual_ordinal_read(intent: IntentFrame, context: Context) -> Result | None:
    """Answer a read-only ordinal follow-up from the authorized prior list."""

    text = intent.utterance.casefold()
    ambiguous_correction = (
        re.search(r"\b(?:other|another)\s+(?:one|item|task|chore|event)\b", text) is not None
    )
    if (
        is_mutation_request(text)
        or not any(
            marker in text
            for marker in ("what about", "tell me about", "which one", "what is", "meant")
        )
        and not (
            "which" in text
            and re.search(r"\b(?:the\s+)?(?:first|second|third|fourth|last)\b", text)
        )
        and not ambiguous_correction
    ):
        return None
    referents = context.values.get("referents")
    those = referents.get("those") if isinstance(referents, dict) else None
    candidates = those.get("candidates") if isinstance(those, dict) else None
    if isinstance(candidates, list) and ambiguous_correction:
        # A correction without a unique structural target must not fall
        # through to semantic action selection.  Keep it in the same
        # authorized collection and ask for the missing identity.
        fact_key = those.get("fact_key") if isinstance(those, dict) else None
        labels = {
            "canonical_items": "grocery item",
            "canonical_tasks": "task",
            "canonical_chores": "chore",
            "events": "event",
        }
        label = labels.get(str(fact_key), "item")
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=f"Which {label} did you mean? Please name it or choose an ordinal.",
            correlation_id=intent.correlation_id,
        )
    item = resolve_obvious_ordinal_item(text, context)
    if item is not None:
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.COMPLETED,
            message=f"Grocery item: {item}",
            evidence={
                "collection": "groceries",
                "authorized_ordinal_item": item,
                **({"canonical_items": candidates} if isinstance(candidates, list) else {}),
            },
            correlation_id=intent.correlation_id,
        )
    for fact_key, label in (
        ("canonical_tasks", "Task"),
        ("canonical_chores", "Chore"),
        ("events", "Event"),
    ):
        referent = resolve_obvious_ordinal(text, context, fact_key)
        if referent is None:
            continue
        title = referent.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        status = referent.get("status") or (
            "completed" if referent.get("completed") is True else "open"
        )
        detail = f"{label}: {title}"
        if isinstance(status, str):
            detail += f" ({status})"
        due_at = referent.get("due_at")
        if isinstance(due_at, str):
            detail += f"; due {due_at}"
        starts_at = referent.get("starts_at")
        if isinstance(starts_at, str):
            detail += f"; starts {starts_at}"
        collection_evidence = {
            "collection": fact_key,
            "authorized_ordinal_referent": referent,
        }
        if isinstance(candidates, list):
            collection_evidence[fact_key] = candidates
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.COMPLETED,
            message=detail,
            evidence=collection_evidence,
            correlation_id=intent.correlation_id,
        )
    return None


def resolve_contextual_remaining(intent: IntentFrame, context: Context) -> Result | None:
    """Return remaining members of an authorized prior collection projection."""

    text = " ".join(intent.utterance.casefold().split()).strip(".!?")
    if is_mutation_request(text) or text not in {
        "what's left",
        "what is left",
        "what remains",
        "what's remaining",
        "what is remaining",
    }:
        return None
    referents = context.values.get("referents")
    those = referents.get("those") if isinstance(referents, dict) else None
    if not isinstance(those, dict) or those.get("fact_key") not in {
        "canonical_items",
        "canonical_tasks",
        "canonical_chores",
    }:
        return None
    candidates = those.get("candidates")
    if not isinstance(candidates, list):
        return None
    fact_key = those["fact_key"]
    remaining: list[Any]
    if fact_key == "canonical_items":
        remaining = [item for item in candidates if isinstance(item, str)]
        evidence = {"canonical_items": remaining}
    elif fact_key == "canonical_chores":
        remaining = [
            item
            for item in candidates
            if isinstance(item, dict) and item.get("completed") is not True
        ]
        evidence = {"chores": remaining}
    else:
        remaining = [
            item
            for item in candidates
            if isinstance(item, dict) and item.get("status") != "completed"
        ]
        evidence = {"canonical_tasks": remaining}
    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="Canonical collection read",
        evidence={"collection": fact_key, **evidence},
        correlation_id=intent.correlation_id,
    )


def run_reference_plan(
    intent: IntentFrame,
    connection: Any,
    principal: Principal,
    manager: PackManager,
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
    task_cards = tuple(manager.retrieve("tasks"))
    cards_by_id = {card.action.action_id: card for card in task_cards}
    try:
        plan_cards = tuple(cards_by_id[action.action_id] for action in plan_actions)
    except KeyError:
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message="A plan step is no longer an available capability",
            correlation_id=intent.correlation_id,
        )
    proposal = ProposedPlan(
        steps=tuple(
            ProposedPlanStep(
                action_ref=action.action_id,
                arguments=action.arguments,
                depends_on=((index - 1,) if index else ()),
            )
            for index, action in enumerate(plan_actions)
        )
    )
    runtimes = {}
    for card in plan_cards:
        action = card.action
        runtimes[action.action_id] = runtime_registry.resolve(card, connection, principal)
    kernel = Kernel(
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
    )
    return kernel.run_proposed_plan(intent, proposal, plan_cards, context=context)


def rewrite_reference_decision(
    intent: IntentFrame,
    decision: Decision,
    cards: tuple[ActionCard, ...],
    context: Context | None = None,
) -> Decision | Result | None:
    """Correct a reference-Pack event proposal when the user named a task destination."""

    if decision.kind is DecisionKind.CLARIFY and context is not None:
        referent = resolve_obvious_ordinal(intent.utterance, context, "canonical_tasks")
        task_card = next(
            (card for card in cards if card.action.action_id == "tasks.complete"), None
        )
        if (
            task_card is not None
            and "complete" in intent.utterance.casefold()
            and isinstance(referent, dict)
            and isinstance(referent.get("title"), str)
            and isinstance(referent.get("task_id"), str)
        ):
            return Decision(
                kind=DecisionKind.ACTION,
                action=task_card.action.model_copy(
                    update={
                        "arguments": {
                            "title": referent["title"],
                            "task_id": referent["task_id"],
                        }
                    }
                ),
                semantic_mode="ACTION",
            )

    if decision.kind is DecisionKind.CLARIFY and is_task_destination_request(intent.utterance):
        task_card = next((card for card in cards if card.action.action_id == "tasks.create"), None)
        title_match = re.search(
            r"^(?:.*?\b(?:add|create|put|place|jot\s+down)\s+)"
            r"(.+?)\s+(?:on|to|into)\s+my\s+(?:to[- ]?do|task)\s+list\b",
            intent.utterance.casefold().strip().rstrip(".!?"),
        )
        if task_card is not None and title_match is not None:
            title = re.sub(
                r"\s+(?:for|on)\s+(?:tomorrow|next\s+week|"
                r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)$",
                "",
                title_match.group(1).strip(),
            ).strip()
            if title:
                arguments: dict[str, Any] = {"title": title}
                due_at = requested_task_due_at(intent.utterance)
                if due_at is not None:
                    arguments["due_at"] = due_at
                return Decision(
                    kind=DecisionKind.ACTION,
                    action=task_card.action.model_copy(update={"arguments": arguments}),
                    semantic_mode="ACTION",
                )

    action = decision.action
    if action is None or action.action_id != "tasks.events.create":
        return None
    if not is_task_destination_request(intent.utterance):
        return None
    task_card = next((card for card in cards if card.action.action_id == "tasks.create"), None)
    if task_card is None:
        return None
    event_title = action.arguments.get("title")
    if not isinstance(event_title, str) or not event_title.strip():
        return Decision(
            kind=DecisionKind.CLARIFY,
            clarification="What should I add to your task list?",
        )
    event_arguments: dict[str, Any] = {"title": event_title}
    due_at = requested_task_due_at(intent.utterance)
    if due_at is not None:
        event_arguments["due_at"] = due_at
    return Decision(
        kind=DecisionKind.ACTION,
        action=task_card.action.model_copy(update={"arguments": event_arguments}),
        semantic_mode="ACTION",
    )


def resolve_reference_pre_model(
    intent: IntentFrame,
    connection: Any,
    principal: Principal,
) -> Result | None:
    """Resolve reference-Pack finance/planning fast paths before cognition."""

    utterance = intent.utterance
    household_store = PostgresHouseholdStore(connection)
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

    explicit_compound_mutation = (
        is_mutation_request(utterance)
        or MultiActionFastPath.task_chore_titles(utterance) is not None
        or MultiActionFastPath.task_event_details(utterance) is not None
    )
    if not explicit_compound_mutation and CrossDomainPlanningFastPath.matches(utterance):
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
    context: Context | None = None,
) -> ActionCard | Result:
    """Apply reference-Pack grounding before generic Core execution.

    The returned ActionCard is still only a proposal.  The shared Kernel owns
    policy, execution, observation, verification, persistence, and completion.
    """

    principal = intent.principal
    if card.action.action_id == "tasks.complete":
        text = intent.utterance.casefold()
        if "chore" in text and "task" not in text:
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message="I could not safely match that chore to a personal task.",
                correlation_id=intent.correlation_id,
            )
        title = card.action.arguments.get("title")
        referent = (
            resolve_obvious_ordinal(intent.utterance, context, "canonical_tasks")
            if context is not None
            else None
        )
        referent_title = referent.get("title") if referent is not None else None
        referent_task_id = referent.get("task_id") if referent is not None else None
        if isinstance(referent_title, str) and referent_title.strip():
            title = referent_title
        if not isinstance(title, str) or not title.strip():
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message=("Name the task to complete, for example: Complete the task buy cat food."),
                correlation_id=intent.correlation_id,
            )
        if referent_title is None and not TaskCompletionFastPath.target_is_grounded(
            intent.utterance, title
        ):
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message=(
                    "Please name the task to complete; I will not infer a target "
                    "from unrelated canonical context."
                ),
                correlation_id=intent.correlation_id,
            )
        tasks = task_store.list(principal)
        grounded_task = None
        if isinstance(referent_task_id, str):
            try:
                parsed_task_id = UUID(referent_task_id)
            except ValueError:
                parsed_task_id = None
            if parsed_task_id is not None:
                grounded_task = next(
                    (task for task in tasks if task.task_id == parsed_task_id), None
                )
            if grounded_task is None:
                return Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.BLOCKED,
                    message="That task is no longer available in the current task list.",
                    correlation_id=intent.correlation_id,
                )
            title = grounded_task.title
        canonical_title = TaskCompletionFastPath.canonical_title(title, tasks)
        grounded_arguments: dict[str, object] = {"title": canonical_title or title}
        if isinstance(referent_task_id, str) and referent_task_id.strip():
            grounded_arguments["task_id"] = referent_task_id
        if card.action.arguments != grounded_arguments:
            card = card.model_copy(
                update={"action": card.action.model_copy(update={"arguments": grounded_arguments})}
            )
            title = str(grounded_arguments["title"])
        if referent_task_id is None:
            completion_result = TaskCompletionFastPath.resolve(intent, title, tasks)
            if completion_result is not None:
                return completion_result

    if card.action.action_id == "tasks.chores.complete":
        text = intent.utterance.casefold()
        if "task" in text and "chore" not in text:
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message="I could not safely match that task to a household chore.",
                correlation_id=intent.correlation_id,
            )
        title = card.action.arguments.get("title")
        referent = (
            resolve_obvious_ordinal(intent.utterance, context, "canonical_chores")
            if context is not None
            else None
        )
        referent_title = referent.get("title") if referent is not None else None
        referent_chore_id = referent.get("chore_id") if referent is not None else None
        if isinstance(referent_title, str) and referent_title.strip():
            title = referent_title
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
        if isinstance(referent_chore_id, str) and referent_chore_id.strip():
            card = card.model_copy(
                update={
                    "action": card.action.model_copy(
                        update={"arguments": {"title": title, "chore_id": referent_chore_id}}
                    )
                }
            )

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
    intent: IntentFrame, card: ActionCard, connection: Any, context: Context | None = None
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
        context,
    )
