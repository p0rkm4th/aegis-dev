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
from .communications import configured_communication_targets
from .compositions import available_compositions
from .contracts import (
    ActionCard,
    ActionSpec,
    ArgumentProvenance,
    ArgumentProvenanceKind,
    Context,
    Decision,
    DecisionKind,
    IntentFrame,
    ObjectiveRequirementProposal,
    ObjectiveSpecProposal,
    ObjectiveState,
    Principal,
    ProposedPlan,
    ProposedPlanStep,
    Result,
)
from .decoding import StrictDecisionDecoder
from .dispatch import ActionExecutorDispatch, ActionVerifierDispatch
from .documents import configured_document_provider
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
from .interaction_context import (
    resolve_obvious_ordinal,
    resolve_obvious_ordinal_item,
    resolve_unique_prior_task_reference,
)
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
    PlanModificationFastPath,
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
    _task_projection,
    ground_task_due_at,
    requested_task_due_at,
)
from .utterance import (
    has_multiple_question_clauses,
    is_correction_request,
    is_mutation_request,
    is_question_request,
    is_task_destination_request,
    strip_context_reset,
    strip_correction_prefix,
)

_WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


def _event_time_is_explicit(utterance: str) -> bool:
    """Require a user-supplied time before creating a calendar event."""

    text = utterance.casefold()
    return bool(
        re.search(
            r"\b(?:today|tomorrow|tonight|morning|afternoon|evening|noon|midnight|"
            r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"next\s+week|at\s+\d|\d{1,2}(?::\d{2})?\s*(?:am|pm)|"
            r"\d{4}-\d{2}-\d{2}|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|"
            r"apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
            r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b)",
            text,
        )
    )


def _utterance_spans(utterance: str, value: object) -> tuple[tuple[int, int], ...]:
    """Return bounded exact spans for a textual argument value."""

    if not isinstance(value, str) or not value.strip():
        return ()
    match = re.search(re.escape(value.strip()), utterance, flags=re.IGNORECASE)
    return (match.span(),) if match is not None else ()


def _ground_argument_provenance(
    intent: IntentFrame, card: ActionCard, context: Context | None = None
) -> ActionCard | Result:
    """Attach typed evidence to reference-Pack arguments before Core execution."""

    provenance: dict[str, ArgumentProvenance] = {}
    for key, value in card.action.arguments.items():
        if (
            card.action.action_id
            in {
                "communications.messages.send",
                "workspace-communications.artifact.send",
                "device-communications.state.send",
            }
            and key in {"target", "channel", "account"}
            and isinstance(value, str)
            and re.fullmatch(
                r"(?:send|text) me (?:(?:the )?grocery list|(?:my )?calendar|"
                r"(?:the )?research (?:on|about) .+|(?:tomorrow's|the) weather|"
                r"the document .+|the health of (?:service )?.+|"
                r"(?:the )?workspace artifact .+|(?:the )?device status)[?!.,]?",
                intent.utterance.strip(),
                flags=re.IGNORECASE,
            )
        ):
            try:
                approved_targets = configured_communication_targets()
            except ValueError:
                approved_targets = frozenset()
            if approved_targets is not None and len(approved_targets) == 1:
                approved_target, approved_channel, approved_account = next(iter(approved_targets))
                expected = {
                    "target": approved_target,
                    "channel": approved_channel,
                    "account": approved_account,
                }[key]
                if value == expected:
                    provenance[key] = ArgumentProvenance(
                        kind=ArgumentProvenanceKind.APPROVED_DEFAULT,
                        default_contract="owner.approved_communication_target.v1",
                    )
                    continue
        if (
            card.action.action_id == "weather.forecast.read"
            and key in {"latitude", "longitude"}
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and re.search(r"\btomorrow\b", intent.utterance, flags=re.IGNORECASE)
        ):
            provenance[key] = ArgumentProvenance(
                kind=ArgumentProvenanceKind.APPROVED_DEFAULT,
                default_contract="owner.weather_coordinates.v1",
            )
            continue
        if card.action.action_id == "weather.forecast.read" and key == "days" and value == 2:
            tomorrow = re.search(r"\btomorrow\b", intent.utterance, flags=re.IGNORECASE)
            if tomorrow is not None:
                provenance[key] = ArgumentProvenance(
                    kind=ArgumentProvenanceKind.DETERMINISTIC_DERIVATION,
                    source_spans=(tomorrow.span(),),
                    derivation="reference.tomorrow_forecast_horizon.v1",
                )
                continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric_spans = tuple(
                match.span()
                for match in re.finditer(
                    r"(?<![\w.])-?(?:\d+(?:\.\d+)?)(?![\w.])", intent.utterance
                )
                if float(match.group(0)) == float(value)
            )
            if numeric_spans:
                provenance[key] = ArgumentProvenance(
                    kind=ArgumentProvenanceKind.EXPLICIT_UTTERANCE,
                    source_spans=numeric_spans[:1],
                )
                continue
        if key == "body_source" and value in {
            "canonical.groceries",
            "canonical.calendar",
            "bounded.research",
            "public.weather",
            "canonical.document",
            "canonical.homelab_health",
            "canonical.workspace_artifact",
            "canonical.device_state",
        }:
            source_phrase = {
                "canonical.groceries": "grocery list",
                "canonical.calendar": "calendar",
                "bounded.research": "research",
                "public.weather": "weather",
                "canonical.document": "document",
                "canonical.homelab_health": "health",
                "canonical.workspace_artifact": "workspace artifact",
                "canonical.device_state": "device status",
            }[value]
            spans = _utterance_spans(intent.utterance, source_phrase)
            if not spans:
                return Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.BLOCKED,
                    message="I could not safely identify the canonical message source.",
                    correlation_id=intent.correlation_id,
                )
            provenance[key] = ArgumentProvenance(
                kind=ArgumentProvenanceKind.DETERMINISTIC_DERIVATION,
                source_spans=spans,
                derivation=(
                    "reference.communication_body_from_groceries.v1"
                    if value == "canonical.groceries"
                    else (
                        "reference.communication_body_from_calendar.v1"
                        if value == "canonical.calendar"
                        else (
                            "reference.communication_body_from_research.v1"
                            if value == "bounded.research"
                            else (
                                "reference.communication_body_from_weather.v1"
                                if value == "public.weather"
                                else (
                                    "reference.communication_body_from_document.v1"
                                    if value == "canonical.document"
                                    else (
                                        "reference.communication_body_from_homelab_health.v1"
                                        if value == "canonical.homelab_health"
                                        else (
                                            "reference.communication_body_from_workspace_artifact.v1"
                                            if value == "canonical.workspace_artifact"
                                            else "reference.communication_body_from_device_state.v1"
                                        )
                                    )
                                )
                            )
                        )
                    )
                ),
            )
            continue
        if key == "workspace_id":
            spans = _utterance_spans(intent.utterance, value)
            if spans:
                provenance[key] = ArgumentProvenance(
                    kind=ArgumentProvenanceKind.EXPLICIT_UTTERANCE,
                    source_spans=spans,
                )
                continue
        if key.endswith("_id"):
            if not isinstance(value, str) or not value.strip():
                return Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.BLOCKED,
                    message="I could not safely ground that canonical target.",
                    correlation_id=intent.correlation_id,
                )
            if key == "entity_id":
                spans = _utterance_spans(intent.utterance, value)
                if spans:
                    provenance[key] = ArgumentProvenance(
                        kind=ArgumentProvenanceKind.EXPLICIT_UTTERANCE,
                        source_spans=spans,
                    )
                    continue
            provenance[key] = ArgumentProvenance(
                kind=ArgumentProvenanceKind.AUTHORIZED_CANONICAL_REFERENT,
                canonical_ref=value,
            )
            continue
        if key.endswith("_at"):
            if not isinstance(value, str):
                return Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.BLOCKED,
                    message="I need a clear date or time for that consequential action.",
                    evidence={"failure": "CONSEQUENTIAL_ARGUMENT_PROVENANCE_UNAVAILABLE"},
                    correlation_id=intent.correlation_id,
                )
            try:
                proposed_temporal = datetime.fromisoformat(value)
            except ValueError:
                proposed_temporal = None
            clock = re.search(
                r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
                intent.utterance,
                flags=re.IGNORECASE,
            )
            if proposed_temporal is None or (
                key == "starts_at"
                and clock is None
                and proposed_temporal.time() != datetime.min.time()
            ):
                return Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.BLOCKED,
                    message="I could not safely ground the requested date or time.",
                    evidence={"failure": "CONSEQUENTIAL_ARGUMENT_PROVENANCE_UNAVAILABLE"},
                    correlation_id=intent.correlation_id,
                )
            if key == "starts_at" and clock is not None:
                hour = int(clock.group(1)) % 12
                if clock.group(3).casefold() == "pm":
                    hour += 12
                minute = int(clock.group(2) or 0)
                if (proposed_temporal.hour, proposed_temporal.minute) != (hour, minute):
                    return Result(
                        objective_id=uuid4(),
                        state=ObjectiveState.BLOCKED,
                        message="I could not safely ground the requested date or time.",
                        evidence={"failure": "CONSEQUENTIAL_ARGUMENT_PROVENANCE_UNAVAILABLE"},
                        correlation_id=intent.correlation_id,
                    )
            spans = tuple(
                match.span()
                for match in re.finditer(
                    r"\b(?:today|tomorrow|tonight|morning|afternoon|evening|noon|midnight|"
                    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|next\s+week|"
                    r"\d{1,2}(?::\d{2})?\s*(?:am|pm)|\d{4}-\d{2}-\d{2})\b",
                    intent.utterance,
                    flags=re.IGNORECASE,
                )
            )
            if not spans:
                return Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.BLOCKED,
                    message="I need an explicit date or time for that consequential action.",
                    evidence={"failure": "CONSEQUENTIAL_ARGUMENT_PROVENANCE_UNAVAILABLE"},
                    correlation_id=intent.correlation_id,
                )
            provenance[key] = ArgumentProvenance(
                kind=ArgumentProvenanceKind.DETERMINISTIC_DERIVATION,
                source_spans=spans[:5],
                derivation="reference.temporal_grounding.v1",
            )
            continue
        if (
            card.action.action_id == "device-controls.devices.command.execute"
            and key == "service"
            and isinstance(value, str)
        ):
            service_match = re.search(
                r"\b(?:turn|switch)\s+(on|off)\b", intent.utterance, flags=re.IGNORECASE
            )
            if service_match is not None and value == f"turn_{service_match.group(1).casefold()}":
                provenance[key] = ArgumentProvenance(
                    kind=ArgumentProvenanceKind.DETERMINISTIC_DERIVATION,
                    source_spans=(service_match.span(),),
                    derivation="reference.device_service.v1",
                )
                continue
        if card.action.action_id == "workspace.artifact.create" and key == "files":
            if not isinstance(value, dict):
                return Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.BLOCKED,
                    message="I could not safely ground the requested workspace files.",
                    correlation_id=intent.correlation_id,
                )
            spans = tuple(
                span
                for path, content in value.items()
                for span_group in (
                    _utterance_spans(intent.utterance, path),
                    _utterance_spans(intent.utterance, content),
                )
                for span in span_group
            )
            if len(spans) != len(value) * 2:
                return Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.BLOCKED,
                    message="I could not safely ground the requested workspace files.",
                    correlation_id=intent.correlation_id,
                )
            provenance[key] = ArgumentProvenance(
                kind=ArgumentProvenanceKind.EXPLICIT_UTTERANCE,
                source_spans=spans,
            )
            continue
        if (
            card.action.action_id == "homelab-reports.inventory.to_workspace"
            and key == "target_path"
            and value == "index.html"
        ):
            span = _utterance_spans(intent.utterance, "homelab")
            if span:
                provenance[key] = ArgumentProvenance(
                    kind=ArgumentProvenanceKind.DETERMINISTIC_DERIVATION,
                    source_spans=span,
                    derivation="reference.homelab_page_target.v1",
                )
                continue
        spans = _utterance_spans(intent.utterance, value)
        if not spans:
            canonical_ref: str | None = None
            if key in {"title", "item", "content"} and context is not None:
                for identity_key in ("task_id", "chore_id", "event_id", "memory_id"):
                    identity = card.action.arguments.get(identity_key)
                    if isinstance(identity, str) and identity.strip():
                        canonical_ref = identity
                        break
                referents = context.values.get("referents")
                candidates = (
                    referents.get("those", {}).get("candidates", [])
                    if isinstance(referents, dict)
                    else []
                )
                matches = [
                    candidate
                    for candidate in candidates
                    if isinstance(candidate, dict)
                    and candidate.get(key, candidate.get("title")) == value
                ]
                if len(matches) == 1:
                    candidate = matches[0]
                    canonical_ref = next(
                        (
                            str(candidate[field])
                            for field in (
                                "task_id",
                                "chore_id",
                                "event_id",
                                "memory_id",
                                "document_id",
                            )
                            if candidate.get(field)
                        ),
                        str(value),
                    )
            if canonical_ref is not None:
                provenance[key] = ArgumentProvenance(
                    kind=ArgumentProvenanceKind.AUTHORIZED_CANONICAL_REFERENT,
                    canonical_ref=canonical_ref,
                )
                continue
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message="I could not safely ground a consequential argument in your request.",
                evidence={"failure": "CONSEQUENTIAL_ARGUMENT_PROVENANCE_UNAVAILABLE"},
                correlation_id=intent.correlation_id,
            )
        provenance[key] = ArgumentProvenance(
            kind=ArgumentProvenanceKind.EXPLICIT_UTTERANCE,
            source_spans=spans,
        )
    return card.model_copy(
        update={"action": card.action.model_copy(update={"argument_provenance": provenance})}
    )


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
        area_details: dict[str, dict[str, Any]] = {}
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
            # Pack metadata is the source of truth for the bounded second
            # level of the Constellation.  The browser receives capability
            # labels, never executable semantics or authority.
            areas: dict[str, list[str]] = {}
            for card in bundle.cards:
                capability = card.action.capability
                area = capability.removeprefix(f"{pack_id}.").split(".", 1)[0]
                areas.setdefault(area, []).append(capability)
            for area, capabilities in sorted(areas.items()):
                area_id = f"{node_id}-area-{area}"
                nodes.append(
                    {
                        "id": area_id,
                        "label": area.replace("-", " ").title(),
                        "detail": (
                            f"{len(capabilities)} capability"
                            if len(capabilities) == 1
                            else f"{len(capabilities)} capabilities"
                        ),
                        "category": "capability",
                        "detail_view": ui.detail_view if ui else None,
                    }
                )
                area_details[area_id] = {
                    "pack": label,
                    "capabilities": sorted(capabilities),
                    "authority": "Core authorization remains required",
                }
                edges.append({"source": node_id, "target": area_id})
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
        # Composition metadata is a bounded semantic layer above Pack views.
        # It is descriptive navigation context only: no composition node is an
        # executable action or an authority grant.
        pack_labels = {
            node["label"]: node["id"] for node in nodes if node.get("id", "").startswith("pack-")
        }
        for composition in available_compositions():
            composition_id = composition.get("id")
            if not isinstance(composition_id, str) or not composition_id:
                continue
            node_id = f"composition-{composition_id}"
            raw_surfaces = composition.get("surfaces", ())
            surfaces = (
                tuple(item for item in raw_surfaces if isinstance(item, str))
                if isinstance(raw_surfaces, (tuple, list))
                else ()
            )
            linked_packs = [
                pack_labels[surface]
                for surface in surfaces
                if isinstance(surface, str) and surface in pack_labels
            ]
            nodes.append(
                {
                    "id": node_id,
                    "label": str(composition.get("label", composition_id)),
                    "detail": str(composition.get("description", "bounded workflow")),
                    "category": "composition",
                    "detail_view": "compositions",
                }
            )
            area_details[node_id] = {
                "surfaces": list(surfaces),
                "authority": str(
                    composition.get("authority", "Core authorization remains required")
                ),
            }
            for pack_id in linked_packs or ["aegis"]:
                edges.append({"source": pack_id, "target": node_id})
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
        details.update(area_details)
        return {"nodes": nodes, "edges": edges, "details": details}
    finally:
        connection.close()


def _display_due_at(value: object) -> str:
    """Render canonical deadlines concisely without changing stored evidence."""

    text = str(value)
    if len(text) <= 10:
        return text
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return text
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone()
    return parsed.strftime("%Y-%m-%d %H:%M %Z").strip()


def reference_format_result(result: Any) -> str:
    """Render reference-Pack canonical evidence for human-facing clients."""

    if result.state.value != "completed":
        return f"Not completed — {result.message}"
    evidence = result.evidence
    if evidence.get("provenance") == "model_generated":
        return str(result.message)
    if evidence.get("referent") == "prior_result":
        return str(result.message)
    if evidence.get("priority_basis") and evidence.get("task") is not None:
        return str(result.message)
    if evidence.get("priority_basis") and evidence.get("authorized_event_priority") is not None:
        return str(result.message)
    if evidence.get("priority_basis") and evidence.get("collection") == "events":
        return str(result.message)
    if isinstance(evidence.get("authorized_next_referent"), dict):
        return str(result.message)
    if any(
        evidence.get(key) is not None
        for key in (
            "authorized_task_focus",
            "authorized_event_focus",
            "authorized_obligation_focus",
            "authorized_chore_focus",
        )
    ):
        return str(result.message)
    if evidence.get("authorized_ordinal_item") is not None:
        return f"Grocery item: {evidence['authorized_ordinal_item']}"
    # Ordinal follow-ups are canonical reads.  Their evidence carries the
    # already-rendered domain label so it must not fall through to the generic
    # mutation-shaped title formatter ("Done — ...").
    if isinstance(evidence.get("authorized_ordinal_referent"), dict):
        return str(result.message)
    if "authorized_other_items" in evidence:
        return str(result.message)
    if isinstance(evidence.get("authorized_other_referent"), dict):
        return str(result.message)
    if isinstance(evidence.get("authorized_relative_referent"), dict):
        return str(result.message)
    if evidence.get("authorized_owned_obligations") is not None:
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
        entries = [
            f"{item['title']} ({item['status']})"
            + (f" due {_display_due_at(item['due_at'])}" if item.get("due_at") else "")
            for item in display_tasks
        ]
        listing = "\n".join(f"• {entry}" for entry in entries)
        if len(tasks) > len(display_tasks):
            listing += f"\n… and {len(tasks) - len(display_tasks)} more"
        return "Tasks: (empty)" if not tasks else "Tasks:\n" + listing
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
        listing = "\n".join(f"• {item['title']} ({item['assignee_id']})" for item in display_chores)
        if len(chores) > len(display_chores):
            listing += f"\n… and {len(chores) - len(display_chores)} more"
        return "Chores: (none)" if not chores else "Chores:\n" + listing
    if evidence.get("events") is not None:
        events = evidence["events"]
        display_events = events[:20]
        listing = "\n".join(f"• {item['title']}" for item in display_events)
        if len(events) > len(display_events):
            listing += f"\n… and {len(events) - len(display_events)} more"
        return "Events: (none)" if not events else "Events:\n" + listing
    if isinstance(evidence.get("workspace_inventory"), list):
        workspaces = evidence["workspace_inventory"]
        entries = []
        for item in workspaces[:20]:
            if not isinstance(item, dict):
                continue
            workspace_id = item.get("workspace_id", "unknown workspace")
            files = item.get("files", [])
            names = ", ".join(str(path) for path in files[:20]) if isinstance(files, list) else ""
            entries.append(f"{workspace_id}: {names or '(empty)'}")
        if not entries:
            return "Workspace artifacts: (none)"
        return "Workspace artifacts:\n" + "\n".join(f"• {entry}" for entry in entries)
    if isinstance(evidence.get("workspace_file"), dict):
        file_record = evidence["workspace_file"]
        path = file_record.get("path", "file")
        content = file_record.get("content")
        if isinstance(content, str):
            return f"Workspace file {path}:\n{content[:20_000]}"
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
            # With the real model enabled, let the strict bounded PLAN
            # contract attempt the proposal.  The no-model path retains the
            # deterministic fail-closed response.
            if not model_enabled:
                return result
        # A recognized bounded compound plan owns its dependent references;
        # leave it for the existing plan validator/Kernel path.  Unrecognized
        # mutation references still fall through to ContextualMutationGuard.
        if (
            MultiActionFastPath.task_chore_titles(intent.utterance) is not None
            or MultiActionFastPath.task_event_details(intent.utterance) is not None
        ):
            return None
    if context is not None:
        result = resolve_contextual_remaining(intent, context)
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
            or resolve_unique_prior_task_reference(intent.utterance, context) is not None
        )
    ):
        return None
    if context is not None and is_correction_request(intent.utterance):
        if context.sources == ("authorized_prior_result",) and not is_question_request(
            strip_correction_prefix(intent.utterance)
        ):
            # A blocked prior turn has no canonical plan or referent to amend.
            # Keep imperative correction language out of model action selection.
            return ContextualMutationGuard.resolve(intent)
        # Let the authorized referent resolver provide a domain-specific
        # clarification; the context-free guard protects only corrections that
        # lack canonical context to resolve against.
        return None
    if context is not None and _ambiguous_temporal_collection_follow_up(intent, context):
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=(
                "Could you clarify what you mean by that date? Are you asking about "
                "tasks, events, or groceries?"
            ),
            correlation_id=intent.correlation_id,
        )
    return ContextualMutationGuard.resolve(intent)


def _ambiguous_temporal_collection_follow_up(intent: IntentFrame, context: Context) -> bool:
    """Keep a date-only follow-up from becoming an ungrounded mutation."""

    if context.sources != ("authorized_canonical_result",):
        return False
    referents = context.values.get("referents")
    those = referents.get("those") if isinstance(referents, dict) else None
    if not isinstance(those, dict) or those.get("fact_key") != "canonical_items":
        return False
    text = " ".join(intent.utterance.casefold().split()).strip(".!?")
    return (
        re.fullmatch(r"(?:what|how) about (?:today|tomorrow|this week|next week)", text) is not None
    )


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
    if "plan_steps" in facts:
        return Context(values=values, sources=context.sources)
    if "canonical_items" in facts or "canonical_tasks" in facts:
        return Context(
            values=values,
            sources=tuple(dict.fromkeys((*context.sources, "authorized_canonical_context"))),
        )
    # The fallback builder may see every reference-Pack projection, but an
    # unrelated general question must not receive that projection as tempting
    # answer material.  Canonical context is relevant only when the utterance
    # identifies an owned collection/state concern or an explicit attention
    # request.  This is context minimization, not an intent router; normal
    # domain fast paths still handle the explicit reads before cognition.
    lowered = utterance.casefold()
    canonical_markers = (
        "task",
        "todo",
        "chore",
        "grocery",
        "groceries",
        "shopping list",
        "obligation",
        "utility",
        "utilities",
        "calendar",
        "event",
        "appointment",
        "need attention",
        "needs attention",
        "my ",
        "i need",
        "what should i",
    )
    if not any(marker in lowered for marker in canonical_markers):
        return Context(values=values, sources=context.sources)
    facts["canonical_items"] = list(
        dict.fromkeys(str(item) for item in household_store.list_groceries(principal))
    )[:20]
    facts["canonical_item_scope"] = "kitchen.shopping_list"
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

    text = utterance.casefold()
    explicit_kitchen = any(marker in text for marker in ("grocery", "groceries", "shopping list"))
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
            and not explicit_kitchen
        ):
            return tuple(manager.retrieve("tasks"))[:10]
    domain = next(
        (
            pack_id
            for marker, pack_id in (
                ("task", "tasks"),
                ("chore", "tasks"),
                ("event", "tasks"),
                ("calendar", "calendar"),
                ("appointment", "calendar"),
                ("grocery", "kitchen"),
                ("grocerie", "kitchen"),
                ("shopping list", "kitchen"),
                ("message", "communications"),
                ("communication", "communications"),
                ("document", "documents"),
                ("file", "documents"),
                ("device", "devices"),
                ("entity", "devices"),
                ("homelab", "homelab"),
                ("service", "homelab"),
                ("network", "network"),
            )
            if marker in text
        ),
        None,
    )
    if is_task_destination_request(utterance):
        if MultiActionFastPath.matches(utterance):
            # A compound request needs every bounded action family from the
            # Tasks Pack; the default five-card shortlist omits events.
            return tuple(manager.retrieve("tasks", limit=10))[:10]
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
    if re.search(r"\bnext\s+one\b", intent.utterance.casefold()):
        next_result = resolve_contextual_ordinal_read(intent, context)
        if next_result is not None:
            return next_result
    modification_result = PlanModificationFastPath.resolve(intent, context)
    if modification_result is not None:
        return modification_result
    progress_result = PlanProgressFastPath.resolve(intent, context)
    if progress_result is not None:
        return progress_result
    recent_action_result = resolve_contextual_recent_action_read(intent, context)
    if recent_action_result is not None:
        return recent_action_result
    ambiguous_collection_result = DomainClarificationFastPath.resolve_ambiguous_collection_status(
        intent
    )
    if ambiguous_collection_result is not None:
        return ambiguous_collection_result
    # A recognized compound mutation must reach the plan runner before any
    # domain read fast path.  Continuation context may contain the word
    # "task" (or a relative date), but that must not collapse a plan into a
    # canonical collection read.
    if (
        MultiActionFastPath.task_chore_titles(intent.utterance) is not None
        or MultiActionFastPath.task_event_details(intent.utterance) is not None
    ):
        return None
    # A compound mutation must reach the bounded plan path.  Do not let the
    # read-only cross-domain planning context claim completion for an objective
    # that explicitly asks Core to create or change multiple records.
    if MultiActionFastPath.matches(intent.utterance):
        return None
    normalized_read = strip_correction_prefix(intent.utterance)
    if not is_mutation_request(intent.utterance) and has_multiple_question_clauses(normalized_read):
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=(
                "That request contains multiple independent reads. "
                "Please ask one at a time so each result stays grounded."
            ),
            correlation_id=intent.correlation_id,
        )
    task_store = PostgresTaskStore(connection)
    household_store = PostgresHouseholdStore(connection)
    snapshot = household_store.read_snapshot(principal)
    personal_obligation_result = resolve_personal_obligation_read(intent, snapshot)
    if personal_obligation_result is not None:
        return personal_obligation_result
    direct_obligation_result = resolve_direct_obligation_ordinal_read(intent, snapshot)
    if direct_obligation_result is not None:
        return direct_obligation_result
    named_obligation_result = resolve_named_obligation_read(intent, snapshot)
    if named_obligation_result is not None:
        return named_obligation_result
    repeat_result = resolve_contextual_repeat_read(intent, context, household_store, task_store)
    if repeat_result is not None:
        return repeat_result
    # Priority language such as "which one is latest" is a semantic event
    # read when the immediately authorized collection is events. Resolve it
    # before the generic ordinal guard, which cannot identify the domain.
    result = resolve_contextual_event_priority_read(intent, context, snapshot)
    if result is not None:
        return result
    result = resolve_contextual_event_relative_read(intent, context, snapshot)
    if result is not None:
        return result
    # Explicit task-priority language such as "handle first" must win over
    # the generic ordinal resolver; both paths still use the same authorized
    # task candidate set and deterministic deadline ordering.
    result = ContextualTaskPriorityFastPath().resolve(intent, context)
    if result is not None:
        return result
    # Chores have no canonical deadline order. Resolve their priority guard
    # before the generic ordinal path so "which one needs attention first"
    # cannot be laundered into selecting the first displayed chore.
    result = ContextualChorePriorityFastPath.resolve(intent, context)
    if result is not None:
        return result
    contextual_ordinal_result = resolve_contextual_ordinal_read(intent, context)
    if contextual_ordinal_result is not None:
        return contextual_ordinal_result
    contextual_task_focus_result = resolve_contextual_task_focus_read(intent, context, task_store)
    if contextual_task_focus_result is not None:
        return contextual_task_focus_result
    contextual_obligation_result = resolve_contextual_obligation_focus_read(
        intent, context, snapshot
    )
    if contextual_obligation_result is not None:
        return contextual_obligation_result
    contextual_chore_result = resolve_contextual_chore_focus_read(intent, context, snapshot)
    if contextual_chore_result is not None:
        return contextual_chore_result
    contextual_grocery_result = resolve_contextual_grocery_membership_read(
        intent, context, household_store
    )
    if contextual_grocery_result is not None:
        return contextual_grocery_result
    contextual_grocery_quantity_result = resolve_contextual_grocery_quantity_read(
        intent, context, household_store
    )
    if contextual_grocery_quantity_result is not None:
        return contextual_grocery_quantity_result
    contextual_grocery_other_result = resolve_contextual_grocery_other_read(
        intent, context, household_store
    )
    if contextual_grocery_other_result is not None:
        return contextual_grocery_other_result
    personal_state = PostgresPersonalStateStore(connection, principal.vault_id).load_for_principal(
        principal
    )
    result = resolve_contextual_event_next_read(intent, context, snapshot)
    if result is not None:
        return result
    result = resolve_contextual_event_focus_read(intent, context, snapshot)
    if result is not None:
        return result
    result = resolve_ambiguous_event_focus_read(intent, context)
    if result is not None:
        return result
    # Resolve an authorized event temporal follow-up before broad personal
    # composers can reinterpret a short correction such as "No, tomorrow."
    # The resolver remains read-only and requires the prior event projection.
    result = resolve_contextual_event_temporal_read(intent, context, snapshot)
    if result is not None:
        return result
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
    if composed_title is None:
        result = ContextualChorePriorityFastPath.resolve(intent, context)
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
        result = ContextualTaskTemporalFastPath().resolve(intent, context, task_store)
        if result is not None:
            return result
        result = ContextualTaskPriorityFastPath().resolve(intent, context)
        if result is not None:
            return result
        result = ContextualCrossDomainPriorityFastPath().resolve(intent, context)
        if result is not None:
            return result
        # A direct task-priority question must not be swallowed by the broad
        # collection read, even when the immediately preceding result was a
        # scalar task mutation rather than a task list.
        result = TaskPriorityFastPath(task_store).resolve(intent)
        if result is not None:
            return result
        result = TaskReadFastPath(task_store).resolve(intent)
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
    ordinal_reference = (
        re.search(
            r"\b(?:the\s+)?(?:first|second|third|fourth|last|next)\s+"
            r"(?:one|task|chore|event)\b",
            text,
        )
        is not None
    )
    unsupported_ordinal_reference = (
        re.search(
            r"\b(?:the\s+)?(?:fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th))\b",
            text,
        )
        is not None
    )
    if (
        is_mutation_request(text)
        or not any(
            marker in text
            for marker in ("what about", "tell me about", "which one", "what is", "show", "meant")
        )
        and not re.search(r"\bshow(?: me)? the task (?:before|after) that\b", text)
        and not (
            "which" in text
            and re.search(r"\b(?:the\s+)?(?:first|second|third|fourth|last)\b", text)
        )
        and not ambiguous_correction
        and not ordinal_reference
        and not unsupported_ordinal_reference
    ):
        return None
    referents = context.values.get("referents")
    those = referents.get("those") if isinstance(referents, dict) else None
    candidates = those.get("candidates") if isinstance(those, dict) else None
    fact_key = those.get("fact_key") if isinstance(those, dict) else None
    relative_match = re.search(r"\b(?:one|task)\s+(before|after)\s+that\b", text)
    next_match = re.search(r"\b(?:the\s+)?next\s+one\b", text)
    selected_task = context.values.get("canonical_facts", {}).get("task")
    if (
        next_match is not None
        and fact_key == "canonical_tasks"
        and isinstance(candidates, list)
        and isinstance(selected_task, dict)
    ):
        selected_indices = [
            index
            for index, candidate in enumerate(candidates)
            if isinstance(candidate, dict)
            and candidate.get("task_id") == selected_task.get("task_id")
        ]
        if len(selected_indices) == 1 and selected_indices[0] + 1 < len(candidates):
            neighbor = candidates[selected_indices[0] + 1]
            if isinstance(neighbor, dict) and isinstance(neighbor.get("title"), str):
                detail = f"Task: {neighbor['title']}"
                status = neighbor.get("status") or (
                    "completed" if neighbor.get("completed") is True else "open"
                )
                if isinstance(status, str):
                    detail += f" ({status})"
                due_at = neighbor.get("due_at")
                if isinstance(due_at, str):
                    detail += f"; due {_display_due_at(due_at)}"
                return Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.COMPLETED,
                    message=detail,
                    evidence={
                        "collection": "canonical_tasks",
                        "authorized_relative_referent": neighbor,
                        "canonical_tasks": candidates,
                        "task": neighbor,
                    },
                    correlation_id=intent.correlation_id,
                )
    if (
        next_match is not None
        and fact_key == "canonical_items"
        and isinstance(candidates, list)
        and candidates
    ):
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=(
                "The grocery list has no canonical next item. "
                "Please name a grocery item or choose an ordinal."
            ),
            correlation_id=intent.correlation_id,
        )
    if (
        relative_match is not None
        and fact_key == "canonical_tasks"
        and isinstance(candidates, list)
        and isinstance(selected_task, dict)
    ):
        selected_indices = [
            index
            for index, candidate in enumerate(candidates)
            if isinstance(candidate, dict)
            and (
                candidate.get("task_id") == selected_task.get("task_id")
                or candidate.get("title") == selected_task.get("title")
            )
        ]
        if len(selected_indices) == 1:
            offset = -1 if relative_match.group(1) == "before" else 1
            neighbor_index = selected_indices[0] + offset
            if 0 <= neighbor_index < len(candidates):
                neighbor = candidates[neighbor_index]
                if isinstance(neighbor, dict) and isinstance(neighbor.get("title"), str):
                    detail = f"Task: {neighbor['title']}"
                    status = neighbor.get("status") or (
                        "completed" if neighbor.get("completed") is True else "open"
                    )
                    if isinstance(status, str):
                        detail += f" ({status})"
                    due_at = neighbor.get("due_at")
                    if isinstance(due_at, str):
                        detail += f"; due {_display_due_at(due_at)}"
                    return Result(
                        objective_id=uuid4(),
                        state=ObjectiveState.COMPLETED,
                        message=detail,
                        evidence={
                            "collection": "canonical_tasks",
                            "authorized_relative_referent": neighbor,
                            "canonical_tasks": candidates,
                            "task": neighbor,
                        },
                        correlation_id=intent.correlation_id,
                    )
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message=(
                    f"There is no task {relative_match.group(1)} that one in the referenced list."
                ),
                correlation_id=intent.correlation_id,
            )
    unsupported_ordinal = re.search(
        r"\b(?:fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th))\b", text
    )
    if (
        unsupported_ordinal is not None
        and isinstance(those, dict)
        and isinstance(candidates, list)
        and candidates
    ):
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=(
                "I can resolve the first four or last item from that list. "
                "Please choose an available ordinal."
            ),
            correlation_id=intent.correlation_id,
        )
    requested_domain = next(
        (
            key
            for key, terms in {
                "canonical_tasks": ("task", "tasks"),
                "canonical_chores": ("chore", "chores"),
                "events": ("event", "events", "calendar", "appointment", "appointments"),
                "canonical_items": ("grocery", "groceries"),
            }.items()
            if re.search(rf"\b(?:{'|'.join(terms)})\b", text)
        ),
        None,
    )
    if requested_domain is not None and fact_key is not None and requested_domain != fact_key:
        # An ordinal word such as "first" is not permission to reinterpret a
        # prior collection in a different domain.
        return None
    if (
        isinstance(candidates, list)
        and len(candidates) == 1
        and " ".join(text.split()).strip(".!?") == "which one"
    ):
        candidate = candidates[0]
        if fact_key == "canonical_items" and isinstance(candidate, str) and candidate.strip():
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.COMPLETED,
                message=f"Grocery item: {candidate}",
                evidence={
                    "collection": "groceries",
                    "authorized_unique_referent": candidate,
                    "canonical_items": candidates,
                },
                correlation_id=intent.correlation_id,
            )
        if isinstance(candidate, dict) and isinstance(candidate.get("title"), str):
            label = {
                "canonical_tasks": "Task",
                "canonical_chores": "Chore",
                "events": "Event",
            }.get(str(fact_key))
            if label is not None:
                return Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.COMPLETED,
                    message=f"{label}: {candidate['title']}",
                    evidence={
                        "collection": fact_key,
                        "authorized_unique_referent": candidate,
                        str(fact_key): candidates,
                    },
                    correlation_id=intent.correlation_id,
                )
    ordinal_match = re.search(r"\b(?:the\s+)?(first|second|third|fourth|last)\b", text)
    if isinstance(candidates, list) and candidates and ordinal_match is not None:
        ordinal_index = {"first": 0, "second": 1, "third": 2, "fourth": 3, "last": -1}[
            ordinal_match.group(1)
        ]
        if ordinal_index >= len(candidates) or ordinal_index < -len(candidates):
            labels = {
                "canonical_items": "grocery items",
                "canonical_tasks": "tasks",
                "canonical_chores": "chores",
                "events": "events",
            }
            label = labels.get(str(fact_key), "items")
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message=f"I found fewer {label} than that. Please choose an available ordinal.",
                correlation_id=intent.correlation_id,
            )
    if (
        isinstance(those, dict)
        and those.get("fact_key") == "canonical_items"
        and re.search(r"\b(?:due|handle|take care of|urgent|priority)\b", text)
        and re.search(r"\b(?:first|next|most)\b", text)
    ):
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=(
                "The grocery list has no canonical priority order. "
                "Please name a grocery item, or ask which task is due first."
            ),
            correlation_id=intent.correlation_id,
        )
    if isinstance(candidates, list) and ambiguous_correction:
        selected_task = context.values.get("canonical_facts", {}).get("task")
        if fact_key == "canonical_tasks" and isinstance(selected_task, dict):
            selected_candidates = [
                candidate
                for candidate in candidates
                if isinstance(candidate, dict)
                and (
                    candidate.get("task_id") == selected_task.get("task_id")
                    or candidate.get("title") == selected_task.get("title")
                )
            ]
            other_candidates = [
                candidate for candidate in candidates if candidate not in selected_candidates
            ]
            if len(selected_candidates) == 1 and len(other_candidates) == 1:
                other = other_candidates[0]
                title = other.get("title") if isinstance(other, dict) else None
                if isinstance(title, str) and title:
                    detail = f"Task: {title}"
                    status = other.get("status") or (
                        "completed" if other.get("completed") is True else "open"
                    )
                    if isinstance(status, str):
                        detail += f" ({status})"
                    due_at = other.get("due_at")
                    if isinstance(due_at, str):
                        detail += f"; due {_display_due_at(due_at)}"
                    return Result(
                        objective_id=uuid4(),
                        state=ObjectiveState.COMPLETED,
                        message=detail,
                        evidence={
                            "collection": "canonical_tasks",
                            "authorized_other_referent": other,
                            "canonical_tasks": candidates,
                            "task": other,
                        },
                        correlation_id=intent.correlation_id,
                    )
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
        ("canonical_obligations", "Obligation"),
    ):
        referent = resolve_obvious_ordinal(text, context, fact_key)
        if referent is None:
            continue
        title = referent.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        detail = f"{label}: {title}"
        if fact_key == "canonical_obligations" and isinstance(referent.get("settled"), bool):
            detail += f" ({'settled' if referent['settled'] else 'unsettled'})"
            responsible_id = referent.get("responsible_id")
            if isinstance(responsible_id, str) and responsible_id:
                detail += f" ({responsible_id})"
        elif fact_key != "events":
            status = referent.get("status") or (
                "completed" if referent.get("completed") is True else "open"
            )
            if isinstance(status, str):
                detail += f" ({status})"
        due_at = referent.get("due_at")
        due_question = fact_key == "canonical_tasks" and "due" in text
        if isinstance(due_at, str):
            detail += f"; due {_display_due_at(due_at)}"
        elif due_question:
            # Answer the temporal part without manufacturing a deadline for
            # an authorized task that has none.
            detail += "; no recorded deadline"
        starts_at = referent.get("starts_at")
        if isinstance(starts_at, str):
            detail += f"; starts {_display_due_at(starts_at)}"
        collection_evidence = {
            "collection": fact_key,
            "authorized_ordinal_referent": referent,
        }
        if isinstance(candidates, list):
            collection_evidence[fact_key] = candidates
        if fact_key == "canonical_tasks":
            collection_evidence["task"] = referent
        elif fact_key == "events":
            # Preserve the selected scalar event focus for a subsequent
            # authorized temporal/date follow-up.
            collection_evidence["event"] = referent
        elif fact_key == "canonical_obligations":
            collection_evidence["obligation"] = referent
        elif fact_key == "canonical_chores":
            collection_evidence["chore"] = referent
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.COMPLETED,
            message=detail,
            evidence=collection_evidence,
            correlation_id=intent.correlation_id,
        )
    return None


def resolve_contextual_grocery_membership_read(
    intent: IntentFrame, context: Context, store: PostgresHouseholdStore
) -> Result | None:
    """Answer a grocery membership follow-up only from authorized grocery context."""

    if context.sources != ("authorized_canonical_result",):
        return None
    text = " ".join(intent.utterance.casefold().split()).strip(".?!")
    if "list" not in text or "still" not in text:
        return None
    referents = context.values.get("referents")
    those = referents.get("those") if isinstance(referents, dict) else None
    if not isinstance(those, dict) or those.get("fact_key") != "canonical_items":
        return None
    prior_items = those.get("candidates")
    if not isinstance(prior_items, list) or not prior_items:
        return None
    current_items = tuple(store.list_groceries(intent.principal))
    unique_current_items = tuple(dict.fromkeys(current_items))
    matches = tuple(
        item
        for item in unique_current_items
        if isinstance(item, str)
        and re.search(rf"\b{re.escape(item.casefold())}\b", text) is not None
    )
    if len(matches) != 1:
        return None
    item = matches[0]
    if not any(
        isinstance(prior_item, str) and prior_item.casefold() == item.casefold()
        for prior_item in prior_items
    ):
        return None
    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message=f"Grocery item: {item} is still on your list.",
        evidence={
            "collection": "groceries",
            "canonical_items": list(current_items),
            "authorized_membership": item,
            "continuation_context": "authorized_prior_result",
        },
        correlation_id=intent.correlation_id,
    )


def resolve_contextual_grocery_quantity_read(
    intent: IntentFrame, context: Context, store: PostgresHouseholdStore
) -> Result | None:
    """Answer a quantity question only after an authorized shopping-list read."""

    if context.sources != ("authorized_canonical_result",):
        return None
    text = " ".join(intent.utterance.casefold().split()).strip(".?!")
    quantity_question = (
        "how much" in text
        or "how many" in text
        or text in {"what is its quantity", "what is the quantity", "what quantity is it"}
    )
    if not quantity_question or (
        "need" not in text
        and text
        not in {
            "how much is it",
            "how many is it",
            "what is its quantity",
            "what is the quantity",
            "what quantity is it",
        }
    ):
        return None
    referents = context.values.get("referents")
    those = referents.get("those") if isinstance(referents, dict) else None
    if not isinstance(those, dict) or those.get("fact_key") != "canonical_items":
        return None
    prior_items = those.get("candidates")
    if not isinstance(prior_items, list) or not prior_items:
        return None
    current_items = tuple(store.list_groceries(intent.principal))
    matches = tuple(
        item
        for item in dict.fromkeys(current_items)
        if isinstance(item, str)
        and re.search(rf"\b{re.escape(item.casefold())}\b", text) is not None
        and any(
            isinstance(prior_item, str) and prior_item.casefold() == item.casefold()
            for prior_item in prior_items
        )
    )
    if (
        not matches
        and len(dict.fromkeys(item.casefold() for item in prior_items if isinstance(item, str)))
        == 1
    ):
        prior_item = next((item for item in prior_items if isinstance(item, str)), None)
        current_unique = tuple(dict.fromkeys(item for item in current_items))
        if (
            prior_item is not None
            and len(current_unique) == 1
            and current_unique[0].casefold() == prior_item.casefold()
        ):
            matches = current_unique
    if len(matches) != 1:
        return None
    item = matches[0]
    quantity = sum(
        1 for current_item in current_items if current_item.casefold() == item.casefold()
    )
    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message=f"Grocery item: {item} (x{quantity}) is on your list.",
        evidence={
            "collection": "groceries",
            "canonical_items": list(current_items),
            "authorized_quantity": {"item": item, "quantity": quantity},
            "continuation_context": "authorized_prior_result",
        },
        correlation_id=intent.correlation_id,
    )


def resolve_contextual_grocery_other_read(
    intent: IntentFrame, context: Context, store: PostgresHouseholdStore
) -> Result | None:
    """Answer a singleton-list ``what else`` follow-up without guessing an anchor."""

    if context.sources != ("authorized_canonical_result",):
        return None
    text = " ".join(intent.utterance.casefold().split()).strip(".?!")
    if "what else" not in text or not ("on it" in text or "on the list" in text):
        return None
    referents = context.values.get("referents")
    those = referents.get("those") if isinstance(referents, dict) else None
    if not isinstance(those, dict) or those.get("fact_key") != "canonical_items":
        return None
    prior_items = those.get("candidates")
    if not isinstance(prior_items, list) or len(prior_items) != 1:
        return None
    anchor = prior_items[0]
    if not isinstance(anchor, str):
        return None
    current_items = tuple(store.list_groceries(intent.principal))
    other_items = tuple(
        item for item in dict.fromkeys(current_items) if item.casefold() != anchor.casefold()
    )
    detail = (
        "No other grocery items are on your list."
        if not other_items
        else "Other groceries: " + ", ".join(other_items)
    )
    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message=detail,
        evidence={
            "collection": "groceries",
            "canonical_items": list(current_items),
            "authorized_other_items": list(other_items),
            "continuation_context": "authorized_prior_result",
        },
        correlation_id=intent.correlation_id,
    )


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


def resolve_contextual_repeat_read(
    intent: IntentFrame,
    context: Context,
    store: PostgresHouseholdStore,
    task_store: PostgresTaskStore | None = None,
) -> Result | None:
    """Repeat an authorized grocery or task collection read against current state."""

    if context.sources != ("authorized_canonical_result",) or is_mutation_request(intent.utterance):
        return None
    text = " ".join(strip_correction_prefix(intent.utterance).casefold().split()).strip(".!?")
    text = re.sub(r"^(?:and|but)\s+", "", text)
    if (
        re.fullmatch(
            r"(?:can you )?(?:show(?: me)?|list|give me) (?:the )?(?:list|it) again|"
            r"(?:can you )?show(?: me)? that",
            text,
        )
        is None
    ):
        return None
    referents = context.values.get("referents")
    those = referents.get("those") if isinstance(referents, dict) else None
    if not isinstance(those, dict):
        return None
    fact_key = those.get("fact_key")
    if fact_key == "canonical_items":
        return GroceryReadFastPath(store).resolve(
            intent.model_copy(update={"utterance": "What groceries do we need?"})
        )
    if fact_key == "events":
        prior_events = those.get("candidates")
        if not isinstance(prior_events, list) or not prior_events:
            return None
        snapshot = store.read_snapshot(intent.principal)
        current_events = snapshot.get("events")
        if not isinstance(current_events, (list, tuple)):
            return None
        grounded_events = []
        for candidate in prior_events:
            if not isinstance(candidate, dict):
                return None
            title = candidate.get("title")
            starts_at = candidate.get("starts_at")
            if not isinstance(title, str) or not isinstance(starts_at, str):
                return None
            matches = [
                event
                for event in current_events
                if event.title.casefold() == title.casefold()
                and event.starts_at.isoformat() == starts_at
            ]
            if len(matches) != 1:
                return None
            event = matches[0]
            grounded_events.append(
                {
                    "event_id": str(event.event_id),
                    "title": event.title,
                    "starts_at": event.starts_at.isoformat(),
                }
            )
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.COMPLETED,
            message="Shared household state read",
            evidence={"collection": "events", "events": grounded_events},
            correlation_id=intent.correlation_id,
        )
    if fact_key != "canonical_tasks" or task_store is None:
        if fact_key != "canonical_chores":
            return None
        prior_chores = those.get("candidates")
        if not isinstance(prior_chores, list) or not prior_chores:
            return None
        snapshot = store.read_snapshot(intent.principal)
        current_chores = snapshot.get("chores")
        if not isinstance(current_chores, (list, tuple)):
            return None
        current_by_id = {str(chore.chore_id): chore for chore in current_chores}
        grounded_chores = []
        for candidate in prior_chores:
            if not isinstance(candidate, dict) or not isinstance(candidate.get("chore_id"), str):
                return None
            chore = current_by_id.get(candidate["chore_id"])
            if chore is None:
                return None
            grounded_chores.append(
                {
                    "chore_id": str(chore.chore_id),
                    "title": chore.title,
                    "assignee_id": chore.assignee_id,
                    "completed": chore.completed,
                }
            )
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.COMPLETED,
            message="Shared household state read",
            evidence={"collection": "canonical_chores", "chores": grounded_chores},
            correlation_id=intent.correlation_id,
        )
    prior_tasks = those.get("candidates")
    if not isinstance(prior_tasks, list) or not prior_tasks:
        return None
    current_tasks = tuple(task_store.list(intent.principal))
    current_by_id = {str(task.task_id): task for task in current_tasks}
    grounded_tasks = []
    for candidate in prior_tasks:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("task_id"), str):
            return None
        task = current_by_id.get(candidate["task_id"])
        if task is None:
            return None
        grounded_tasks.append(_task_projection(task))
    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="Canonical task list read",
        evidence={"collection": "canonical_tasks", "canonical_tasks": grounded_tasks},
        correlation_id=intent.correlation_id,
    )


def resolve_contextual_event_temporal_read(
    intent: IntentFrame, context: Context, snapshot: dict[str, object]
) -> Result | None:
    """Apply a date-only read follow-up to the authorized event collection."""

    if context.sources != ("authorized_canonical_result",):
        return None
    referents = context.values.get("referents")
    those = referents.get("those") if isinstance(referents, dict) else None
    if not isinstance(those, dict) or those.get("fact_key") != "events":
        facts = context.values.get("canonical_facts")
        if not isinstance(facts, dict) or not isinstance(facts.get("events"), list):
            return None
        those = {"fact_key": "events", "candidates": facts["events"]}
    text = " ".join(strip_correction_prefix(intent.utterance).casefold().split()).strip(".!?")
    text = re.sub(r"^(?:and|but)\s+", "", text)
    temporal_terms = ("today", "tomorrow", "this weekend", "next week", *_WEEKDAYS)
    temporal = next(
        (term for term in temporal_terms if text in {f"what about {term}", term}),
        None,
    )
    if temporal is None:
        return None
    follow_up = intent.model_copy(update={"utterance": f"What events are happening {temporal}?"})
    return HouseholdReadFastPath(snapshot).resolve(follow_up)


def resolve_contextual_event_next_read(
    intent: IntentFrame, context: Context, snapshot: dict[str, object]
) -> Result | None:
    """Answer a bounded next-event question from an authorized event list."""

    if context.sources != ("authorized_canonical_result",):
        return None
    text = " ".join(strip_correction_prefix(intent.utterance).casefold().split()).strip(".!?")
    text = re.sub(r"^(?:and|but)\s+", "", text)
    if text not in {
        "when is the next one",
        "what is the next one",
        "what about the next one",
        "which is next",
        "what is happening next",
        "what's happening next",
    }:
        return None
    referents = context.values.get("referents")
    those = referents.get("those") if isinstance(referents, dict) else None
    if isinstance(those, dict) and those.get("fact_key") == "events":
        candidates = those.get("candidates")
        facts = context.values.get("canonical_facts")
        compact_events = facts.get("events") if isinstance(facts, dict) else None
        if isinstance(compact_events, list):
            # Ordinal referents preserve display order, but "next" needs the
            # separately compacted chronological candidates when available.
            candidates = compact_events
    else:
        facts = context.values.get("canonical_facts")
        candidates = facts.get("events") if isinstance(facts, dict) else None
    if not isinstance(candidates, list):
        return None
    now = datetime.now(timezone.utc)
    upcoming: list[tuple[datetime, dict[str, Any]]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("starts_at"), str):
            continue
        try:
            starts_at = datetime.fromisoformat(candidate["starts_at"])
        except ValueError:
            continue
        if starts_at.tzinfo is None:
            starts_at = starts_at.replace(tzinfo=timezone.utc)
        starts_at = starts_at.astimezone(timezone.utc)
        if starts_at >= now:
            upcoming.append((starts_at, candidate))
    if not upcoming:
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message="I cannot find an upcoming event in the referenced calendar.",
            correlation_id=intent.correlation_id,
        )
    starts_at, event = min(upcoming, key=lambda item: item[0])
    title = event.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message=f"Event: {title}; starts {_display_due_at(starts_at.isoformat())}",
        evidence={
            "collection": "events",
            "authorized_next_referent": event,
            "event": event,
            "canonical_events": candidates,
            "snapshot_space_id": snapshot.get("space_id"),
        },
        correlation_id=intent.correlation_id,
    )


def resolve_contextual_task_focus_read(
    intent: IntentFrame, context: Context, store: PostgresTaskStore
) -> Result | None:
    """Read back one task focus from an authorized scalar result."""

    if context.sources != ("authorized_canonical_result",) or is_mutation_request(intent.utterance):
        return None
    text = " ".join(strip_correction_prefix(intent.utterance).casefold().split()).strip(".!?")
    text = re.sub(r"^(?:and|but)\s+", "", text)
    if not (
        (
            text.startswith(("show ", "can you show ", "tell me about ", "tell me more about "))
            and "that" in text
        )
        or text
        in {
            "what about that one",
            "is that still open",
            "is it done",
            "is that done",
            "is it complete",
            "is that complete",
            "is that one still open",
            "is it still open",
            "when is that due",
            "when is it due",
            "what is its deadline",
            "what is the deadline",
        }
    ):
        return None
    facts = context.values.get("canonical_facts")
    focus = facts.get("task") if isinstance(facts, dict) else None
    if not isinstance(focus, dict) or not isinstance(focus.get("title"), str):
        return None
    current = tuple(store.list(intent.principal))
    task_id = focus.get("task_id")
    matches = [
        task for task in current if isinstance(task_id, str) and str(task.task_id) == task_id
    ]
    if not matches:
        title = focus["title"].casefold()
        matches = [task for task in current if task.title.casefold() == title]
    if len(matches) != 1:
        message = (
            "I can no longer find that canonical task."
            if not matches
            else "I found multiple canonical tasks with that title; please name the task."
        )
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=message,
            correlation_id=intent.correlation_id,
        )
    task = matches[0]
    detail = f"Task: {task.title} ({task.status.value})"
    if task.due_at is not None:
        detail += f"; due {_display_due_at(task.due_at.isoformat())}"
    elif "deadline" in text or "due" in text:
        detail += "; no recorded deadline"
    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message=detail,
        evidence={
            "collection": "canonical_tasks",
            "authorized_task_focus": {
                "task_id": str(task.task_id),
                "title": task.title,
                "status": task.status.value,
            },
            # Preserve the conventional scalar focus key so a further
            # authorized follow-up can continue the same referent.
            "task": {
                "task_id": str(task.task_id),
                "title": task.title,
                "status": task.status.value,
            },
        },
        correlation_id=intent.correlation_id,
    )


def resolve_contextual_obligation_focus_read(
    intent: IntentFrame, context: Context, snapshot: dict[str, object]
) -> Result | None:
    """Read settled state from one authorized canonical obligation focus."""

    if context.sources != ("authorized_canonical_result",) or is_mutation_request(intent.utterance):
        return None
    text = " ".join(strip_correction_prefix(intent.utterance).casefold().split()).strip(".!?")
    text = re.sub(r"^(?:and|but)\s+", "", text)
    if text not in {
        "is it settled",
        "is that settled",
        "has it been settled",
        "was it settled",
        "who is responsible",
        "who is responsible for it",
        "who handles it",
        "who is it assigned to",
        "who is assigned to it",
        "how much is it",
        "how much is owed",
        "what is the amount",
        "what is its amount",
        "what is owed for it",
    }:
        return None
    facts = context.values.get("canonical_facts")
    focus = facts.get("obligation") if isinstance(facts, dict) else None
    if not isinstance(focus, dict) or not isinstance(focus.get("title"), str):
        return None
    obligations = snapshot.get("obligations")
    current = tuple(obligations) if isinstance(obligations, (list, tuple)) else ()
    obligation_id = focus.get("obligation_id")
    matches = [
        obligation
        for obligation in current
        if isinstance(obligation_id, str) and str(obligation.obligation_id) == obligation_id
    ]
    if not matches:
        title = focus["title"].casefold()
        matches = [
            obligation
            for obligation in current
            if obligation.title.casefold() == title
            and obligation.responsible_id == focus.get("responsible_id")
        ]
    if len(matches) != 1:
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message="I can no longer find one unambiguous canonical obligation for that reference.",
            correlation_id=intent.correlation_id,
        )
    obligation = matches[0]
    if text in {
        "who is responsible",
        "who is responsible for it",
        "who handles it",
        "who is it assigned to",
        "who is assigned to it",
    }:
        message = f"Obligation: {obligation.title} is assigned to {obligation.responsible_id}"
    elif text in {
        "how much is it",
        "how much is owed",
        "what is the amount",
        "what is its amount",
        "what is owed for it",
    }:
        message = f"Obligation: {obligation.title} is ${obligation.amount / 100:.2f}"
    else:
        state = "settled" if obligation.settled else "unsettled"
        message = f"Obligation: {obligation.title} is {state} ({obligation.responsible_id})"
    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message=message,
        evidence={
            "collection": "canonical_obligations",
            "authorized_obligation_focus": {
                "obligation_id": str(obligation.obligation_id),
                "title": obligation.title,
                "responsible_id": obligation.responsible_id,
                "settled": obligation.settled,
                "amount": obligation.amount,
            },
            "obligation": {
                "obligation_id": str(obligation.obligation_id),
                "title": obligation.title,
                "responsible_id": obligation.responsible_id,
                "settled": obligation.settled,
                "amount": obligation.amount,
            },
        },
        correlation_id=intent.correlation_id,
    )


def resolve_direct_obligation_ordinal_read(
    intent: IntentFrame, snapshot: dict[str, object]
) -> Result | None:
    """Resolve an explicit obligation ordinal against current canonical state."""

    text = " ".join(intent.utterance.casefold().split()).strip(".!?")
    if is_mutation_request(text) or not re.search(r"\bobligations?\b", text):
        return None
    ordinal = re.search(r"\b(?:the\s+)?(first|second|third|fourth|last)\b", text)
    if ordinal is None or not re.search(r"\b(?:show|which|what|tell me about)\b", text):
        return None
    obligations = snapshot.get("obligations")
    outstanding = (
        [
            obligation
            for obligation in obligations
            if hasattr(obligation, "title") and not obligation.settled
        ]
        if isinstance(obligations, (list, tuple))
        else []
    )
    index = {"first": 0, "second": 1, "third": 2, "fourth": 3, "last": -1}[ordinal.group(1)]
    if not outstanding or index >= len(outstanding) or index < -len(outstanding):
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=(
                "I found fewer outstanding obligations than that. "
                "Please choose an available ordinal."
            ),
            correlation_id=intent.correlation_id,
        )
    obligation = outstanding[index]
    referent = {
        "obligation_id": str(obligation.obligation_id),
        "title": obligation.title,
        "responsible_id": obligation.responsible_id,
        "settled": bool(obligation.settled),
        "amount": obligation.amount,
    }
    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message=f"Obligation: {obligation.title} (unsettled) ({obligation.responsible_id})",
        evidence={
            "collection": "canonical_obligations",
            "authorized_ordinal_referent": referent,
            "obligation": referent,
            "canonical_obligations": [
                {
                    "obligation_id": str(item.obligation_id),
                    "title": item.title,
                    "responsible_id": item.responsible_id,
                    "settled": bool(item.settled),
                    "amount": item.amount,
                }
                for item in outstanding
            ],
        },
        correlation_id=intent.correlation_id,
    )


def resolve_personal_obligation_read(
    intent: IntentFrame, snapshot: dict[str, object]
) -> Result | None:
    """Read the authenticated owner's outstanding canonical obligations."""

    text = " ".join(intent.utterance.casefold().split()).strip(".!?")
    if is_mutation_request(text) or text not in {"what do i owe", "how much do i owe"}:
        return None
    obligations = snapshot.get("obligations")
    owned = (
        [
            obligation
            for obligation in obligations
            if hasattr(obligation, "responsible_id")
            and obligation.responsible_id == intent.principal.id
            and not obligation.settled
        ]
        if isinstance(obligations, (list, tuple))
        else []
    )
    if not owned:
        message = "You have no outstanding obligations assigned to you."
    elif len(owned) == 1:
        obligation = owned[0]
        message = f"You owe ${obligation.amount / 100:.2f} for {obligation.title}."
    else:
        total = sum(obligation.amount for obligation in owned)
        details = "; ".join(
            f"{obligation.title} (${obligation.amount / 100:.2f})" for obligation in owned
        )
        message = f"You owe ${total / 100:.2f} across: {details}."
    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message=message,
        evidence={
            "collection": "canonical_obligations",
            "authorized_owner": intent.principal.id,
            "authorized_owned_obligations": [
                {
                    "obligation_id": str(obligation.obligation_id),
                    "title": obligation.title,
                    "amount": obligation.amount,
                    "responsible_id": obligation.responsible_id,
                    "settled": bool(obligation.settled),
                }
                for obligation in owned
            ],
        },
        correlation_id=intent.correlation_id,
    )


def resolve_named_obligation_read(
    intent: IntentFrame, snapshot: dict[str, object]
) -> Result | None:
    """Resolve a read about one uniquely named canonical obligation."""

    text = " ".join(intent.utterance.casefold().split()).strip(".!?")
    if is_mutation_request(text):
        return None
    obligations = snapshot.get("obligations")
    current = tuple(obligations) if isinstance(obligations, (list, tuple)) else ()
    matches = [
        obligation
        for obligation in current
        if hasattr(obligation, "title")
        and re.search(rf"\b{re.escape(obligation.title.casefold())}\b", text)
    ]
    if len(matches) != 1:
        return None
    if not re.search(
        r"\b(?:amount|owe|owes|owed|settled|paid|responsible|assigned|handles)\b", text
    ):
        return None
    obligation = matches[0]
    if re.search(r"\b(?:amount|owe|owed)\b", text):
        message = f"Obligation: {obligation.title} is ${obligation.amount / 100:.2f}"
    elif re.search(r"\b(?:responsible|assigned|handles|owes)\b", text):
        message = f"Obligation: {obligation.title} is assigned to {obligation.responsible_id}"
    else:
        state = "settled" if obligation.settled else "unsettled"
        message = f"Obligation: {obligation.title} is {state} ({obligation.responsible_id})"
    referent = {
        "obligation_id": str(obligation.obligation_id),
        "title": obligation.title,
        "responsible_id": obligation.responsible_id,
        "settled": bool(obligation.settled),
        "amount": obligation.amount,
    }
    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message=message,
        evidence={
            "collection": "canonical_obligations",
            "authorized_obligation_focus": referent,
            "obligation": referent,
        },
        correlation_id=intent.correlation_id,
    )


def resolve_contextual_chore_focus_read(
    intent: IntentFrame, context: Context, snapshot: dict[str, object]
) -> Result | None:
    """Read assignee from one authorized canonical chore focus."""

    if context.sources != ("authorized_canonical_result",) or is_mutation_request(intent.utterance):
        return None
    text = " ".join(strip_correction_prefix(intent.utterance).casefold().split()).strip(".!?")
    text = re.sub(r"^(?:and|but)\s+", "", text)
    if text not in {
        "who is assigned",
        "who is assigned to it",
        "who handles it",
        "what is its assignee",
        "who is its assignee",
        "who is assigned to this",
        "is it done",
        "is that done",
        "is it complete",
        "is that complete",
    }:
        return None
    facts = context.values.get("canonical_facts")
    focus = facts.get("chore") if isinstance(facts, dict) else None
    if not isinstance(focus, dict) or not isinstance(focus.get("title"), str):
        return None
    chores = snapshot.get("chores")
    current = tuple(chores) if isinstance(chores, (list, tuple)) else ()
    chore_id = focus.get("chore_id")
    matches = [
        chore for chore in current if isinstance(chore_id, str) and str(chore.chore_id) == chore_id
    ]
    if not matches:
        title = focus["title"].casefold()
        matches = [
            chore
            for chore in current
            if chore.title.casefold() == title and chore.assignee_id == focus.get("assignee_id")
        ]
    if len(matches) != 1:
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message="I can no longer find one unambiguous canonical chore for that reference.",
            correlation_id=intent.correlation_id,
        )
    chore = matches[0]
    if text in {"is it done", "is that done", "is it complete", "is that complete"}:
        status = "complete" if chore.completed else "open"
        message = f"Chore: {chore.title} is {status}"
    else:
        message = f"Chore: {chore.title} is assigned to {chore.assignee_id}"
    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message=message,
        evidence={
            "collection": "canonical_chores",
            "authorized_chore_focus": {
                "chore_id": str(chore.chore_id),
                "title": chore.title,
                "assignee_id": chore.assignee_id,
                "completed": chore.completed,
            },
            "chore": {
                "chore_id": str(chore.chore_id),
                "title": chore.title,
                "assignee_id": chore.assignee_id,
                "completed": chore.completed,
            },
        },
        correlation_id=intent.correlation_id,
    )


def resolve_contextual_event_priority_read(
    intent: IntentFrame, context: Context, snapshot: dict[str, object]
) -> Result | None:
    """Select the earliest/latest event from an authorized calendar result."""

    if context.sources != ("authorized_canonical_result",) or is_mutation_request(intent.utterance):
        return None
    text = " ".join(strip_correction_prefix(intent.utterance).casefold().split()).strip(".!?")
    text = re.sub(r"^(?:and|but)\s+", "", text)
    latest = text in {"which event is latest", "what event is latest", "which one is latest"}
    earliest = text in {
        "which event is earliest",
        "what event is earliest",
        "which one is earliest",
    }
    if not latest and not earliest:
        return None
    facts = context.values.get("canonical_facts")
    candidates = facts.get("events") if isinstance(facts, dict) else None
    referents = context.values.get("referents")
    those = referents.get("those") if isinstance(referents, dict) else None
    if not isinstance(candidates, list) and isinstance(those, dict):
        candidates = those.get("candidates") if those.get("fact_key") == "events" else None
    if not isinstance(candidates, list):
        return None
    dated: list[tuple[datetime, dict[str, Any]]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("starts_at"), str):
            continue
        try:
            starts_at = datetime.fromisoformat(candidate["starts_at"])
        except ValueError:
            continue
        if starts_at.tzinfo is None:
            starts_at = starts_at.replace(tzinfo=timezone.utc)
        dated.append((starts_at.astimezone(timezone.utc), candidate))
    if not dated:
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message="I cannot order the referenced events because they have no valid times.",
            correlation_id=intent.correlation_id,
        )
    starts_at, event = (max if latest else min)(dated, key=lambda item: item[0])
    title = event.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    label = "latest" if latest else "earliest"
    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message=(
            f"Based on the {label} recorded time: {title}; "
            f"starts {_display_due_at(starts_at.isoformat())}"
        ),
        evidence={
            "collection": "events",
            "priority_basis": f"authorized_prior_result_{label}_starts_at",
            "authorized_event_priority": event,
            "event": event,
            "canonical_events": candidates,
            "snapshot_space_id": snapshot.get("space_id"),
        },
        correlation_id=intent.correlation_id,
    )


def resolve_contextual_event_relative_read(
    intent: IntentFrame, context: Context, snapshot: dict[str, object]
) -> Result | None:
    """Resolve an earlier-event follow-up from an authorized scalar event focus."""

    if context.sources != ("authorized_canonical_result",) or is_mutation_request(intent.utterance):
        return None
    text = " ".join(strip_correction_prefix(intent.utterance).casefold().split()).strip(".!?")
    text = re.sub(r"^(?:and|but)\s+", "", text)
    earlier_query = text in {
        "earlier one",
        "what about the earlier one",
        "what about an earlier one",
        "what about the event before that",
        "what about the one before that",
        "what about an event before that",
        "show the event before that",
        "show me the event before that",
    }
    after_query = text in {
        "what about the event after that",
        "what about the one after that",
        "what about an event after that",
        "show the event after that",
        "show me the event after that",
    }
    if not earlier_query and not after_query:
        return None
    facts = context.values.get("canonical_facts")
    focus = facts.get("event") if isinstance(facts, dict) else None
    if not isinstance(focus, dict) or not isinstance(focus.get("event_id"), str):
        return None
    events = snapshot.get("events")
    current = tuple(events) if isinstance(events, (list, tuple)) else ()
    anchor_matches = [event for event in current if str(event.event_id) == focus["event_id"]]
    if len(anchor_matches) != 1:
        return None
    anchor = anchor_matches[0]
    anchor_time = anchor.starts_at
    if anchor_time.tzinfo is None:
        anchor_time = anchor_time.replace(tzinfo=timezone.utc)
    related = []
    for event in current:
        event_time = event.starts_at
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        if (earlier_query and event_time < anchor_time) or (
            after_query and event_time > anchor_time
        ):
            related.append((event_time, event))
    if not related:
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=(
                "I cannot find an earlier canonical event for that reference."
                if earlier_query
                else "I cannot find a later canonical event for that reference."
            ),
            correlation_id=intent.correlation_id,
        )
    related.sort(key=lambda item: item[0], reverse=earlier_query)
    if len(related) > 1 and related[0][0] == related[1][0]:
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message="I found multiple equally earlier canonical events; please name the event.",
            correlation_id=intent.correlation_id,
        )
    _event_time, selected = related[0]
    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message=(
            f"Event: {selected.title}; starts {_display_due_at(selected.starts_at.isoformat())}"
        ),
        evidence={
            "collection": "events",
            "authorized_relative_referent": {
                "relation": "earlier" if earlier_query else "later",
                "event_id": str(selected.event_id),
                "title": selected.title,
                "starts_at": selected.starts_at.isoformat(),
            },
            "event": {
                "event_id": str(selected.event_id),
                "title": selected.title,
                "starts_at": selected.starts_at.isoformat(),
            },
        },
        correlation_id=intent.correlation_id,
    )


def resolve_contextual_event_focus_read(
    intent: IntentFrame, context: Context, snapshot: dict[str, object]
) -> Result | None:
    """Read the time of one event from an authorized scalar event focus."""

    if context.sources != ("authorized_canonical_result",) or is_mutation_request(intent.utterance):
        return None
    text = " ".join(strip_correction_prefix(intent.utterance).casefold().split()).strip(".!?")
    text = re.sub(r"^(?:and|but)\s+", "", text)
    if text not in {
        "is that still open",
        "when does it start",
        "when does that start",
        "when is that",
        "when is it",
        "what time is it",
        "what day is that",
        "what day is it",
        "what date is that",
        "what is its date",
        "what is the date",
        "what date is it",
        "is it still scheduled",
        "is that still scheduled",
    }:
        return None
    facts = context.values.get("canonical_facts")
    focus = facts.get("event") if isinstance(facts, dict) else None
    if not isinstance(focus, dict) or not isinstance(focus.get("title"), str):
        return None
    events = snapshot.get("events")
    current = tuple(events) if isinstance(events, (list, tuple)) else ()
    event_id = focus.get("event_id")
    matches = [
        event for event in current if isinstance(event_id, str) and str(event.event_id) == event_id
    ]
    if not matches:
        title = focus["title"].casefold()
        matches = [event for event in current if event.title.casefold() == title]
    if len(matches) != 1:
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message="I can no longer find one unambiguous canonical event for that reference.",
            correlation_id=intent.correlation_id,
        )
    event = matches[0]
    starts_at = event.starts_at
    if starts_at.tzinfo is None:
        starts_at = starts_at.replace(tzinfo=timezone.utc)
    status_query = text in {
        "is that still open",
        "is it still scheduled",
        "is that still scheduled",
    }
    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message=(
            f"Event: {event.title} is scheduled; starts {_display_due_at(starts_at.isoformat())}"
            if status_query
            else f"Event: {event.title}; starts {_display_due_at(starts_at.isoformat())}"
        ),
        evidence={
            "collection": "events",
            "authorized_event_focus": {
                "event_id": str(event.event_id),
                "title": event.title,
                "starts_at": event.starts_at.isoformat(),
            },
            "event": {
                "event_id": str(event.event_id),
                "title": event.title,
                "starts_at": event.starts_at.isoformat(),
            },
        },
        correlation_id=intent.correlation_id,
    )


def resolve_ambiguous_event_focus_read(intent: IntentFrame, context: Context) -> Result | None:
    """Keep an event-list pronoun from being mistaken for a new event write."""

    if context.sources != ("authorized_canonical_result",):
        return None
    text = " ".join(strip_correction_prefix(intent.utterance).casefold().split()).strip(".!?")
    text = re.sub(r"^(?:and|but)\s+", "", text)
    if text not in {
        "when is it",
        "when is that",
        "what time is it",
        "what day is it",
        "what date is it",
        "is it still scheduled",
        "is that still scheduled",
    }:
        return None
    facts = context.values.get("canonical_facts")
    if isinstance(facts, dict) and isinstance(facts.get("event"), dict):
        return None
    referents = context.values.get("referents")
    those = referents.get("those") if isinstance(referents, dict) else None
    if not isinstance(those, dict) or those.get("fact_key") != "events":
        return None
    candidates = those.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.BLOCKED,
        message="Which event did you mean? Please name the event or choose an ordinal.",
        correlation_id=intent.correlation_id,
    )


def resolve_contextual_recent_action_read(intent: IntentFrame, context: Context) -> Result | None:
    """Read the immediately preceding canonical action result.

    This is a bounded referent, not a new semantic router: it only exposes a
    title already returned by the authorized prior Result and never performs
    or authorizes another action.
    """

    if context.sources != ("authorized_canonical_result",):
        return None
    text = " ".join(intent.utterance.casefold().split()).strip(".!?")
    if "just" not in text or not text.startswith(("what did", "what was", "what have")):
        return None
    facts = context.values.get("canonical_facts")
    if not isinstance(facts, dict):
        return None
    title = facts.get("title")
    collection = facts.get("collection")
    if not isinstance(title, str) or not title.strip() or not isinstance(collection, str):
        return None
    labels = {
        "tasks": "task",
        "chores": "chore",
        "events": "event",
        "groceries": "grocery item",
    }
    label = labels.get(collection)
    if label is None:
        return None
    status = facts.get("status")
    if status == "completed":
        message = f"{label.title()}: {title} (completed)"
    else:
        message = f"{label.title()}: {title}"
    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message=message,
        evidence={
            "collection": collection,
            "title": title,
            "status": status,
            "referent": "prior_result",
        },
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
    grounded_plan_actions: list[ActionSpec] = []
    for action, card in zip(plan_actions, plan_cards, strict=True):
        grounded = ground_reference_action_runtime(
            intent,
            card.model_copy(update={"action": action}),
            connection,
            context,
        )
        if isinstance(grounded, Result):
            return grounded
        grounded_plan_actions.append(grounded.action)
    plan_actions = tuple(grounded_plan_actions)
    plan_cards = tuple(
        card.model_copy(update={"action": action})
        for card, action in zip(plan_cards, plan_actions, strict=True)
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
    return kernel.run_proposed_plan(
        intent,
        proposal,
        plan_cards,
        context=context,
        objective_spec=ObjectiveSpecProposal(
            requirements=tuple(
                ObjectiveRequirementProposal(action_ref=step.action_ref, arguments=step.arguments)
                for step in proposal.steps
            )
        ),
    )


def rewrite_reference_decision(
    intent: IntentFrame,
    decision: Decision,
    cards: tuple[ActionCard, ...],
    context: Context | None = None,
) -> Decision | Result | None:
    """Correct a reference-Pack event proposal when the user named a task destination."""

    if decision.kind is DecisionKind.ACTION and MultiActionFastPath.matches(intent.utterance):
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=(
                "This request contains multiple actions, but I could not form a complete "
                "verified plan. No action was executed; please separate the requests."
            ),
            correlation_id=intent.correlation_id,
        )

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
        or MultiActionFastPath.matches(utterance)
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
    if card.action.action_id == "workspace.artifact.read":
        read_arguments = dict(card.action.arguments)
        provenance: dict[str, ArgumentProvenance] = {}
        for key, value in read_arguments.items():
            if not isinstance(value, str) or not _utterance_spans(intent.utterance, value):
                return Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.BLOCKED,
                    message="Name the Workspace ID and relative file path explicitly.",
                    correlation_id=intent.correlation_id,
                )
            provenance[key] = ArgumentProvenance(
                kind=ArgumentProvenanceKind.EXPLICIT_UTTERANCE,
                source_spans=_utterance_spans(intent.utterance, value),
            )
        return card.model_copy(
            update={
                "action": card.action.model_copy(
                    update={"arguments": read_arguments, "argument_provenance": provenance}
                )
            }
        )
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
        if referent is None and context is not None:
            referent = resolve_unique_prior_task_reference(intent.utterance, context)
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

    if card.action.action_id == "tasks.events.create" and not _event_time_is_explicit(
        intent.utterance
    ):
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message="What date and time should I use for that event? I won't infer one.",
            correlation_id=intent.correlation_id,
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

    return _ground_argument_provenance(intent, card, context)


def ground_reference_action_runtime(
    intent: IntentFrame, card: ActionCard, connection: Any, context: Context | None = None
) -> ActionCard | Result:
    """Load reference Pack state before applying its canonical grounding rules."""

    principal = intent.principal
    if card.action.action_id in {
        "documents.export_to_workspace",
        "documents.summarize_to_workspace",
    } or (
        card.action.action_id == "communications.messages.send"
        and card.action.arguments.get("body_source") == "canonical.document"
    ):
        documents = configured_document_provider().list_documents()
        matches = [
            document
            for document in documents
            if document.title.casefold() in intent.utterance.casefold()
        ]
        if len(matches) != 1:
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message="Name one authorized document to export; no document was changed.",
                correlation_id=intent.correlation_id,
            )
        base_values = dict(context.values) if context is not None else {}
        base_values["referents"] = {
            "those": {
                "fact_key": "authorized_documents",
                "candidates": [
                    {
                        "document_id": matches[0].document_id,
                        "title": matches[0].title,
                    }
                ],
            }
        }
        if context is not None:
            context.values["referents"] = base_values["referents"]
        context = Context(values=base_values, sources=("authorized_canonical_result",))
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
    grounded = ground_reference_action(
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
    if card.action.action_id in {
        "documents.export_to_workspace",
        "documents.summarize_to_workspace",
    } and not isinstance(grounded, Result):
        document_id = (
            next(
                (
                    candidate["document_id"]
                    for candidate in context.values.get("referents", {})
                    .get("those", {})
                    .get("candidates", [])
                    if isinstance(candidate, dict) and isinstance(candidate.get("document_id"), str)
                ),
                None,
            )
            if context is not None
            else None
        )
        if document_id is not None:
            provenance = dict(grounded.action.argument_provenance)
            provenance["document_id"] = ArgumentProvenance(
                kind=ArgumentProvenanceKind.AUTHORIZED_CANONICAL_REFERENT,
                canonical_ref=document_id,
            )
            grounded = grounded.model_copy(
                update={
                    "action": grounded.action.model_copy(
                        update={
                            "arguments": {
                                **grounded.action.arguments,
                                "document_id": document_id,
                            },
                            "argument_provenance": provenance,
                        }
                    )
                }
            )
    return grounded
