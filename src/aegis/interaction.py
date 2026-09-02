"""Reusable AEGIS application interaction boundary for all clients."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, cast
from uuid import UUID, uuid4

from .audit import PostgresAuditLog
from .contracts import (
    ActionCard,
    Context,
    Decision,
    DecisionKind,
    IntentFrame,
    ModelRequest,
    Objective,
    ObjectiveState,
    Principal,
    Result,
    WorkingSet,
)
from .decoding import InvalidDecision, StrictDecisionDecoder
from .embeddings import OllamaEmbeddingProvider, PostgresMemoryVectorIndex
from .finance import FinanceLedger, FinanceReadFastPath, PostgresFinanceSnapshotStore
from .gateway_rpc import OpenClawWebSocketChannel
from .household import (
    Chore,
    ChoreCompletionFastPath,
    HouseholdObligation,
    HouseholdReadFastPath,
    PostgresChoreExecutor,
    PostgresChoreVerifier,
    PostgresEventExecutor,
    PostgresEventVerifier,
    PostgresHouseholdStore,
)
from .identity import PostgresSpacePolicy, Role
from .kernel import Kernel, _FixedActionModel
from .ollama import OllamaHttpTransport, OllamaProvider
from .openclaw import OpenClawExecutor
from .pack_lifecycle import PackManager, PackStatus, PostgresPackStore
from .pack_runtime import PackRuntimeRegistry
from .personal import PersonalMemoryFastPath, PostgresPersonalStateStore
from .planning import (
    ContextualMutationGuard,
    CrossDomainPlanningFastPath,
    DomainClarificationFastPath,
    MultiActionFastPath,
    PersonalChoreComposer,
    PersonalMemoryChoreComposer,
    PersonalMemoryTaskComposer,
    PersonalTaskComposer,
)
from .projections import SharedObligation
from .reference_packs import (
    OpenClawGroceryExecutor,
    OpenClawGroceryVerifier,
    PostgresGroceryListExecutor,
    PostgresGroceryListVerifier,
    reference_bundles,
)
from .store import PostgresObjectiveStore
from .tasks import (
    PostgresTaskExecutor,
    PostgresTaskListExecutor,
    PostgresTaskListVerifier,
    PostgresTaskStore,
    PostgresTaskVerifier,
    TaskCompletionFastPath,
    TaskIntentClarificationFastPath,
    TaskPriorityFastPath,
    TaskReadFastPath,
    ground_task_due_at,
    requested_task_due_at,
)
from .utterance import (
    has_multiple_question_clauses,
    is_question_request,
    is_task_destination_request,
)

_MAX_CONTEXT_TURN_CHARS = 500
_MAX_CONTEXT_CANDIDATES = 10


class InteractionInputError(ValueError):
    """A safe, actionable request-shape error from a client-facing selector."""


class InteractionDependencies:
    """Infrastructure callbacks supplied by the composition root."""

    def __init__(
        self,
        connect: Callable[[str], Any],
        required: Callable[[str], str],
        apply_migrations: Callable[[Any], None],
        ensure_local_identity: Callable[[Any, Principal], None],
        select_action: Callable[[str, PackManager], tuple[str, ActionCard]],
        openclaw_channel: Callable[[], OpenClawWebSocketChannel],
        local_identity: Callable[[], bool],
        model_provider: Callable[[], Any] | None = None,
        capability_retriever: Callable[[str, PackManager], tuple[ActionCard, ...]] | None = None,
        runtime_registry: PackRuntimeRegistry | None = None,
    ) -> None:
        self.connect = connect
        self.required = required
        self.apply_migrations = apply_migrations
        self.ensure_local_identity = ensure_local_identity
        self.select_action = select_action
        self.openclaw_channel = openclaw_channel
        self.local_identity = local_identity
        self.model_provider = model_provider
        self.capability_retriever = capability_retriever
        self.runtime_registry = runtime_registry


def _compact_context_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Keep only bounded canonical facts for a follow-up model working set."""

    compact: dict[str, Any] = {}
    for key in ("canonical_items", "canonical_tasks", "title", "item"):
        value = evidence.get(key)
        if isinstance(value, (list, tuple)):
            if key == "canonical_items":
                compact[key] = list(dict.fromkeys(str(item) for item in value))[:20]
            else:
                compact[key] = list(value[:20])
        elif isinstance(value, str):
            compact[key] = value
    planning = evidence.get("planning")
    if isinstance(planning, dict):
        compact_planning: dict[str, Any] = {}
        for key in ("open_tasks", "open_obligations", "memories", "affordability"):
            value = planning.get(key)
            if isinstance(value, (list, tuple)):
                compact_planning[key] = list(value[:10])
            elif isinstance(value, dict):
                compact_planning[key] = value
        if compact_planning:
            compact["planning"] = compact_planning
    return compact


def _authorized_context_evidence(context: Context) -> dict[str, Any]:
    """Carry only the authorized working-set facts needed by a later turn."""

    raw = context.values.get("canonical_facts")
    if not isinstance(raw, dict):
        return {}
    return _compact_context_evidence(raw)


def _with_continuation_context(result: Result, context: Context) -> Result:
    """Trace a follow-up to authorized context without changing its identity."""

    prior_objective_id = context.values.get("prior_objective_id")
    if not isinstance(prior_objective_id, str):
        return result
    evidence = dict(result.evidence)
    evidence["continuation_of_objective_id"] = prior_objective_id
    evidence["continuation_context"] = "authorized_prior_result"
    return result.model_copy(update={"evidence": evidence})


def _context_from_prior_result(
    objective_store: Any, correlation_id: UUID | None, principal: Principal
) -> Context:
    """Resolve follow-up context only through the authorized canonical store."""

    if correlation_id is None:
        return Context()
    objective = objective_store.get_objective_by_correlation(correlation_id, principal)
    getter = getattr(objective_store, "get_result_for_correlation", None)
    if objective is None or not callable(getter):
        return Context()
    result = getter(correlation_id, principal)
    if result is None:
        return Context()
    evidence = _compact_context_evidence(result.evidence)
    if not evidence:
        return Context()
    referents: dict[str, Any] = {}
    for fact_key in ("canonical_items", "canonical_tasks", "canonical_obligations"):
        candidates = evidence.get(fact_key)
        if isinstance(candidates, list) and candidates:
            referents["those"] = {
                "source": "canonical_facts",
                "fact_key": fact_key,
                "candidates": candidates[:_MAX_CONTEXT_CANDIDATES],
            }
            break
    return Context(
        values={
            "prior_correlation_id": str(correlation_id),
            "prior_objective_id": str(objective.id),
            "prior_state": result.state.value,
            "recent_turns": [
                {
                    "role": "user",
                    "utterance": objective.intent.utterance[:_MAX_CONTEXT_TURN_CHARS],
                    "correlation_id": str(correlation_id),
                }
            ],
            "referents": referents,
            "canonical_facts": evidence,
        },
        sources=("authorized_canonical_result",),
    )


def _fallback_working_context(
    context: Context,
    task_store: PostgresTaskStore,
    household_store: PostgresHouseholdStore,
    principal: Principal,
    utterance: str,
) -> Context:
    """Add a small authorized candidate set for bounded intent resolution."""

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
    tasks = list(task_store.list(principal))
    priority_request = ("which" in utterance.casefold() and "first" in utterance.casefold()) or any(
        term in utterance.casefold() for term in ("prioritize", "priority", "focus")
    )
    if priority_request:
        tasks = [task for task in tasks if task.status.value == "open"]
        tasks.sort(
            key=lambda task: (
                task.due_at is None,
                task.due_at.isoformat() if task.due_at is not None else "",
            )
        )
    query_terms = {
        term
        for term in utterance.casefold().split()
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
    ]
    if not selected:
        selected = tasks
    facts["canonical_tasks"] = [
        {
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


class _RuntimePolicy:
    def allows(self, request: Any) -> bool:
        return bool(request.action.action_id == "kitchen.groceries.add")


class _NoApproval:
    def required(self, request: Any) -> bool:
        return False

    def approved(self, request: Any) -> bool:
        return True


class _ActionExecutorDispatch:
    """Dispatch plan steps to their existing Pack executor adapters."""

    def __init__(self, delegates: dict[str, Any]) -> None:
        self.delegates = delegates

    def execute(self, request: Any) -> Any:
        try:
            delegate = self.delegates[request.action.action_id]
        except KeyError as exc:
            raise ValueError("plan contains an unsupported action") from exc
        observation = delegate.execute(request)
        return observation.model_copy(update={"action_id": request.action.action_id})


class _ActionVerifierDispatch:
    """Dispatch plan verification to the matching canonical verifier."""

    def __init__(self, delegates: dict[str, Any]) -> None:
        self.delegates = delegates

    def verify(self, observation: Any, contract: Any) -> Any:
        action_id = observation.action_id
        if not isinstance(action_id, str) or action_id not in self.delegates:
            raise ValueError("plan verifier is unavailable")
        return self.delegates[action_id].verify(observation, contract)


class InteractionBoundary:
    """Canonical application interaction service used by every client."""

    def __init__(self, dependencies: InteractionDependencies) -> None:
        self.dependencies = dependencies

    def _fallback_cards(
        self, manager: PackManager, utterance: str, context: Context | None = None
    ) -> tuple[ActionCard, ...]:
        """Offer a bounded capability vocabulary; metadata remains Core-owned."""

        text = utterance.casefold()
        facts = (context.values if context is not None else {}).get("canonical_facts", {})
        if is_task_destination_request(utterance):
            # An explicit task-list destination is a bounded capability hint;
            # do not offer calendar actions for a request the user directed to
            # their to-do collection.
            return tuple(manager.retrieve("tasks"))[:10]
        if isinstance(facts, dict):
            if (
                isinstance(facts.get("canonical_items"), list)
                and context is not None
                and "authorized_canonical_context" in context.sources
            ):
                # A contextual grocery question is answered from the
                # authorized projection; do not offer a mutation card that
                # could turn an unresolved reference into a write.
                return ()
            if (
                isinstance(facts.get("canonical_tasks"), list)
                and context is not None
                and "authorized_canonical_context" in context.sources
            ):
                return tuple(manager.retrieve("tasks"))[:10]
        if self.dependencies.capability_retriever is not None:
            try:
                semantic_cards = self.dependencies.capability_retriever(utterance, manager)
            except Exception:
                # Retrieval is an optimization; a provider outage must not
                # bypass the bounded model/decoder or change authority.
                semantic_cards = ()
            if semantic_cards:
                write_cards = tuple(
                    card
                    for card in semantic_cards
                    if any(
                        permission.endswith(".write")
                        for permission in card.action.required_permissions
                    )
                )
                if write_cards:
                    namespace = write_cards[0].action.action_id.split(".", 1)[0]
                    scoped_cards = tuple(
                        card
                        for card in semantic_cards
                        if card.action.action_id.split(".", 1)[0] == namespace
                    )
                    if scoped_cards:
                        return scoped_cards[:10]
                return tuple(semantic_cards)[:10]
        # Domain retrieval is only a candidate reduction. Action meaning and
        # arguments still come from the bounded model proposal and decoder.
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
        cards = manager.retrieve(domain) if domain is not None else manager.enabled_cards()
        return tuple(cards)[:10]

    def _fallback_decision(
        self, intent: IntentFrame, cards: tuple[ActionCard, ...], context: Context
    ) -> Decision | Result | None:
        if self.dependencies.model_provider is None:
            return None
        try:
            provider = self.dependencies.model_provider()
            if cards:
                classification_values = {
                    key: value for key, value in context.values.items() if key != "canonical_facts"
                }
                classification_request = ModelRequest(
                    working_set=WorkingSet(
                        intent=intent,
                        context=context.model_copy(update={"values": classification_values}),
                    ),
                    # Classification may choose only a semantic mode, but it
                    # needs the bounded candidate vocabulary to distinguish a
                    # list destination from a request to create or complete.
                    # Core still decodes the later action proposal against the
                    # same cards and retains all authority gates.
                    action_cards=cards,
                    classification_only=True,
                )
                classification_response = provider.decide(classification_request)
                semantic_mode = (
                    classification_response.raw.get("semantic_mode")
                    if isinstance(classification_response.raw, dict)
                    else None
                )
                if semantic_mode == "ACTION" and isinstance(classification_response.raw, dict):
                    # Some providers return a bounded action reference even
                    # during the mode pass. Reuse it only when it exactly
                    # names a supplied ActionCard and its arguments are a
                    # declared subset. This prevents a second cognition pass
                    # from drifting a clear mutation into a nearby read;
                    # grounding, policy, execution, and verification remain
                    # below this proposal boundary.
                    action_ref = classification_response.raw.get("action_ref")
                    action_arguments = classification_response.raw.get("action_arguments", {})
                    selected_card = next(
                        (card for card in cards if card.action.action_id == action_ref), None
                    )
                    if selected_card is not None and isinstance(action_arguments, dict):
                        declared = set(selected_card.argument_keys)
                        if set(action_arguments).issubset(declared):
                            return Decision(
                                kind=DecisionKind.ACTION,
                                action=selected_card.action.model_copy(
                                    update={"arguments": dict(action_arguments)}
                                ),
                                semantic_mode="ACTION",
                            )
                if semantic_mode in {"GENERATION", "READ"}:
                    if semantic_mode == "GENERATION":
                        generated_answer = (
                            classification_response.raw.get("answer")
                            if isinstance(classification_response.raw, dict)
                            else None
                        )
                        if isinstance(generated_answer, str) and generated_answer.strip():
                            return Decision(
                                kind=DecisionKind.ANSWER,
                                answer=generated_answer,
                                semantic_mode="GENERATION",
                            )
                    answer_context = context
                    if semantic_mode == "GENERATION":
                        answer_context = context.model_copy(
                            update={
                                "values": {
                                    key: value
                                    for key, value in context.values.items()
                                    if key != "canonical_facts"
                                }
                            }
                        )
                    answer_request = ModelRequest(
                        working_set=WorkingSet(intent=intent, context=answer_context),
                        action_cards=(),
                    )
                    answer = StrictDecisionDecoder().decode(
                        provider.decide(answer_request), (), allow_argument_proposals=False
                    )
                    if answer.kind is DecisionKind.ANSWER:
                        return answer.model_copy(update={"semantic_mode": semantic_mode})
            routing_only = any(
                permission.endswith(".write")
                for card in cards
                for permission in card.action.required_permissions
            )
            routing_context = context
            if routing_only:
                routing_values = {
                    key: value for key, value in context.values.items() if key != "canonical_facts"
                }
                routing_context = context.model_copy(update={"values": routing_values})
            request = ModelRequest(
                working_set=WorkingSet(intent=intent, context=routing_context),
                action_cards=cards,
                allow_argument_proposals=True,
                routing_only=routing_only,
            )
            decoder = StrictDecisionDecoder()
            decision: Decision | None = None
            for attempt in range(2):
                response = provider.decide(request)
                try:
                    decision = decoder.decode(
                        response, request.action_cards, allow_argument_proposals=True
                    )
                    break
                except InvalidDecision as error:
                    # Repair only an empty benign answer with no capability
                    # cards. This remains bounded cognition and cannot turn a
                    # malformed action into an executable proposal.
                    raw = response.raw
                    if not (
                        attempt == 0
                        and isinstance(raw, dict)
                        and raw.get("kind") == DecisionKind.ANSWER.value
                        and not isinstance(raw.get("answer"), str)
                    ):
                        raise error
                    request = request.model_copy(update={"action_cards": ()})
            if decision is None:
                raise InvalidDecision("model answer repair did not produce a decision")
            if routing_only and decision.kind is DecisionKind.ANSWER:
                reconsidered = decoder.decode(
                    provider.decide(request), request.action_cards, allow_argument_proposals=True
                )
                if reconsidered.kind is not decision.kind:
                    return Decision(
                        kind=DecisionKind.CLARIFY,
                        clarification=(
                            "I am not confident whether you want to change something or "
                            "ask about it. Please clarify the intended action."
                        ),
                    )
                if reconsidered.kind is DecisionKind.ANSWER:
                    decision = reconsidered
                else:
                    decision = reconsidered
            if routing_only and decision.kind is DecisionKind.ANSWER:
                request = request.model_copy(
                    update={
                        "working_set": WorkingSet(intent=intent, context=context),
                        "routing_only": False,
                    }
                )
                decision = decoder.decode(
                    provider.decide(request), request.action_cards, allow_argument_proposals=True
                )
            if decision.kind is DecisionKind.ACTION and decision.action is not None:
                selected_card = next(
                    (card for card in cards if card.action.action_id == decision.action.action_id),
                    None,
                )
                if (
                    selected_card is not None
                    and selected_card.action.action_id == "tasks.events.create"
                    and is_task_destination_request(intent.utterance)
                ):
                    task_cards = tuple(
                        card for card in cards if card.action.action_id == "tasks.create"
                    )
                    if task_cards:
                        title = decision.action.arguments.get("title")
                        if not isinstance(title, str) or not title.strip():
                            return Decision(
                                kind=DecisionKind.CLARIFY,
                                clarification="What should I add to your task list?",
                            )
                        task_arguments: dict[str, Any] = {"title": title}
                        due_at = requested_task_due_at(intent.utterance)
                        if due_at is not None:
                            task_arguments["due_at"] = due_at
                        decision = Decision(
                            kind=DecisionKind.ACTION,
                            action=task_cards[0].action.model_copy(
                                update={"arguments": task_arguments}
                            ),
                            semantic_mode="ACTION",
                        )
                        selected_card = task_cards[0]
                if (
                    is_question_request(intent.utterance)
                    and selected_card is not None
                    and any(
                        permission.endswith(".write")
                        for permission in selected_card.action.required_permissions
                    )
                ):
                    read_cards = tuple(
                        card
                        for card in cards
                        if not any(
                            permission.endswith(".write")
                            for permission in card.action.required_permissions
                        )
                    )
                    if read_cards:
                        read_request = ModelRequest(
                            working_set=WorkingSet(intent=intent, context=context),
                            action_cards=read_cards,
                        )
                        read_decision = decoder.decode(
                            provider.decide(read_request),
                            read_cards,
                            allow_argument_proposals=False,
                        )
                        if (
                            read_decision.kind is not DecisionKind.ACTION
                            or read_decision.action is not None
                        ):
                            return read_decision
                action = decision.action
                if action is None:
                    return decision
                card = next(
                    (card for card in cards if card.action.action_id == action.action_id),
                    None,
                )
                if card is not None and card.argument_keys and not action.arguments:
                    # A small model can identify the capability but omit its
                    # object argument. Re-ask with the already-selected card;
                    # this remains bounded cognition, not phrase extraction.
                    focused = request.model_copy(update={"action_cards": (card,)})
                    decision = decoder.decode(
                        provider.decide(focused), (card,), allow_argument_proposals=True
                    )
            return decision
        except InvalidDecision:
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message="I could not safely interpret that request. Please rephrase it.",
                evidence={"provenance": "model_boundary", "authoritative": False},
                correlation_id=intent.correlation_id,
                retryable=True,
            )
        except Exception as exc:
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.FAILED,
                message="Model unavailable; request can be retried",
                evidence={
                    "provenance": "model_boundary",
                    "authoritative": False,
                    "error_type": type(exc).__name__,
                },
                correlation_id=intent.correlation_id,
                retryable=True,
            )

    def _model(self) -> Any:
        if self.dependencies.model_provider is not None:
            return self.dependencies.model_provider()
        return OllamaProvider(
            os.environ.get("AEGIS_OLLAMA_MODEL", "qwen3:8b"),
            OllamaHttpTransport(self.dependencies.required("AEGIS_OLLAMA_URL")),
        )

    def run(
        self,
        utterance: str,
        principal: Principal,
        correlation_id: UUID | None = None,
        context_correlation_id: UUID | None = None,
    ) -> Result:
        connection = self.dependencies.connect(self.dependencies.required("AEGIS_DATABASE_URL"))
        channel: OpenClawWebSocketChannel | None = None
        try:
            self.dependencies.apply_migrations(connection)
            if self.dependencies.local_identity():
                self.dependencies.ensure_local_identity(connection, principal)
            intent = IntentFrame(
                principal=principal,
                utterance=utterance,
                correlation_id=correlation_id or uuid4(),
            )
            objective_store = PostgresObjectiveStore(connection)
            context = _context_from_prior_result(objective_store, context_correlation_id, principal)
            recovered_plan = objective_store.get_objective_by_correlation(
                intent.correlation_id, principal
            )
            if recovered_plan is None and objective_store.correlation_bound(intent.correlation_id):
                return Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.BLOCKED,
                    message="request correlation is unavailable",
                    correlation_id=intent.correlation_id,
                )
            if recovered_plan is not None and recovered_plan.steps:
                prior_plan_result = objective_store.get_result(f"plan:{intent.correlation_id}")
                if (
                    prior_plan_result is not None
                    and prior_plan_result.state is ObjectiveState.COMPLETED
                    and not prior_plan_result.retryable
                ):
                    return prior_plan_result

            def persist_fast_result(result: Result) -> Result:
                result = _with_continuation_context(result, context)
                objective_store.save_objective(
                    Objective(
                        id=result.objective_id,
                        intent=intent,
                        correlation_id=intent.correlation_id,
                        state=result.state,
                    )
                )
                objective_store.save_result(f"interaction:{intent.correlation_id}", result)
                return result

            if recovered_plan is not None and not recovered_plan.steps:
                prior_interaction_result = objective_store.get_result(
                    f"interaction:{intent.correlation_id}"
                )
                if prior_interaction_result is not None and not prior_interaction_result.retryable:
                    return prior_interaction_result
            recovered_plan_actions = (
                recovered_plan.steps
                if recovered_plan is not None and recovered_plan.steps
                else None
            )
            if recovered_plan_actions is None:
                multi_action_result = MultiActionFastPath.resolve(intent)
                if multi_action_result is not None:
                    return persist_fast_result(multi_action_result)
            if self.dependencies.model_provider is None:
                domain_clarification = DomainClarificationFastPath.resolve(intent)
                if domain_clarification is not None:
                    return persist_fast_result(domain_clarification)
            contextual_mutation = ContextualMutationGuard.resolve(intent)
            if contextual_mutation is not None:
                return persist_fast_result(contextual_mutation)
            if FinanceReadFastPath.needs_purchase_amount(utterance):
                return persist_fast_result(
                    Result(
                        objective_id=uuid4(),
                        state=ObjectiveState.BLOCKED,
                        message=(
                            "What purchase amount should I compare with your available "
                            "balance and obligations?"
                        ),
                        correlation_id=intent.correlation_id,
                    )
                )
            household_store = PostgresHouseholdStore(connection)
            if recovered_plan_actions is None and CrossDomainPlanningFastPath.matches(utterance):
                task_store = PostgresTaskStore(connection)
                personal_state = PostgresPersonalStateStore(
                    connection, principal.vault_id
                ).load_for_principal(principal)
                household_snapshot = household_store.read_snapshot(principal)
                obligations = tuple(
                    SharedObligation(item.title, item.amount)
                    for item in cast(
                        tuple[HouseholdObligation, ...],
                        household_snapshot.get("obligations", ()),
                    )
                    if not item.settled
                )
                finance: dict[str, Any] | None = None
                if FinanceReadFastPath.matches(utterance):
                    finance_result = FinanceReadFastPath(
                        FinanceLedger(PostgresFinanceSnapshotStore(connection))
                    ).resolve(intent, obligations)
                    if finance_result is not None:
                        finance = finance_result.evidence
                planning_result = CrossDomainPlanningFastPath(
                    personal_state,
                    household_snapshot,
                    task_store.list(principal),
                    finance,
                ).resolve(intent)
                if planning_result is not None:
                    return persist_fast_result(planning_result)
            if recovered_plan_actions is None and FinanceReadFastPath.matches(utterance):
                snapshot = household_store.read_snapshot(principal)
                household_obligations = cast(
                    tuple[HouseholdObligation, ...], snapshot.get("obligations", ())
                )
                obligations = tuple(
                    SharedObligation(item.title, item.amount)
                    for item in household_obligations
                    if not item.settled
                )
                finance_result = FinanceReadFastPath(
                    FinanceLedger(PostgresFinanceSnapshotStore(connection))
                ).resolve(intent, obligations)
                if finance_result is not None:
                    PostgresAuditLog(connection).append(
                        "finance.affordability.read",
                        principal.id,
                        {
                            "purchase_cents": finance_result.evidence["purchase_cents"],
                            "shared_obligations_cents": finance_result.evidence[
                                "shared_obligations_cents"
                            ],
                            "affordable": finance_result.evidence["affordable"],
                        },
                    )
                    return persist_fast_result(finance_result)
            task_store = PostgresTaskStore(connection)
            personal_state = PostgresPersonalStateStore(
                connection, principal.vault_id
            ).load_for_principal(principal)
            goal_task_title, goal_task_error = PersonalTaskComposer.resolve(
                utterance, personal_state
            )
            goal_chore_title, goal_chore_error = PersonalChoreComposer.resolve(
                utterance, personal_state
            )
            memory_task_title, memory_task_error = PersonalMemoryTaskComposer.resolve(
                utterance, personal_state
            )
            memory_chore_title, memory_chore_error = PersonalMemoryChoreComposer.resolve(
                utterance, personal_state
            )
            composer_errors = tuple(
                error
                for error in (
                    goal_task_error,
                    goal_chore_error,
                    memory_task_error,
                    memory_chore_error,
                )
                if error is not None
            )
            if composer_errors:
                return persist_fast_result(
                    Result(
                        objective_id=uuid4(),
                        state=ObjectiveState.BLOCKED,
                        message=composer_errors[0],
                        correlation_id=intent.correlation_id,
                    )
                )
            composed_title = (
                goal_task_title or goal_chore_title or memory_task_title or memory_chore_title
            )
            household_snapshot = household_store.read_snapshot(principal)
            if recovered_plan_actions is None and CrossDomainPlanningFastPath.matches(utterance):
                planning_result = CrossDomainPlanningFastPath(
                    personal_state, household_snapshot, task_store.list(principal)
                ).resolve(intent)
                if planning_result is not None:
                    return persist_fast_result(planning_result)
            if (
                recovered_plan_actions is None
                and composed_title is None
                and HouseholdReadFastPath.matches(utterance)
            ):
                household_result = HouseholdReadFastPath(household_snapshot).resolve(intent)
                if household_result is not None:
                    return persist_fast_result(household_result)
            if recovered_plan_actions is None and composed_title is None:
                if self.dependencies.model_provider is None:
                    task_clarification = TaskIntentClarificationFastPath.resolve(intent)
                    if task_clarification is not None:
                        return persist_fast_result(task_clarification)
                task_result = TaskReadFastPath(task_store).resolve(intent)
                if task_result is not None:
                    return persist_fast_result(task_result)
                priority_result = TaskPriorityFastPath(task_store).resolve(intent)
                if priority_result is not None:
                    return persist_fast_result(priority_result)
            semantic_enabled = os.environ.get("AEGIS_SEMANTIC_MEMORY", "0").lower() in {
                "1",
                "true",
                "yes",
            }
            if semantic_enabled:
                embedding_provider = OllamaEmbeddingProvider(
                    os.environ.get("AEGIS_EMBEDDING_MODEL", "nomic-embed-text"),
                    self.dependencies.required("AEGIS_OLLAMA_URL"),
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
            if recovered_plan_actions is None and composed_title is None:
                memory_result = memory_fast_path.resolve(intent)
                if memory_result is not None:
                    return persist_fast_result(memory_result)
            manager = PackManager(store=PostgresPackStore(connection))
            for bundle in reference_bundles():
                try:
                    manager.status(bundle.manifest.pack_id)
                    installed_bundle = manager._bundles[bundle.manifest.pack_id]
                    installed_ids = {card.action.action_id for card in installed_bundle.cards}
                    required_ids = {card.action.action_id for card in bundle.cards}
                    if not required_ids.issubset(installed_ids) or tuple(
                        installed_bundle.cards
                    ) != tuple(bundle.cards):
                        # Persisted Pack metadata is a contract, not merely an
                        # ID cache. Refresh it when bounded argument/schema
                        # details change so old runtimes cannot misroute safely.
                        manager.remove(bundle.manifest.pack_id)
                        manager.discover(bundle)
                except KeyError:
                    manager.discover(bundle)
            for pack_id in ("tasks", "kitchen"):
                if manager.status(pack_id) is PackStatus.DISCOVERED:
                    manager.install(
                        pack_id,
                        frozenset(manager._bundles[pack_id].manifest.permissions),
                    )
                    manager.enable(pack_id)
                elif manager.status(pack_id) is PackStatus.INSTALLED:
                    manager.enable(pack_id)
            plan_titles = MultiActionFastPath.task_chore_titles(utterance)
            if recovered_plan_actions is not None:
                plan_actions = recovered_plan_actions
            elif plan_titles is not None:
                task_card = next(
                    card
                    for card in manager.retrieve("tasks")
                    if card.action.action_id == "tasks.create"
                )
                chore_card = next(
                    card
                    for card in manager.retrieve("tasks")
                    if card.action.action_id == "tasks.chores.create"
                )
                task_action = task_card.action.model_copy(
                    update={"arguments": {"title": plan_titles[0]}}
                )
                chore_action = chore_card.action.model_copy(
                    update={"arguments": {"title": plan_titles[1]}}
                )
                plan_actions = (task_action, chore_action)
            elif (event_details := MultiActionFastPath.task_event_details(utterance)) is not None:
                task_card = next(
                    card
                    for card in manager.retrieve("tasks")
                    if card.action.action_id == "tasks.create"
                )
                event_card = next(
                    card
                    for card in manager.retrieve("tasks")
                    if card.action.action_id == "tasks.events.create"
                )
                task_action = task_card.action.model_copy(
                    update={"arguments": {"title": event_details[0]}}
                )
                event_action = event_card.action.model_copy(
                    update={
                        "arguments": {
                            "title": event_details[1],
                            "starts_at": event_details[2],
                        }
                    }
                )
                plan_actions = (task_action, event_action)
            else:
                plan_actions = None
            if plan_actions is not None:
                principal_store = PostgresHouseholdStore(connection)
                kernel = Kernel(
                    self._model(),
                    StrictDecisionDecoder(),
                    PostgresSpacePolicy(
                        connection,
                        {"tasks.write": frozenset({Role.OWNER, Role.MEMBER})},
                    ),
                    _ActionExecutorDispatch(
                        {
                            "tasks.create": PostgresTaskExecutor(task_store, principal),
                            "tasks.chores.create": PostgresChoreExecutor(
                                principal_store, principal
                            ),
                            "tasks.events.create": PostgresEventExecutor(
                                principal_store, principal
                            ),
                        }
                    ),
                    _ActionVerifierDispatch(
                        {
                            "tasks.create": PostgresTaskVerifier(task_store, principal),
                            "tasks.chores.create": PostgresChoreVerifier(
                                principal_store, principal
                            ),
                            "tasks.events.create": PostgresEventVerifier(
                                principal_store, principal
                            ),
                        }
                    ),
                    store=PostgresObjectiveStore(connection),
                    audit=PostgresAuditLog(connection),
                )
                return kernel.run_sequence(intent, plan_actions, context=context)
            try:
                if self.dependencies.model_provider is not None:
                    raise InteractionInputError("semantic action resolution required")
                _domain, card = self.dependencies.select_action(utterance, manager)
            except InteractionInputError as exc:
                fallback_context = _fallback_working_context(
                    context, task_store, household_store, principal, utterance
                )
                fallback = self._fallback_decision(
                    intent,
                    self._fallback_cards(manager, utterance, fallback_context),
                    fallback_context,
                )
                if isinstance(fallback, Result):
                    return persist_fast_result(fallback)
                if isinstance(fallback, Decision):
                    if fallback.kind is DecisionKind.ANSWER:
                        answer_evidence: dict[str, Any] = {
                            "provenance": "model_generated",
                            "authoritative": False,
                            "answer_mode": fallback.semantic_mode,
                        }
                        authorized_facts = _authorized_context_evidence(fallback_context)
                        if authorized_facts:
                            answer_evidence.update(authorized_facts)
                            answer_evidence["context_provenance"] = "authorized_working_set"
                        return persist_fast_result(
                            Result(
                                objective_id=uuid4(),
                                state=ObjectiveState.COMPLETED,
                                message=fallback.answer or "",
                                evidence=answer_evidence,
                                correlation_id=intent.correlation_id,
                            )
                        )
                    if fallback.kind is DecisionKind.CLARIFY:
                        clarification = fallback.clarification or "Please clarify your request."
                        if has_multiple_question_clauses(utterance):
                            # A model clarification must not import a domain
                            # from the authorized working set when the user
                            # asked multiple independent questions. Keep the
                            # safety result while making the next step useful.
                            clarification = (
                                "That request contains multiple independent questions. "
                                "Please ask one at a time so I can answer each from "
                                "authorized information."
                            )
                        return persist_fast_result(
                            Result(
                                objective_id=uuid4(),
                                state=ObjectiveState.BLOCKED,
                                message=clarification,
                                correlation_id=intent.correlation_id,
                            )
                        )
                    if fallback.kind is DecisionKind.NEED_CONTEXT:
                        return persist_fast_result(
                            Result(
                                objective_id=uuid4(),
                                state=ObjectiveState.BLOCKED,
                                message="I need more context to safely interpret that request.",
                                correlation_id=intent.correlation_id,
                            )
                        )
                    if fallback.kind is DecisionKind.ACTION and fallback.action is not None:
                        fallback_card = next(
                            (
                                candidate
                                for candidate in self._fallback_cards(
                                    manager,
                                    utterance,
                                    fallback_context,
                                )
                                if candidate.action.action_id == fallback.action.action_id
                            ),
                            None,
                        )
                        if fallback_card is None:
                            return persist_fast_result(
                                Result(
                                    objective_id=uuid4(),
                                    state=ObjectiveState.BLOCKED,
                                    message=(
                                        "I could not safely match that request to an "
                                        "available capability."
                                    ),
                                    correlation_id=intent.correlation_id,
                                    retryable=True,
                                )
                            )
                        # Keep the model's bounded arguments while restoring the
                        # canonical card metadata that the decoder validated.
                        card = fallback_card.model_copy(update={"action": fallback.action})
                    else:
                        return persist_fast_result(
                            Result(
                                objective_id=uuid4(),
                                state=ObjectiveState.BLOCKED,
                                message=fallback.reason or str(fallback.kind),
                                correlation_id=intent.correlation_id,
                            )
                        )
                else:
                    domain_clarification = DomainClarificationFastPath.resolve(intent)
                    if domain_clarification is not None:
                        return persist_fast_result(domain_clarification)
                    return persist_fast_result(
                        Result(
                            objective_id=uuid4(),
                            state=ObjectiveState.BLOCKED,
                            message=str(exc),
                            correlation_id=intent.correlation_id,
                        )
                    )
            principal_store = PostgresHouseholdStore(connection)
            if card.action.action_id == "tasks.complete":
                title = card.action.arguments.get("title")
                if not isinstance(title, str) or not title.strip():
                    return persist_fast_result(
                        Result(
                            objective_id=uuid4(),
                            state=ObjectiveState.BLOCKED,
                            message=(
                                "Name the task to complete, for example: "
                                "Complete the task buy cat food."
                            ),
                            correlation_id=intent.correlation_id,
                        )
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
                    return persist_fast_result(completion_result)
            if card.action.action_id == "tasks.chores.complete":
                title = card.action.arguments.get("title")
                if not isinstance(title, str) or not title.strip():
                    return persist_fast_result(
                        Result(
                            objective_id=uuid4(),
                            state=ObjectiveState.BLOCKED,
                            message=(
                                "Name the chore to complete, for example: "
                                "Complete the chore clean the kitchen."
                            ),
                            correlation_id=intent.correlation_id,
                        )
                    )
                household_snapshot = principal_store.read_snapshot(principal)
                completion_result = ChoreCompletionFastPath.resolve(
                    intent,
                    title,
                    cast(tuple[Chore, ...], household_snapshot["chores"]),
                )
                if completion_result is not None:
                    return persist_fast_result(completion_result)
            if goal_task_title is not None and card.action.action_id == "tasks.create":
                card = card.model_copy(
                    update={
                        "action": card.action.model_copy(
                            update={"arguments": {"title": goal_task_title}}
                        )
                    }
                )
            elif goal_chore_title is not None and card.action.action_id == "tasks.chores.create":
                card = card.model_copy(
                    update={
                        "action": card.action.model_copy(
                            update={"arguments": {"title": goal_chore_title}}
                        )
                    }
                )
            elif memory_task_title is not None and card.action.action_id == "tasks.create":
                arguments: dict[str, Any] = {"title": memory_task_title}
                due_at = requested_task_due_at(utterance)
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
                    return persist_fast_result(
                        Result(
                            objective_id=uuid4(),
                            state=ObjectiveState.BLOCKED,
                            message="I need a clear deadline before adding that task.",
                            correlation_id=intent.correlation_id,
                        )
                    )
                grounded, due_at = ground_task_due_at(utterance, proposed_due_at)
                if not grounded:
                    return persist_fast_result(
                        Result(
                            objective_id=uuid4(),
                            state=ObjectiveState.BLOCKED,
                            message=(
                                "What deadline should I use for that task? "
                                "I won't infer one from context."
                            ),
                            correlation_id=intent.correlation_id,
                        )
                    )
                arguments = dict(card.action.arguments)
                if due_at is None:
                    arguments.pop("due_at", None)
                else:
                    arguments["due_at"] = due_at
                card = card.model_copy(
                    update={"action": card.action.model_copy(update={"arguments": arguments})}
                )
            if self.dependencies.runtime_registry is not None:
                runtime = self.dependencies.runtime_registry.resolve(card, connection, principal)
                executor = runtime.executor
                verifier = runtime.verifier
                permissions = runtime.permissions
            elif card.action.action_id == "kitchen.groceries.add":
                channel = self.dependencies.openclaw_channel()
                executor = OpenClawExecutor(
                    OpenClawGroceryExecutor(
                        channel,
                        os.environ.get("AEGIS_LIVE_GROCERY_PATH", "/tmp/aegis-alpha-groceries.tsv"),
                        principal_store,
                        principal,
                    ),
                    _RuntimePolicy(),
                    _NoApproval(),
                )
                verifier = OpenClawGroceryVerifier(principal_store, principal)
                permissions = {"kitchen.write": frozenset({Role.OWNER, Role.MEMBER})}
            elif card.action.action_id == "kitchen.groceries.list":
                executor = PostgresGroceryListExecutor(principal_store, principal)
                verifier = PostgresGroceryListVerifier(principal_store, principal)
                permissions = {"kitchen.read": frozenset({Role.OWNER, Role.MEMBER})}
            elif card.action.action_id in {"tasks.create", "tasks.complete"}:
                executor = PostgresTaskExecutor(task_store, principal)
                verifier = PostgresTaskVerifier(task_store, principal)
                permissions = {"tasks.write": frozenset({Role.OWNER, Role.MEMBER})}
            elif card.action.action_id == "tasks.chores.create":
                executor = PostgresChoreExecutor(principal_store, principal)
                verifier = PostgresChoreVerifier(principal_store, principal)
                permissions = {"tasks.write": frozenset({Role.OWNER, Role.MEMBER})}
            elif card.action.action_id == "tasks.chores.complete":
                executor = PostgresChoreExecutor(principal_store, principal)
                verifier = PostgresChoreVerifier(principal_store, principal)
                permissions = {"tasks.write": frozenset({Role.OWNER, Role.MEMBER})}
            elif card.action.action_id == "tasks.events.create":
                executor = PostgresEventExecutor(principal_store, principal)
                verifier = PostgresEventVerifier(principal_store, principal)
                permissions = {"tasks.write": frozenset({Role.OWNER, Role.MEMBER})}
            else:
                executor = PostgresTaskListExecutor(task_store, principal)
                verifier = PostgresTaskListVerifier(task_store, principal)
                permissions = {"tasks.read": frozenset({Role.OWNER, Role.MEMBER})}
            kernel = Kernel(
                # The bounded model proposal was already decoded and
                # canonicalized above. Reuse it as a fixed proposal rather
                # than invoking cognition a second time with legacy decoder
                # settings or allowing the action to drift.
                _FixedActionModel(card.action),
                StrictDecisionDecoder(),
                PostgresSpacePolicy(connection, permissions),
                executor,
                verifier,
                store=PostgresObjectiveStore(connection),
                audit=PostgresAuditLog(connection),
            )
            return kernel.run(intent, (card,), context=context)
        finally:
            if channel is not None:
                channel.close()
            connection.close()
