"""Reusable AEGIS application interaction boundary for all clients."""

from __future__ import annotations

import os
from collections.abc import Callable
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
from .kernel import Kernel
from .ollama import OllamaHttpTransport, OllamaProvider
from .openclaw import OpenClawExecutor
from .pack_lifecycle import PackManager, PackStatus, PostgresPackStore
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
    TaskReadFastPath,
    requested_task_due_at,
)


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
    ) -> None:
        self.connect = connect
        self.required = required
        self.apply_migrations = apply_migrations
        self.ensure_local_identity = ensure_local_identity
        self.select_action = select_action
        self.openclaw_channel = openclaw_channel
        self.local_identity = local_identity
        self.model_provider = model_provider


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
    return Context(
        values={
            "prior_correlation_id": str(correlation_id),
            "prior_objective_id": str(objective.id),
            "prior_state": result.state.value,
            "canonical_facts": evidence,
        },
        sources=("authorized_canonical_result",),
    )


def _fallback_working_context(
    context: Context, task_store: PostgresTaskStore, principal: Principal, utterance: str
) -> Context:
    """Add a small authorized candidate set for bounded intent resolution."""

    values = dict(context.values)
    facts = dict(values.get("canonical_facts", {}))
    if "canonical_items" in facts or "canonical_tasks" in facts:
        return Context(
            values=values,
            sources=tuple(dict.fromkeys((*context.sources, "authorized_canonical_context"))),
        )
    tasks = list(task_store.list(principal))
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
        {"title": task.title, "status": task.status.value} for task in selected[:20]
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

    @staticmethod
    def _fallback_cards(
        manager: PackManager, utterance: str, context: Context | None = None
    ) -> tuple[ActionCard, ...]:
        """Offer a bounded capability vocabulary; metadata remains Core-owned."""

        text = utterance.casefold()
        facts = (context.values if context is not None else {}).get("canonical_facts", {})
        if isinstance(facts, dict):
            if isinstance(facts.get("canonical_items"), list):
                # A contextual grocery question is answered from the
                # authorized projection; do not offer a mutation card that
                # could turn an unresolved reference into a write.
                return ()
            if isinstance(facts.get("canonical_tasks"), list):
                return tuple(manager.retrieve("tasks"))[:10]
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
            request = ModelRequest(
                working_set=WorkingSet(intent=intent, context=context),
                action_cards=cards,
                allow_argument_proposals=True,
            )
            decoder = StrictDecisionDecoder()
            response = provider.decide(request)
            try:
                decision = decoder.decode(response, cards, allow_argument_proposals=True)
            except InvalidDecision as first_error:
                # Repair an empty benign answer with no capability cards. This
                # keeps the retry semantic and bounded without allowing a
                # malformed answer to become a completed Result.
                raw = response.raw
                if not (
                    isinstance(raw, dict)
                    and raw.get("kind") == DecisionKind.ANSWER.value
                    and not isinstance(raw.get("answer"), str)
                ):
                    raise first_error
                focused = request.model_copy(update={"action_cards": ()})
                decision = decoder.decode(
                    provider.decide(focused), (), allow_argument_proposals=True
                )
            if decision.kind is DecisionKind.ACTION and decision.action is not None:
                card = next(
                    (card for card in cards if card.action.action_id == decision.action.action_id),
                    None,
                )
                if card is not None and card.argument_keys and not decision.action.arguments:
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
                    context, task_store, principal, utterance
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
                        return persist_fast_result(
                            Result(
                                objective_id=uuid4(),
                                state=ObjectiveState.COMPLETED,
                                message=fallback.answer or "",
                                evidence={
                                    "provenance": "model_generated",
                                    "authoritative": False,
                                },
                                correlation_id=intent.correlation_id,
                            )
                        )
                    if fallback.kind is DecisionKind.CLARIFY:
                        return persist_fast_result(
                            Result(
                                objective_id=uuid4(),
                                state=ObjectiveState.BLOCKED,
                                message=fallback.clarification or "Please clarify your request.",
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
                completion_result = TaskCompletionFastPath.resolve(
                    intent, title, task_store.list(principal)
                )
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
            if card.action.action_id == "kitchen.groceries.add":
                channel = self.dependencies.openclaw_channel()
                executor: Any = OpenClawExecutor(
                    OpenClawGroceryExecutor(
                        channel,
                        os.environ.get("AEGIS_LIVE_GROCERY_PATH", "/tmp/aegis-alpha-groceries.tsv"),
                        principal_store,
                        principal,
                    ),
                    _RuntimePolicy(),
                    _NoApproval(),
                )
                verifier: Any = OpenClawGroceryVerifier(principal_store, principal)
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
                self._model(),
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
