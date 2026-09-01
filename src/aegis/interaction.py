"""Reusable AEGIS application interaction boundary for all clients."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, cast
from uuid import UUID, uuid4

from .audit import PostgresAuditLog
from .contracts import ActionCard, IntentFrame, Principal, Result
from .decoding import StrictDecisionDecoder
from .embeddings import OllamaEmbeddingProvider, PostgresMemoryVectorIndex
from .finance import FinanceLedger, FinanceReadFastPath, PostgresFinanceSnapshotStore
from .gateway_rpc import OpenClawWebSocketChannel
from .household import (
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
from .planning import CrossDomainPlanningFastPath, DomainClarificationFastPath, MultiActionFastPath
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
    TaskReadFastPath,
)


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
    ) -> None:
        self.connect = connect
        self.required = required
        self.apply_migrations = apply_migrations
        self.ensure_local_identity = ensure_local_identity
        self.select_action = select_action
        self.openclaw_channel = openclaw_channel
        self.local_identity = local_identity


class _RuntimePolicy:
    def allows(self, request: Any) -> bool:
        return bool(request.action.action_id == "kitchen.groceries.add")


class _NoApproval:
    def required(self, request: Any) -> bool:
        return False

    def approved(self, request: Any) -> bool:
        return True


class InteractionBoundary:
    """Canonical application interaction service used by every client."""

    def __init__(self, dependencies: InteractionDependencies) -> None:
        self.dependencies = dependencies

    def run(
        self, utterance: str, principal: Principal, correlation_id: UUID | None = None
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
            multi_action_result = MultiActionFastPath.resolve(intent)
            if multi_action_result is not None:
                return multi_action_result
            domain_clarification = DomainClarificationFastPath.resolve(intent)
            if domain_clarification is not None:
                return domain_clarification
            household_store = PostgresHouseholdStore(connection)
            if CrossDomainPlanningFastPath.matches(utterance):
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
                    return planning_result
            if FinanceReadFastPath.matches(utterance):
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
                    return finance_result
            task_store = PostgresTaskStore(connection)
            personal_state = PostgresPersonalStateStore(
                connection, principal.vault_id
            ).load_for_principal(principal)
            household_snapshot = household_store.read_snapshot(principal)
            if CrossDomainPlanningFastPath.matches(utterance):
                planning_result = CrossDomainPlanningFastPath(
                    personal_state, household_snapshot, task_store.list(principal)
                ).resolve(intent)
                if planning_result is not None:
                    return planning_result
            if HouseholdReadFastPath.matches(utterance):
                household_result = HouseholdReadFastPath(household_snapshot).resolve(intent)
                if household_result is not None:
                    return household_result
            task_result = TaskReadFastPath(task_store).resolve(intent)
            if task_result is not None:
                return task_result
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
            memory_result = memory_fast_path.resolve(intent)
            if memory_result is not None:
                return memory_result
            manager = PackManager(store=PostgresPackStore(connection))
            for bundle in reference_bundles():
                try:
                    manager.status(bundle.manifest.pack_id)
                    installed_ids = {
                        card.action.action_id
                        for card in manager._bundles[bundle.manifest.pack_id].cards
                    }
                    required_ids = {card.action.action_id for card in bundle.cards}
                    if not required_ids.issubset(installed_ids):
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
            _domain, card = self.dependencies.select_action(utterance, manager)
            principal_store = PostgresHouseholdStore(connection)
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
            elif card.action.action_id == "tasks.create":
                executor = PostgresTaskExecutor(task_store, principal)
                verifier = PostgresTaskVerifier(task_store, principal)
                permissions = {"tasks.write": frozenset({Role.OWNER, Role.MEMBER})}
            elif card.action.action_id == "tasks.chores.create":
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
                OllamaProvider(
                    os.environ.get("AEGIS_OLLAMA_MODEL", "qwen3:8b"),
                    OllamaHttpTransport(self.dependencies.required("AEGIS_OLLAMA_URL")),
                ),
                StrictDecisionDecoder(),
                PostgresSpacePolicy(connection, permissions),
                executor,
                verifier,
                store=PostgresObjectiveStore(connection),
                audit=PostgresAuditLog(connection),
            )
            return kernel.run(intent, (card,))
        finally:
            if channel is not None:
                channel.close()
            connection.close()
