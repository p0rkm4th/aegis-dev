import json
import os
from datetime import datetime, timedelta, timezone
from threading import Event
from uuid import uuid4, uuid5

import pytest

import aegis.ollama as ollama_module
from aegis.ambient import (
    AmbientService,
    BackgroundTask,
    Notification,
    OpenClawAmbientPlatform,
    SqliteAmbientState,
)
from aegis.audit import AuditError, AuditLog, SqliteAuditLog
from aegis.backup import backup_sqlite, restore_sqlite
from aegis.card_collecting import (
    CardCollection,
    CardCollectionExecutor,
    CardCollectionVerifier,
    card_collection_card,
)
from aegis.config import AegisConfig
from aegis.contracts import (
    ActionCard,
    ActionSpec,
    AuthorizationRequest,
    Context,
    Decision,
    DecisionKind,
    ExecutionRequest,
    ExternalEffectAssurance,
    IntentFrame,
    ModelRequest,
    ModelResponse,
    ObjectiveRequirement,
    ObjectiveRequirementProposal,
    ObjectiveSpec,
    ObjectiveSpecProposal,
    ObjectiveState,
    Observation,
    PolicyDecision,
    Principal,
    ProposalFailureEvidence,
    ProposalFailureKind,
    ProposedPlan,
    ProposedPlanStep,
    Result,
    VerificationContract,
    VerificationResult,
    WorkingSet,
)
from aegis.decoding import InvalidDecision, StrictDecisionDecoder
from aegis.devices import DeviceCommand, HomeAssistantAdapter
from aegis.evaluation import DecisionEvaluationHarness, EvaluationCase
from aegis.finance import (
    Account,
    AffordabilityProjection,
    FinanceLedger,
    FinanceReadFastPath,
    FinanceSnapshot,
    PostgresFinanceSnapshotStore,
    Transaction,
)
from aegis.gateway_rpc import (
    CorrelatedRpcClient,
    OpenClawGatewayRpc,
    OpenClawWebSocketChannel,
    RpcProtocolError,
    RpcResponse,
)
from aegis.health import HealthService
from aegis.homelab import FixtureHomelabRuntime, HomelabPack, Host, PostgresHomelabStore, Service
from aegis.household import (
    Chore,
    ChoreCompletionFastPath,
    ContextualChorePriorityFastPath,
    HouseholdEvent,
    HouseholdObligation,
    HouseholdReadFastPath,
    HouseholdSpace,
    PostgresHouseholdStore,
)
from aegis.household_proactivity import HouseholdProactivity, HouseholdSignals
from aegis.identity import (
    InMemoryAuthorization,
    KeycloakIdentityProvider,
    KeycloakOIDCClient,
    Membership,
    OpenFGAAuthorization,
    PostgresExternalPrincipalResolver,
    Resource,
    Role,
    Space,
    Vault,
)
from aegis.kernel import Kernel
from aegis.migrations import validate_migrations
from aegis.model_router import BaselineMetrics, ConfiguredModelRouter, ModelUnavailable
from aegis.network import (
    AuthorizedNetworkScope,
    DiscoveredDevice,
    HomelabInventory,
    NetworkScopePolicy,
    ScopeDenied,
)
from aegis.ollama import OllamaHttpTransport, OllamaProvider, OllamaResponseError
from aegis.openclaw import GatewayDisconnected, OpenClawExecutor, ReconnectingGatewayClient
from aegis.osint import (
    CapabilityGap,
    Forge,
    ForgeLifecycle,
    ForgeStatus,
    Investigation,
    PostgresInvestigationStore,
)
from aegis.pack_lifecycle import PackBundle, PackManager, PackManifest, PackStatus
from aegis.personal import (
    PersonalMemoryFastPath,
    PersonalState,
    PostgresPersonalStateStore,
    Provenance,
)
from aegis.planning import (
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
)
from aegis.projections import (
    HouseholdProjection,
    PostgresProjectionStore,
    PrivacyProjectionService,
    PrivateContribution,
    SharedObligation,
)
from aegis.reference_packs import (
    OpenClawGroceryExecutor,
    OpenClawGroceryVerifier,
    OpenClawHomelabExecutor,
    OpenClawNetworkProbeExecutor,
    ReferenceExecutor,
    ReferenceVerifier,
    ReferenceWorld,
    reference_packs,
)
from aegis.registry import CapabilityRegistry
from aegis.security_lab import PostgresSecurityLabStore, SecurityLab
from aegis.store import PostgresObjectiveStore, SqliteObjectiveStore
from aegis.tasks import (
    ContextualTaskPriorityFastPath,
    ContextualTaskTemporalFastPath,
    PostgresTaskExecutor,
    PostgresTaskStore,
    PostgresTaskVerifier,
    Task,
    TaskCompletionFastPath,
    TaskReadFastPath,
    TaskStatus,
)
from aegis.utterance import is_mutation_request, strip_context_reset


def test_contextual_task_priority_uses_only_authorized_prior_tasks():
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [
                        {"title": "later task", "status": "open", "due_at": "2026-09-05"},
                        {"title": "first task", "status": "open", "due_at": "2026-09-02"},
                        {"title": "done task", "status": "completed", "due_at": "2026-09-01"},
                    ],
                }
            }
        }
    )
    result = ContextualTaskPriorityFastPath().resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which of those should I do first?",
        ),
        context,
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.message.endswith("first task")
    assert result.evidence["priority_basis"] == "authorized_prior_result_earliest_due_at"


def test_contextual_task_priority_uses_latest_authorized_prior_task():
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [
                        {"title": "soon task", "status": "open", "due_at": "2026-09-02"},
                        {"title": "latest task", "status": "open", "due_at": "2026-09-05"},
                    ],
                }
            }
        }
    )
    result = ContextualTaskPriorityFastPath().resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which task is due latest?",
        ),
        context,
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.message.endswith("latest task")
    assert result.evidence["priority_basis"] == "authorized_prior_result_latest_due_at"


def test_contextual_task_temporal_followup_uses_authorized_task_domain():
    class Store:
        def list(self, _principal):
            return (Task(uuid4(), "apartment", "tomorrow task", "alice"),)

    context = Context(
        values={
            "referents": {
                "those": {"fact_key": "canonical_tasks", "candidates": [{"title": "old"}]}
            }
        },
        sources=("authorized_canonical_result",),
    )
    result = ContextualTaskTemporalFastPath().resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What about tomorrow?",
        ),
        context,
        Store(),
    )

    assert result is not None
    assert result.evidence["due_filter"] == "tomorrow"


def test_task_temporal_read_accepts_explicit_correction_prefix():
    class Store:
        def list(self, _principal):
            return (Task(uuid4(), "space", "tomorrow task", "alice"),)

    assert TaskReadFastPath.matches("No, I meant what tasks are due tomorrow?")


def test_contextual_task_temporal_followup_uses_cross_domain_planning_tasks():
    class Store:
        def list(self, _principal):
            return (Task(uuid4(), "apartment", "tomorrow task", "alice"),)

    context = Context(
        values={
            "canonical_facts": {
                "planning": {
                    "open_tasks": [
                        {"title": "tomorrow task", "due_at": "2026-09-04T00:00:00+00:00"}
                    ]
                }
            }
        },
        sources=("authorized_canonical_result",),
    )
    result = ContextualTaskTemporalFastPath().resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What about tomorrow?",
        ),
        context,
        Store(),
    )

    assert result is not None
    assert result.evidence["due_filter"] == "tomorrow"


def test_contextual_task_temporal_followup_uses_authorized_priority_task_focus():
    class Store:
        def list(self, _principal):
            return (
                Task(
                    uuid4(),
                    "apartment",
                    "tomorrow task",
                    "alice",
                    due_at=datetime.now(timezone.utc) + timedelta(days=1),
                ),
                Task(
                    uuid4(),
                    "apartment",
                    "completed tomorrow task",
                    "alice",
                    due_at=datetime.now(timezone.utc) + timedelta(days=1),
                    status=TaskStatus.COMPLETED,
                ),
            )

    context = Context(
        values={"canonical_facts": {"task": {"title": "current priority task"}}},
        sources=("authorized_canonical_result",),
    )
    result = ContextualTaskTemporalFastPath().resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What about tomorrow?",
        ),
        context,
        Store(),
    )

    assert result is not None
    assert result.evidence["due_filter"] == "tomorrow"
    assert [item["title"] for item in result.evidence["canonical_tasks"]] == ["tomorrow task"]


def test_contextual_chore_priority_never_invents_deadline_order():
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_chores",
                    "candidates": [{"title": "clean kitchen", "completed": False}],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )
    result = ContextualChorePriorityFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which one should I start with?",
        ),
        context,
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "no canonical deadlines" in result.message


def test_contextual_chore_priority_blocks_attention_wording():
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_chores",
                    "candidates": [{"title": "clean kitchen", "completed": False}],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )
    result = ContextualChorePriorityFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which one needs attention first?",
        ),
        context,
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "no canonical deadlines" in result.message


def test_contextual_chore_next_never_invents_deadline_order():
    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_chores",
                    "candidates": [{"title": "clean kitchen", "completed": False}],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )
    result = ContextualChorePriorityFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which chore is next?",
        ),
        context,
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "no canonical deadlines" in result.message


def test_correction_without_explicit_mutation_cannot_become_new_action():
    result = ContextualMutationGuard.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="No, I meant my task list",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "explicit action" in result.message


def test_explicit_mutation_after_correction_remains_available_to_normal_grounding():
    assert (
        ContextualMutationGuard.resolve(
            IntentFrame(
                principal=Principal(id="alice", vault_id="alice-vault"),
                utterance="Actually, add a task to check the mailbox",
            )
        )
        is None
    )


class Model:
    def __init__(self, response):
        self.response = response

    def decide(self, request):
        return self.response


class Decoder:
    def __init__(self, decision):
        self.decision = decision

    def decode(self, response, cards):
        assert len(cards) <= 5
        return self.decision


class Policy:
    def __init__(self, result):
        self.result = result

    def authorize(self, request):
        return self.result


class Executor:
    def __init__(self):
        self.calls = 0

    def execute(self, request):
        self.calls += 1
        return Observation(
            execution_id=uuid4(), evidence={"accepted": True}, command_succeeded=True
        )


class Verifier:
    def __init__(self, verified):
        self.verified = verified

    def verify(self, observation, contract):
        return VerificationResult(
            verified=self.verified,
            evidence={"readback": self.verified},
            reason="verified" if self.verified else "postcondition failed",
        )


class FastPath:
    def __init__(self, result):
        self.result = result

    def resolve(self, intent):
        return self.result


def intent():
    return IntentFrame(principal=Principal(id="alice", vault_id="alice-vault"), utterance="do it")


def test_answer_is_deterministically_completed_without_execution():
    ex = Executor()
    k = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.ANSWER, answer="done")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        ex,
        Verifier(True),
    )
    result = k.run(intent())
    assert result.message == "done"
    assert ex.calls == 0


def test_kernel_run_sequence_replays_persisted_steps_without_duplicate_execution(tmp_path):
    from aegis.store import SqliteObjectiveStore

    store = SqliteObjectiveStore(str(tmp_path / "sequence.sqlite"))
    actions = (
        ActionSpec(
            action_id="first",
            capability="first",
            verification=VerificationContract(kind="readback"),
        ),
        ActionSpec(
            action_id="second",
            capability="second",
            verification=VerificationContract(kind="readback"),
        ),
    )
    ex = Executor()
    first = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        ex,
        Verifier(True),
        store=store,
    ).run_sequence(intent(), actions)

    replay_executor = Executor()
    replay = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        replay_executor,
        Verifier(True),
        store=store,
    ).run_sequence(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="a different plan wording",
            correlation_id=first.correlation_id,
        ),
        actions,
    )

    assert first.state is ObjectiveState.COMPLETED
    assert len(first.evidence["steps"]) == 2
    assert replay == first
    assert ex.calls == 2
    assert replay_executor.calls == 0


def test_kernel_persists_core_owned_objective_spec_and_validated_plan(tmp_path):
    from aegis.store import SqliteObjectiveStore

    store = SqliteObjectiveStore(str(tmp_path / "objective-completeness.sqlite"))
    cards = tuple(
        ActionCard(
            action=ActionSpec(
                action_id=action_id,
                capability=action_id,
                arguments={},
                verification=VerificationContract(kind="readback"),
            ),
            summary=action_id,
            relevance=1,
        )
        for action_id in ("first", "second")
    )
    objective_spec = ObjectiveSpec(
        requirements=tuple(ObjectiveRequirement(action_ref=card.action.action_id) for card in cards)
    )
    proposal = ProposedPlan(
        steps=tuple(ProposedPlanStep(action_ref=card.action.action_id) for card in cards)
    )
    ex = Executor()
    result = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        ex,
        Verifier(True),
        store=store,
    ).run_proposed_plan(intent(), proposal, cards, objective_spec=objective_spec)

    persisted = store.get_objective(result.objective_id)
    assert result.state is ObjectiveState.COMPLETED
    assert persisted is not None
    assert persisted.objective_spec == objective_spec
    assert persisted.validated_plan is not None
    assert persisted.validated_plan.objective_id == persisted.id == result.objective_id
    assert [step.requirement_id for step in persisted.validated_plan.steps] == [
        requirement.requirement_id for requirement in objective_spec.requirements
    ]
    assert all(step.depends_on == () for step in persisted.validated_plan.steps)
    legacy_retry = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        Executor(),
        Verifier(True),
        store=store,
    ).run_sequence(
        intent().model_copy(update={"correlation_id": result.correlation_id}),
        tuple(card.action for card in cards),
    )
    assert legacy_retry.state is ObjectiveState.BLOCKED
    store.close()


def test_kernel_assigns_requirement_ids_to_untrusted_spec_proposals(tmp_path):
    from aegis.store import SqliteObjectiveStore

    store = SqliteObjectiveStore(str(tmp_path / "proposal-identity.sqlite"))
    card = ActionCard(
        action=ActionSpec(
            action_id="tasks.create",
            capability="tasks.write",
            verification=VerificationContract(kind="readback"),
        ),
        summary="Create a task",
        relevance=1,
    )
    spec = ObjectiveSpecProposal(
        requirements=(ObjectiveRequirementProposal(action_ref="tasks.create"),)
    )
    result = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        Executor(),
        Verifier(True),
        store=store,
    ).run_proposed_plan(
        intent(),
        ProposedPlan(steps=(ProposedPlanStep(action_ref="tasks.create"),)),
        (card,),
        objective_spec=spec,
    )

    persisted = store.get_objective(result.objective_id)
    assert persisted is not None and persisted.objective_spec is not None
    assert persisted.objective_spec.requirements[0].requirement_id == uuid5(
        uuid5(result.correlation_id, "objective-completeness"), "objective-requirement:0"
    )


def test_requirement_bound_objective_survives_provider_replacement_without_reinterpretation(
    tmp_path,
):
    from aegis.store import SqliteObjectiveStore

    store = SqliteObjectiveStore(str(tmp_path / "provider-replacement.sqlite"))
    cards = tuple(
        ActionCard(
            action=ActionSpec(
                action_id=action_id,
                capability=action_id,
                verification=VerificationContract(kind="readback"),
            ),
            summary=action_id,
            relevance=1,
        )
        for action_id in ("first", "second")
    )
    objective_spec = ObjectiveSpec(
        requirements=tuple(ObjectiveRequirement(action_ref=card.action.action_id) for card in cards)
    )
    proposal = ProposedPlan(
        steps=tuple(ProposedPlanStep(action_ref=card.action.action_id) for card in cards)
    )
    original = intent()
    first_executor = Executor()
    first = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        first_executor,
        Verifier(True),
        store=store,
    ).run_proposed_plan(original, proposal, cards, objective_spec=objective_spec)

    replacement_executor = Executor()
    replacement = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="replacement provider unused")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        replacement_executor,
        Verifier(True),
        store=store,
    ).run_proposed_plan(
        original.model_copy(update={"utterance": "replacement provider retry"}),
        proposal,
        cards,
        objective_spec=objective_spec,
    )

    persisted = store.get_objective(first.objective_id)
    assert first.state is ObjectiveState.COMPLETED
    assert replacement == first
    assert replacement_executor.calls == 0
    assert persisted is not None
    assert persisted.objective_spec == objective_spec
    assert persisted.validated_plan is not None
    assert [step.requirement_id for step in persisted.validated_plan.steps] == [
        requirement.requirement_id for requirement in objective_spec.requirements
    ]
    store.close()


def test_requirement_bound_plan_stops_when_next_step_is_revoked(tmp_path):
    from aegis.store import SqliteObjectiveStore

    class AllowFirstOnly:
        calls = 0

        def authorize(self, _request):
            self.calls += 1
            return PolicyDecision(
                allowed=self.calls == 1,
                reason="revoked" if self.calls > 1 else "ok",
            )

    store = SqliteObjectiveStore(str(tmp_path / "revoked-objective.sqlite"))
    cards = tuple(
        ActionCard(
            action=ActionSpec(
                action_id=action_id,
                capability=action_id,
                verification=VerificationContract(kind="readback"),
            ),
            summary=action_id,
            relevance=1,
        )
        for action_id in ("first", "second")
    )
    objective_spec = ObjectiveSpec(
        requirements=tuple(ObjectiveRequirement(action_ref=card.action.action_id) for card in cards)
    )
    executor = Executor()
    result = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        AllowFirstOnly(),
        executor,
        Verifier(True),
        store=store,
    ).run_proposed_plan(
        intent(),
        ProposedPlan(
            steps=tuple(ProposedPlanStep(action_ref=card.action.action_id) for card in cards)
        ),
        cards,
        objective_spec=objective_spec,
    )

    assert result.state is ObjectiveState.BLOCKED
    assert executor.calls == 1
    assert len(result.evidence["steps"]) == 2
    assert result.evidence["steps"][0]["state"] == ObjectiveState.COMPLETED.value
    assert result.evidence["steps"][1]["state"] == ObjectiveState.BLOCKED.value


def test_requirement_bound_plan_recovers_after_verified_step_persistence_crash(tmp_path):
    from aegis.store import SqliteObjectiveStore

    class CrashAfterFirstStep(SqliteObjectiveStore):
        crashed = False

        def save_result(self, key, result):
            if key.endswith(":first") and not self.crashed:
                self.crashed = True
                raise RuntimeError("simulated requirement result persistence crash")
            super().save_result(key, result)

    store = CrashAfterFirstStep(str(tmp_path / "requirement-recovery.sqlite"))
    cards = tuple(
        ActionCard(
            action=ActionSpec(
                action_id=action_id,
                capability=action_id,
                verification=VerificationContract(kind="readback"),
            ),
            summary=action_id,
            relevance=1,
        )
        for action_id in ("first", "second")
    )
    objective_spec = ObjectiveSpec(
        requirements=tuple(ObjectiveRequirement(action_ref=card.action.action_id) for card in cards)
    )
    proposal = ProposedPlan(
        steps=tuple(ProposedPlanStep(action_ref=card.action.action_id) for card in cards)
    )
    original_intent = intent()
    first_executor = Executor()
    try:
        Kernel(
            Model(object()),
            Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
            Policy(PolicyDecision(allowed=True, reason="ok")),
            first_executor,
            Verifier(True),
            store=store,
        ).run_proposed_plan(original_intent, proposal, cards, objective_spec=objective_spec)
    except RuntimeError as exc:
        assert "persistence crash" in str(exc)
    else:
        raise AssertionError("simulated requirement persistence crash was not raised")

    retry_executor = Executor()
    recovered = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        retry_executor,
        Verifier(True),
        store=store,
    ).run_proposed_plan(
        original_intent.model_copy(update={"utterance": "resume objective"}),
        proposal,
        cards,
        objective_spec=objective_spec,
    )

    assert recovered.state is ObjectiveState.COMPLETED
    assert first_executor.calls == 1
    assert retry_executor.calls == 1
    store.close()


def test_requirement_bound_dependencies_gate_only_dependent_steps(tmp_path):
    from aegis.store import SqliteObjectiveStore

    class AllowExceptFirst:
        def authorize(self, request):
            allowed = request.action.capability != "first"
            return PolicyDecision(allowed=allowed, reason="ok" if allowed else "revoked")

    store = SqliteObjectiveStore(str(tmp_path / "dependency-readiness.sqlite"))
    cards = tuple(
        ActionCard(
            action=ActionSpec(
                action_id=action_id,
                capability=action_id,
                verification=VerificationContract(kind="readback"),
            ),
            summary=action_id,
            relevance=1,
        )
        for action_id in ("first", "independent", "dependent")
    )
    objective_spec = ObjectiveSpec(
        requirements=tuple(ObjectiveRequirement(action_ref=card.action.action_id) for card in cards)
    )
    proposal = ProposedPlan(
        steps=(
            ProposedPlanStep(action_ref="first"),
            ProposedPlanStep(action_ref="independent"),
            ProposedPlanStep(action_ref="dependent", depends_on=(0,)),
        )
    )
    executor = Executor()
    result = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        AllowExceptFirst(),
        executor,
        Verifier(True),
        store=store,
    ).run_proposed_plan(intent(), proposal, cards, objective_spec=objective_spec)

    assert result.state is ObjectiveState.BLOCKED
    assert executor.calls == 1
    states = [step["state"] for step in result.evidence["steps"]]
    assert states == ["blocked", "completed", "blocked"]
    store.close()


@pytest.mark.skipif(
    not os.environ.get("AEGIS_TEST_DATABASE_URL"), reason="requires disposable PostgreSQL"
)
def test_postgres_persists_requirement_bound_plan_across_store_instance():
    import psycopg

    principal = Principal(id=f"objective-{uuid4()}", vault_id=f"vault-{uuid4()}")
    url = os.environ["AEGIS_TEST_DATABASE_URL"]
    setup = psycopg.connect(url)
    setup.execute(
        "INSERT INTO aegis_principals (id, external_subject) VALUES (%s, %s)",
        (principal.id, principal.id),
    )
    setup.execute(
        "INSERT INTO vaults (id, owner_principal_id) VALUES (%s, %s)",
        (principal.vault_id, principal.id),
    )
    setup.commit()
    setup.close()
    store_connection = psycopg.connect(url)
    card = ActionCard(
        action=ActionSpec(
            action_id="tasks.create",
            capability="tasks.write",
            verification=VerificationContract(kind="readback"),
        ),
        summary="Create a task",
        relevance=1,
    )
    objective_spec = ObjectiveSpec(requirements=(ObjectiveRequirement(action_ref="tasks.create"),))
    request_intent = IntentFrame(principal=principal, utterance="add a task")
    result = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        Executor(),
        Verifier(True),
        store=PostgresObjectiveStore(store_connection),
    ).run_proposed_plan(
        request_intent,
        ProposedPlan(steps=(ProposedPlanStep(action_ref="tasks.create"),)),
        (card,),
        objective_spec=objective_spec,
    )
    read_connection = None
    try:
        read_connection = psycopg.connect(url)
        persisted = PostgresObjectiveStore(read_connection).get_objective(result.objective_id)
        assert result.state is ObjectiveState.COMPLETED
        assert persisted is not None
        assert persisted.objective_spec == objective_spec
        assert persisted.validated_plan is not None
    finally:
        if read_connection is not None:
            read_connection.close()
        store_connection.close()


@pytest.mark.skipif(
    not os.environ.get("AEGIS_TEST_DATABASE_URL"), reason="requires disposable PostgreSQL"
)
def test_postgres_requirement_plan_restart_does_not_replay_verified_step():
    import psycopg

    principal = Principal(id=f"restart-{uuid4()}", vault_id=f"vault-{uuid4()}")
    url = os.environ["AEGIS_TEST_DATABASE_URL"]
    setup = psycopg.connect(url)
    setup.execute(
        "INSERT INTO aegis_principals (id, external_subject) VALUES (%s, %s)",
        (principal.id, principal.id),
    )
    setup.execute(
        "INSERT INTO vaults (id, owner_principal_id) VALUES (%s, %s)",
        (principal.vault_id, principal.id),
    )
    setup.commit()
    setup.close()
    cards = tuple(
        ActionCard(
            action=ActionSpec(
                action_id=action_id,
                capability=action_id,
                verification=VerificationContract(kind="readback"),
            ),
            summary=action_id,
            relevance=1,
        )
        for action_id in ("first", "second")
    )
    objective_spec = ObjectiveSpec(
        requirements=tuple(ObjectiveRequirement(action_ref=card.action.action_id) for card in cards)
    )
    proposal = ProposedPlan(
        steps=tuple(ProposedPlanStep(action_ref=card.action.action_id) for card in cards)
    )
    original_intent = IntentFrame(principal=principal, utterance="do both")

    class CrashStore(PostgresObjectiveStore):
        crashed = False

        def save_result(self, key, result):
            if key.endswith(":first") and not self.crashed:
                self.crashed = True
                raise RuntimeError("simulated PostgreSQL restart")
            super().save_result(key, result)

    first_connection = psycopg.connect(url)
    first_executor = Executor()
    try:
        Kernel(
            Model(object()),
            Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
            Policy(PolicyDecision(allowed=True, reason="ok")),
            first_executor,
            Verifier(True),
            store=CrashStore(first_connection),
        ).run_proposed_plan(original_intent, proposal, cards, objective_spec=objective_spec)
    except RuntimeError as exc:
        assert "restart" in str(exc)
    else:
        raise AssertionError("simulated PostgreSQL restart was not raised")
    finally:
        first_connection.close()

    retry_connection = psycopg.connect(url)
    retry_executor = Executor()
    try:
        recovered = Kernel(
            Model(object()),
            Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
            Policy(PolicyDecision(allowed=True, reason="ok")),
            retry_executor,
            Verifier(True),
            store=PostgresObjectiveStore(retry_connection),
        ).run_proposed_plan(
            original_intent.model_copy(update={"utterance": "resume after restart"}),
            proposal,
            cards,
            objective_spec=objective_spec,
        )
        assert recovered.state is ObjectiveState.COMPLETED
        assert first_executor.calls == 1
        assert retry_executor.calls == 1
    finally:
        retry_connection.close()


@pytest.mark.skipif(
    not os.environ.get("AEGIS_TEST_DATABASE_URL"), reason="requires disposable PostgreSQL"
)
def test_postgres_dependency_readiness_preserves_independent_branch():
    import psycopg

    principal = Principal(id=f"dependency-{uuid4()}", vault_id=f"vault-{uuid4()}")
    url = os.environ["AEGIS_TEST_DATABASE_URL"]
    setup = psycopg.connect(url)
    setup.execute(
        "INSERT INTO aegis_principals (id, external_subject) VALUES (%s, %s)",
        (principal.id, principal.id),
    )
    setup.execute(
        "INSERT INTO vaults (id, owner_principal_id) VALUES (%s, %s)",
        (principal.vault_id, principal.id),
    )
    setup.commit()
    setup.close()

    class AllowExceptFirst:
        def authorize(self, request):
            allowed = request.action.capability != "first"
            return PolicyDecision(allowed=allowed, reason="ok" if allowed else "revoked")

    cards = tuple(
        ActionCard(
            action=ActionSpec(
                action_id=action_id,
                capability=action_id,
                verification=VerificationContract(kind="readback"),
            ),
            summary=action_id,
            relevance=1,
        )
        for action_id in ("first", "independent", "dependent")
    )
    objective_spec = ObjectiveSpec(
        requirements=tuple(ObjectiveRequirement(action_ref=card.action.action_id) for card in cards)
    )
    proposal = ProposedPlan(
        steps=(
            ProposedPlanStep(action_ref="first"),
            ProposedPlanStep(action_ref="independent"),
            ProposedPlanStep(action_ref="dependent", depends_on=(0,)),
        )
    )
    connection = psycopg.connect(url)
    executor = Executor()
    try:
        result = Kernel(
            Model(object()),
            Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
            AllowExceptFirst(),
            executor,
            Verifier(True),
            store=PostgresObjectiveStore(connection),
        ).run_proposed_plan(
            IntentFrame(principal=principal, utterance="prepare everything"),
            proposal,
            cards,
            objective_spec=objective_spec,
        )
    finally:
        connection.close()

    assert result.state is ObjectiveState.BLOCKED
    assert executor.calls == 1
    assert [step["state"] for step in result.evidence["steps"]] == [
        "blocked",
        "completed",
        "blocked",
    ]


def test_kernel_run_sequence_recovers_after_crash_before_aggregate_persistence(tmp_path):
    from aegis.store import SqliteObjectiveStore

    class CrashBeforeAggregateStore(SqliteObjectiveStore):
        fail_aggregate = True

        def save_result(self, key, result):
            if self.fail_aggregate and key.startswith("plan:"):
                raise RuntimeError("simulated crash before plan aggregate")
            super().save_result(key, result)

    store = CrashBeforeAggregateStore(str(tmp_path / "sequence-crash.sqlite"))
    actions = (
        ActionSpec(
            action_id="first",
            capability="first",
            verification=VerificationContract(kind="readback"),
        ),
        ActionSpec(
            action_id="second",
            capability="second",
            verification=VerificationContract(kind="readback"),
        ),
    )
    first_executor = Executor()
    first_intent = intent()
    try:
        Kernel(
            Model(object()),
            Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
            Policy(PolicyDecision(allowed=True, reason="ok")),
            first_executor,
            Verifier(True),
            store=store,
        ).run_sequence(first_intent, actions)
    except RuntimeError as exc:
        assert "aggregate" in str(exc)
    else:
        raise AssertionError("simulated aggregate persistence crash was not raised")

    store.fail_aggregate = False
    replay_executor = Executor()
    recovered = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        replay_executor,
        Verifier(True),
        store=store,
    ).run_sequence(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="resume the plan",
            correlation_id=first_intent.correlation_id,
        ),
        actions,
    )

    assert recovered.state is ObjectiveState.COMPLETED
    assert recovered.retryable is False
    assert len(recovered.evidence["steps"]) == 2
    assert first_executor.calls == 2
    assert replay_executor.calls == 0
    store.close()


def test_kernel_run_sequence_recovers_after_child_observation_persistence_crash(tmp_path):
    from aegis.store import SqliteObjectiveStore

    class CrashOnFirstChildResultStore(SqliteObjectiveStore):
        fail_child_result = True

        def save_result(self, key, result):
            if self.fail_child_result and key.endswith(":first"):
                raise RuntimeError("simulated crash after first child observation")
            super().save_result(key, result)

    store = CrashOnFirstChildResultStore(str(tmp_path / "sequence-child-crash.sqlite"))
    actions = (
        ActionSpec(
            action_id="first",
            capability="first",
            verification=VerificationContract(kind="readback"),
        ),
        ActionSpec(
            action_id="second",
            capability="second",
            verification=VerificationContract(kind="readback"),
        ),
    )
    original_intent = intent()
    first_executor = Executor()
    try:
        Kernel(
            Model(object()),
            Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
            Policy(PolicyDecision(allowed=True, reason="ok")),
            first_executor,
            Verifier(True),
            store=store,
        ).run_sequence(original_intent, actions)
    except RuntimeError as exc:
        assert "child observation" in str(exc)
    else:
        raise AssertionError("simulated child result persistence crash was not raised")

    store.fail_child_result = False
    replay_executor = Executor()
    recovered = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        replay_executor,
        Verifier(True),
        store=store,
    ).run_sequence(
        original_intent.model_copy(update={"utterance": "resume after interruption"}),
        actions,
    )

    assert recovered.state is ObjectiveState.COMPLETED
    assert [step["state"] for step in recovered.evidence["steps"]] == [
        ObjectiveState.COMPLETED.value,
        ObjectiveState.COMPLETED.value,
    ]
    assert first_executor.calls == 1
    assert replay_executor.calls == 1
    store.close()


def test_kernel_run_sequence_reverifies_successful_observation_after_temporary_failure(tmp_path):
    from aegis.store import SqliteObjectiveStore

    class ToggleVerifier:
        def __init__(self, verified):
            self.verified = verified

        def verify(self, observation, contract):
            return VerificationResult(
                verified=self.verified,
                evidence={"readback": self.verified},
                reason="verified" if self.verified else "postcondition failed",
            )

    store = SqliteObjectiveStore(str(tmp_path / "sequence-verification-retry.sqlite"))
    actions = (
        ActionSpec(
            action_id="first",
            capability="first",
            verification=VerificationContract(kind="readback"),
        ),
        ActionSpec(
            action_id="second",
            capability="second",
            verification=VerificationContract(kind="readback"),
        ),
    )
    original_intent = intent()
    first_executor = Executor()
    failed = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        first_executor,
        ToggleVerifier(False),
        store=store,
    ).run_sequence(original_intent, actions)

    retry_executor = Executor()
    recovered = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        retry_executor,
        ToggleVerifier(True),
        store=store,
    ).run_sequence(
        original_intent.model_copy(update={"utterance": "retry after access was restored"}),
        actions,
    )

    assert failed.state is ObjectiveState.FAILED
    assert recovered.state is ObjectiveState.COMPLETED
    assert [step["state"] for step in recovered.evidence["steps"]] == [
        ObjectiveState.COMPLETED.value,
        ObjectiveState.COMPLETED.value,
    ]
    assert first_executor.calls == 1
    assert retry_executor.calls == 1
    store.close()


def test_kernel_run_sequence_authorizes_each_step_independently():
    class SecondStepDeniedPolicy:
        def __init__(self):
            self.calls = 0

        def authorize(self, request):
            self.calls += 1
            return PolicyDecision(
                allowed=self.calls == 1,
                reason="second step denied" if self.calls == 2 else "ok",
            )

    actions = (
        ActionSpec(
            action_id="first",
            capability="first",
            verification=VerificationContract(kind="readback"),
        ),
        ActionSpec(
            action_id="second",
            capability="second",
            verification=VerificationContract(kind="readback"),
        ),
    )
    ex = Executor()
    result = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        SecondStepDeniedPolicy(),
        ex,
        Verifier(True),
    ).run_sequence(intent(), actions)

    assert result.state is ObjectiveState.BLOCKED
    assert "step 2 of 2" in result.message
    assert result.evidence["steps"][0]["state"] == ObjectiveState.COMPLETED.value
    assert result.evidence["steps"][1]["state"] == ObjectiveState.BLOCKED.value
    assert ex.calls == 1


def test_kernel_run_sequence_explicit_retry_resumes_after_step_authorization_denial(tmp_path):
    from aegis.store import SqliteObjectiveStore

    class Policy:
        def __init__(self, allow_second):
            self.allow_second = allow_second
            self.calls = 0

        def authorize(self, request):
            self.calls += 1
            return PolicyDecision(
                allowed=self.allow_second or self.calls == 1,
                reason="second step denied" if not self.allow_second else "ok",
            )

    store = SqliteObjectiveStore(str(tmp_path / "sequence-denial-retry.sqlite"))
    actions = (
        ActionSpec(
            action_id="first",
            capability="first",
            verification=VerificationContract(kind="readback"),
        ),
        ActionSpec(
            action_id="second",
            capability="second",
            verification=VerificationContract(kind="readback"),
        ),
    )
    original_intent = intent()
    first_executor = Executor()
    denied = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        Policy(False),
        first_executor,
        Verifier(True),
        store=store,
    ).run_sequence(original_intent, actions)

    retry_executor = Executor()
    recovered = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        Policy(True),
        retry_executor,
        Verifier(True),
        store=store,
    ).run_sequence(
        original_intent.model_copy(update={"utterance": "retry after authorization changed"}),
        actions,
    )

    assert denied.state is ObjectiveState.BLOCKED
    assert recovered.state is ObjectiveState.COMPLETED
    assert first_executor.calls == 1
    assert retry_executor.calls == 1
    store.close()


def test_kernel_run_sequence_does_not_replay_plan_result_to_wrong_principal(tmp_path):
    from aegis.store import SqliteObjectiveStore

    store = SqliteObjectiveStore(str(tmp_path / "private-sequence.sqlite"))
    actions = (
        ActionSpec(
            action_id="private",
            capability="private",
            verification=VerificationContract(kind="readback"),
        ),
    )
    owner_intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance="do private work",
        correlation_id=uuid4(),
    )
    owner_result = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        Executor(),
        Verifier(True),
        store=store,
    ).run_sequence(owner_intent, actions)

    other_result = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        Executor(),
        Verifier(True),
        store=store,
    ).run_sequence(
        IntentFrame(
            principal=Principal(id="bob", vault_id="bob-vault"),
            utterance="do private work",
            correlation_id=owner_intent.correlation_id,
        ),
        actions,
    )

    assert owner_result.state is ObjectiveState.COMPLETED
    assert other_result.state is ObjectiveState.BLOCKED
    assert other_result.objective_id != owner_result.objective_id
    assert "unavailable" in other_result.message
    store.close()


def test_kernel_run_sequence_rejects_same_principal_plan_rebinding(tmp_path):
    from aegis.store import SqliteObjectiveStore

    store = SqliteObjectiveStore(str(tmp_path / "bound-sequence.sqlite"))
    first_action = ActionSpec(
        action_id="first",
        capability="first",
        verification=VerificationContract(kind="readback"),
    )
    second_action = ActionSpec(
        action_id="second",
        capability="second",
        verification=VerificationContract(kind="readback"),
    )
    original_intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance="do the first thing",
        correlation_id=uuid4(),
    )
    kernel = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.BLOCKED, reason="unused")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        Executor(),
        Verifier(True),
        store=store,
    )
    original = kernel.run_sequence(original_intent, (first_action,))
    rebound = kernel.run_sequence(
        original_intent.model_copy(update={"utterance": "do the second thing"}),
        (second_action,),
    )

    assert original.state is ObjectiveState.COMPLETED
    assert rebound.state is ObjectiveState.BLOCKED
    assert rebound.objective_id == original.objective_id
    assert "different action sequence" in rebound.message
    store.close()


def test_policy_denial_cannot_be_bypassed_by_model_action():
    ex = Executor()
    action = ActionSpec(
        action_id="restart",
        capability="homelab.restart",
        verification=VerificationContract(kind="health"),
    )
    k = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.ACTION, action=action)),
        Policy(PolicyDecision(allowed=False, reason="not authorized")),
        ex,
        Verifier(True),
    )
    assert k.run(intent()).state.value == "blocked"
    assert ex.calls == 0


def test_successful_command_without_verification_is_not_success():
    ex = Executor()
    action = ActionSpec(action_id="write", capability="test.write")
    k = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.ACTION, action=action)),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        ex,
        Verifier(True),
    )
    assert k.run(intent()).state.value == "failed"
    assert [event.event_type for event in k.audit.events] == [
        "objective.created",
        "action.observed",
        "result.failed",
    ]


def test_failed_postcondition_is_failed():
    action = ActionSpec(
        action_id="restart",
        capability="test.restart",
        verification=VerificationContract(kind="health"),
    )
    k = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.ACTION, action=action)),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        Executor(),
        Verifier(False),
    )
    result = k.run(intent())
    assert result.state.value == "failed"
    assert result.retryable is True


def test_executor_exception_is_persisted_as_unknown_and_not_replayed():
    class FailingExecutor:
        calls = 0

        def execute(self, request):
            self.calls += 1
            raise RuntimeError("private gateway detail")

    action = ActionSpec(
        action_id="write",
        capability="test.write",
        verification=VerificationContract(kind="readback"),
    )
    executor = FailingExecutor()
    k = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.ACTION, action=action)),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        executor,
        Verifier(True),
    )

    first = k.run(intent())
    second = k.run(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="do it",
            correlation_id=first.correlation_id,
        )
    )

    assert first.state is ObjectiveState.OBSERVED
    assert first.retryable is False
    assert second == first
    assert executor.calls == 1
    assert first.evidence["outcome"] == "unknown"
    assert first.evidence["assurance"] == "OUTCOME_UNKNOWN"
    assert first.evidence["error_type"] == "RuntimeError"
    assert "private gateway detail" not in str(first)
    assert [event.event_type for event in k.audit.events][-1] == "result.observed"
    assert "result.failed" not in [event.event_type for event in k.audit.events]


def test_unknown_outcome_reconciles_without_reexecuting_mutation():
    class AmbiguousExecutor:
        calls = 0

        def execute(self, _request):
            self.calls += 1
            raise TimeoutError("provider stopped responding")

    class Reconciler:
        def verify(self, _observation, _contract):
            return VerificationResult(verified=False, evidence={}, reason="still unknown")

        def reconcile(self, _observation, _contract):
            return VerificationResult(
                verified=True,
                evidence={"provider_reconciliation": True},
                reason="provider state confirmed the expected effect",
            )

    executor = AmbiguousExecutor()
    verifier = Reconciler()
    action = ActionSpec(
        action_id="remote.write",
        capability="remote.write",
        verification=VerificationContract(kind="custom", expected={"effect": "on"}),
    )
    k = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.ACTION, action=action)),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        executor,
        verifier,
    )

    first = k.run(intent())
    second = k.run(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="do it",
            correlation_id=first.correlation_id,
        )
    )

    assert first.state is ObjectiveState.OBSERVED
    assert second.state is ObjectiveState.COMPLETED
    assert second.evidence["reconciled"] is True
    assert executor.calls == 1


def test_typed_unknown_assurance_is_observed_even_without_evidence_marker():
    class ProviderOutcomeExecutor:
        def execute(self, request):
            return Observation(
                execution_id=request.action_id,
                evidence={"provider": "fault-fixture"},
                command_succeeded=False,
                assurance=ExternalEffectAssurance.OUTCOME_UNKNOWN,
            )

    action = ActionSpec(
        action_id="remote.write",
        capability="remote.write",
        verification=VerificationContract(kind="custom", expected={"effect": "on"}),
    )
    result = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.ACTION, action=action)),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        ProviderOutcomeExecutor(),
        Verifier(False),
    ).run(intent())
    assert result.state is ObjectiveState.OBSERVED
    assert result.retryable is False
    assert result.evidence["assurance"] == "OUTCOME_UNKNOWN"


def test_verifier_exception_is_a_truthful_failed_result():
    class FailingVerifier:
        def verify(self, observation, contract):
            raise RuntimeError("private readback detail")

    action = ActionSpec(
        action_id="write",
        capability="test.write",
        verification=VerificationContract(kind="readback"),
    )
    k = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.ACTION, action=action)),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        Executor(),
        FailingVerifier(),
    )

    result = k.run(intent())

    assert result.state is ObjectiveState.FAILED
    assert (
        result.message == "verification unavailable; external state must be checked before retrying"
    )
    assert result.evidence["verification"] == "unavailable"
    assert result.evidence["error_type"] == "RuntimeError"
    assert "private readback detail" not in str(result)


def test_replay_does_not_execute_side_effect_twice():
    ex = Executor()
    action = ActionSpec(
        action_id="write",
        capability="test.write",
        verification=VerificationContract(kind="readback"),
    )
    k = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.ACTION, action=action)),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        ex,
        Verifier(True),
    )
    first = k.run(intent())
    second = k.run(
        first
        and IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="do it",
            correlation_id=first.correlation_id,
        )
    )
    assert first.state.value == "completed"
    assert second.state.value == "completed"
    assert ex.calls == 1


def test_restart_reuses_persisted_action_after_crash_before_result(tmp_path):
    store = SqliteObjectiveStore(str(tmp_path / "objective.sqlite"))
    original_intent = intent()
    action = ActionSpec(
        action_id="write",
        capability="test.write",
        verification=VerificationContract(kind="readback"),
    )

    class CrashBeforeResultStore:
        def __init__(self):
            self.failed = False

        def save_result(self, key, result):
            if not self.failed:
                self.failed = True
                raise RuntimeError("simulated process interruption")
            store.save_result(key, result)

        def __getattr__(self, name):
            return getattr(store, name)

    first_store = CrashBeforeResultStore()
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        Kernel(
            Model(object()),
            Decoder(Decision(kind=DecisionKind.ACTION, action=action)),
            Policy(PolicyDecision(allowed=True, reason="ok")),
            Executor(),
            Verifier(True),
            store=first_store,
        ).run(original_intent)

    class MustNotPlan:
        def decide(self, _request):
            raise AssertionError("recovery must not invoke the model")

    recovered = Kernel(
        MustNotPlan(),
        Decoder(Decision(kind=DecisionKind.CLARIFY, clarification="unused")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        Executor(),
        Verifier(True),
        store=store,
    ).run(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="different wording after restart",
            correlation_id=original_intent.correlation_id,
        )
    )

    assert recovered.state is ObjectiveState.COMPLETED
    assert recovered.message == "verified"


def test_registry_returns_at_most_five_relevant_cards():
    cards = tuple(
        ActionCard(
            action=ActionSpec(action_id=f"homelab-{i}", capability="homelab.read"),
            summary=str(i),
            relevance=i / 10,
        )
        for i in range(7)
    )
    assert len(CapabilityRegistry(cards).retrieve("homelab")) == 5


def test_decoder_rejects_invented_action():
    card = ActionCard(
        action=ActionSpec(action_id="safe", capability="test.safe"),
        summary="safe",
        relevance=1,
    )
    response = {"kind": "ACTION", "action": {"action_id": "delete", "capability": "danger"}}
    try:
        StrictDecisionDecoder().decode(type("Response", (), {"raw": response})(), (card,))
    except InvalidDecision:
        pass
    else:
        raise AssertionError("invented action was accepted")


def test_decoder_requires_text_for_clarification():
    response = {"kind": "CLARIFY", "clarification": "   "}
    try:
        StrictDecisionDecoder().decode(type("Response", (), {"raw": response})(), ())
    except InvalidDecision:
        pass
    else:
        raise AssertionError("empty clarification was accepted")


def test_decoder_rejects_semantic_mode_that_contradicts_decision_kind():
    action = {
        "kind": "ACTION",
        "semantic_mode": "READ",
        "action_ref": "safe",
        "action_arguments": {},
    }
    answer = {"kind": "ANSWER", "semantic_mode": "ACTION", "answer": "ok"}
    clarify = {"kind": "CLARIFY", "semantic_mode": "GENERATION", "clarification": "Which?"}
    card = ActionCard(
        action=ActionSpec(action_id="safe", capability="test.safe"),
        summary="safe",
        relevance=1,
    )
    for raw in (action, answer, clarify):
        try:
            StrictDecisionDecoder().decode(
                type("Response", (), {"raw": raw})(),
                (card,),
                allow_argument_proposals=True,
            )
        except InvalidDecision:
            continue
        raise AssertionError("contradictory semantic mode was accepted")


def test_decoder_accepts_bounded_answer_context_focus_but_not_action_focus():
    answer = StrictDecisionDecoder().decode(
        type(
            "Response",
            (),
            {
                "raw": {
                    "kind": "ANSWER",
                    "semantic_mode": "READ",
                    "answer": "The open tasks are listed above.",
                    "context_focus": "canonical_tasks",
                }
            },
        )(),
        (),
    )
    assert answer.context_focus == "canonical_tasks"

    try:
        StrictDecisionDecoder().decode(
            type(
                "Response",
                (),
                {
                    "raw": {
                        "kind": "ACTION",
                        "action_ref": "tasks.complete",
                        "action_arguments": {"title": "review backups"},
                        "context_focus": "canonical_tasks",
                    }
                },
            )(),
            (),
            allow_argument_proposals=True,
        )
    except InvalidDecision:
        pass
    else:
        raise AssertionError("action proposal accepted conversational context metadata")


def test_decoder_rejects_empty_answer_content():
    response = {"kind": "ANSWER", "reason": "model explanation without an answer"}
    try:
        StrictDecisionDecoder().decode(type("Response", (), {"raw": response})(), ())
    except InvalidDecision:
        pass
    else:
        raise AssertionError("empty answer was accepted")


def test_decoder_canonicalizes_single_card_copy_errors_without_accepting_invented_actions():
    card = ActionCard(
        action=ActionSpec(
            action_id="tasks.create",
            capability="tasks.write",
            arguments={"title": "canonical title"},
            verification=VerificationContract(kind="readback"),
        ),
        summary="create task",
        relevance=1,
    )
    malformed_copy = {
        "kind": "ACTION",
        "action": {"action_id": "tasks.create", "capability": "wrong"},
    }

    decision = StrictDecisionDecoder().decode(
        type("Response", (), {"raw": malformed_copy})(), (card,)
    )

    assert decision.action == card.action


def test_decoder_accepts_only_declared_bounded_action_arguments():
    card = ActionCard(
        action=ActionSpec(
            action_id="tasks.complete",
            capability="tasks.complete",
            required_permissions=("tasks.write",),
            verification=VerificationContract(kind="readback"),
        ),
        summary="complete a task",
        relevance=1,
        argument_keys=("title",),
    )
    response = {
        "kind": "ACTION",
        "action": {
            "action_id": "tasks.complete",
            "capability": "tasks.complete",
            "arguments": {"title": "get gud scrub"},
            "required_permissions": ["tasks.write"],
            "verification": {"kind": "readback", "expected": {}},
        },
    }
    decision = StrictDecisionDecoder().decode(
        type("Response", (), {"raw": response})(),
        (card,),
        allow_argument_proposals=True,
    )
    assert decision.action == card.action.model_copy(
        update={"arguments": {"title": "get gud scrub"}}
    )

    unsafe = dict(response)
    unsafe["action"] = dict(response["action"])
    unsafe["action"]["arguments"] = {"title": "get gud scrub", "permission": "owner"}
    try:
        StrictDecisionDecoder().decode(
            type("Response", (), {"raw": unsafe})(),
            (card,),
            allow_argument_proposals=True,
        )
    except InvalidDecision:
        pass
    else:
        raise AssertionError("undeclared model argument was accepted")


def test_decoder_expands_bounded_action_reference_into_canonical_card():
    card = ActionCard(
        action=ActionSpec(
            action_id="tasks.complete",
            capability="tasks.complete",
            required_permissions=("tasks.write",),
            verification=VerificationContract(kind="readback"),
        ),
        summary="complete a task",
        relevance=1,
        argument_keys=("title",),
    )
    response = {
        "kind": "ACTION",
        "action_ref": "tasks.complete",
        "action_arguments": {"title": "review backups"},
    }

    decision = StrictDecisionDecoder().decode(
        type("Response", (), {"raw": response})(),
        (card,),
        allow_argument_proposals=True,
    )

    assert decision.action == card.action.model_copy(
        update={"arguments": {"title": "review backups"}}
    )


def test_decoder_rejects_unknown_bounded_action_reference():
    card = ActionCard(
        action=ActionSpec(action_id="tasks.list", capability="tasks.read"),
        summary="show tasks",
        relevance=1,
    )
    response = {"kind": "ACTION", "action_ref": "tasks.delete"}

    try:
        StrictDecisionDecoder().decode(
            type("Response", (), {"raw": response})(), (card,), allow_argument_proposals=True
        )
    except InvalidDecision:
        pass
    else:
        raise AssertionError("unknown action reference was accepted")


def test_kernel_blocks_malformed_model_without_execution():
    ex = Executor()
    k = Kernel(
        Model(type("Response", (), {"raw": "not-json"})()),
        StrictDecisionDecoder(),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        ex,
        Verifier(True),
    )
    result = k.run(intent())
    assert result.state.value == "blocked"
    assert ex.calls == 0


def test_model_exception_becomes_terminal_retryable_result():
    class FailingModel:
        def decide(self, request):
            raise RuntimeError("private provider detail")

    k = Kernel(
        FailingModel(),
        Decoder(Decision(kind=DecisionKind.ANSWER, answer="unused")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        Executor(),
        Verifier(True),
    )

    result = k.run(intent())

    assert result.state is ObjectiveState.FAILED
    assert result.message == "Model unavailable; request can be retried"
    assert result.evidence == {"model": "unavailable", "error_type": "RuntimeError"}
    assert "private provider detail" not in str(result)
    assert k.store.get_objective(result.objective_id).state is ObjectiveState.FAILED
    assert [event.event_type for event in k.audit.events] == [
        "objective.created",
        "model.failed",
    ]


def test_same_correlation_can_retry_after_pre_decision_model_failure():
    class SwitchableModel:
        def __init__(self):
            self.failure = True

        def decide(self, request):
            if self.failure:
                raise RuntimeError("provider unavailable")
            return object()

    action = ActionSpec(
        action_id="write",
        capability="test.write",
        verification=VerificationContract(kind="readback"),
    )
    model = SwitchableModel()
    store = SqliteObjectiveStore(":memory:")
    first_kernel = Kernel(
        model,
        Decoder(Decision(kind=DecisionKind.ACTION, action=action)),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        Executor(),
        Verifier(True),
        store=store,
    )
    correlation_id = uuid4()
    first = first_kernel.run(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="do it",
            correlation_id=correlation_id,
        )
    )
    model.failure = False
    second = first_kernel.run(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="do it",
            correlation_id=correlation_id,
        )
    )

    assert first.state is ObjectiveState.FAILED
    assert second.state is ObjectiveState.COMPLETED
    store.close()


def test_deterministic_fast_path_bypasses_model():
    class ExplodingModel:
        def decide(self, request):
            raise AssertionError("model should not be called")

    expected = Result(
        objective_id=uuid4(),
        state="completed",
        message="three tasks",
        correlation_id=uuid4(),
    )
    k = Kernel(
        ExplodingModel(),
        Decoder(Decision(kind=DecisionKind.ANSWER, answer="unused")),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        Executor(),
        Verifier(True),
        fast_path=FastPath(expected),
    )
    assert k.run(intent()) == expected


def test_three_reference_packs_use_one_generic_execution_pipeline():
    world = ReferenceWorld()
    executor = ReferenceExecutor(world)
    verifier = ReferenceVerifier(world)
    principal = Principal(id="alice", vault_id="alice-vault")
    representative_pack_ids = ("tasks", "kitchen", "homelab")
    for pack in tuple(
        pack for pack in reference_packs() if pack.manifest.pack_id in representative_pack_ids
    ):
        card = pack.cards[0]
        action = card.action.model_copy(
            update={
                "arguments": {"title": "Call landlord"}
                if pack.manifest.pack_id == "tasks"
                else {"item": "rice"}
                if pack.manifest.pack_id == "kitchen"
                else {"service": "test-service"}
            }
        )
        decision = Decision(kind=DecisionKind.ACTION, action=action)
        kernel = Kernel(
            Model(object()),
            Decoder(decision),
            Policy(PolicyDecision(allowed=True, reason="scope permits action")),
            executor,
            verifier,
        )
        result = kernel.run(
            IntentFrame(principal=principal, utterance=pack.manifest.pack_id),
            (ActionCard(action=action, summary=card.summary, relevance=card.relevance),),
        )
        assert result.state.value == "completed"
    assert world.tasks == [{"title": "Call landlord", "status": "open"}]
    assert world.groceries == ["rice"]
    assert world.services["test-service"] == "healthy"


def test_openclaw_runtime_deny_prevents_gateway_call():
    class Client:
        def __init__(self):
            self.calls = 0

        def execute(self, request):
            self.calls += 1
            raise AssertionError("denied request crossed Gateway boundary")

    class DenyRuntime:
        def allows(self, request):
            return False

    class NoApproval:
        def required(self, request):
            return False

        def approved(self, request):
            return False

    client = Client()
    request = ExecutionRequest(
        objective_id=uuid4(),
        action_id=uuid4(),
        action=ActionSpec(action_id="safe", capability="safe"),
        idempotency_key="k",
    )
    observation = OpenClawExecutor(client, DenyRuntime(), NoApproval()).execute(request)
    assert not observation.command_succeeded
    assert client.calls == 0


def test_gateway_reconnect_reuses_same_idempotency_request_when_safe():
    class Transport:
        def __init__(self):
            self.calls = 0
            self.reconnected = False
            self.requests = []

        def execute(self, request):
            self.calls += 1
            self.requests.append(request.idempotency_key)
            if self.calls == 1:
                raise GatewayDisconnected()
            return Observation(
                execution_id=request.action_id, evidence={"verified": True}, command_succeeded=True
            )

        def retry_is_safe(self, request):
            return True

        def reconnect(self):
            self.reconnected = True

        def cancel(self, request):
            raise AssertionError("unexpected cancellation")

    transport = Transport()
    request = ExecutionRequest(
        objective_id=uuid4(),
        action_id=uuid4(),
        action=ActionSpec(action_id="safe", capability="safe"),
        idempotency_key="stable-key",
    )
    observation = ReconnectingGatewayClient(transport).execute(request)
    assert observation.command_succeeded
    assert transport.reconnected
    assert transport.requests == ["stable-key", "stable-key"]


def test_gateway_disconnect_with_unknown_outcome_is_not_retried():
    class Transport:
        def __init__(self):
            self.calls = 0

        def execute(self, request):
            self.calls += 1
            raise GatewayDisconnected()

        def retry_is_safe(self, request):
            return False

        def reconnect(self):
            raise AssertionError("unknown outcome must not reconnect and replay")

        def cancel(self, request):
            raise AssertionError("unexpected cancellation")

    request = ExecutionRequest(
        objective_id=uuid4(),
        action_id=uuid4(),
        action=ActionSpec(action_id="unsafe", capability="unsafe"),
        idempotency_key="unknown-key",
    )
    observation = ReconnectingGatewayClient(Transport()).execute(request)
    assert observation.evidence["outcome"] == "unknown"


def test_gateway_cancel_is_explicit_and_never_executes():
    class Transport:
        def __init__(self):
            self.cancelled = False

        def execute(self, request):
            raise AssertionError("cancelled request was executed")

        def retry_is_safe(self, request):
            return False

        def reconnect(self):
            raise AssertionError("unexpected reconnect")

        def cancel(self, request):
            self.cancelled = True

    transport = Transport()
    request = ExecutionRequest(
        objective_id=uuid4(),
        action_id=uuid4(),
        action=ActionSpec(action_id="safe", capability="safe"),
        idempotency_key="cancel-key",
    )
    event = Event()
    event.set()
    observation = ReconnectingGatewayClient(transport).execute(request, event)
    assert observation.evidence["transport"] == "cancelled"
    assert transport.cancelled


def test_postgres_request_status_prefers_persisted_result_for_replayed_correlation():
    correlation = uuid4()
    objective_id = uuid4()

    class Cursor:
        def fetchone(self):
            return (str(objective_id), "completed", "canonical readback verified", False)

    class Connection:
        def __init__(self):
            self.params = None

        def execute(self, _query, params=()):
            self.params = params
            return Cursor()

    connection = Connection()
    status = PostgresObjectiveStore(connection).get_request_status(
        correlation,
        Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )

    assert status == (objective_id, ObjectiveState.COMPLETED, "canonical readback verified", False)
    assert connection.params == (
        str(correlation),
        "alice",
        "alice-vault",
        str(correlation),
        "alice",
    )


def test_postgres_result_replay_does_not_leak_storage_retry_marker():
    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="verified",
        evidence={"canonical": True},
        correlation_id=uuid4(),
        retryable=False,
    )

    class Cursor:
        def fetchone(self):
            return (
                str(result.correlation_id),
                str(result.objective_id),
                result.state.value,
                {"canonical": True, "retryable": False},
                result.message,
            )

    class Connection:
        def execute(self, query, params=()):
            assert "FROM results" in query
            return Cursor()

    replayed = PostgresObjectiveStore(Connection()).get_result("correlation:write")

    assert replayed == result
    assert "retryable" not in replayed.evidence


def test_sqlite_store_survives_kernel_restart_without_duplicate_side_effect(tmp_path):
    store = SqliteObjectiveStore(str(tmp_path / "aegis.sqlite"))
    world = ReferenceWorld()
    action = ActionSpec(
        action_id="kitchen.groceries.add",
        capability="kitchen.groceries.write",
        arguments={"item": "rice"},
        verification=VerificationContract(kind="readback"),
    )
    correlation = uuid4()
    first_kernel = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.ACTION, action=action)),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        ReferenceExecutor(world),
        ReferenceVerifier(world),
        store=store,
    )
    first = first_kernel.run(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="add rice",
            correlation_id=correlation,
        ),
        (ActionCard(action=action, summary="add", relevance=1),),
    )
    second_kernel = Kernel(
        Model(object()),
        Decoder(Decision(kind=DecisionKind.ACTION, action=action)),
        Policy(PolicyDecision(allowed=True, reason="ok")),
        ReferenceExecutor(world),
        ReferenceVerifier(world),
        store=store,
    )
    second = second_kernel.run(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="add rice",
            correlation_id=correlation,
        ),
        (ActionCard(action=action, summary="add", relevance=1),),
    )
    assert first.state.value == "completed"
    assert second.state.value == "completed"
    assert world.groceries == ["rice"]
    store.close()


def test_gateway_rpc_requires_matching_response_correlation():
    class Channel:
        def send(self, request):
            return RpcResponse(request_id=uuid4(), result={"ok": True})

    try:
        CorrelatedRpcClient(Channel()).call("agent", {"runId": "run-1"})
    except RpcProtocolError as exc:
        assert "correlation" in str(exc)
    else:
        raise AssertionError("mismatched Gateway response was accepted")


def test_gateway_rpc_hides_remote_error_text():
    class Channel:
        def send(self, request):
            return RpcResponse(
                request_id=request.request_id,
                error={"code": "permission_denied", "message": "gateway-password=secret"},
            )

    try:
        CorrelatedRpcClient(Channel()).call("agent")
    except RpcProtocolError as exc:
        assert str(exc) == "Gateway RPC failed: permission_denied"
        assert "secret" not in str(exc)
    else:
        raise AssertionError("Gateway error was accepted")


def test_gateway_transport_error_hides_exception_text():
    class Socket:
        def recv(self):
            raise RuntimeError("gateway-password=secret")

    try:
        OpenClawWebSocketChannel("ws://gateway", "token")._receive_response(Socket(), "request-1")
    except RpcProtocolError as exc:
        assert str(exc) == "Gateway transport failed: RuntimeError"
        assert "secret" not in str(exc)
    else:
        raise AssertionError("Gateway transport error was accepted")


def test_gateway_rpc_named_methods_preserve_documented_method_names():
    class Channel:
        def __init__(self):
            self.methods = []

        def send(self, request):
            self.methods.append(request.method)
            return RpcResponse(request_id=request.request_id, result={"status": "accepted"})

    channel = Channel()
    rpc = OpenClawGatewayRpc(CorrelatedRpcClient(channel))
    assert rpc.agent({"runId": "r"})["status"] == "accepted"
    assert rpc.agent_wait({"runId": "r"})["status"] == "accepted"
    assert rpc.cancel({"runId": "r"})["status"] == "accepted"
    assert channel.methods == ["agent", "agent.wait", "agent.cancel"]


def test_gateway_channel_buffers_events_seen_while_waiting_for_response():
    class Socket:
        def __init__(self):
            self.frames = iter(
                [
                    '{"type":"event","event":"terminal.data","payload":{"data":"marker"}}',
                    '{"type":"res","id":"request-1","ok":true,"payload":{"ok":true}}',
                ]
            )

        def recv(self):
            return next(self.frames)

    channel = OpenClawWebSocketChannel("ws://gateway", "token", persistent=True)
    socket = Socket()
    assert channel._receive_response(socket, "request-1")["ok"] is True
    channel._socket = socket
    assert channel.receive_event("terminal.data")["data"] == "marker"


def test_persistent_gateway_event_failure_discards_dead_socket():
    class Socket:
        def recv(self):
            raise RuntimeError("connection reset")

        def close(self):
            pass

    channel = OpenClawWebSocketChannel("ws://gateway", "token", persistent=True)
    channel._socket = Socket()

    try:
        channel.receive_event("terminal.data")
    except RpcProtocolError as exc:
        assert str(exc) == "Gateway event transport failed: RuntimeError"
    else:
        raise AssertionError("Gateway event failure was accepted")
    assert channel._socket is None


def test_discarding_gateway_socket_drops_connection_local_events():
    channel = OpenClawWebSocketChannel("ws://gateway", "token", persistent=True)
    channel._pending_events.append(
        {"type": "event", "event": "terminal.data", "payload": {"data": "stale"}}
    )

    channel._discard_socket()

    assert channel._pending_events == []


def test_closing_gateway_socket_drops_connection_local_events():
    channel = OpenClawWebSocketChannel("ws://gateway", "token", persistent=True)
    channel._pending_events.append(
        {"type": "event", "event": "terminal.data", "payload": {"data": "stale"}}
    )

    channel.close()

    assert channel._pending_events == []


def test_gateway_close_discards_socket_when_socket_close_fails():
    class Socket:
        def close(self):
            raise RuntimeError("close failed")

    channel = OpenClawWebSocketChannel("ws://gateway", "token", persistent=True)
    channel._socket = Socket()

    channel.close()

    assert channel._socket is None


@pytest.mark.parametrize(
    "executor_factory, action",
    [
        (
            lambda channel: OpenClawGroceryExecutor(channel, "/tmp/aegis-groceries"),
            ActionSpec(
                action_id="kitchen.groceries.add",
                capability="kitchen.groceries.write",
                arguments={"item": "rice"},
            ),
        ),
        (
            lambda channel: OpenClawHomelabExecutor(channel, {"api": "true"}),
            ActionSpec(
                action_id="homelab.service.restart",
                capability="homelab.write",
                arguments={"service": "api"},
            ),
        ),
        (
            lambda channel: OpenClawNetworkProbeExecutor(channel),
            ActionSpec(
                action_id="network.probe",
                capability="network.read",
                arguments={"address": "127.0.0.1", "port": 80},
            ),
        ),
    ],
)
def test_gateway_protocol_failure_becomes_unknown_observation(executor_factory, action):
    class Channel:
        persistent = True

        def send(self, request):
            raise RpcProtocolError("Gateway RPC failed: gateway_error")

        def receive_event(self, event_name):
            raise AssertionError("protocol failure should stop before event read")

    request = ExecutionRequest(
        objective_id=uuid4(),
        action_id=uuid4(),
        action=action,
        idempotency_key="stable-ambiguous-key",
    )
    observation = executor_factory(Channel()).execute(request)

    assert not observation.command_succeeded
    assert observation.evidence == {
        "gateway": "openclaw",
        "outcome": "unknown",
        "idempotency_key": "stable-ambiguous-key",
    }


def test_vault_and_space_authorization_is_structural_and_revocable():
    auth = InMemoryAuthorization()
    auth.add_vault(Vault("alice-vault", "alice"))
    auth.add_vault(Vault("bob-vault", "bob"))
    auth.add_space(Space("apartment", "Apartment"))
    auth.add_membership(Membership("bob", "apartment", Role.MEMBER))
    auth.add_resource(Resource("alice-bank", "alice-vault"))
    auth.add_resource(Resource("rent", "alice-vault", "apartment"))
    alice = Principal(id="alice", vault_id="alice-vault")
    bob = Principal(id="bob", vault_id="bob-vault")
    assert auth.can_read(alice, "alice-bank").allowed
    assert not auth.can_read(bob, "alice-bank").allowed
    assert auth.can_read(bob, "rent").allowed
    auth.revoke("bob", "apartment")
    assert not auth.can_read(bob, "rent").allowed


def test_keycloak_claim_mapping_requires_explicit_aegis_scope_claims():
    principal = KeycloakIdentityProvider().principal_from_claims(
        {"sub": "alice", "aegis_vault_id": "alice-vault", "aegis_space_ids": ["apartment"]}
    )
    assert principal.id == "alice"
    assert principal.space_ids == ("apartment",)
    for claims in ({"sub": "alice"}, {"aegis_vault_id": "vault"}):
        try:
            KeycloakIdentityProvider().principal_from_claims(claims)
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete identity claims were accepted")


def test_keycloak_oidc_client_maps_userinfo_and_fails_closed(monkeypatch):
    import json

    import aegis.identity as identity_module

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return json.dumps(
                {
                    "sub": "alice",
                    "aegis_vault_id": "alice-vault",
                    "aegis_space_ids": ["apartment"],
                }
            ).encode()

    def fake_urlopen(request, timeout):
        assert request.full_url.endswith("/realms/aegis/protocol/openid-connect/userinfo")
        assert request.get_header("Authorization") == "Bearer token"
        assert timeout == 3.0
        return Response()

    monkeypatch.setattr(identity_module, "urlopen", fake_urlopen)
    client = KeycloakOIDCClient("http://keycloak/realms/aegis", timeout=3.0)
    principal = client.principal_from_access_token("token")
    assert principal == Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    try:
        client.principal_from_access_token("")
    except ValueError:
        pass
    else:
        raise AssertionError("empty access token was accepted")


def test_keycloak_oidc_client_resolves_immutable_subject_to_canonical_principal(
    monkeypatch,
):
    import json

    import aegis.identity as identity_module

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return json.dumps(
                {
                    "sub": "keycloak-subject",
                    "aegis_vault_id": "alice-vault",
                    "aegis_space_ids": ["apartment"],
                }
            ).encode()

    monkeypatch.setattr(identity_module, "urlopen", lambda request, timeout: Response())
    client = KeycloakOIDCClient(
        "http://keycloak/realms/aegis",
        principal_resolver=lambda subject: "alice" if subject == "keycloak-subject" else "",
    )

    assert client.principal_from_access_token("token") == Principal(
        id="alice", vault_id="alice-vault", space_ids=("apartment",)
    )


def test_postgres_external_principal_resolver_fails_closed_when_unprovisioned():
    class Cursor:
        def fetchone(self):
            return None

    class Connection:
        def execute(self, query, params):
            assert "external_subject = %s" in query
            assert params == ("unknown-subject",)
            return Cursor()

        def close(self):
            pass

    resolver = PostgresExternalPrincipalResolver(lambda url: Connection(), "postgresql://db")
    try:
        resolver("unknown-subject")
    except PermissionError as exc:
        assert str(exc) == "external identity is not provisioned in AEGIS"
    else:
        raise AssertionError("unprovisioned external identity was accepted")


def test_postgres_external_principal_resolver_contains_store_failure():
    def connect(_url):
        raise RuntimeError("password=private-secret")

    resolver = PostgresExternalPrincipalResolver(connect, "postgresql://db")
    try:
        resolver("subject")
    except RuntimeError as exc:
        assert str(exc) == "canonical identity mapping is unavailable"
        assert "private-secret" not in str(exc)
    else:
        raise AssertionError("identity store failure was exposed")


def test_openfga_adapter_is_fail_closed_on_relationship_denial():
    class Client:
        def check(self, user, relation, object_id):
            assert (user, relation, object_id) == ("user:bob", "can_read", "resource:alice-bank")
            return False

    decision = OpenFGAAuthorization(Client()).can_read(
        Principal(id="bob", vault_id="bob-vault"), "alice-bank"
    )
    assert not decision.allowed


def test_privacy_projection_exposes_only_allowlisted_derived_household_fields():
    class Policy:
        def may_derive(self, requester, owner_id, space_id):
            return (
                requester.id == "bob" and space_id == "apartment" and owner_id in {"alice", "bob"}
            )

    result = PrivacyProjectionService(Policy()).build(
        Principal(id="bob", vault_id="bob-vault"),
        "apartment",
        ("alice", "bob"),
        (SharedObligation("rent", 2000), SharedObligation("utilities", 200)),
        (
            PrivateContribution("alice", 900, "alice-private-finance"),
            PrivateContribution("bob", 400, "bob-private-finance"),
        ),
    )
    assert isinstance(result, HouseholdProjection)
    assert result.obligation_total == 2200
    assert result.settlements == {"alice": -200, "bob": -700}
    assert "alice-private-finance" not in repr(result)


def test_privacy_projection_denies_unapproved_private_input_before_derivation():
    class Deny:
        def may_derive(self, requester, owner_id, space_id):
            return False

    try:
        PrivacyProjectionService(Deny()).build(
            Principal(id="bob", vault_id="bob-vault"),
            "apartment",
            ("alice", "bob"),
            (SharedObligation("rent", 2000),),
            (PrivateContribution("alice", 1000, "alice-balance"),),
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("unauthorized private data entered projection")


def test_audit_log_is_hash_chained_and_detects_tampering():
    log = AuditLog()
    first = log.append("objective.created", "alice", {"domain": "kitchen"})
    log.append("action.verified", "alice", {"result": "readback"}, objective_id=first.event_id)
    assert log.verify()
    log.events[0].payload["domain"] = "finance"
    assert not log.verify()


def test_audit_log_rejects_secret_fields_at_ingress():
    try:
        AuditLog().append("action.observed", "alice", {"access_token": "do-not-store"})
    except AuditError:
        pass
    else:
        raise AssertionError("secret entered semantic audit log")


def test_persistent_audit_log_reloads_and_detects_database_tampering(tmp_path):
    path = str(tmp_path / "audit.sqlite")
    log = SqliteAuditLog(path)
    log.append("objective.created", "alice", {"domain": "tasks"})
    log.close()
    reloaded = SqliteAuditLog(path)
    assert reloaded.verify()
    reloaded.close()
    tampered = SqliteAuditLog(path)
    tampered.connection.execute("UPDATE audit_events SET payload = ?", ('{"domain":"finance"}',))
    tampered.connection.commit()
    tampered.close()
    assert not SqliteAuditLog(path).verify()


def test_pack_lifecycle_requires_explicit_permissions_and_enablement():
    card = ActionCard(
        action=ActionSpec(
            action_id="cards.read",
            capability="cards.read",
            required_permissions=("cards.read",),
            verification=VerificationContract(kind="readback"),
        ),
        summary="Read cards",
        relevance=1,
    )
    bundle = PackBundle(
        manifest=PackManifest(pack_id="cards", version="0.1.0", permissions=("cards.read",)),
        cards=(card,),
    )
    manager = PackManager()
    manager.discover(bundle)
    assert manager.status("cards") is PackStatus.DISCOVERED
    try:
        manager.install("cards")
    except PermissionError:
        pass
    else:
        raise AssertionError("Pack installed without permission grant")
    manager.install("cards", frozenset({"cards.read"}))
    assert manager.enabled_cards() == ()
    manager.enable("cards")
    assert manager.enabled_cards() == (card,)
    manager.disable("cards")
    assert manager.enabled_cards() == ()
    assert manager.retrieve("cards") == ()
    manager.enable("cards")
    assert manager.retrieve("cards") == (card,)
    assert manager.audit.verify()
    assert [event.event_type for event in manager.audit.events] == [
        "pack.discovered",
        "pack.installed",
        "pack.enabled",
        "pack.disabled",
        "pack.enabled",
    ]


def test_capability_registry_semantic_retrieval_ranks_bounded_pack_cards():
    cards = (
        ActionCard(
            action=ActionSpec(
                action_id="kitchen.groceries.list", capability="kitchen.groceries.read"
            ),
            summary="Show groceries and food needed from the store",
            relevance=1,
        ),
        ActionCard(
            action=ActionSpec(action_id="tasks.list", capability="tasks.read"),
            summary="Show outstanding tasks and obligations",
            relevance=1,
        ),
    )

    class Embedder:
        def embed(self, texts):
            return tuple(
                (1.0, 0.0) if index == 0 or "grocery" in text or "food" in text else (0.0, 1.0)
                for index, text in enumerate(texts)
            )

    retrieved = CapabilityRegistry(cards).retrieve_semantic("what should i buy", Embedder())

    assert retrieved == (cards[0], cards[1])


def test_capability_registry_semantic_retrieval_exposes_non_authoritative_scores():
    cards = (
        ActionCard(
            action=ActionSpec(action_id="first", capability="test.read"),
            summary="first",
            relevance=0.1,
        ),
        ActionCard(
            action=ActionSpec(action_id="second", capability="test.read"),
            summary="second",
            relevance=0.2,
        ),
    )

    class Embedder:
        def embed(self, texts):
            return tuple(
                (1.0, 0.0) if index in (0, 1) else (0.0, 1.0) for index, _ in enumerate(texts)
            )

    matches = CapabilityRegistry(cards).retrieve_semantic_with_scores("query", Embedder())

    assert [match.card.action.action_id for match in matches] == ["first", "second"]
    assert matches[0].score > matches[1].score


def test_household_read_fast_path_does_not_capture_explicit_task_completion():
    from aegis.household import HouseholdReadFastPath

    assert HouseholdReadFastPath.matches("What chores are on the list?")
    assert HouseholdReadFastPath.matches("Give me the chores I still have to do")
    assert HouseholdReadFastPath.matches("What do I have scheduled tomorrow?")
    assert HouseholdReadFastPath.matches("And what is on my calendar?")
    assert not HouseholdReadFastPath.matches("What tasks are scheduled tomorrow?")
    assert not HouseholdReadFastPath.matches(
        "What do I need to take care of before the apartment inspection?"
    )
    assert not HouseholdReadFastPath.matches("Finish the task called the quarterly inspection.")


def test_pack_validation_rejects_cross_namespace_or_unverified_mutation():
    cross_namespace = PackBundle(
        manifest=PackManifest(pack_id="cards", version="0.1.0"),
        cards=(
            ActionCard(
                action=ActionSpec(action_id="finance.read", capability="finance.read"),
                summary="wrong namespace",
                relevance=1,
            ),
        ),
    )
    unverified = PackBundle(
        manifest=PackManifest(pack_id="cards", version="0.1.0", permissions=("cards.write",)),
        cards=(
            ActionCard(
                action=ActionSpec(
                    action_id="cards.write",
                    capability="cards.write",
                    required_permissions=("cards.write",),
                ),
                summary="unverified mutation",
                relevance=1,
            ),
        ),
    )
    for bundle in (cross_namespace, unverified):
        try:
            PackManager().discover(bundle)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe Pack manifest was accepted")


def test_model_router_prefers_available_local_provider_and_records_provenance():
    class Provider:
        def __init__(self, provider_id, local, available):
            self.provider_id = provider_id
            self.local = local
            self._available = available

        def available(self):
            return self._available

        def decide(self, request):
            return type("Response", (), {"raw": {"kind": "ANSWER", "answer": self.provider_id}})()

    router = ConfiguredModelRouter(
        (Provider("cloud", False, True), Provider("ollama/qwen3:8b", True, True)),
        allow_cloud=True,
    )
    response = router.decide(type("Request", (), {})())
    assert response.raw["answer"] == "cloud"
    assert router.traces[0].provider_id == "cloud"
    local_router = ConfiguredModelRouter(
        (Provider("cloud", False, True), Provider("ollama/qwen3:8b", True, True)),
    )
    assert local_router.decide(type("Request", (), {})()).raw["answer"] == "ollama/qwen3:8b"


def test_model_router_fails_closed_when_only_disallowed_provider_exists():
    class Cloud:
        provider_id = "cloud"
        local = False

        def available(self):
            return True

        def decide(self, request):
            raise AssertionError("cloud provider was used without explicit opt-in")

    try:
        ConfiguredModelRouter((Cloud(),)).decide(type("Request", (), {})())
    except ModelUnavailable:
        pass
    else:
        raise AssertionError("privacy policy was bypassed")


def test_baseline_metrics_are_compact_and_measurable():
    metrics = BaselineMetrics()
    metrics.record(success=True, schema_valid=True, latency_ms=10)
    metrics.record(success=False, schema_valid=True, false_completion=True, latency_ms=20)
    assert metrics.summary() == {
        "cases": 2,
        "success_rate": 0.5,
        "schema_valid_rate": 1.0,
        "false_completions": 1,
        "security_errors": 0,
        "average_latency_ms": 15,
    }


def test_ollama_provider_repairs_malformed_json_once():
    class Transport:
        def __init__(self):
            self.calls = []

        def chat(self, payload):
            self.calls.append(payload)
            content = "not-json" if len(self.calls) == 1 else '{"kind":"ANSWER","answer":"ok"}'
            return {"message": {"content": content}}

    transport = Transport()
    provider = OllamaProvider("qwen3:8b", transport)
    response = provider.decide(
        ModelRequest(working_set=WorkingSet(intent=intent()), action_cards=())
    )
    assert response.raw == {"kind": "ANSWER", "answer": "ok"}
    assert response.model_calls == 2
    assert response.latency_ms is not None
    assert provider.last_response_metrics["model_calls"] == 2
    assert len(transport.calls) == 2
    assert provider.request_mode_counts == {"ordinary_decision": 1}
    assert "invalid" in transport.calls[1]["messages"][0]["content"]
    assert transport.calls[0]["think"] is False
    assert transport.calls[0]["format"]["title"] == "Decision"
    assert transport.calls[0]["format"]["$defs"]["ActionSpec"]["required"] == [
        "action_id",
        "capability",
        "arguments",
        "required_permissions",
        "verification",
    ]
    assert "single_card_rule" in transport.calls[0]["messages"][0]["content"]


def test_ollama_classification_schema_requires_semantic_mode():
    class Transport:
        def chat(self, payload):
            return {
                "message": {"content": '{"kind":"ANSWER","semantic_mode":"READ","answer":"ok"}'}
            }

    provider = OllamaProvider("qwen3:8b", Transport())
    response = provider.decide(
        ModelRequest(
            working_set=WorkingSet(intent=intent()),
            action_cards=(),
            classification_only=True,
        )
    )

    assert response.raw["semantic_mode"] == "READ"


def test_ollama_schema_requires_semantic_mode_for_all_provider_decisions():
    schema = OllamaProvider._decision_schema(
        ModelRequest(working_set=WorkingSet(intent=intent()), action_cards=())
    )

    assert "semantic_mode" in schema["required"]


def test_ollama_effect_schema_is_segmentation_only():
    schema = OllamaProvider._decision_schema(
        ModelRequest(
            working_set=WorkingSet(intent=intent()),
            action_cards=(),
            objective_effect_only=True,
        )
    )
    properties = schema["properties"]["effects"]["items"]["properties"]
    assert set(properties) == {"effect_text", "source_span", "polarity"}
    assert schema["properties"]["effects"]["items"]["required"] == [
        "effect_text",
        "source_span",
    ]


def test_ollama_http_transport_rejects_non_http_urls():
    try:
        OllamaHttpTransport("unix:///run/ollama.sock")
    except ValueError:
        pass
    else:
        raise AssertionError("non-HTTP Ollama transport URL was accepted")


def test_ollama_http_transport_resolves_model_digest_from_inventory(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"models":[{"name":"qwen3:8b","digest":"sha256:abc"}]}'

    monkeypatch.setattr(ollama_module, "urlopen", lambda request, timeout: Response())

    transport = OllamaHttpTransport("http://ollama.example:11434")

    assert transport.model_digest("qwen3:8b") == "sha256:abc"
    assert transport.model_digest("missing:latest") is None


def test_ollama_provider_does_not_retry_beyond_bound():
    class Transport:
        def __init__(self):
            self.calls = 0

        def chat(self, payload):
            self.calls += 1
            return {"message": {"content": "not-json"}}

    transport = Transport()
    provider = OllamaProvider("qwen3:8b", transport)
    try:
        provider.decide(ModelRequest(working_set=WorkingSet(intent=intent()), action_cards=()))
    except OllamaResponseError:
        pass
    else:
        raise AssertionError("malformed Ollama output was accepted")
    assert transport.calls == 2


def test_ollama_compact_card_view_omits_policy_internals():
    card = ActionCard(
        action=ActionSpec(
            action_id="tasks.create",
            capability="tasks.write",
            required_permissions=("tasks.write",),
            verification=VerificationContract(kind="readback"),
        ),
        summary="Create a task",
        relevance=0.9,
        argument_keys=("title",),
        argument_descriptions={"title": "task title"},
    )
    request = ModelRequest(working_set=WorkingSet(intent=intent()), action_cards=(card,))
    prompt = OllamaProvider("qwen3:8b", object(), compact_action_cards=True)._prompt(request)
    payload = json.loads(prompt)
    visible = payload["action_cards"][0]
    assert visible == {
        "action_id": "tasks.create",
        "capability": "tasks.write",
        "operation": "write",
        "summary": "Create a task",
        "relevance": 0.9,
        "argument_keys": ["title"],
        "argument_descriptions": {"title": "task title"},
    }
    assert "required_permissions" not in visible
    assert "verification" not in visible


def test_ollama_plan_prompt_separates_plan_and_action_shapes():
    request = ModelRequest(
        working_set=WorkingSet(intent=intent()),
        action_cards=(),
        allow_argument_proposals=True,
        allow_plan_proposals=True,
    )

    payload = json.loads(OllamaProvider("qwen3:8b", object())._prompt(request))

    assert "plan object" in payload["plan_rule"]
    assert "must not contain action_ref, action_arguments, or action" in payload["plan_rule"]
    assert "every independent state change requested by the user" in payload["plan_rule"]
    assert "Keep arguments scoped to their own step" in payload["plan_rule"]
    assert "optional or 'if clearly stated'" in payload["argument_proposal_rule"]


def test_ollama_schema_does_not_accept_model_provenance_metadata():
    schema = OllamaProvider._decision_schema(
        ModelRequest(working_set=WorkingSet(intent=intent()), action_cards=())
    )

    action_schema = schema["$defs"]["ActionSpec"]
    assert "argument_provenance" not in action_schema["properties"]


def test_ollama_repair_prompt_exposes_validator_stage_only_as_context():
    request = ModelRequest(
        working_set=WorkingSet(intent=intent()),
        action_cards=(),
        proposal_repair_only=True,
        repair_validator_stage="requested_effect_structural_coverage",
        proposal_failure=ProposalFailureEvidence(kind=ProposalFailureKind.BAD_SOURCE_SPAN),
    )
    payload = json.loads(OllamaProvider("qwen3:8b", object())._prompt(request))
    assert payload["repair_validator_stage"] == "requested_effect_structural_coverage"
    assert payload["proposal_failure"]["kind"] == "BAD_SOURCE_SPAN"


def test_ollama_structural_repair_prompt_requires_plan_attempt():
    request = ModelRequest(
        working_set=WorkingSet(intent=intent()),
        action_cards=(),
        proposal_repair_only=True,
        repair_validator_stage="proposal_repair",
        proposal_failure=ProposalFailureEvidence(
            kind=ProposalFailureKind.UNACCOUNTED_STRUCTURAL_ANCHOR
        ),
    )

    payload = json.loads(OllamaProvider("qwen3:8b", object())._prompt(request))

    assert "return a PLAN" in payload["instruction"]
    assert "Do not return CLARIFY merely" in payload["instruction"]


def test_ollama_structural_repair_prompt_exposes_grounded_effect_contract():
    request = ModelRequest(
        working_set=WorkingSet(
            intent=intent(),
            context=Context(values={"grounded_requested_effects": [{"effect_text": "do A"}]}),
        ),
        action_cards=(),
        proposal_repair_only=True,
        repair_validator_stage="proposal_repair",
        proposal_failure=ProposalFailureEvidence(
            kind=ProposalFailureKind.UNACCOUNTED_STRUCTURAL_ANCHOR
        ),
    )

    payload = json.loads(OllamaProvider("qwen3:8b", object())._prompt(request))

    assert "grounded_requested_effects" in payload["instruction"]
    assert (
        "one corresponding plan step and objective requirement for each" in payload["instruction"]
    )


def test_ollama_mapping_repair_schema_is_objective_spec():
    schema = OllamaProvider._decision_schema(
        ModelRequest(
            working_set=WorkingSet(intent=intent()),
            action_cards=(),
            proposal_repair_only=True,
            objective_interpretation_only=True,
        )
    )

    assert "requirements" in schema["properties"]
    assert "kind" not in schema["properties"]


def test_ollama_fidelity_prompt_has_no_plan_authority():
    proposal = ObjectiveSpecProposal(
        requirements=(ObjectiveRequirementProposal(action_ref="tasks.create"),)
    )
    request = ModelRequest(
        working_set=WorkingSet(intent=intent()),
        action_cards=(),
        objective_fidelity_only=True,
        objective_spec_proposal=proposal,
    )

    payload = json.loads(OllamaProvider("qwen3:8b", object())._prompt(request))

    assert payload["objective_fidelity_only"] is True
    assert "not to any plan" in payload["instruction"]
    assert payload["objective_spec_proposal"] == proposal.model_dump(mode="json")


def test_openclaw_grocery_verifier_rejects_duplicate_external_records(tmp_path):
    key = "correlation:kitchen.groceries.add"
    path = tmp_path / "groceries.tsv"
    path.write_text(f"{key}|rice\n{key}|rice\n", encoding="utf-8")
    observation = Observation(
        execution_id=uuid4(),
        evidence={"external_state_path": str(path), "idempotency_key": key},
        command_succeeded=True,
    )
    result = OpenClawGroceryVerifier().verify(observation, VerificationContract(kind="readback"))
    assert not result.verified
    assert result.evidence["external_records_for_key"] == 2


def test_openclaw_grocery_verifier_requires_canonical_household_readback(tmp_path):
    key = "correlation:kitchen.groceries.add"
    path = tmp_path / "groceries.tsv"
    path.write_text(f"{key}|rice\n", encoding="utf-8")

    class Canonical:
        def grocery_recorded(self, principal, item, idempotency_key):
            return False

    observation = Observation(
        execution_id=uuid4(),
        evidence={"external_state_path": str(path), "idempotency_key": key},
        command_succeeded=True,
    )
    result = OpenClawGroceryVerifier(Canonical(), Principal(id="alice", vault_id="vault")).verify(
        observation, VerificationContract(kind="readback")
    )
    assert not result.verified
    assert result.evidence["canonical_grocery_verified"] is False


def test_decision_evaluation_harness_measures_valid_and_rejected_cases():
    card = ActionCard(
        action=ActionSpec(
            action_id="tasks.create",
            capability="tasks.create",
            verification=VerificationContract(kind="readback"),
        ),
        summary="create task",
        relevance=1,
    )
    cases = (
        EvaluationCase(
            "answer",
            ModelResponse(raw={"kind": "ANSWER", "answer": "three tasks"}),
            (),
            expected_kind=DecisionKind.ANSWER,
        ),
        EvaluationCase(
            "action",
            ModelResponse(raw={"kind": "ACTION", "action": card.action.model_dump()}),
            (card,),
            expected_kind=DecisionKind.ACTION,
            expected_action_id="tasks.create",
        ),
        EvaluationCase(
            "invented",
            ModelResponse(raw={"kind": "ACTION", "action": {"action_id": "root"}}),
            (card,),
            expect_rejection=True,
        ),
    )
    summary = DecisionEvaluationHarness().run(cases).summary()
    assert summary["cases"] == 3
    assert summary["success_rate"] == 1.0
    assert summary["security_errors"] == 0


def test_personal_state_resolves_aliases_and_preserves_provenance():
    from datetime import datetime, timezone

    state = PersonalState()
    server = state.add_entity("Atlas", ("the server", "atlas host"))
    when = datetime(2026, 8, 30, 23, 0, tzinfo=timezone.utc)
    memory = state.add_memory(
        "Worked on the backup architecture", when, Provenance.EXPLICIT_USER, (server.entity_id,)
    )
    assert state.resolve_entity("THE SERVER") == server
    assert state.memories_between(
        datetime(2026, 8, 30, 22, 0, tzinfo=timezone.utc),
        datetime(2026, 8, 31, 1, 0, tzinfo=timezone.utc),
        server.entity_id,
    ) == (memory,)
    assert memory.provenance is Provenance.EXPLICIT_USER


def test_personal_memory_search_is_ranked_current_and_provenance_preserving():
    from datetime import datetime, timezone

    state = PersonalState()
    entity = state.add_entity("Backup Architecture", ("backup",))
    first = state.add_memory(
        "Discussed backup architecture",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        Provenance.EXPLICIT_USER,
        (entity.entity_id,),
    )
    state.add_memory(
        "Reviewed unrelated recipe",
        datetime(2026, 1, 2, tzinfo=timezone.utc),
        Provenance.INFERRED,
    )
    corrected = state.correct_memory(
        first.memory_id,
        "Decided on backup architecture",
        datetime(2026, 1, 3, tzinfo=timezone.utc),
    )
    assert state.search_memories("backup architecture") == (corrected,)
    assert corrected.provenance is Provenance.CORRECTED
    try:
        state.search_memories("backup", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive memory search limit was accepted")


def test_personal_memory_fast_path_returns_grounded_result_without_model():
    from datetime import datetime, timezone

    state = PersonalState()
    state.add_memory(
        "Worked on the backup architecture",
        datetime(2026, 8, 31, 1, tzinfo=timezone.utc),
        Provenance.EXPLICIT_USER,
    )
    state.add_memory(
        "Worked on an unrelated older task",
        datetime(2026, 8, 29, tzinfo=timezone.utc),
        Provenance.EXPLICIT_USER,
    )
    result = PersonalMemoryFastPath(
        state, now=datetime(2026, 8, 31, 12, tzinfo=timezone.utc)
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What was I working on last night?",
        )
    )
    assert result is not None
    assert result.state.value == "completed"
    assert result.evidence["memories"][0]["provenance"] == "explicit_user"
    assert len(result.evidence["memories"]) == 1
    assert (
        PersonalMemoryFastPath(state).resolve(
            IntentFrame(principal=Principal(id="alice", vault_id="alice-vault"), utterance="hello")
        )
        is None
    )


def test_personal_context_fast_path_returns_projects_and_goals_without_model():
    from datetime import datetime, timezone

    state = PersonalState()
    now = datetime(2026, 8, 30, tzinfo=timezone.utc)
    project = state.add_project("Backup architecture", now)
    state.add_goal("Finish the restore drill", now, project.project_id)
    principal = Principal(id="alice", vault_id="alice-vault")

    projects = PersonalMemoryFastPath(state).resolve(
        IntentFrame(principal=principal, utterance="What projects am I working on?")
    )
    goals = PersonalMemoryFastPath(state).resolve(
        IntentFrame(principal=principal, utterance="Show my goals")
    )

    assert projects is not None
    assert projects.evidence["projects"] == [
        {"name": "Backup architecture", "created_at": now.isoformat()}
    ]
    assert goals is not None
    assert goals.evidence["goals"][0]["project"] == "Backup architecture"


def test_personal_memory_fast_path_resolves_entity_alias_queries():
    from datetime import datetime, timezone

    state = PersonalState()
    entity = state.add_entity("OpenClaw", ("gateway",))
    state.add_memory(
        "The gateway uses a paired device for terminal execution",
        datetime(2026, 8, 30, tzinfo=timezone.utc),
        Provenance.OBSERVED,
        (entity.entity_id,),
    )

    result = PersonalMemoryFastPath(state).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What do I know about the gateway?",
        )
    )

    assert result is not None
    assert result.evidence["memories"][0]["provenance"] == "observed"


def test_personal_memory_fast_path_does_not_claim_vague_general_followup():
    from datetime import datetime, timezone

    state = PersonalState()
    state.add_memory(
        "Investigated the server backup failure",
        datetime(2026, 8, 30, tzinfo=timezone.utc),
        Provenance.OBSERVED,
    )

    result = PersonalMemoryFastPath(state).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What about that server problem?",
        )
    )

    assert result is None


def test_personal_memory_fast_path_yields_to_explicit_mutation_requests():
    state = PersonalState()
    state.add_memory(
        "The apartment inspection is tomorrow",
        datetime(2026, 8, 30, tzinfo=timezone.utc),
        Provenance.OBSERVED,
    )

    result = PersonalMemoryFastPath(state).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Create an event for the apartment inspection date from my memory.",
        )
    )

    assert result is None


def test_mutation_request_recognizes_jot_down_action_language():
    assert is_mutation_request("Could you jot down replace the porch bulb on my to-do list?")
    assert is_mutation_request("Schedule a meeting for Friday.")
    assert is_mutation_request("Could you book a meeting for Friday?")


def test_context_reset_prefix_preserves_the_new_objective():
    assert strip_context_reset("Actually, never mind, what groceries do I need?") == (
        "what groceries do i need?"
    )


def test_personal_memory_correction_supersedes_old_record():
    from datetime import datetime, timezone

    state = PersonalState()
    first = state.add_memory(
        "Use daily backups", datetime(2026, 8, 30, tzinfo=timezone.utc), Provenance.INFERRED
    )
    corrected = state.correct_memory(
        first.memory_id, "Use hourly backups", datetime(2026, 8, 31, tzinfo=timezone.utc)
    )
    assert first.superseded_by == corrected.memory_id
    records = state.memories_between(
        datetime(2026, 8, 29, tzinfo=timezone.utc),
        datetime(2026, 9, 1, tzinfo=timezone.utc),
    )
    assert records == (corrected,)


def test_personal_state_serialization_survives_reload_with_supersession():
    from datetime import datetime, timezone

    state = PersonalState()
    entity = state.add_entity("Atlas", ("server",))
    original = state.add_memory(
        "old plan",
        datetime(2026, 8, 30, tzinfo=timezone.utc),
        Provenance.EXPLICIT_USER,
        (entity.entity_id,),
    )
    corrected = state.correct_memory(
        original.memory_id, "new plan", datetime(2026, 8, 31, tzinfo=timezone.utc)
    )
    restored = PersonalState.from_json(state.to_json())
    assert restored.resolve_entity("SERVER") == entity
    assert restored.memories[original.memory_id].superseded_by == corrected.memory_id
    assert restored.memories[corrected.memory_id].provenance is Provenance.CORRECTED


def test_personal_state_serialization_preserves_projects_and_goals():
    from datetime import datetime, timezone

    state = PersonalState()
    when = datetime(2026, 8, 31, tzinfo=timezone.utc)
    project = state.add_project("AEGIS", when)
    goal = state.add_goal("Ship the first beta", when, project.project_id)
    restored = PersonalState.from_json(state.to_json())
    assert restored.projects[project.project_id] == project
    assert restored.goals[goal.goal_id] == goal


def test_personal_state_rejects_goal_with_unknown_project():
    from datetime import datetime, timezone

    state = PersonalState()
    try:
        state.add_goal("orphan", datetime.now(timezone.utc), uuid4())
    except ValueError:
        pass
    else:
        raise AssertionError("goal with unknown project was accepted")


def test_personal_state_rejects_unknown_schema():
    try:
        PersonalState.from_json('{"schema_version":99,"entities":[],"memories":[]}')
    except ValueError:
        pass
    else:
        raise AssertionError("unknown personal-state schema was accepted")


def test_postgres_personal_store_writes_supersession_after_all_memory_rows():
    from datetime import datetime, timezone

    class Connection:
        def __init__(self):
            self.calls = []
            self.commits = 0

        def execute(self, query, params=()):
            self.calls.append((query, params))

        def commit(self):
            self.commits += 1

    state = PersonalState()
    first = state.add_memory(
        "old", datetime(2026, 8, 30, tzinfo=timezone.utc), Provenance.EXPLICIT_USER
    )
    corrected = state.correct_memory(
        first.memory_id, "new", datetime(2026, 8, 31, tzinfo=timezone.utc)
    )
    connection = Connection()
    PostgresPersonalStateStore(connection, "alice-vault").save(state)

    memory_insert_indexes = [
        index
        for index, (query, _) in enumerate(connection.calls)
        if "INSERT INTO personal_memories" in query
    ]
    update_index = next(
        index
        for index, (query, _) in enumerate(connection.calls)
        if "UPDATE personal_memories SET superseded_by" in query
    )
    assert len(memory_insert_indexes) == 2
    assert max(memory_insert_indexes) < update_index
    assert connection.calls[update_index][1][0] == str(corrected.memory_id)
    assert connection.commits == 1


def test_postgres_personal_store_rejects_non_owner_vault_read():
    class Result:
        def fetchone(self):
            return None

    class Connection:
        def execute(self, query, params=()):
            assert "owner_principal_id" in query
            assert params == ("alice-vault", "bob")
            return Result()

    with pytest.raises(PermissionError, match="Vault access denied"):
        PostgresPersonalStateStore(Connection(), "alice-vault").load_for_principal(
            Principal(id="bob", vault_id="alice-vault")
        )


def test_household_space_supports_shared_workflows_for_active_members():
    from datetime import datetime, timezone

    alice = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    bob = Principal(id="bob", vault_id="bob-vault", space_ids=("apartment",))
    space = HouseholdSpace("apartment", {"alice", "bob"})
    space.add_grocery(alice, "rice")
    space.add_chore(alice, Chore("dishes", "Wash dishes", "bob"))
    space.add_event(
        alice,
        HouseholdEvent(
            "inspection", "Apartment inspection", datetime(2026, 9, 2, tzinfo=timezone.utc)
        ),
    )
    space.add_obligation(alice, HouseholdObligation("rent", "September rent", 2000, "bob"))
    snapshot = space.snapshot(bob)
    assert snapshot["groceries"] == ("rice",)
    assert len(snapshot["chores"]) == 1
    assert len(snapshot["events"]) == 1
    assert len(snapshot["obligations"]) == 1


def test_household_read_fast_path_returns_only_shared_allowlisted_fields():
    alice = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    space = HouseholdSpace("apartment", {alice.id})
    space.add_obligation(alice, HouseholdObligation("utilities", "Utilities", 18500, alice.id))
    result = HouseholdReadFastPath(space.snapshot(alice)).resolve(
        IntentFrame(principal=alice, utterance="Who still needs to handle utilities?")
    )

    assert result is not None
    assert result.evidence["obligations"] == [
        {
            "title": "Utilities",
            "responsible_id": "alice",
            "amount": 18500,
            "settled": False,
        }
    ]
    assert "private" not in repr(result.evidence).lower()
    assert not HouseholdReadFastPath.matches("Create a chore to clean the kitchen")
    assert not HouseholdReadFastPath.matches("Please create a task for the apartment inspection")


def test_household_read_fast_path_filters_explicit_chore_status():
    alice = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    space = HouseholdSpace("apartment", {alice.id})
    space.add_chore(alice, Chore("done", "Wash dishes", "alice", completed=True))
    space.add_chore(alice, Chore("open", "Sweep floor", "alice"))

    result = HouseholdReadFastPath(space.snapshot(alice)).resolve(
        IntentFrame(principal=alice, utterance="Show my completed chores")
    )

    assert result is not None
    assert result.evidence["status_filter"] == "completed"
    assert result.evidence["chores"] == [
        {
            "chore_id": "done",
            "title": "Wash dishes",
            "assignee_id": "alice",
            "completed": True,
        }
    ]


def test_household_read_fast_path_filters_natural_event_date_requests():
    from datetime import timedelta

    alice = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    space = HouseholdSpace("apartment", {alice.id})
    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
    space.add_event(alice, HouseholdEvent("tomorrow", "Apartment inspection", tomorrow))
    space.add_event(
        alice,
        HouseholdEvent("later", "Dentist appointment", tomorrow + timedelta(days=2)),
    )

    assert HouseholdReadFastPath.matches("What do I have scheduled tomorrow?")
    result = HouseholdReadFastPath(space.snapshot(alice)).resolve(
        IntentFrame(principal=alice, utterance="What do I have scheduled tomorrow?")
    )

    assert result is not None
    assert result.evidence["date_filter"] == "tomorrow"
    assert result.evidence["events"] == [
        {"title": "Apartment inspection", "starts_at": tomorrow.isoformat()}
    ]


def test_household_chore_priority_request_does_not_return_the_whole_list():
    alice = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    space = HouseholdSpace("apartment", {alice.id})
    space.add_chore(alice, Chore("kitchen", "clean the kitchen", "alice"))

    result = HouseholdReadFastPath(space.snapshot(alice)).resolve(
        IntentFrame(principal=alice, utterance="Which chore is due first?")
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "no deadlines" in result.message


def test_household_read_fast_path_filters_events_in_next_week_window():
    from datetime import timedelta

    alice = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    space = HouseholdSpace("apartment", {alice.id})
    next_week = datetime.now(timezone.utc) + timedelta(days=8)
    space.add_event(alice, HouseholdEvent("next-week", "Team sync", next_week))
    space.add_event(
        alice,
        HouseholdEvent("later", "Quarterly review", next_week + timedelta(days=7)),
    )

    result = HouseholdReadFastPath(space.snapshot(alice)).resolve(
        IntentFrame(principal=alice, utterance="Which appointments do I have next week?")
    )

    assert result is not None
    assert result.evidence["date_filter"] == "next_week"
    assert result.evidence["events"] == [{"title": "Team sync", "starts_at": next_week.isoformat()}]


def test_household_read_fast_path_filters_events_by_weekday():
    from datetime import timedelta

    alice = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    space = HouseholdSpace("apartment", {alice.id})
    now = datetime.now(timezone.utc)
    days_until_friday = (4 - now.weekday()) % 7
    friday = now + timedelta(days=days_until_friday)
    space.add_event(alice, HouseholdEvent("friday", "Friday appointment", friday))
    space.add_event(alice, HouseholdEvent("other", "Other appointment", friday + timedelta(days=1)))

    result = HouseholdReadFastPath(space.snapshot(alice)).resolve(
        IntentFrame(principal=alice, utterance="What appointments do I have on Friday?")
    )

    assert result is not None
    assert result.evidence["date_filter"] == "weekday:friday"
    assert result.evidence["events"] == [
        {"title": "Friday appointment", "starts_at": friday.isoformat()}
    ]


def test_cross_domain_planning_fast_path_keeps_personal_and_shared_context():
    from datetime import datetime, timezone

    alice = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    personal = PersonalState()
    project = personal.add_project("Backup architecture", datetime(2026, 8, 1, tzinfo=timezone.utc))
    personal.add_goal(
        "Finish the restore drill", datetime(2026, 8, 2, tzinfo=timezone.utc), project.project_id
    )
    space = HouseholdSpace("apartment", {alice.id})
    space.add_obligation(alice, HouseholdObligation("utilities", "Utilities", 120, alice.id))
    task = Task(uuid4(), "apartment", "Review backup runbook", "alice")

    utterance = (
        "Considering my personal goals and current household obligations, "
        "what should I prioritize this week?"
    )
    result = CrossDomainPlanningFastPath(personal, space.snapshot(alice), (task,)).resolve(
        IntentFrame(principal=alice, utterance=utterance)
    )

    assert result is not None
    planning = result.evidence["planning"]
    assert planning["goals"][0]["description"] == "Finish the restore drill"
    assert planning["open_obligations"] == [{"title": "Utilities", "responsible_id": "alice"}]
    assert planning["open_tasks"][0]["title"] == "Review backup runbook"
    assert planning["open_chores"] == []
    assert planning["priority_candidates"] == [
        "household obligation: Utilities",
        "personal goal: Finish the restore drill",
        "task: Review backup runbook",
    ]
    assert len(planning["goals"]) <= 5
    assert len(planning["open_obligations"]) <= 5
    assert len(planning["open_tasks"]) <= 5


def test_contextual_cross_domain_priority_uses_authorized_task_deadline():
    context = Context(
        values={
            "canonical_facts": {
                "planning": {
                    "open_tasks": [
                        {
                            "task_id": "later",
                            "title": "later task",
                            "due_at": "2026-09-05T12:00:00+00:00",
                        },
                        {
                            "task_id": "first",
                            "title": "first task",
                            "due_at": "2026-09-03T12:00:00+00:00",
                        },
                    ],
                    "open_chores": [{"chore_id": "chore", "title": "clean kitchen"}],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )
    result = ContextualCrossDomainPriorityFastPath().resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="which of these is the best place to begin?",
        ),
        context,
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.message.endswith("first task")
    assert result.evidence["priority_basis"] == "authorized_prior_result_earliest_task_deadline"


def test_cross_domain_planning_recognizes_plural_utilities_context():
    assert CrossDomainPlanningFastPath.matches(
        "Considering my personal memory, Utilities, and open tasks, what should I prioritize?"
    )


def test_cross_domain_planning_recognizes_which_should_i_read_composition():
    assert CrossDomainPlanningFastPath.matches(
        "Please tell me what you remember about the backup and which household obligation "
        "I should handle first."
    )


def test_cross_domain_planning_recognizes_conjoined_finance_memory_read():
    assert CrossDomainPlanningFastPath.matches(
        "Can I afford $50 after Utilities, and what do I remember about the backup?"
    )


def test_cross_domain_planning_never_absorbs_compound_mutation():
    alice = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    result = CrossDomainPlanningFastPath(PersonalState(), {}, ()).resolve(
        IntentFrame(
            principal=alice,
            utterance="Put checking the furnace on my task list and make wiping the porch a chore",
        )
    )

    assert result is None


def test_domain_clarification_does_not_preempt_cross_domain_planning_vocabulary():
    assert not DomainClarificationFastPath.matches(
        "Given what I remember about backup rotation, what should I prioritize "
        "before utilities, and is a $75 purchase safe?"
    )


def test_cross_domain_planning_includes_only_relevant_current_memories():
    from datetime import datetime, timezone

    alice = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    personal = PersonalState()
    personal.add_memory(
        "The server backup runbook needs a restore drill",
        datetime(2026, 8, 3, tzinfo=timezone.utc),
        Provenance.OBSERVED,
    )
    personal.add_memory(
        "The unrelated vacation idea can wait",
        datetime(2026, 8, 4, tzinfo=timezone.utc),
        Provenance.OBSERVED,
    )
    personal.add_memory(
        "A backup note",
        datetime(2026, 8, 5, tzinfo=timezone.utc),
        Provenance.OBSERVED,
    )
    space = HouseholdSpace("apartment", {alice.id})
    space.add_obligation(alice, HouseholdObligation("utilities", "Utilities", 120, alice.id))
    result = CrossDomainPlanningFastPath(
        personal,
        space.snapshot(alice),
        (Task(uuid4(), "apartment", "Review backup runbook", "alice"),),
    ).resolve(
        IntentFrame(
            principal=alice,
            utterance=(
                "Considering my server backup memory, Utilities, and open tasks, "
                "what should I prioritize?"
            ),
        )
    )

    assert result is not None
    assert result.evidence["planning"]["memories"] == [
        {
            "content": "The server backup runbook needs a restore drill",
            "occurred_at": "2026-08-03T00:00:00+00:00",
            "provenance": "observed",
        }
    ]


def test_cross_domain_planning_fast_path_includes_only_derived_finance_fields():
    from datetime import datetime, timezone

    alice = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    personal = PersonalState()
    personal.add_goal("Finish the restore drill", datetime(2026, 8, 2, tzinfo=timezone.utc))
    space = HouseholdSpace("apartment", {alice.id})
    space.add_obligation(alice, HouseholdObligation("utilities", "Utilities", 120, alice.id))

    result = CrossDomainPlanningFastPath(
        personal,
        space.snapshot(alice),
        (Task(uuid4(), "apartment", "Review backup runbook", "alice"),),
        {
            "affordable": True,
            "purchase_cents": 5000,
            "shared_obligations_cents": 120,
            "shortfall_cents": 0,
            "balance_cents": 999999,
        },
    ).resolve(
        IntentFrame(
            principal=alice,
            utterance="Can I afford $50 after Utilities, and what tasks remain?",
        )
    )

    assert result is not None
    planning = result.evidence["planning"]
    assert planning["affordability"] == {
        "affordable": True,
        "purchase_cents": 5000,
        "shared_obligations_cents": 120,
        "shortfall_cents": 0,
    }
    assert "balance_cents" not in repr(planning)


def test_multi_action_fast_path_extracts_bounded_task_chore_plan():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    intent = IntentFrame(
        principal=principal,
        utterance="Create a task and a chore to prepare for the apartment inspection.",
    )

    result = MultiActionFastPath.resolve(intent)

    assert result is None
    assert MultiActionFastPath.task_chore_titles(intent.utterance) == (
        "prepare for the apartment inspection",
        "prepare for the apartment inspection",
    )
    assert not MultiActionFastPath.matches("Create a task to buy cat food")


def test_multi_action_fast_path_preserves_distinct_task_and_chore_titles():
    assert MultiActionFastPath.task_chore_titles(
        "Create a task to review the backup and a chore to clean the utility closet."
    ) == ("review the backup", "clean the utility closet")


def test_contextual_event_mutation_blocks_unsupported_memory_date_resolution():
    result = ContextualMutationGuard.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Create an event for the inspection date from my memory.",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "Provide the event date and time directly" in result.message


@pytest.mark.parametrize(
    "utterance",
    [
        "add the first two",
        "complete those",
        "remove it from my list",
        "go ahead and do it",
        "handle that",
    ],
)
def test_contextual_reference_mutation_fails_closed_instead_of_becoming_a_literal_title(
    utterance,
):
    result = ContextualMutationGuard.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance=utterance,
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "will not guess" in result.message


def test_temporal_next_week_is_not_mistaken_for_a_referent():
    assert (
        ContextualMutationGuard.resolve(
            IntentFrame(
                principal=Principal(id="alice", vault_id="alice-vault"),
                utterance="Add a task to inspect the sunroom latch next week",
            )
        )
        is None
    )
    blocked = ContextualMutationGuard.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Add the next one",
        )
    )
    assert blocked is not None
    assert "will not guess" in blocked.message


def test_multi_action_fast_path_still_blocks_unbounded_compound_requests():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    result = MultiActionFastPath.resolve(
        IntentFrame(
            principal=principal,
            utterance="Create a task and add groceries to prepare for the inspection.",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "multiple actions" in result.message


def test_multi_action_fast_path_blocks_read_plus_mutation_compounds():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    result = MultiActionFastPath.resolve(
        IntentFrame(
            principal=principal,
            utterance="Can I afford $50 after Utilities, and add bread to groceries?",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "multiple actions" in result.message


def test_multi_action_fast_path_blocks_unresolved_action_plus_mutation():
    result = MultiActionFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
            utterance="Can you handle the backup and add rice to groceries?",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "multiple actions" in result.message


def test_multi_action_fast_path_blocks_sequential_mutations():
    result = MultiActionFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
            utterance="Create a task to review the backup, then add rice to groceries.",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "multiple actions" in result.message


def test_multi_action_fast_path_blocks_as_well_as_mutations():
    result = MultiActionFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
            utterance="Add rice to groceries as well as create a task to review the backup.",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "multiple actions" in result.message


def test_multi_action_fast_path_blocks_unresolved_review_plus_mutation():
    result = MultiActionFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
            utterance="Review the backup and add rice to groceries.",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "multiple actions" in result.message


def test_multi_action_fast_path_blocks_while_mutations():
    result = MultiActionFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
            utterance="Add rice to groceries while creating a task to review the backup.",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "multiple actions" in result.message


def test_multi_action_fast_path_extracts_temporal_task_event_plan():
    details = MultiActionFastPath.task_event_details(
        "Create a task and an event for apartment inspection tomorrow."
    )

    assert details is not None
    assert details[0] == "apartment inspection"
    assert details[1] == "apartment inspection"
    assert details[2].endswith("+00:00")


def test_multi_action_fast_path_preserves_distinct_task_and_event_titles():
    details = MultiActionFastPath.task_event_details(
        "Create a task to review the backup and an event to inspect the utility closet tomorrow."
    )

    assert details is not None
    assert details[0:2] == ("review the backup", "inspect the utility closet")
    assert details[2].endswith("+00:00")


def test_multi_action_fast_path_accepts_scheduled_appointment_language():
    details = MultiActionFastPath.task_event_details(
        "Add a task to inspect the fuse box and schedule an appointment to inspect the "
        "fuse box tomorrow."
    )

    assert details is not None
    assert details[0:2] == ("inspect the fuse box", "inspect the fuse box")
    assert details[2].endswith("+00:00")


def test_multi_action_fast_path_grounds_weekday_event_date():
    details = MultiActionFastPath.task_event_details(
        "Add a task to check the linens and schedule an appointment to inspect the smoke alarm "
        "Friday."
    )

    assert details is not None
    assert details[1] == "inspect the smoke alarm"
    scheduled = datetime.fromisoformat(details[2])
    now = datetime.now(timezone.utc)
    assert scheduled.date() > now.date()
    assert scheduled.weekday() == 4


def test_multi_action_fast_path_resolves_dependent_event_pronoun():
    details = MultiActionFastPath.task_event_details(
        "Add a task to check the water softener and schedule an appointment to check it tomorrow."
    )

    assert details is not None
    assert details[0:2] == ("check the water softener", "check the water softener")
    assert details[2].endswith("+00:00")


def test_domain_clarification_fast_path_handles_underspecified_request():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    intent = IntentFrame(principal=principal, utterance="Can you take care of the house stuff?")

    result = DomainClarificationFastPath.resolve(intent)

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "more direction" in result.message


def test_domainless_collection_status_clarifies_without_guessing():
    ambiguous = DomainClarificationFastPath.resolve_ambiguous_collection_status(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Could you show me what is still open?",
        )
    )

    assert ambiguous is not None
    assert ambiguous.state is ObjectiveState.BLOCKED
    assert ambiguous.message == (
        "What should I check: open tasks, household chores, groceries, or memories?"
    )
    assert (
        DomainClarificationFastPath.resolve_ambiguous_collection_status(
            IntentFrame(
                principal=Principal(id="alice", vault_id="alice-vault"),
                utterance="Show my open tasks",
            )
        )
        is None
    )


def test_domainless_take_care_read_clarifies_collection_without_guessing():
    result = DomainClarificationFastPath.resolve_ambiguous_collection_status(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Anything I need to take care of today?",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert result.message == (
        "What should I check: open tasks, household chores, groceries, or memories?"
    )


def test_domainless_pending_read_clarifies_collection_without_guessing():
    result = DomainClarificationFastPath.resolve_ambiguous_collection_status(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Can you show me what is still pending?",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert result.message == (
        "What should I check: open tasks, household chores, groceries, or memories?"
    )


def test_domainless_work_recommendation_clarifies_without_guessing():
    for utterance in ("What should I work on?", "What should I tackle?"):
        result = DomainClarificationFastPath.resolve_ambiguous_collection_status(
            IntentFrame(
                principal=Principal(id="alice", vault_id="alice-vault"),
                utterance=utterance,
            )
        )

        assert result is not None
        assert result.state is ObjectiveState.BLOCKED
        assert result.message == (
            "What should I check: open tasks, household chores, groceries, or memories?"
        )


def test_prioritized_work_recommendation_remains_outside_collection_clarification():
    result = DomainClarificationFastPath.resolve_ambiguous_collection_status(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What should I work on next?",
        )
    )

    assert result is None


def test_implicit_ordinal_mutation_reference_blocks_before_model_selection():
    result = ContextualMutationGuard.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Take care of the first one.",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "will not guess" in result.message


def test_correction_against_persisted_plan_cannot_rewrite_or_execute_history():
    result = PlanModificationFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Actually, change the task to a different one.",
        ),
        Context(
            sources=("authorized_canonical_result",),
            values={
                "canonical_facts": {
                    "plan_steps": [{"index": 0, "state": "completed"}],
                }
            },
        ),
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "explicit new action" in result.message


def test_paraphrased_compound_mutation_is_structurally_plan_shaped():
    assert MultiActionFastPath.matches(
        "Could you set up my Friday prep by creating a task to wipe down the entryway, "
        "adding a chore to check the lights, and scheduling an appointment to inspect the "
        "furnace next Friday?"
    )


def test_domain_clarification_fast_path_gives_reminder_guidance():
    result = DomainClarificationFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Can you make sure I remember to review the backup?",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "reminder as a task" in result.message
    assert "Create a task to review the backup" in result.message


def test_domain_clarification_fast_path_intercepts_bare_reminder_before_cognition():
    bare = DomainClarificationFastPath.resolve_underspecified_reminder(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Can you remind me to check the mail tomorrow?",
        )
    )
    assert bare is not None
    assert bare.state is ObjectiveState.BLOCKED
    assert "reminder as a task" in bare.message

    explicit = DomainClarificationFastPath.resolve_underspecified_reminder(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Put a reminder on my list to inspect the backup",
        )
    )
    assert explicit is None


def test_personal_memory_fast_path_accepts_natural_recent_activity_language():
    state = PersonalState()
    state.add_memory(
        "reviewed the restore drill",
        datetime.now(timezone.utc),
        Provenance.EXPLICIT_USER,
    )
    result = PersonalMemoryFastPath(state).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What have I been up to recently?",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["memories"][0]["content"] == "reviewed the restore drill"


def test_personal_task_composer_grounds_task_in_unique_canonical_goal():
    from datetime import datetime, timezone

    personal = PersonalState()
    personal.add_goal("Finish the restore drill", datetime(2026, 9, 1, tzinfo=timezone.utc))

    title, error = PersonalTaskComposer.resolve("Turn my restore drill goal into a task", personal)

    assert title == "Finish the restore drill"
    assert error is None


def test_personal_task_composer_does_not_guess_between_goals():
    from datetime import datetime, timezone

    personal = PersonalState()
    now = datetime(2026, 9, 1, tzinfo=timezone.utc)
    personal.add_goal("Finish the restore drill", now)
    personal.add_goal("Review the backup architecture", now)

    title, error = PersonalTaskComposer.resolve("Create a task from my goal", personal)

    assert title is None
    assert error == "Which personal goal should I turn into a task? Please name the goal."


def test_personal_chore_composer_grounds_shared_chore_in_goal():
    from datetime import datetime, timezone

    personal = PersonalState()
    personal.add_goal("Finish the restore drill", datetime(2026, 9, 1, tzinfo=timezone.utc))

    title, error = PersonalChoreComposer.resolve(
        "Turn my restore drill goal into a chore", personal
    )

    assert title == "Finish the restore drill"
    assert error is None


def test_personal_memory_task_composer_grounds_task_in_current_memory():
    from datetime import datetime, timezone

    personal = PersonalState()
    personal.add_memory(
        "Review the backup architecture",
        datetime(2026, 9, 1, tzinfo=timezone.utc),
        Provenance.OBSERVED,
    )

    title, error = PersonalMemoryTaskComposer.resolve(
        "Turn my backup architecture memory into a task", personal
    )

    assert title == "Review the backup architecture"
    assert error is None


def test_personal_memory_task_composer_clarifies_equal_memory_matches():
    from datetime import datetime, timezone

    personal = PersonalState()
    occurred_at = datetime(2026, 9, 1, tzinfo=timezone.utc)
    personal.add_memory("Review the server backup", occurred_at, Provenance.OBSERVED)
    personal.add_memory("Investigate the server backup", occurred_at, Provenance.OBSERVED)

    title, error = PersonalMemoryTaskComposer.resolve(
        "Turn my server backup memory into a task", personal
    )

    assert title is None
    assert error == "Which personal memory should I turn into a task? Please name the memory."


def test_personal_memory_chore_composer_grounds_shared_chore_in_current_memory():
    from datetime import datetime, timezone

    personal = PersonalState()
    personal.add_memory(
        "The garage filter needs replacing",
        datetime(2026, 9, 1, tzinfo=timezone.utc),
        Provenance.OBSERVED,
    )

    title, error = PersonalMemoryChoreComposer.resolve(
        "Turn my garage filter memory into a chore", personal
    )

    assert title == "The garage filter needs replacing"
    assert error is None


def test_postgres_household_store_reloads_shared_state_without_persisting_membership():
    from datetime import datetime, timezone

    class Cursor:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class Connection:
        def __init__(self):
            self.payload = None
            self.commits = 0

        def execute(self, query, params=()):
            if query.startswith("SELECT payload"):
                return Cursor((self.payload,) if self.payload is not None else None)
            self.payload = params[1]
            return Cursor(None)

        def commit(self):
            self.commits += 1

    alice = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    bob = Principal(id="bob", vault_id="bob-vault", space_ids=("apartment",))
    when = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
    space = HouseholdSpace("apartment", {alice.id, bob.id})
    space.add_grocery(alice, "rice")
    chore = Chore("chore-1", "Dishes", bob.id)
    event = HouseholdEvent("event-1", "Inspection", when)
    obligation = HouseholdObligation("obligation-1", "Utilities", 120, bob.id)
    space.add_chore(alice, chore)
    space.add_event(alice, event)
    space.add_obligation(alice, obligation)

    connection = Connection()
    store = PostgresHouseholdStore(connection)
    store.save(space)
    restored = store.load("apartment", {alice.id})

    assert restored.groceries == ["rice"]
    assert restored.chores == {chore.chore_id: chore}
    assert restored.events == {event.event_id: event}
    assert restored.obligations == {obligation.obligation_id: obligation}
    assert restored.members == {alice.id}
    assert connection.commits == 1
    try:
        restored.snapshot(bob)
    except PermissionError:
        pass
    else:
        raise AssertionError("reloaded household state accepted a non-member")


def test_household_space_rejects_nonmember_and_private_data_is_not_accepted():
    outsider = Principal(id="mallory", vault_id="mallory-vault", space_ids=())
    space = HouseholdSpace("apartment", {"alice", "bob"})
    for operation in (
        lambda: space.add_grocery(outsider, "private-bank-balance"),
        lambda: space.snapshot(outsider),
    ):
        try:
            operation()
        except PermissionError:
            pass
        else:
            raise AssertionError("nonmember accessed shared household state")


def test_postgres_task_store_persists_lifecycle_and_requires_active_membership():
    from datetime import datetime, timezone

    class Cursor:
        def __init__(self, rows):
            self.rows = rows

        def fetchone(self):
            return self.rows[0] if self.rows else None

        def fetchall(self):
            return self.rows

    class Connection:
        def __init__(self):
            self.rows = {}
            self.commits = 0

        def execute(self, query, params=()):
            if "FROM space_memberships" in query:
                return Cursor([(1,)] if params[0] in {"alice", "bob"} else [])
            if query.startswith("INSERT INTO tasks"):
                self.rows[str(params[0])] = tuple(params)
            elif query.startswith("SELECT id") and "WHERE idempotency_key" in query:
                row = next((row for row in self.rows.values() if row[7] == params[0]), None)
                return Cursor([row] if row else [])
            elif query.startswith("SELECT id") and "WHERE id" in query:
                row = self.rows.get(str(params[0]))
                return Cursor([row] if row else [])
            elif query.startswith("SELECT id"):
                return Cursor(list(self.rows.values()))
            return Cursor([])

        def commit(self):
            self.commits += 1

    alice = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    bob = Principal(id="bob", vault_id="bob-vault", space_ids=("apartment",))
    connection = Connection()
    store = PostgresTaskStore(connection)
    task = store.create(alice, "Inspect backups", datetime(2026, 9, 2, tzinfo=timezone.utc), bob.id)
    assert task.status is TaskStatus.OPEN
    assert (
        store.create(
            alice,
            task.title,
            task.due_at,
            task.assignee_id,
            idempotency_key=task.idempotency_key,
        )
        == task
    )
    try:
        store.create(alice, "Changed title", idempotency_key=task.idempotency_key)
    except ValueError:
        pass
    else:
        raise AssertionError("idempotency key was rebound to changed task arguments")
    assert store.list(bob)[0] == task
    assert store.complete(bob, task.task_id).status is TaskStatus.COMPLETED
    try:
        store.list(Principal(id="mallory", vault_id="mallory-vault"))
    except PermissionError:
        pass
    else:
        raise AssertionError("task store accepted a non-member")


def test_task_executor_and_verifier_use_generic_observation_contract():
    class Store:
        def __init__(self):
            self.task = None
            self.key = None

        def create(self, principal, title, idempotency_key=None):
            self.key = idempotency_key
            self.task = Task(
                uuid4(), "apartment", title, principal.id, idempotency_key=idempotency_key
            )
            return self.task

        def get(self, principal, task_id):
            return self.task if self.task and self.task.task_id == task_id else None

    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    store = Store()
    executor = PostgresTaskExecutor(store, principal)
    request = ExecutionRequest(
        objective_id=uuid4(),
        action_id=uuid4(),
        action=ActionSpec(
            action_id="tasks.create",
            capability="tasks.create",
            arguments={"title": "Review backups"},
            verification=VerificationContract(kind="readback"),
        ),
        idempotency_key="correlation:tasks.create",
    )
    observation = executor.execute(request)
    assert observation.command_succeeded
    assert store.key == request.idempotency_key
    verification = PostgresTaskVerifier(store, principal).verify(
        observation, request.action.verification
    )
    assert verification.verified
    assert verification.evidence["canonical_status"] == TaskStatus.OPEN.value


def test_task_executor_and_verifier_preserve_due_at():
    class Store:
        def __init__(self):
            self.task = None

        def create(self, principal, title, due_at=None, idempotency_key=None):
            self.task = Task(
                uuid4(),
                "apartment",
                title,
                principal.id,
                due_at=due_at,
                idempotency_key=idempotency_key or "",
            )
            return self.task

        def get(self, _principal, task_id):
            return self.task if self.task and self.task.task_id == task_id else None

    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    due_at = datetime(2026, 9, 2, 22, 0, tzinfo=timezone.utc)
    request = ExecutionRequest(
        objective_id=uuid4(),
        action_id=uuid4(),
        action=ActionSpec(
            action_id="tasks.create",
            capability="tasks.create",
            arguments={"title": "Review the restore drill", "due_at": due_at.isoformat()},
            verification=VerificationContract(kind="readback"),
        ),
        idempotency_key="correlation:tasks.create.due",
    )
    store = Store()

    observation = PostgresTaskExecutor(store, principal).execute(request)
    verification = PostgresTaskVerifier(store, principal).verify(
        observation, request.action.verification
    )

    assert observation.command_succeeded
    assert observation.evidence["due_at"] == due_at.isoformat()
    assert verification.verified


def test_task_completion_executor_resolves_unique_title_and_verifies_status():
    class Store:
        def __init__(self):
            self.task = Task(uuid4(), "apartment", "Verify backup retention", "alice")

        def list(self, _principal):
            return (self.task,)

        def complete(self, _principal, task_id):
            assert task_id == self.task.task_id
            self.task = Task(
                self.task.task_id,
                self.task.space_id,
                self.task.title,
                self.task.created_by,
                status=TaskStatus.COMPLETED,
            )
            return self.task

        def get(self, _principal, task_id):
            return self.task if task_id == self.task.task_id else None

    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    store = Store()
    request = ExecutionRequest(
        objective_id=uuid4(),
        action_id=uuid4(),
        action=ActionSpec(
            action_id="tasks.complete",
            capability="tasks.complete",
            arguments={"title": "verify backup retention."},
            verification=VerificationContract(kind="readback"),
        ),
        idempotency_key="correlation:tasks.complete",
    )

    observation = PostgresTaskExecutor(store, principal).execute(request)
    verification = PostgresTaskVerifier(store, principal).verify(
        observation, request.action.verification
    )

    assert observation.command_succeeded
    assert observation.evidence["status"] == TaskStatus.COMPLETED.value
    assert verification.verified
    assert verification.evidence["canonical_status"] == TaskStatus.COMPLETED.value


def test_task_completion_executor_blocks_ambiguous_title():
    class Store:
        def list(self, _principal):
            return (
                Task(uuid4(), "apartment", "Check the drill", "alice"),
                Task(uuid4(), "apartment", "Check the drill", "alice"),
            )

    request = ExecutionRequest(
        objective_id=uuid4(),
        action_id=uuid4(),
        action=ActionSpec(
            action_id="tasks.complete",
            capability="tasks.complete",
            arguments={"title": "check the drill"},
            verification=VerificationContract(kind="readback"),
        ),
        idempotency_key="correlation:tasks.complete.ambiguous",
    )

    observation = PostgresTaskExecutor(
        Store(), Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    ).execute(request)

    assert not observation.command_succeeded
    assert observation.evidence["ambiguous_task_title"] is True


def test_task_completion_executor_accepts_grounded_task_id_for_duplicate_titles():
    first = Task(uuid4(), "apartment", "Check the drill", "alice")
    second = Task(uuid4(), "apartment", "Check the drill", "alice")

    class Store:
        def list(self, _principal):
            return (first, second)

        def get(self, _principal, task_id):
            return next((task for task in (first, second) if task.task_id == task_id), None)

        def complete(self, _principal, task_id):
            assert task_id == first.task_id
            return first.__class__(
                first.task_id,
                first.space_id,
                first.title,
                first.created_by,
                first.assignee_id,
                first.due_at,
                TaskStatus.COMPLETED,
                first.idempotency_key,
            )

    request = ExecutionRequest(
        objective_id=uuid4(),
        action_id=uuid4(),
        action=ActionSpec(
            action_id="tasks.complete",
            capability="tasks.complete",
            arguments={"title": "Check the drill", "task_id": str(first.task_id)},
            verification=VerificationContract(kind="readback"),
        ),
        idempotency_key="correlation:tasks.complete.grounded-id",
    )

    observation = PostgresTaskExecutor(
        Store(), Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    ).execute(request)

    assert observation.command_succeeded
    assert observation.evidence["task_id"] == str(first.task_id)
    assert observation.evidence["status"] == TaskStatus.COMPLETED.value


def test_task_completion_fast_path_clarifies_missing_or_duplicate_titles():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    intent = IntentFrame(principal=principal, utterance="Complete the task Check the drill")
    tasks = (
        Task(uuid4(), "apartment", "Check the drill", "alice"),
        Task(uuid4(), "apartment", "Check the drill", "alice"),
    )

    duplicate = TaskCompletionFastPath.resolve(intent, "check the drill", tasks)
    missing = TaskCompletionFastPath.resolve(intent, "missing task", tasks)

    assert duplicate is not None
    assert duplicate.state is ObjectiveState.BLOCKED
    assert "multiple tasks" in duplicate.message
    assert missing is not None
    assert "couldn't find" in missing.message


def test_task_completion_grounding_does_not_import_unmentioned_canonical_title():
    from aegis.tasks import TaskCompletionFastPath

    assert TaskCompletionFastPath.target_is_grounded(
        "The library card errand is done", "renew the library card"
    )
    assert not TaskCompletionFastPath.target_is_grounded("Finish the backup", "restore drill")


def test_task_completion_canonicalizes_unique_leading_article_omission():
    tasks = (Task(uuid4(), "apartment", "the annual inspection", "alice"),)

    assert TaskCompletionFastPath.canonical_title("annual inspection", tasks) == (
        "the annual inspection"
    )


def test_task_completion_canonicalizes_unique_prefix_reference():
    tasks = (
        Task(
            uuid4(),
            "apartment",
            "browser activity refresh verification 2026-09-01",
            "alice",
        ),
    )

    assert (
        TaskCompletionFastPath.canonical_title("browser activity refresh verification", tasks)
        == "browser activity refresh verification 2026-09-01"
    )


def test_chore_completion_fast_path_clarifies_duplicate_titles():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    intent = IntentFrame(principal=principal, utterance="Complete the chore Clean the closet")
    chores = (
        Chore("chore-1", "Clean the closet", "alice"),
        Chore("chore-2", "Clean the closet", "alice"),
    )

    result = ChoreCompletionFastPath.resolve(intent, "clean the closet", chores)

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "multiple chores" in result.message


def test_finance_ledger_keeps_private_accounts_and_allows_explicit_derived_contribution():
    from datetime import datetime, timezone

    ledger = FinanceLedger()
    ledger.record_snapshot(
        FinanceSnapshot(
            "alice",
            (Account("checking", "alice", 500_000),),
            (Transaction("t1", "checking", -2500, datetime.now(timezone.utc), "groceries"),),
        )
    )
    alice = Principal(id="alice", vault_id="alice-vault")
    bob = Principal(id="bob", vault_id="bob-vault")
    assert ledger.total_balance(alice, "alice") == 500_000
    contribution = ledger.derived_contribution(alice, "alice", "apartment", 2500, True)
    assert contribution.owner_id == "alice"
    assert contribution.source_resource_id == "finance-contribution:apartment"
    try:
        ledger.private_snapshot(bob, "alice")
    except PermissionError:
        pass
    else:
        raise AssertionError("private finance account crossed Vault boundary")


def test_postgres_finance_store_reloads_snapshot_and_keeps_owner_boundary():
    from datetime import datetime, timezone

    class Cursor:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class Connection:
        def __init__(self, payload=None):
            self.payload = payload
            self.provider_id = None
            self.captured_at = None
            self.commits = 0

        def execute(self, query, params=()):
            if query.startswith("SELECT payload"):
                if self.payload is None:
                    return Cursor(None)
                return Cursor((self.payload, self.provider_id, self.captured_at))
            self.payload, self.provider_id, self.captured_at = params[1:]
            return Cursor(None)

        def commit(self):
            self.commits += 1

    captured = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
    snapshot = FinanceSnapshot(
        "alice",
        (Account("checking", "alice", 500_000),),
        (Transaction("transaction-1", "checking", -2500, captured, "groceries"),),
        "sandbox-bank",
        captured,
    )
    writer = Connection()
    PostgresFinanceSnapshotStore(writer).save(snapshot)
    reader = Connection(writer.payload)
    reader.provider_id = writer.provider_id
    reader.captured_at = writer.captured_at
    ledger = FinanceLedger(PostgresFinanceSnapshotStore(reader))
    restored = ledger.private_snapshot(Principal(id="alice", vault_id="alice-vault"), "alice")
    assert restored == snapshot
    assert ledger.total_balance(Principal(id="alice", vault_id="alice-vault"), "alice") == 500_000
    try:
        ledger.private_snapshot(Principal(id="bob", vault_id="bob-vault"), "alice")
    except PermissionError:
        pass
    else:
        raise AssertionError("durable finance snapshot crossed owner boundary")
    assert writer.commits == 1


def test_finance_projection_requires_explicit_derive_authorization():
    ledger = FinanceLedger()
    ledger.record_snapshot(FinanceSnapshot("alice", (Account("checking", "alice", 1),)))
    try:
        ledger.derived_contribution(
            Principal(id="alice", vault_id="alice-vault"), "alice", "apartment", 1, False
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("finance data was projected without authorization")


def test_cross_domain_affordability_returns_derived_fields_without_private_balance():
    ledger = FinanceLedger()
    ledger.record_snapshot(FinanceSnapshot("alice", (Account("checking", "alice", 500_000),)))
    projection = ledger.assess_affordability(
        Principal(id="alice", vault_id="alice-vault"),
        "alice",
        100_000,
        (SharedObligation("rent", 200_000),),
        reserve_cents=100_000,
    )
    assert isinstance(projection, AffordabilityProjection)
    assert projection.affordable
    assert projection.shared_obligations_cents == 200_000
    assert projection.shortfall_cents == 0
    assert "balance" not in projection.__dict__


def test_finance_read_fast_path_recognizes_purchase_safety_language():
    assert FinanceReadFastPath.matches("Is a $75 purchase safe?")
    assert not FinanceReadFastPath.matches("Is the backup purchase safe?")


def test_finance_fast_path_yields_compound_questions_to_bounded_cognition():
    assert not FinanceReadFastPath.matches(
        "What do I still need to handle, and can I afford a $5 purchase?"
    )
    assert FinanceReadFastPath.matches("Can I afford a $5 purchase?")
    assert FinanceReadFastPath.matches("Would spending twenty dollars be okay for me?")
    assert FinanceReadFastPath.amount_cents("Would spending twenty five dollars be okay?") == 2500
    assert FinanceReadFastPath.amount_cents("spend thirty-five bucks") == 3500
    assert FinanceReadFastPath.matches("Is it okay to spend thirty-five bucks?")


def test_finance_fast_path_blocks_unsupported_general_balance_reads():
    assert FinanceReadFastPath.unsupported_balance_read("What is my current balance?")
    assert not FinanceReadFastPath.unsupported_balance_read("Can I afford a $5 purchase?")


def test_household_projection_handles_multi_member_settlement_remainder():
    class Policy:
        def may_derive(self, requester, owner_id, space_id):
            return True

    projection = PrivacyProjectionService(Policy()).build(
        Principal(id="alice", vault_id="alice-vault"),
        "apartment",
        ("alice", "bob", "cara"),
        (SharedObligation("utilities", 100),),
        (),
    )
    assert projection.equal_share == 33
    assert projection.settlements == {"alice": -34, "bob": -33, "cara": -33}


def test_postgres_projection_store_roundtrips_allowlisted_fields_and_checks_membership():
    import json

    class Cursor:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class Connection:
        def __init__(self):
            self.payload = None
            self.commits = 0

        def execute(self, query, params=()):
            if query.startswith("SELECT payload"):
                return Cursor((self.payload,) if self.payload is not None else None)
            self.payload = json.loads(params[1])
            return Cursor(None)

        def commit(self):
            self.commits += 1

    projection = HouseholdProjection(
        "apartment", 2200, 1100, {"alice": 500}, {"alice": -600, "bob": 1100}
    )
    connection = Connection()
    store = PostgresProjectionStore(connection)
    store.save(projection)
    restored = store.load(
        "apartment",
        Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
        {"alice", "bob"},
    )
    assert restored == projection
    assert "balance" not in connection.payload
    try:
        store.load("apartment", Principal(id="mallory", vault_id="mallory-vault"), {"alice", "bob"})
    except PermissionError:
        pass
    else:
        raise AssertionError("projection store accepted a non-member")
    assert connection.commits == 1


def test_finance_snapshot_exposes_provider_provenance_only_to_owner():
    from datetime import datetime, timezone

    ledger = FinanceLedger()
    captured = datetime(2026, 8, 31, tzinfo=timezone.utc)
    ledger.record_snapshot(
        FinanceSnapshot(
            "alice", (Account("checking", "alice", 1),), provider_id="sandbox", captured_at=captured
        )
    )
    alice = Principal(id="alice", vault_id="alice-vault")
    bob = Principal(id="bob", vault_id="bob-vault")
    assert ledger.provenance(alice, "alice") == {
        "provider_id": "sandbox",
        "captured_at": captured.isoformat(),
    }
    try:
        ledger.provenance(bob, "alice")
    except PermissionError:
        pass
    else:
        raise AssertionError("finance provenance crossed Vault boundary")


def test_network_reachability_and_discovery_do_not_grant_action_authority():
    inventory = HomelabInventory()
    inventory.record_discovery(DiscoveredDevice("10.0.0.5", "atlas", ("https",)))
    inventory.record_discovery(DiscoveredDevice("10.0.1.5", "unknown", ("ssh",)))
    inventory.add_scope(AuthorizedNetworkScope("home-lab", ("10.0.0.0/24",), "owned lab"))
    assert inventory.authorized_devices("home-lab")[0].address == "10.0.0.5"
    inventory.require_action_scope("home-lab", "10.0.0.5")
    try:
        inventory.require_action_scope("home-lab", "10.0.1.5")
    except ScopeDenied:
        pass
    else:
        raise AssertionError("discovered device outside scope was authorized")


def test_inactive_or_unknown_network_scope_fails_closed():
    inventory = HomelabInventory()
    inventory.add_scope(AuthorizedNetworkScope("lab", ("192.168.1.0/24",), "lab", active=False))
    for scope_id, address in (("lab", "192.168.1.2"), ("missing", "192.168.1.2")):
        try:
            inventory.require_action_scope(scope_id, address)
        except ScopeDenied:
            pass
        else:
            raise AssertionError("inactive or unknown network scope allowed action")


def test_security_lab_requires_explicit_scope_before_recording_evidence():
    inventory = HomelabInventory()
    inventory.add_scope(AuthorizedNetworkScope("owned-lab", ("192.0.2.0/24",), "CTF lab"))
    lab = SecurityLab(inventory)
    finding = lab.record_finding(
        "owned-lab", "192.0.2.10", "service banner observed", ("banner=fixture",)
    )
    assert finding.target == "192.0.2.10"
    try:
        lab.record_finding("owned-lab", "192.0.3.10", "reachable target", ("ping",))
    except PermissionError:
        pass
    else:
        raise AssertionError("reachable but out-of-scope target was accepted")


def test_postgres_security_lab_store_reloads_owner_scoped_finding():
    import json

    class Cursor:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class Connection:
        def __init__(self):
            self.row = None
            self.commits = 0

        def execute(self, query, params=()):
            if query.startswith("SELECT 1"):
                return Cursor((1,))
            if query.startswith("SELECT owner_id"):
                return Cursor(self.row)
            self.row = (params[1], json.loads(params[3]))
            return Cursor(None)

        def commit(self):
            self.commits += 1

    inventory = HomelabInventory()
    inventory.add_scope(AuthorizedNetworkScope("lab", ("192.0.2.0/24",), "fixture lab"))
    finding = SecurityLab(inventory).record_finding(
        "lab", "192.0.2.10", "fixture service observed", ("banner=fixture",)
    )
    connection = Connection()
    store = PostgresSecurityLabStore(connection)
    alice = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    store.save(alice, finding)
    assert store.load(alice, finding.finding_id) == finding
    try:
        store.load(Principal(id="bob", vault_id="bob-vault"), finding.finding_id)
    except PermissionError:
        pass
    else:
        raise AssertionError("security finding crossed owner boundary")
    assert connection.commits == 1


def test_network_scope_policy_denies_reachable_target_outside_persisted_scope():
    class BasePolicy:
        def authorize(self, request):
            return PolicyDecision(allowed=True, reason="Space permission allows probe")

    class Store:
        def load(self, principal):
            inventory = HomelabInventory()
            inventory.add_scope(AuthorizedNetworkScope("lab", ("127.0.0.0/8",), "owned lab"))
            return inventory

    request = AuthorizationRequest(
        principal=Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
        objective_id=uuid4(),
        action=ActionSpec(
            action_id="network.probe",
            capability="network.probe",
            arguments={"address": "192.0.2.1", "port": 80, "scope_id": "lab"},
            required_permissions=("network.read",),
            verification=VerificationContract(kind="health"),
        ),
    )

    decision = NetworkScopePolicy(BasePolicy(), Store()).authorize(request)
    assert not decision.allowed


def test_fixture_homelab_runtime_requires_restart_before_health():
    service = Service("plex", "host", "Plex", "http://127.0.0.1:32400")
    provider = FixtureHomelabRuntime()

    assert provider.health(service) is False
    assert provider.restart(service) is True
    assert provider.health(service) is True


def test_homelab_restart_requires_scope_and_health_verification():
    class Runtime:
        def __init__(self, healthy):
            self.healthy = healthy
            self.restarts = 0

        def restart(self, service):
            self.restarts += 1
            return True

        def health(self, service):
            return self.healthy

    inventory = HomelabInventory()
    inventory.add_scope(AuthorizedNetworkScope("lab", ("10.0.0.0/24",), "owned lab"))
    runtime = Runtime(True)
    pack = HomelabPack(inventory, runtime)
    pack.add_host(Host("atlas", "10.0.0.5", "atlas"))
    pack.add_service(Service("plex", "atlas", "Plex", "https://10.0.0.5/health"))
    result = pack.restart_service("lab", "plex")
    assert result.attempted and result.verified and runtime.restarts == 1
    runtime.healthy = False
    failed = pack.restart_service("lab", "plex")
    assert failed.attempted and not failed.verified


def test_postgres_homelab_store_reloads_space_inventory():
    import json

    class Cursor:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class Connection:
        def __init__(self):
            self.payload = None
            self.commits = 0

        def execute(self, query, params=()):
            if query.startswith("SELECT 1"):
                return Cursor((1,))
            if query.startswith("SELECT payload"):
                return Cursor((self.payload,) if self.payload is not None else None)
            self.payload = json.loads(params[1])
            return Cursor(None)

        def commit(self):
            self.commits += 1

    class Runtime:
        def restart(self, service):
            return True

        def health(self, service):
            return True

    connection = Connection()
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    pack = HomelabPack(HomelabInventory(), Runtime())
    pack.add_host(Host("atlas", "192.0.2.10", "atlas", {"ram_gb": 64}))
    pack.add_service(Service("plex", "atlas", "Plex", "http://192.0.2.10:32400/health"))
    store = PostgresHomelabStore(connection)
    store.save(principal, pack)
    restored = store.load(principal, Runtime())
    assert restored.hosts["atlas"].resources == {"ram_gb": 64}
    assert restored.services["plex"].health_endpoint.endswith("/health")
    assert connection.commits == 1


def test_homelab_restart_outside_authorized_scope_never_reaches_runtime():
    class Runtime:
        def restart(self, service):
            raise AssertionError("out-of-scope service was restarted")

        def health(self, service):
            return False

    inventory = HomelabInventory()
    inventory.add_scope(AuthorizedNetworkScope("lab", ("10.0.0.0/24",), "owned lab"))
    pack = HomelabPack(inventory, Runtime())
    pack.add_host(Host("other", "10.0.1.5", "other"))
    pack.add_service(Service("ssh", "other", "SSH", "https://10.0.1.5/health"))
    try:
        pack.restart_service("lab", "ssh")
    except ScopeDenied:
        pass
    else:
        raise AssertionError("out-of-scope Homelab action was allowed")


def test_osint_findings_require_grounded_sources():
    from datetime import datetime, timezone

    investigation = Investigation()
    source = investigation.add_source(
        "https://example.test/report", "Report", datetime.now(timezone.utc)
    )
    finding = investigation.add_finding("The report states X", (source.source_id,), 0.9)
    assert finding.source_ids == (source.source_id,)
    try:
        investigation.add_finding("Unsupported claim", (uuid4(),), 1)
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported OSINT finding was accepted")


def test_postgres_osint_store_reloads_sources_and_denies_other_owner():
    import json

    class Cursor:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class Connection:
        def __init__(self):
            self.row = None
            self.commits = 0

        def execute(self, query, params=()):
            if query.startswith("SELECT owner_id"):
                return Cursor(self.row)
            self.row = (params[1], json.loads(params[2]))
            return Cursor(None)

        def commit(self):
            self.commits += 1

    investigation = Investigation()
    source = investigation.add_source(
        "https://example.test/source", "Fixture source", datetime.now(timezone.utc)
    )
    finding = investigation.add_finding("Observed fixture fact", (source.source_id,), 0.8)
    connection = Connection()
    store = PostgresInvestigationStore(connection)
    alice = Principal(id="alice", vault_id="alice-vault")
    store.save(alice, investigation)
    assert store.load(alice, investigation.investigation_id).findings[finding.finding_id] == finding
    try:
        store.load(Principal(id="bob", vault_id="bob-vault"), investigation.investigation_id)
    except PermissionError:
        pass
    else:
        raise AssertionError("OSINT investigation crossed owner boundary")
    assert connection.commits == 1


def test_forge_proposes_capability_gap_but_cannot_self_install():
    proposal = Forge().propose(CapabilityGap("trading cards", "alice"))
    assert proposal.pack_id.startswith("generated-")
    assert proposal.requires_approval
    try:
        Forge().install(proposal, approved=False)
    except PermissionError:
        pass
    else:
        raise AssertionError("Forge installed without explicit approval")


def test_forge_lifecycle_requires_provenance_sandbox_tests_and_owner_approval():
    proposal = Forge().propose(CapabilityGap("trading cards", "alice"))
    lifecycle = ForgeLifecycle()
    record = lifecycle.validate(
        proposal, license_class="INTERFACE_ONLY", sandbox_passed=True, tests_passed=True
    )
    assert record.status is ForgeStatus.VALIDATED
    try:
        lifecycle.approve(proposal.pack_id, owner_approved=False)
    except PermissionError:
        pass
    else:
        raise AssertionError("Forge proposal was approved without owner consent")
    try:
        lifecycle.install(proposal.pack_id)
    except PermissionError:
        pass
    else:
        raise AssertionError("rejected Forge proposal was installed")


def test_forge_rejects_unknown_license_or_failed_sandbox():
    proposal = Forge().propose(CapabilityGap("cards", "alice"))
    for kwargs in (
        {"license_class": "UNKNOWN", "sandbox_passed": True, "tests_passed": True},
        {"license_class": "COPY_SAFE", "sandbox_passed": False, "tests_passed": True},
    ):
        try:
            ForgeLifecycle().validate(proposal, **kwargs)
        except (PermissionError, ValueError):
            pass
        else:
            raise AssertionError("unsafe Forge proposal was validated")


def test_forge_trading_card_proposal_materializes_only_after_approval():
    proposal = Forge().propose(CapabilityGap("trading cards", "alice", ("collection.write",)))
    lifecycle = ForgeLifecycle()
    with_approval = lifecycle.validate(
        proposal, license_class="INTERFACE_ONLY", sandbox_passed=True, tests_passed=True
    )
    assert with_approval.status is ForgeStatus.VALIDATED
    try:
        lifecycle.to_bundle(proposal.pack_id, ())
    except PermissionError:
        pass
    else:
        raise AssertionError("uninstalled Forge proposal became a Pack")

    lifecycle.approve(proposal.pack_id, owner_approved=True)
    lifecycle.install(proposal.pack_id)
    card = card_collection_card(proposal.pack_id)
    bundle = lifecycle.to_bundle(proposal.pack_id, (card,))
    manager = PackManager()
    manager.discover(bundle)
    manager.install(proposal.pack_id, frozenset({"collection.write"}))
    manager.enable(proposal.pack_id)
    assert manager.retrieve("trading") == (card,)

    collection = CardCollection()
    executor = CardCollectionExecutor(collection, card.action.action_id)
    verifier = CardCollectionVerifier(collection)
    action = card.action.model_copy(update={"arguments": {"name": "Pikachu"}})
    request = ExecutionRequest(
        objective_id=uuid4(), action_id=uuid4(), action=action, idempotency_key="card-1"
    )
    observation = executor.execute(request)
    assert verifier.verify(observation, action.verification).verified
    assert collection.items == [{"name": "Pikachu", "set": None, "number": None}]


def test_ambient_suggestions_do_not_execute_actions_and_platform_is_policy_gated():
    class Platform:
        def __init__(self):
            self.notifications = []
            self.tasks = []

        def deliver_notification(self, notification):
            self.notifications.append(notification)

        def schedule_background(self, task):
            self.tasks.append(task)

        def cancel_background(self, task):
            self.tasks.remove(task)

    class Policy:
        def __init__(self, allowed):
            self.allowed = allowed

        def allow_notification(self, notification):
            return self.allowed

        def allow_background_task(self, task):
            return self.allowed

    platform = Platform()
    service = AmbientService(platform, Policy(False))
    suggestion = service.propose(
        "rice is low",
        "Add rice to groceries",
        ActionSpec(action_id="kitchen.groceries.add", capability="kitchen.groceries.add"),
    )
    assert suggestion.proposed_action is not None
    assert platform.notifications == []
    try:
        service.deliver(
            Notification(recipient_ids=("alice",), text="Rice is low", correlation_id=uuid4())
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("ambient notification bypassed policy")
    try:
        service.schedule(
            BackgroundTask(
                run_at=datetime.now(timezone.utc),
                task_type="suggestion.refresh",
                idempotency_key="ambient-1",
            )
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("ambient task bypassed policy")

    allowed = AmbientService(platform, Policy(True))
    notification = Notification(
        recipient_ids=("alice",), text="Rice is low", correlation_id=uuid4()
    )
    allowed.deliver(notification)
    assert platform.notifications == [notification]

    assert allowed.deliver(notification) is False
    assert (
        allowed.schedule(
            BackgroundTask(
                run_at=datetime.now(timezone.utc),
                task_type="suggestion.refresh",
                idempotency_key="ambient-3",
            )
        )
        is True
    )
    assert (
        allowed.schedule(
            BackgroundTask(
                run_at=datetime.now(timezone.utc),
                task_type="suggestion.refresh",
                idempotency_key="ambient-3",
            )
        )
        is False
    )


def test_home_assistant_adapter_reads_state_and_never_bypasses_command_policy():
    class Gateway:
        def __init__(self):
            self.commands = []

        def get_state(self, entity_id):
            return {"state": "on", "attributes": {"friendly_name": "Lamp"}}

        def call_service(self, command):
            self.commands.append(command)

    class Policy:
        def __init__(self, allowed):
            self.allowed = allowed

        def allow_command(self, command):
            return self.allowed

    gateway = Gateway()
    adapter = HomeAssistantAdapter(gateway, Policy(False))
    state = adapter.read_state("light.lamp", datetime.now(timezone.utc))
    assert state.state == "on"
    assert state.provenance is Provenance.OBSERVED
    command = DeviceCommand(entity_id="light.lamp", service="turn_off")
    try:
        adapter.execute(command, datetime.now(timezone.utc))
    except PermissionError:
        pass
    else:
        raise AssertionError("Home Assistant command bypassed policy")
    assert gateway.commands == []

    allowed_adapter = HomeAssistantAdapter(gateway, Policy(True))
    verified = allowed_adapter.execute(
        command.model_copy(update={"expected_state": "on"}), datetime.now(timezone.utc)
    )
    assert verified.accepted is True
    assert verified.verified is True
    unverified = allowed_adapter.execute(command, datetime.now(timezone.utc))
    assert unverified.accepted is True
    assert unverified.verified is False


def test_ambient_platform_failure_releases_claim_for_safe_retry():
    class Platform:
        def __init__(self):
            self.fail = True
            self.tasks = []

        def deliver_notification(self, notification):
            pass

        def schedule_background(self, task):
            if self.fail:
                self.fail = False
                raise RuntimeError("gateway unavailable")
            self.tasks.append(task)

        def cancel_background(self, task):
            self.tasks.remove(task)

    class Policy:
        def allow_notification(self, notification):
            return True

        def allow_background_task(self, task):
            return True

    platform = Platform()
    service = AmbientService(platform, Policy())
    task = BackgroundTask(
        run_at=datetime.now(timezone.utc), task_type="suggestion.refresh", idempotency_key="retry"
    )
    try:
        service.schedule(task)
    except RuntimeError:
        pass
    else:
        raise AssertionError("flaky ambient platform unexpectedly succeeded")
    assert service.schedule(task) is True


def test_sqlite_ambient_state_survives_restart_and_suppresses_replay(tmp_path):
    path = str(tmp_path / "ambient.sqlite")
    first = SqliteAmbientState(path)
    assert first.claim("background:stable") is True
    first.close()
    second = SqliteAmbientState(path)
    assert second.claim("background:stable") is False
    second.release("background:stable")
    assert second.claim("background:stable") is True
    second.close()


def test_household_proactivity_composes_grounded_suggestion_without_executing():
    suggestion = HouseholdProactivity.suggest_meal(
        HouseholdSignals(
            space_id="apartment",
            low_groceries=("rice",),
            expiring_ingredients=("chicken",),
            shared_day="Wednesday",
        )
    )
    assert suggestion is not None
    assert "chicken" in suggestion.text
    assert suggestion.proposed_action is not None
    assert suggestion.proposed_action.arguments == {"item": "rice"}
    assert (
        HouseholdProactivity.suggest_meal(
            HouseholdSignals(space_id="apartment", low_groceries=("rice",))
        )
        is None
    )


def test_sqlite_backup_restore_preserves_canonical_objective_state(tmp_path):
    source_path = str(tmp_path / "source.sqlite")
    backup_path = str(tmp_path / "backup.sqlite")
    restored_path = str(tmp_path / "restored.sqlite")
    original = SqliteAmbientState(source_path)
    assert original.claim("notification:one") is True
    original.close()
    backup_sqlite(source_path, backup_path)
    restore_sqlite(backup_path, restored_path)
    restored = SqliteAmbientState(restored_path)
    assert restored.claim("notification:one") is False
    restored.close()


def test_config_requires_infrastructure_and_does_not_expose_database_secret(monkeypatch):
    for name, value in {
        "AEGIS_DATABASE_URL": "postgres://user:password@example/db",
        "AEGIS_KEYCLOAK_URL": "https://keycloak.test",
        "AEGIS_OPENFGA_URL": "https://openfga.test",
        "AEGIS_OPENCLAW_GATEWAY_URL": "wss://openclaw.test",
    }.items():
        monkeypatch.setenv(name, value)
    config = AegisConfig.from_environment()
    assert config.database_url.get_secret_value().endswith("/db")
    assert "password" not in str(config)


def test_config_allows_local_alpha_without_optional_services(monkeypatch):
    monkeypatch.delenv("AEGIS_KEYCLOAK_URL", raising=False)
    monkeypatch.delenv("AEGIS_OPENFGA_URL", raising=False)
    monkeypatch.delenv("AEGIS_OPENCLAW_GATEWAY_URL", raising=False)
    monkeypatch.delenv("AEGIS_ENVIRONMENT", raising=False)
    monkeypatch.setenv("AEGIS_DATABASE_URL", "postgres://user:password@example/db")

    config = AegisConfig.from_environment()

    assert config.environment == "development"
    assert config.keycloak_url is None
    assert config.openfga_url is None
    assert config.openclaw_gateway_url is None


def test_config_requires_authority_services_in_production(monkeypatch):
    monkeypatch.setenv("AEGIS_DATABASE_URL", "postgres://user:password@example/db")
    monkeypatch.setenv("AEGIS_ENVIRONMENT", "production")
    monkeypatch.delenv("AEGIS_KEYCLOAK_URL", raising=False)
    monkeypatch.delenv("AEGIS_OPENFGA_URL", raising=False)

    with pytest.raises(ValueError, match="production configuration requires"):
        AegisConfig.from_environment()


def test_health_report_separates_required_readiness_from_optional_health():
    report = HealthService(
        (
            ("postgres", True, True, "connected"),
            ("home_assistant", False, False, "not configured"),
        )
    ).report()
    assert report.healthy is False
    assert report.ready is True
    assert report.components[1].required is False

    failed = HealthService((("postgres", False, True, "unavailable"),)).report()
    assert failed.healthy is False
    assert failed.ready is False


def test_release_truth_rejects_inconsistent_runtime_pointers():
    from aegis.release_truth import validate_state_pointers

    errors = validate_state_pointers(
        {
            "repository_head_sha": "abcdef1",
            "deterministic_green_sha": "abcdef1",
            "last_pushed_sha": "abcdef1",
            "hosted_ci_green_sha": "abcdef1",
            "installed_release_sha": None,
            "running_release_sha": "abcdef1",
            "live_green_sha": "abcdef1",
        }
    )
    assert "running_release_sha requires installed_release_sha" in errors


def test_release_truth_rejects_owner_runtime_pointer_drift():
    from aegis.release_truth import validate_state_pointers

    errors = validate_state_pointers(
        {
            "repository_head_sha": "abcdef1",
            "deterministic_green_sha": "abcdef1",
            "last_pushed_sha": "abcdef1",
            "hosted_ci_green_sha": "abcdef1",
            "installed_release_sha": "1234567",
            "running_release_sha": "1234567",
            "live_green_sha": "abcdef1",
            "owner_dogfood_runtime": {
                "repository_head_sha": "abcdef1",
                "deterministic_green_sha": "abcdef1",
                "installed_release_sha": "7654321",
                "running_release_sha": "1234567",
                "live_green_sha": "abcdef1",
            },
        }
    )
    assert (
        "installed_release_sha disagrees with owner_dogfood_runtime.installed_release_sha" in errors
    )


def test_runtime_identity_does_not_expose_configuration_secrets(monkeypatch):
    from aegis import cli

    monkeypatch.setenv("AEGIS_RELEASE_SHA", "abcdef1")
    monkeypatch.setenv("AEGIS_OLLAMA_URL", "https://user:secret@example.test:11434/path")
    monkeypatch.setenv("AEGIS_OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setattr(
        cli,
        "_postgres_health",
        lambda _url: cli.ComponentHealth(
            name="postgres", healthy=True, required=True, detail="connected"
        ),
    )
    monkeypatch.setattr(
        cli,
        "_ollama_health",
        lambda _url, _model: cli.ComponentHealth(
            name="ollama", healthy=True, required=True, detail="available"
        ),
    )
    monkeypatch.setattr(cli, "_identity_health", lambda: (True, "local development identity mode"))
    identity = cli._runtime_report().runtime
    assert identity is not None
    assert identity.release_sha == "abcdef1"
    assert "secret" not in identity.endpoint
    assert identity.endpoint == "https://example.test:11434/path"


def test_migration_manifest_is_contiguous_and_nonempty():
    assert validate_migrations() == (
        "001_initial.sql",
        "002_audit_hash_chain.sql",
        "003_pack_installations.sql",
        "004_personal_state.sql",
        "005_household_state.sql",
        "006_finance_snapshots.sql",
        "007_household_projections.sql",
        "008_tasks.sql",
        "009_task_idempotency.sql",
        "010_network_state.sql",
        "011_personal_memory_vectors.sql",
        "012_osint_investigations.sql",
        "013_homelab_inventory.sql",
        "014_security_lab_findings.sql",
        "015_pack_upgrade_candidates.sql",
        "016_audit_chain_head.sql",
    )


def test_openclaw_ambient_adapter_preserves_correlation_and_idempotency():
    class Gateway:
        def __init__(self):
            self.calls = []

        def notify(self, params):
            self.calls.append(("notify", params))

        def schedule(self, params):
            self.calls.append(("schedule", params))

        def cancel(self, params):
            self.calls.append(("cancel", params))

    gateway = Gateway()
    platform = OpenClawAmbientPlatform(gateway)
    notification = Notification(
        recipient_ids=("alice",), text="Rice is low", correlation_id=uuid4()
    )
    task = BackgroundTask(
        run_at=datetime.now(timezone.utc),
        task_type="suggestion.refresh",
        idempotency_key="ambient-2",
    )
    platform.deliver_notification(notification)
    platform.schedule_background(task)
    platform.cancel_background(task)
    assert gateway.calls[0] == ("notify", notification.model_dump(mode="json"))
    assert gateway.calls[1] == ("schedule", task.model_dump(mode="json"))
    assert gateway.calls[2] == (
        "cancel",
        {"task_id": str(task.task_id), "idempotency_key": "ambient-2"},
    )


def test_openclaw_gateway_notification_uses_confirmed_system_event_method():
    class Client:
        def __init__(self):
            self.calls = []

        def call(self, method, params):
            self.calls.append((method, params))
            return {"queued": True}

    client = Client()
    gateway = OpenClawGatewayRpc(client)
    assert gateway.notify({"text": "Apartment inspection tomorrow", "wake": False}) == {
        "queued": True
    }
    assert client.calls == [
        ("system-event", {"text": "Apartment inspection tomorrow", "wake": False})
    ]
