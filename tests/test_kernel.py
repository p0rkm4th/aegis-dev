from datetime import datetime, timezone
from threading import Event
from uuid import uuid4

import pytest

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
    Decision,
    DecisionKind,
    ExecutionRequest,
    IntentFrame,
    ModelRequest,
    ModelResponse,
    ObjectiveState,
    Observation,
    PolicyDecision,
    Principal,
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
from aegis.homelab import HomelabPack, Host, PostgresHomelabStore, Service
from aegis.household import (
    Chore,
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
    CrossDomainPlanningFastPath,
    DomainClarificationFastPath,
    MultiActionFastPath,
    PersonalChoreComposer,
    PersonalMemoryChoreComposer,
    PersonalMemoryTaskComposer,
    PersonalTaskComposer,
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
    PostgresTaskExecutor,
    PostgresTaskStore,
    PostgresTaskVerifier,
    Task,
    TaskStatus,
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
    store.close()


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
    assert k.run(intent()).state.value == "failed"


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

    assert first.state is ObjectiveState.FAILED
    assert second == first
    assert executor.calls == 1
    assert first.evidence["outcome"] == "unknown"
    assert first.evidence["error_type"] == "RuntimeError"
    assert "private gateway detail" not in str(first)


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
    for pack in reference_packs()[:3]:
        card = pack.cards[0]
        action = card.action.model_copy(
            update={
                "arguments": {"title": "Call landlord"}
                if pack.pack_id == "tasks"
                else {"item": "rice"}
                if pack.pack_id == "kitchen"
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
            IntentFrame(principal=principal, utterance=pack.pack_id),
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
    assert len(transport.calls) == 2
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


def test_ollama_http_transport_rejects_non_http_urls():
    try:
        OllamaHttpTransport("unix:///run/ollama.sock")
    except ValueError:
        pass
    else:
        raise AssertionError("non-HTTP Ollama transport URL was accepted")


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


def test_personal_memory_fast_path_resolves_contextual_reference():
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

    assert result is not None
    assert result.evidence["memories"][0]["content"] == ("Investigated the server backup failure")


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
    assert planning["priority_candidates"] == [
        "household obligation: Utilities",
        "personal goal: Finish the restore drill",
        "task: Review backup runbook",
    ]
    assert len(planning["goals"]) <= 5
    assert len(planning["open_obligations"]) <= 5
    assert len(planning["open_tasks"]) <= 5


def test_cross_domain_planning_recognizes_plural_utilities_context():
    assert CrossDomainPlanningFastPath.matches(
        "Considering my personal memory, Utilities, and open tasks, what should I prioritize?"
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


def test_multi_action_fast_path_extracts_temporal_task_event_plan():
    details = MultiActionFastPath.task_event_details(
        "Create a task and an event for apartment inspection tomorrow."
    )

    assert details is not None
    assert details[0] == "apartment inspection"
    assert details[1].endswith("+00:00")


def test_domain_clarification_fast_path_handles_underspecified_request():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    intent = IntentFrame(principal=principal, utterance="Can you take care of the house stuff?")

    result = DomainClarificationFastPath.resolve(intent)

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "more direction" in result.message


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
