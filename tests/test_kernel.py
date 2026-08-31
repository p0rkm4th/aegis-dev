from threading import Event
from uuid import uuid4

from aegis.audit import AuditError, AuditLog, SqliteAuditLog
from aegis.contracts import (
    ActionCard,
    ActionSpec,
    Decision,
    DecisionKind,
    ExecutionRequest,
    IntentFrame,
    Observation,
    PolicyDecision,
    Principal,
    Result,
    VerificationContract,
    VerificationResult,
)
from aegis.decoding import InvalidDecision, StrictDecisionDecoder
from aegis.gateway_rpc import CorrelatedRpcClient, OpenClawGatewayRpc, RpcProtocolError, RpcResponse
from aegis.identity import (
    InMemoryAuthorization,
    KeycloakIdentityProvider,
    Membership,
    OpenFGAAuthorization,
    Resource,
    Role,
    Space,
    Vault,
)
from aegis.kernel import Kernel
from aegis.openclaw import GatewayDisconnected, OpenClawExecutor, ReconnectingGatewayClient
from aegis.pack_lifecycle import PackBundle, PackManager, PackManifest, PackStatus
from aegis.projections import (
    HouseholdProjection,
    PrivacyProjectionService,
    PrivateContribution,
    SharedObligation,
)
from aegis.reference_packs import (
    ReferenceExecutor,
    ReferenceVerifier,
    ReferenceWorld,
    reference_packs,
)
from aegis.registry import CapabilityRegistry
from aegis.store import SqliteObjectiveStore


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
    for pack in reference_packs():
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
            action_id="cards.read", capability="cards.read", required_permissions=("cards.read",)
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
