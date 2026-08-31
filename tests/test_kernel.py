from threading import Event
from uuid import uuid4

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
from aegis.kernel import Kernel
from aegis.openclaw import GatewayDisconnected, OpenClawExecutor, ReconnectingGatewayClient
from aegis.reference_packs import (
    ReferenceExecutor,
    ReferenceVerifier,
    ReferenceWorld,
    reference_packs,
)
from aegis.registry import CapabilityRegistry


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
