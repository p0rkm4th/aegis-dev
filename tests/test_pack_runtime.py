from __future__ import annotations

import json
from uuid import uuid4

import pytest

from aegis.contracts import (
    ActionCard,
    ActionSpec,
    AuthorizationRequest,
    ExecutionRequest,
    IntentFrame,
    ModelRequest,
    ModelResponse,
    Observation,
    PolicyDecision,
    Principal,
    VerificationContract,
    VerificationResult,
)
from aegis.decoding import StrictDecisionDecoder
from aegis.dispatch import ActionExecutorDispatch, ActionVerifierDispatch
from aegis.interaction import _ActionExecutorDispatch, _ActionVerifierDispatch
from aegis.kernel import Kernel
from aegis.pack_runtime import ActionRuntime, PackRuntimeRegistry
from aegis.web import BrowserApp


def test_third_party_pack_runtime_is_registered_by_action_contract_not_domain_branch():
    card = ActionCard(
        action=ActionSpec(
            action_id="weather.read",
            capability="weather.read",
            required_permissions=("weather.read",),
            verification=VerificationContract(kind="custom"),
        ),
        summary="Read the current weather",
        relevance=1,
    )
    executor = object()
    verifier = object()
    registry = PackRuntimeRegistry()
    registry.register(
        card.action.action_id,
        lambda _connection, _principal: ActionRuntime(
            executor=executor,
            verifier=verifier,
            permissions={"weather.read": frozenset()},
        ),
    )

    resolved = registry.resolve(card, object(), Principal(id="alice", vault_id="alice-vault"))

    assert resolved.executor is executor
    assert resolved.verifier is verifier
    assert tuple(resolved.permissions) == ("weather.read",)


def test_third_party_runtime_cannot_omit_action_permissions():
    card = ActionCard(
        action=ActionSpec(
            action_id="weather.write",
            capability="weather.write",
            required_permissions=("weather.write",),
        ),
        summary="Change a weather preference",
        relevance=1,
    )
    registry = PackRuntimeRegistry()
    registry.register(
        card.action.action_id,
        lambda _connection, _principal: ActionRuntime(object(), object(), {}),
    )

    with pytest.raises(PermissionError, match="does not cover"):
        registry.resolve(card, object(), Principal(id="alice", vault_id="alice-vault"))


def test_pack_registration_is_atomic_and_covers_every_declared_card():
    cards = tuple(
        ActionCard(
            action=ActionSpec(action_id=action_id, capability=action_id),
            summary=action_id,
            relevance=1,
        )
        for action_id in ("weather.read", "weather.write")
    )
    registry = PackRuntimeRegistry()

    with pytest.raises(ValueError, match="match its ActionCards"):
        registry.register_pack(
            cards,
            {"weather.read": lambda _c, _p: ActionRuntime(None, None, {})},
        )

    registry.register_pack(
        cards,
        {
            card.action.action_id: lambda _connection, _principal: ActionRuntime(
                object(), object(), {}
            )
            for card in cards
        },
    )
    assert registry.resolve(cards[0], object(), Principal(id="alice", vault_id="v"))


def test_third_party_pack_read_and_verified_write_use_core_dispatch():
    state: dict[str, str] = {}

    class Executor:
        def execute(self, request: ExecutionRequest) -> Observation:
            if request.action.action_id == "weather.note.write":
                state["note"] = str(request.action.arguments["note"])
            return Observation(
                execution_id=request.action_id,
                action_id=request.action.action_id,
                evidence={"state": dict(state)},
                command_succeeded=True,
            )

    class Verifier:
        def verify(
            self, observation: Observation, _contract: VerificationContract
        ) -> VerificationResult:
            return VerificationResult(
                verified=observation.command_succeeded,
                evidence=observation.evidence,
                reason="verified toy Pack state",
            )

    class Policy:
        def authorize(self, request: AuthorizationRequest) -> PolicyDecision:
            return PolicyDecision(allowed=True, reason=f"allowed {request.action.capability}")

    class Model:
        def __init__(self, action: ActionSpec):
            self.action = action

        def decide(self, _request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                raw={"kind": "ACTION", "action": self.action.model_dump(mode="json")}
            )

    principal = Principal(id="alice", vault_id="alice-vault")
    write = ActionCard(
        action=ActionSpec(
            action_id="weather.note.write",
            capability="weather.write",
            arguments={"note": "bring a coat"},
            required_permissions=("weather.write",),
            verification=VerificationContract(kind="custom"),
        ),
        summary="Record a weather note",
        relevance=1,
        argument_keys=("note",),
    )
    read = write.model_copy(
        update={
            "action": write.action.model_copy(
                update={
                    "action_id": "weather.note.read",
                    "capability": "weather.read",
                    "arguments": {},
                }
            ),
            "summary": "Read the weather note",
        }
    )
    registry = PackRuntimeRegistry()
    for card in (read, write):
        permission = card.action.required_permissions[0]
        registry.register(
            card.action.action_id,
            lambda _connection, _principal, permission=permission: ActionRuntime(
                Executor(), Verifier(), {permission: frozenset()}
            ),
        )

    write_runtime = registry.resolve(write, object(), principal)
    write_result = Kernel(
        Model(write.action),
        StrictDecisionDecoder(),
        Policy(),
        _ActionExecutorDispatch({write.action.action_id: write_runtime.executor}),
        _ActionVerifierDispatch({write.action.action_id: write_runtime.verifier}),
    ).run(IntentFrame(principal=principal, utterance="save a weather note"), (write,))
    assert write_result.state.value == "completed"
    assert state["note"] == "bring a coat"

    read_runtime = registry.resolve(read, object(), principal)
    read_result = Kernel(
        Model(read.action),
        StrictDecisionDecoder(),
        Policy(),
        _ActionExecutorDispatch({read.action.action_id: read_runtime.executor}),
        _ActionVerifierDispatch({read.action.action_id: read_runtime.verifier}),
    ).run(IntentFrame(principal=principal, utterance="read the weather note"), (read,))
    assert read_result.state.value == "completed"
    assert read_result.evidence["state"]["note"] == "bring a coat"


def test_generic_dispatch_routes_by_declared_action_contract():
    class Delegate:
        def execute(self, request: ExecutionRequest) -> Observation:
            return Observation(
                execution_id=request.action_id,
                evidence={"action": request.action.action_id},
                command_succeeded=True,
            )

        def verify(
            self, observation: Observation, _contract: VerificationContract
        ) -> VerificationResult:
            return VerificationResult(
                verified=observation.command_succeeded,
                evidence=observation.evidence,
                reason="generic dispatch verification",
            )

    action = ActionSpec(action_id="weather.note.write", capability="weather.write")
    delegate = Delegate()
    observation = ActionExecutorDispatch({action.action_id: delegate}).execute(
        ExecutionRequest(
            objective_id=uuid4(),
            action_id=uuid4(),
            action=action,
            idempotency_key="weather-1",
        )
    )
    assert observation.action_id == action.action_id
    verified = ActionVerifierDispatch({action.action_id: delegate}).verify(
        observation, VerificationContract(kind="custom")
    )
    assert verified.verified is True


def test_third_party_pack_read_and_verified_write_cross_browser_boundary():
    """A browser client can use a Pack runtime without client-owned semantics."""

    state: dict[str, str] = {}
    principal = Principal(id="alice", vault_id="alice-vault")
    read = ActionCard(
        action=ActionSpec(
            action_id="weather.note.read",
            capability="weather.read",
            required_permissions=("weather.read",),
            verification=VerificationContract(kind="custom"),
        ),
        summary="Read a weather note",
        relevance=1,
    )
    write = ActionCard(
        action=ActionSpec(
            action_id="weather.note.write",
            capability="weather.write",
            arguments={"note": "bring a coat"},
            required_permissions=("weather.write",),
            verification=VerificationContract(kind="custom"),
        ),
        summary="Record a weather note",
        relevance=1,
        argument_keys=("note",),
    )

    class Executor:
        def execute(self, request: ExecutionRequest) -> Observation:
            if request.action.action_id == write.action.action_id:
                state["note"] = str(request.action.arguments["note"])
            return Observation(
                execution_id=uuid4(),
                action_id=request.action.action_id,
                evidence={"note": state.get("note", "")},
                command_succeeded=True,
            )

    class Verifier:
        def verify(
            self, observation: Observation, _contract: VerificationContract
        ) -> VerificationResult:
            return VerificationResult(
                verified=observation.command_succeeded,
                evidence=observation.evidence,
                reason="verified browser Pack state",
            )

    class Policy:
        def authorize(self, _request: AuthorizationRequest) -> PolicyDecision:
            return PolicyDecision(allowed=True, reason="toy Pack policy allowed")

    class Model:
        def __init__(self, action: ActionSpec):
            self.action = action

        def decide(self, _request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                raw={"kind": "ACTION", "action": self.action.model_dump(mode="json")}
            )

    registry = PackRuntimeRegistry()
    for card in (read, write):
        permission = card.action.required_permissions[0]
        registry.register(
            card.action.action_id,
            lambda _connection, _principal, permission=permission: ActionRuntime(
                Executor(), Verifier(), {permission: frozenset()}
            ),
        )

    def interaction(utterance: str, caller: Principal, correlation_id):
        card = write if utterance.startswith("write") else read
        runtime = registry.resolve(card, object(), caller)
        result = Kernel(
            Model(card.action),
            StrictDecisionDecoder(),
            Policy(),
            _ActionExecutorDispatch({card.action.action_id: runtime.executor}),
            _ActionVerifierDispatch({card.action.action_id: runtime.verifier}),
        ).run(
            IntentFrame(
                principal=caller,
                utterance=utterance,
                correlation_id=correlation_id,
            ),
            (card,),
        )
        return {
            "message": result.message,
            "state": result.state.value,
            "detail": result.message,
            "objective_id": str(result.objective_id),
        }

    app = BrowserApp(principal, interaction, lambda _caller: {"nodes": []})
    write_status, _, write_body = app.dispatch(
        "POST",
        "/api/message",
        json.dumps({"utterance": "write bring a coat", "correlation_id": str(uuid4())}).encode(),
    )
    read_status, _, read_body = app.dispatch(
        "POST",
        "/api/message",
        json.dumps({"utterance": "read the note", "correlation_id": str(uuid4())}).encode(),
    )

    assert write_status == 200
    assert read_status == 200
    assert json.loads(write_body)["state"] == "completed"
    assert json.loads(read_body)["state"] == "completed"
    assert state["note"] == "bring a coat"
