from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from aegis.capability_retrieval import retrieve_action_cards
from aegis.contracts import (
    ActionCard,
    ActionSpec,
    ArgumentGroundingRule,
    ArgumentProvenance,
    ArgumentProvenanceKind,
    Context,
    GroundingProposal,
    IntentFrame,
    ModelResponse,
    ObjectiveRequirement,
    ObjectiveSpec,
    ObjectiveState,
    Principal,
    ProposedPlan,
    ProposedPlanStep,
    Result,
    StructuralAnchor,
    StructuralCoverageSignal,
    VerificationContract,
)
from aegis.interaction import InteractionBoundary, InteractionDependencies, InteractionInputError
from aegis.store import InMemoryObjectiveStore


def test_model_fallback_reuses_one_bounded_retrieval_working_set(monkeypatch):
    card = ActionCard(
        action=ActionSpec(action_id="weather.note.write", capability="weather.write"),
        summary="Record a weather note",
        relevance=1,
    )
    retriever_calls = []

    class Provider:
        def decide(self, _request):
            return SimpleNamespace(raw={"kind": "ACTION", "action_ref": card.action.action_id})

    class Store:
        def get_objective_by_correlation(self, _correlation_id, _principal):
            return None

        def correlation_bound(self, _correlation_id):
            return False

        def get_result_for_correlation(self, _correlation_id, _principal):
            return None

        def get_result(self, _key):
            return None

    class Manager:
        def reconcile(self, _bundles, _enabled):
            return None

    class RuntimeRegistry:
        def resolve(self, _card, _connection, _principal):
            return SimpleNamespace(
                executor=object(), verifier=object(), permissions=frozenset(), cleanup=None
            )

    class FixedKernel:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, _intent, _cards, context=None):
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.COMPLETED,
                message="verified",
                correlation_id=_intent.correlation_id,
            )

    monkeypatch.setattr("aegis.interaction.PostgresObjectiveStore", lambda _connection: Store())
    monkeypatch.setattr("aegis.interaction.PostgresAuditLog", lambda _connection: object())
    monkeypatch.setattr("aegis.interaction.PackManager", lambda **_kwargs: Manager())
    monkeypatch.setattr("aegis.interaction.Kernel", FixedKernel)

    boundary = InteractionBoundary(
        InteractionDependencies(
            connect=lambda _url: SimpleNamespace(close=lambda: None),
            required=lambda _name: "unused",
            apply_migrations=lambda _connection: None,
            ensure_local_identity=lambda _connection, _principal: None,
            select_action=lambda _utterance, _manager: (_ for _ in ()).throw(
                InteractionInputError("semantic action resolution required")
            ),
            openclaw_channel=lambda: None,
            local_identity=lambda: False,
            model_provider=lambda: Provider(),
            capability_retriever=lambda _utterance, _manager: (
                retriever_calls.append(True) or (card,)
            ),
            runtime_registry=RuntimeRegistry(),
        )
    )

    result = boundary.run(
        "record a weather note",
        Principal(id="alice", vault_id="alice-vault"),
        correlation_id=uuid4(),
    )

    assert result.state is ObjectiveState.COMPLETED
    assert len(retriever_calls) == 1


def test_unknown_pack_grounding_runs_through_boundary_and_replay_is_idempotent(monkeypatch):
    state = {"level": 0}
    executions = 0
    proposal_level = 40
    policy_allowed = True
    utterance = "Set the test light level to 40"
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("home",))
    card = ActionCard(
        action=ActionSpec(
            action_id="thirdparty.light.set_level",
            capability="thirdparty.light.set_level",
            required_permissions=("thirdparty.light.write",),
            verification=VerificationContract(kind="custom"),
        ),
        summary="Set a test light level",
        relevance=1,
        argument_keys=("level",),
        argument_grounding={
            "level": ArgumentGroundingRule(
                permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
            )
        },
    )

    class Executor:
        def execute(self, request):
            nonlocal executions
            executions += 1
            state["level"] = request.action.arguments["level"]
            from aegis.contracts import Observation

            return Observation(
                execution_id=request.action_id,
                action_id=request.action.action_id,
                evidence={"level": state["level"]},
                command_succeeded=True,
            )

    class Verifier:
        def verify(self, observation, _contract):
            from aegis.contracts import VerificationResult

            return VerificationResult(
                verified=observation.evidence["level"] == 40,
                evidence=observation.evidence,
                reason="third-party light state verified",
            )

    class Policy:
        def authorize(self, _request):
            from aegis.contracts import PolicyDecision

            return PolicyDecision(allowed=policy_allowed, reason="test permission allowed")

    def ground(intent, proposed, _connection, _context):
        value = proposed.action.arguments["level"]
        return GroundingProposal(
            argument_key="level",
            proposed_value=value,
            provenance=ArgumentProvenance(
                kind=ArgumentProvenanceKind.EXPLICIT_UTTERANCE,
                source_spans=((intent.utterance.rfind(str(value)), len(intent.utterance)),),
            ),
        )

    from aegis.audit import AuditLog
    from aegis.pack_runtime import ActionRuntime, PackRuntimeRegistry

    registry = PackRuntimeRegistry()
    registry.register(
        card.action.action_id,
        lambda _connection, _principal: ActionRuntime(
            Executor(), Verifier(), {"thirdparty.light.write": frozenset()}, grounder=ground
        ),
    )
    store = InMemoryObjectiveStore()
    audit = AuditLog()
    monkeypatch.setattr("aegis.interaction.PostgresObjectiveStore", lambda _connection: store)
    monkeypatch.setattr("aegis.interaction.PostgresAuditLog", lambda _connection: audit)
    monkeypatch.setattr("aegis.interaction.PostgresSpacePolicy", lambda *_args: Policy())
    monkeypatch.setattr("aegis.interaction.PackManager", lambda **_kwargs: SimpleNamespace())

    class Provider:
        def decide(self, request):
            assert request.classification_only
            return ModelResponse(
                raw={
                    "kind": "ACTION",
                    "semantic_mode": "ACTION",
                    "action_ref": card.action.action_id,
                    "action_arguments": {"level": proposal_level},
                }
            )

    dependencies = InteractionDependencies(
        connect=lambda _url: SimpleNamespace(close=lambda: None),
        required=lambda _name: "test",
        apply_migrations=lambda _connection: None,
        ensure_local_identity=lambda _connection, _principal: None,
        select_action=lambda _utterance, _manager: (_ for _ in ()).throw(
            InteractionInputError("semantic action resolution required")
        ),
        openclaw_channel=lambda: None,
        local_identity=lambda: False,
        model_provider=lambda: Provider(),
        capability_retriever=lambda _utterance, _manager: (card,),
        runtime_registry=registry,
        structural_parser=lambda _utterance: StructuralCoverageSignal(
            anchors=(StructuralAnchor(source_span=(0, 1), kind="clause"),)
        ),
    )
    boundary = InteractionBoundary(dependencies)
    correlation = uuid4()
    first = boundary.run(utterance, principal, correlation_id=correlation)
    second = boundary.run(utterance, principal, correlation_id=correlation)

    assert first.state is ObjectiveState.COMPLETED, first.message
    assert second.state is ObjectiveState.COMPLETED
    assert state["level"] == 40
    assert executions == 1

    plan_result = boundary._run_proposed_plan(
        IntentFrame(principal=principal, utterance=utterance),
        ProposedPlan(
            steps=(
                ProposedPlanStep(
                    action_ref=card.action.action_id,
                    arguments={"level": 40},
                ),
            )
        ),
        (card,),
        object(),
        principal,
        Context(),
        ObjectiveSpec(
            requirements=(
                ObjectiveRequirement(action_ref=card.action.action_id, arguments={"level": 40}),
            )
        ),
    )
    assert plan_result.state is ObjectiveState.COMPLETED
    assert executions == 2

    proposal_level = 41
    forged = boundary.run(utterance, principal, correlation_id=uuid4())
    assert forged.state is ObjectiveState.BLOCKED
    assert executions == 2

    proposal_level = 40
    policy_allowed = False
    revoked = boundary.run(utterance, principal, correlation_id=uuid4())
    assert revoked.state is ObjectiveState.BLOCKED
    assert executions == 2


def test_multiple_write_candidates_drop_same_namespace_read_noise():
    writes = tuple(
        ActionCard(
            action=ActionSpec(
                action_id=action_id,
                capability=action_id,
                required_permissions=("tasks.write",),
            ),
            summary=action_id,
            relevance=1,
        )
        for action_id in ("tasks.create", "tasks.chores.create")
    )
    read = ActionCard(
        action=ActionSpec(
            action_id="tasks.list",
            capability="tasks.list",
            required_permissions=("tasks.read",),
        ),
        summary="tasks.list",
        relevance=1,
    )

    class Dependencies:
        capability_retriever = staticmethod(lambda _utterance, _manager: writes + (read,))
        fallback_card_selector = None

    cards = retrieve_action_cards(Dependencies(), object(), "compound request")

    assert tuple(card.action.action_id for card in cards) == (
        "tasks.create",
        "tasks.chores.create",
    )


def test_structural_plurality_widens_a_single_write_shortlist_without_authority():
    selected = ActionCard(
        action=ActionSpec(
            action_id="tasks.create",
            capability="tasks.write",
            required_permissions=("tasks.write",),
        ),
        summary="Create a task",
        relevance=1,
    )
    second = ActionCard(
        action=ActionSpec(
            action_id="kitchen.groceries.add",
            capability="kitchen.write",
            required_permissions=("kitchen.write",),
        ),
        summary="Add groceries",
        relevance=1,
    )

    class Dependencies:
        capability_retriever = staticmethod(lambda _utterance, _manager: (selected,))
        structural_parser = staticmethod(
            lambda _utterance: type("Signal", (), {"anchors": (1, 2), "negation_spans": ()})()
        )
        fallback_card_selector = None

    class Manager:
        @staticmethod
        def enabled_cards():
            return (selected, second)

    cards = retrieve_action_cards(Dependencies(), Manager(), "create a task and add groceries")

    assert tuple(card.action.action_id for card in cards) == (
        "tasks.create",
        "kitchen.groceries.add",
    )
