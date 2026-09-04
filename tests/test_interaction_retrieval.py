from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from aegis.capability_retrieval import retrieve_action_cards
from aegis.contracts import ActionCard, ActionSpec, ObjectiveState, Principal, Result
from aegis.interaction import InteractionBoundary, InteractionDependencies, InteractionInputError


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
