from __future__ import annotations

import pytest

from aegis.contracts import ActionCard, ActionSpec, Principal, VerificationContract
from aegis.pack_runtime import ActionRuntime, PackRuntimeRegistry


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
