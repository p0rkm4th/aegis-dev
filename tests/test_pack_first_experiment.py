from __future__ import annotations

import pytest

from aegis.audit import AuditLog
from aegis.contracts import ActionCard, ActionSpec
from aegis.pack_first_experiment import (
    PackRouterStatus,
    compact_pack_catalog,
    measure_incumbent,
    measure_pack_first,
    measure_retrieval_assisted_pack_first,
    parse_router_response,
    router_prompt,
    selected_cards,
    validate_selected_packs,
)
from aegis.pack_lifecycle import PackBundle, PackManager, PackManifest, PackStatus, PackUI


def dynamic_manager() -> PackManager:
    manager = PackManager(audit=AuditLog())
    manager.discover(
        PackBundle(
            manifest=PackManifest(
                pack_id="dynamic-weather",
                version="0.1.0",
                ui=PackUI(label="Weather", category="owner utility"),
            ),
            cards=(
                ActionCard(
                    action=ActionSpec(
                        action_id="dynamic-weather.read",
                        capability="dynamic-weather.read",
                    ),
                    summary="Read bounded weather conditions",
                    relevance=0.9,
                ),
            ),
        )
    )
    return manager


def test_dynamic_pack_is_discovered_and_routed_without_core_special_case():
    manager = dynamic_manager()
    catalog = compact_pack_catalog(manager)

    assert catalog[0].pack_id == "dynamic-weather"
    assert catalog[0].status is PackStatus.DISCOVERED
    assert "owner utility" in router_prompt("what is the weather", catalog)

    manager.install("dynamic-weather", frozenset())
    with pytest.raises(ValueError, match="not enabled"):
        validate_selected_packs(
            parse_router_response({"status": "SELECTED", "selected_pack_ids": ["dynamic-weather"]}),
            manager,
        )

    manager.enable("dynamic-weather")
    response = parse_router_response(
        {"status": "SELECTED", "selected_pack_ids": ["dynamic-weather"]}
    )
    assert validate_selected_packs(response, manager) == ("dynamic-weather",)
    assert selected_cards(manager, response.selected_pack_ids)[0].action.action_id == (
        "dynamic-weather.read"
    )


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"status": "SELECTED", "selected_pack_ids": ["missing"]}, "unknown Pack"),
        ({"status": "SELECTED", "selected_pack_ids": ["dynamic-weather"]}, "not enabled"),
        ({"status": "SELECTED", "selected_pack_ids": ["dynamic-weather"]}, "not enabled"),
        ({"status": "NO_VALID_PACK", "selected_pack_ids": ["dynamic-weather"]}, "non-selected"),
        ({"status": "SELECTED", "selected_pack_ids": []}, "requires a Pack"),
    ],
)
def test_pack_router_fails_closed_before_execution(payload, message):
    manager = dynamic_manager()
    if payload["selected_pack_ids"] == ["missing"]:
        manager.install("dynamic-weather", frozenset())
        manager.enable("dynamic-weather")
    with pytest.raises(ValueError, match=message):
        response = parse_router_response(payload)
        validate_selected_packs(response, manager)


def test_router_measurements_keep_pack_first_context_compact_and_exclude_owner_data():
    manager = dynamic_manager()
    manager.install("dynamic-weather", frozenset())
    manager.enable("dynamic-weather")
    seen: list[str] = []

    def router(prompt: str):
        seen.append(prompt)
        return {"status": "SELECTED", "selected_pack_ids": ["dynamic-weather"]}

    pure = measure_pack_first(
        "what is the weather? secret-grocery-value",
        manager,
        router,
        frozenset({"dynamic-weather"}),
    )
    assisted = measure_retrieval_assisted_pack_first(
        "what is the weather?",
        manager,
        router,
        lambda _utterance, _manager: ("dynamic-weather",),
        frozenset({"dynamic-weather"}),
    )

    assert pure.variant == "pack_first"
    assert assisted.variant == "retrieval_assisted_pack_first"
    assert pure.valid and pure.correct
    assert assisted.valid and assisted.correct
    assert len(seen) == 2
    assert "dynamic-weather" in seen[0]
    assert "secret-grocery-value" in seen[0]
    assert "quantity" not in seen[0]
    assert pure.context_bytes == 0


def test_incumbent_is_measured_as_control_without_router_call():
    manager = dynamic_manager()
    manager.install("dynamic-weather", frozenset())
    manager.enable("dynamic-weather")
    measurement = measure_incumbent(
        "what is the weather",
        lambda _utterance: selected_cards(manager, ("dynamic-weather",)),
        frozenset({"dynamic-weather"}),
    )

    assert measurement.variant == "incumbent"
    assert measurement.status is PackRouterStatus.SELECTED
    assert measurement.correct
    assert measurement.model_calls == 0
