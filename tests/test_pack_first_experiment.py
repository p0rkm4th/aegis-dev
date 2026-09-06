from __future__ import annotations

import pytest

from aegis.audit import AuditLog
from aegis.contracts import ActionCard, ActionSpec
from aegis.pack_first_experiment import (
    PackCase,
    PackRouterStatus,
    compact_pack_catalog,
    measure_incumbent,
    measure_pack_first,
    measure_retrieval_assisted_pack_first,
    parse_router_response,
    router_prompt,
    run_pack_tournament,
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


def corpus_manager() -> PackManager:
    manager = dynamic_manager()
    for pack_id, label, capability in (
        ("dynamic-food", "Food", "read"),
        ("dynamic-finance", "Finance", "read"),
        ("dynamic-homelab", "Homelab", "read"),
        ("dynamic-forge", "Forge", "review"),
    ):
        manager.discover(
            PackBundle(
                manifest=PackManifest(
                    pack_id=pack_id,
                    version="0.1.0",
                    ui=PackUI(label=label, category="owner utility"),
                ),
                cards=(
                    ActionCard(
                        action=ActionSpec(
                            action_id=f"{pack_id}.{capability}",
                            capability=f"{pack_id}.{capability}",
                        ),
                        summary=f"Bounded {label} capability",
                        relevance=0.9,
                    ),
                ),
            )
        )
    for pack_id in (
        "dynamic-weather",
        "dynamic-food",
        "dynamic-finance",
        "dynamic-homelab",
        "dynamic-forge",
    ):
        manager.install(pack_id, frozenset())
        manager.enable(pack_id)
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


def test_tournament_runner_reports_each_variant_without_changing_production():
    manager = dynamic_manager()
    manager.install("dynamic-weather", frozenset())
    manager.enable("dynamic-weather")

    def router(_prompt: str):
        return {"status": "SELECTED", "selected_pack_ids": ["dynamic-weather"]}

    report = run_pack_tournament(
        (PackCase("what is the weather", frozenset({"dynamic-weather"})),),
        manager,
        lambda _utterance: selected_cards(manager, ("dynamic-weather",)),
        router,
        lambda _utterance, _manager: ("dynamic-weather",),
    )

    assert {item.variant for item in report.measurements} == {
        "incumbent",
        "pack_first",
        "retrieval_assisted_pack_first",
    }
    assert report.metrics["pack_first"]["routing_recall"] == 1.0
    assert report.metrics["pack_first"]["context_bytes_per_case"] == 0.0


def test_tournament_corpus_covers_owner_reality_domains_and_unsupported_requests():
    manager = corpus_manager()
    cases = (
        PackCase("what groceries do we need", frozenset({"dynamic-food"})),
        PackCase("show my spending", frozenset({"dynamic-finance"})),
        PackCase("why is Plex down", frozenset({"dynamic-homelab"})),
        PackCase("review a capability need", frozenset({"dynamic-forge"})),
        PackCase("tell me a joke", frozenset()),
    )

    def route_for(text: str) -> str | None:
        lowered = text.casefold()
        return next(
            (
                pack_id
                for term, pack_id in (
                    ("groceries", "dynamic-food"),
                    ("spending", "dynamic-finance"),
                    ("plex", "dynamic-homelab"),
                    ("capability need", "dynamic-forge"),
                )
                if term in lowered
            ),
            None,
        )

    def router(prompt: str):
        pack_id = route_for(prompt)
        return (
            {"status": "SELECTED", "selected_pack_ids": [pack_id]}
            if pack_id
            else {"status": "NO_VALID_PACK", "selected_pack_ids": []}
        )

    def incumbent(utterance: str):
        pack_id = route_for(utterance)
        return selected_cards(manager, (pack_id,)) if pack_id else ()

    report = run_pack_tournament(
        cases,
        manager,
        incumbent,
        router,
        lambda utterance, _manager: (route_for(utterance) or "dynamic-weather",),
    )

    assert len(report.measurements) == 15
    assert report.metrics["incumbent"]["routing_recall"] == 1.0
    assert report.metrics["pack_first"]["routing_recall"] == 1.0
    assert report.metrics["retrieval_assisted_pack_first"]["routing_recall"] == 1.0
