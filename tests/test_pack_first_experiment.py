from __future__ import annotations

import pytest

from aegis.audit import AuditLog
from aegis.contracts import ActionCard, ActionSpec
from aegis.pack_first_experiment import (
    OWNER_PACK_CORPUS,
    PackCase,
    PackFollowUpContext,
    PackRouterStatus,
    compact_pack_catalog,
    is_pack_lifecycle_request,
    measure_incumbent,
    measure_pack_first,
    measure_retrieval_assisted_pack_first,
    parse_router_response,
    route_with_one_context_retry,
    router_prompt,
    run_pack_tournament,
    selected_cards,
    semantic_pack_prefilter,
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


def test_router_catalog_excludes_non_enabled_packs_but_lifecycle_catalog_remains_inspectable():
    manager = dynamic_manager()
    lifecycle_catalog = compact_pack_catalog(manager)
    router_catalog = compact_pack_catalog(manager, enabled_only=True)

    assert lifecycle_catalog[0].status is PackStatus.DISCOVERED
    assert router_catalog == ()

    manager.install("dynamic-weather", frozenset())
    installed_catalog = compact_pack_catalog(manager, enabled_only=True)
    assert installed_catalog == ()

    manager.enable("dynamic-weather")
    enabled_catalog = compact_pack_catalog(manager, enabled_only=True)
    assert [entry.pack_id for entry in enabled_catalog] == ["dynamic-weather"]


def test_semantic_pack_prefilter_reuses_enabled_card_retrieval_only():
    manager = dynamic_manager()

    class Embedder:
        def embed(self, texts):
            return tuple(
                (1.0, 0.0) if index == 0 or "weather" in text else (0.0, 1.0)
                for index, text in enumerate(texts)
            )

    assert semantic_pack_prefilter("what is the weather", manager, Embedder()) == ()
    manager.install("dynamic-weather", frozenset())
    manager.enable("dynamic-weather")
    assert semantic_pack_prefilter("what is the weather", manager, Embedder()) == (
        "dynamic-weather",
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


def test_owner_corpus_includes_adversarial_followups_without_core_phrase_routes():
    utterances = {case.utterance for case in OWNER_PACK_CORPUS}
    assert "we ran out of milk" in utterances
    assert "show me the other one" in utterances
    assert "nah, tomorrow" in utterances
    assert "restart that instead" in utterances
    assert any(case.expected_pack_ids is None for case in OWNER_PACK_CORPUS)
    assert any(case.expected_pack_ids == frozenset() for case in OWNER_PACK_CORPUS)


def test_owner_corpus_covers_real_frontier_variants_without_sensitive_context():
    cases = {case.utterance: case.expected_pack_ids for case in OWNER_PACK_CORPUS}
    assert cases["which pantry items are low"] == frozenset({"dynamic-food"})
    assert cases["can I afford eighty dollars for groceries"] == frozenset({"dynamic-finance"})
    assert cases["which services are unhealthy"] == frozenset({"dynamic-homelab"})
    assert cases["what services are down"] == frozenset({"dynamic-homelab"})
    assert cases["show authorized network observations"] == frozenset({"dynamic-homelab"})
    assert cases["what should I do about this unsupported capability"] == frozenset(
        {"dynamic-forge"}
    )
    assert cases["show grocery needs and whether eighty dollars is affordable"] == frozenset(
        {"dynamic-food", "dynamic-finance"}
    )
    assert cases["can I spend eighty dollars on groceries"] == frozenset({"dynamic-finance"})
    assert cases[
        "can I keep tonight's grocery trip under eighty dollars based on what we need"
    ] == frozenset({"dynamic-food", "dynamic-finance"})
    assert cases["what groceries do we still need"] == frozenset({"dynamic-food"})
    assert cases["what's in the pantry right now"] == frozenset({"dynamic-food"})
    assert cases["did we already buy oat milk"] == frozenset({"dynamic-food"})
    assert cases[
        "can I keep tonight's grocery trip under eighty dollars based on what we still need"
    ] == frozenset({"dynamic-food", "dynamic-finance"})
    assert cases[
        "we are running low on groceries, can I spend eighty dollars tonight, "
        "and check why Plex is down"
    ] == frozenset({"dynamic-food", "dynamic-finance", "dynamic-homelab"})
    assert cases["check groceries then tell me if Plex is down"] == frozenset(
        {"dynamic-food", "dynamic-homelab"}
    )


def test_pack_lifecycle_language_is_experimental_fail_closed_preflight():
    assert is_pack_lifecycle_request("install this unknown Pack")
    assert is_pack_lifecycle_request("approve the capability")
    assert not is_pack_lifecycle_request("show my groceries")

    manager = corpus_manager()
    measurement = measure_pack_first(
        "install this unknown Pack",
        manager,
        lambda _prompt: {"status": "SELECTED", "selected_pack_ids": ["dynamic-weather"]},
        frozenset(),
    )
    assert measurement.status is PackRouterStatus.UNSUPPORTED
    assert measurement.model_calls == 0
    assert measurement.correct


def test_follow_up_uses_one_bounded_context_retry_only():
    manager = corpus_manager()
    prompts: list[str] = []

    def router(prompt: str):
        prompts.append(prompt)
        if len(prompts) == 1:
            return {"status": "NEED_CONTEXT", "selected_pack_ids": []}
        return {"status": "SELECTED", "selected_pack_ids": ["dynamic-homelab"]}

    response, calls = route_with_one_context_retry(
        "restart that instead",
        manager,
        router,
        PackFollowUpContext(
            prior_pack_ids=("dynamic-homelab",),
            canonical_result_type="service",
            referent_ids=("plex",),
            objective_requirements=("inspect health",),
        ),
    )
    assert response.selected_pack_ids == ("dynamic-homelab",)
    assert calls == 2
    assert "plex" in prompts[1]
    assert "owner" not in prompts[1].casefold() or "authority" in prompts[1].casefold()


def test_follow_up_does_not_retry_without_bounded_context():
    manager = corpus_manager()
    calls = 0

    def router(_prompt: str):
        nonlocal calls
        calls += 1
        return {"status": "NEED_CONTEXT", "selected_pack_ids": []}

    response, count = route_with_one_context_retry("do the other one", manager, router)
    assert response.status is PackRouterStatus.NEED_CONTEXT
    assert count == calls == 1


def test_tournament_metrics_preserve_fail_closed_category_distribution():
    manager = dynamic_manager()
    manager.install("dynamic-weather", frozenset())
    manager.enable("dynamic-weather")

    report = run_pack_tournament(
        (PackCase("what is the weather", frozenset({"dynamic-weather"})),),
        manager,
        lambda _utterance: (),
        lambda _prompt: {"status": "SELECTED", "selected_pack_ids": ["missing-pack"]},
        lambda _utterance, _manager: (),
    )

    metrics = report.metrics["pack_first"]
    assert metrics["invalid_measurement_rate"] == 1.0
    assert metrics["failure_category.ValueError"] == 1.0
