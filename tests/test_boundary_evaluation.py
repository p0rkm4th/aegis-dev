import json
from pathlib import Path


def test_frozen_semantic_corpora_assign_every_case_to_a_family() -> None:
    for filename in ("semantic_dev.json", "semantic_heldout.json"):
        cases = json.loads((Path("evaluation") / filename).read_text(encoding="utf-8"))
        assert cases
        assert all(isinstance(case.get("family"), str) and case["family"] for case in cases)
        assert all(isinstance(case.get("phenomena"), list) and case["phenomena"] for case in cases)
        assert all(
            case.get("provenance") in {"manual", "owner_harvest", "transformed"} for case in cases
        )
        assert all(isinstance(case.get("expected_mutation"), bool) for case in cases)
        assert all(
            case.get("semantic_mode") in {"READ", "ACTION", "GENERATION", "CLARIFY"}
            for case in cases
        )


def test_development_and_heldout_corpora_keep_distinct_case_ids() -> None:
    development = {
        case["id"]
        for case in json.loads(Path("evaluation/semantic_dev.json").read_text(encoding="utf-8"))
    }
    heldout = {
        case["id"]
        for case in json.loads(Path("evaluation/semantic_heldout.json").read_text(encoding="utf-8"))
    }

    assert development.isdisjoint(heldout)


def test_evaluation_boundary_uses_pack_composition_for_fallback_selection() -> None:
    source = Path("scripts/evaluate_boundary.py").read_text(encoding="utf-8")
    assert "fallback_card_selector=reference_fallback_cards" in source
    assert "from aegis.reference_interaction import reference_fallback_cards" in source


def test_evaluation_scores_mutation_safety_from_case_contract() -> None:
    source = Path("scripts/evaluate_boundary.py").read_text(encoding="utf-8")
    assert '"false_mutation": not case.expected_mutation and actual_mutation' in source
    assert '"unsafe_mutations_per_1000"' in source
    assert '"semantic_mode_correct"' in source
