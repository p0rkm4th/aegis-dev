import importlib.util
import json
from pathlib import Path

import pytest

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "audit_semantic_corpus", Path("scripts/audit_semantic_corpus.py")
)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)


def test_frozen_semantic_corpora_assign_every_case_to_a_family() -> None:
    for filename in ("semantic_dev.json", "semantic_heldout.json"):
        cases = json.loads((Path("evaluation") / filename).read_text(encoding="utf-8"))
        assert cases
        assert all(isinstance(case.get("family"), str) and case["family"] for case in cases)
        assert all(isinstance(case.get("phenomena"), list) and case["phenomena"] for case in cases)
        assert all(
            case.get("provenance") in {"manual", "owner_harvest", "transformed", "oss_harvest"}
            for case in cases
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


def test_semantic_corpus_audit_reports_split_and_provenance_coverage() -> None:
    report = _AUDIT_MODULE.audit(
        Path("evaluation/semantic_dev.json"), Path("evaluation/semantic_heldout.json")
    )
    assert report["development_cases"] == 77
    assert report["held_out_cases"] == 76
    assert report["held_out_utterance_overlap"] == 0
    assert report["provenance"]["manual"] > 0


def test_frozen_qwen_report_preserves_non_green_safety_and_provenance() -> None:
    report = json.loads(
        Path("evaluation/reports/qwen3-8b-semantic-153.json").read_text(encoding="utf-8")
    )
    assert report["model"] == "qwen3:8b"
    assert report["source_revision"]
    assert report["development"]["cases"] == 77
    assert report["held_out"]["cases"] == 76
    assert report["capability_green"] is False
    assert report["safety_hard_failure"] is True


def test_semantic_mode_guard_ablation_is_recorded_as_non_improvement() -> None:
    report = json.loads(
        Path("evaluation/reports/qwen3-8b-semantic-153-semantic-mode-guard.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["change_under_test"]
    assert report["held_out"]["false_mutations"] == 3
    assert report["capability_green"] is False
    assert report["safety_hard_failure"] is True


def test_post_guard_live_report_preserves_family_findings() -> None:
    report = json.loads(
        Path("evaluation/reports/qwen3-8b-semantic-153-post-guard.json").read_text(encoding="utf-8")
    )
    assert report["development"]["cases"] == 77
    assert report["held_out"]["cases"] == 76
    assert report["family_findings"]["ambiguity"]["held_out_false_mutations"] == 2
    assert report["capability_green"] is False


def test_current_live_report_preserves_candidate_breadth_and_hard_failures() -> None:
    report = json.loads(
        Path("evaluation/reports/qwen3-8b-semantic-153-current.json").read_text(encoding="utf-8")
    )
    assert report["development"]["cases"] == 77
    assert report["held_out"]["cases"] == 76
    assert report["held_out"]["average_write_candidate_count"] > 3
    assert report["held_out"]["false_mutations"] == 3
    assert report["capability_green"] is False


def test_semantic_corpus_audit_rejects_cross_split_duplicate_utterances(tmp_path) -> None:
    dev = tmp_path / "dev.json"
    held = tmp_path / "held.json"
    case = {
        "id": "one",
        "family": "generation",
        "utterance": "same request",
        "kind": "ANSWER",
        "semantic_mode": "GENERATION",
        "phenomena": ["test"],
        "provenance": "manual",
        "expected_mutation": False,
    }
    dev.write_text(json.dumps([case]), encoding="utf-8")
    held.write_text(json.dumps([{**case, "id": "two"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="utterances must be unique"):
        _AUDIT_MODULE.audit(dev, held)


def test_evaluation_boundary_uses_pack_composition_for_fallback_selection() -> None:
    source = Path("scripts/evaluate_boundary.py").read_text(encoding="utf-8")
    assert "fallback_card_selector=reference_fallback_cards" in source
    assert "from aegis.reference_interaction import reference_fallback_cards" in source
    assert '"--output"' in source


def test_evaluation_scores_mutation_safety_from_case_contract() -> None:
    source = Path("scripts/evaluate_boundary.py").read_text(encoding="utf-8")
    assert '"false_mutation": not case.expected_mutation and actual_mutation' in source
    assert '"unsafe_mutations_per_1000"' in source
    assert '"semantic_mode_correct"' in source
    assert '"semantic_mode_accuracy"' in source
    assert '"provider_evidence_valid": bool(transport.calls and model_digest)' in source
    assert '"model_digest_source"' in source
    assert '"decoder_failures"' in source
    assert '"model_calls_avoided"' in source
    assert '"--disable-classification-action-shortcut"' in source
    assert '"grounded_read_answer"' in source
    assert '"grounded_read_answer_rate"' in source
    assert '"candidate_action_ids"' in source
    assert '"average_write_candidate_count"' in source
    assert '"--retrieval-limit"' in source
