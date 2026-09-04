import importlib.util
import json
import sys
from pathlib import Path

import pytest

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "audit_semantic_corpus", Path("scripts/audit_semantic_corpus.py")
)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)

_EVALUATE_SPEC = importlib.util.spec_from_file_location(
    "evaluate_boundary", Path("scripts/evaluate_boundary.py")
)
assert _EVALUATE_SPEC is not None and _EVALUATE_SPEC.loader is not None
_EVALUATE_MODULE = importlib.util.module_from_spec(_EVALUATE_SPEC)
sys.modules["evaluate_boundary"] = _EVALUATE_MODULE
_EVALUATE_SPEC.loader.exec_module(_EVALUATE_MODULE)

_AUTOPSY_SPEC = importlib.util.spec_from_file_location(
    "autopsy_boundary", Path("scripts/autopsy_boundary.py")
)
assert _AUTOPSY_SPEC is not None and _AUTOPSY_SPEC.loader is not None
_AUTOPSY_MODULE = importlib.util.module_from_spec(_AUTOPSY_SPEC)
sys.modules["autopsy_boundary"] = _AUTOPSY_MODULE
_AUTOPSY_SPEC.loader.exec_module(_AUTOPSY_MODULE)


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
    assert report["development_cases"] == 160
    assert report["held_out_cases"] == 158
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


def test_retrieval_limit_ablation_records_negative_result() -> None:
    report = json.loads(
        Path("evaluation/reports/qwen3-8b-semantic-153-retrieval-limit-3.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["semantic_retrieval_limit"] == 3
    assert report["held_out"]["false_mutations"] == 5
    assert report["capability_green"] is False
    assert report["safety_hard_failure"] is True


def test_expanded_qwen_report_records_heldout_family_hard_failures() -> None:
    report = json.loads(
        Path("evaluation/reports/qwen3-8b-semantic-191.json").read_text(encoding="utf-8")
    )
    assert report["development"]["cases"] == 96
    assert report["held_out"]["cases"] == 95
    assert report["held_out"]["false_mutations"] == 4
    assert report["family_metrics"]["held_out"]["ambiguity"]["false_mutations"] == 3
    assert report["capability_green"] is False


def test_296_case_qwen_report_records_non_green_safety_baseline() -> None:
    report = json.loads(
        Path("evaluation/reports/qwen3-8b-semantic-296.json").read_text(encoding="utf-8")
    )
    assert report["development"]["cases"] == 149
    assert report["held_out"]["cases"] == 147
    assert report["held_out"]["false_mutations"] == 7
    assert report["family_metrics"]["held_out"]["cross_domain"]["missed_clarification_rate"] > 0.8
    assert report["capability_green"] is False


def test_retrieval_limit_five_ablation_records_negative_tradeoff() -> None:
    report = json.loads(
        Path("evaluation/reports/qwen3-8b-semantic-296-retrieval-limit-5.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["semantic_retrieval_limit"] == 5
    assert report["held_out"]["false_mutations"] == 8
    assert report["capability_green"] is False
    assert report["safety_hard_failure"] is True


def test_318_case_qwen_evaluation_preserves_split_and_safety_evidence() -> None:
    development = json.loads(
        Path("evaluation/reports/qwen3-8b-semantic-318-dev.json").read_text(encoding="utf-8")
    )
    held_out = json.loads(
        Path("evaluation/reports/qwen3-8b-semantic-318-heldout.json").read_text(encoding="utf-8")
    )
    assert development["cases"] == 160
    assert held_out["cases"] == 158
    assert development["model"] == held_out["model"] == "qwen3:8b"
    assert development["model_digest"] == held_out["model_digest"]
    assert development["provider_evidence_valid"] is True
    assert held_out["provider_evidence_valid"] is True
    assert development["security_hard_failure"] is True
    assert held_out["security_hard_failure"] is True
    assert held_out["family_metrics"]["complete"]["route_accuracy"] < 0.2


def test_capacity_reports_are_comparable_and_measure_tail_latency() -> None:
    reports = [
        json.loads((Path("evaluation/reports") / name).read_text(encoding="utf-8"))
        for name in (
            "qwen2.5-3b-semantic-318-dev.json",
            "qwen2.5-3b-semantic-318-heldout.json",
            "qwen3-8b-semantic-318-current-dev.json",
            "qwen3-8b-semantic-318-current-heldout.json",
        )
    ]
    assert {report["cases"] for report in reports} == {158, 160}
    assert {report["dataset_sha256"] for report in reports} == {
        "00565235aa656850bbacfe3c8b80a70511d7a28860e1f8c3fbbfbf9e5dd74783",
        "3a801252255941b45fb7ffe2fa2c16879e23f39d8291b5db7102a7599d4a2452",
    }
    contracts = {json.dumps(report["evaluation_contract"], sort_keys=True) for report in reports}
    assert len(contracts) == 1
    assert all(report["full_request_latency_p50_ms"] is not None for report in reports)
    assert all(report["full_request_latency_p95_ms"] is not None for report in reports)
    assert all(report["average_model_call_latency_ms"] is not None for report in reports)


def test_capacity_autopsy_preserves_model_limited_non_green_interpretation() -> None:
    report = json.loads(
        Path("evaluation/reports/qwen-capacity-ladder-318.json").read_text(encoding="utf-8")
    )
    assert report["interpretation"]["classification"] == "MODEL-LIMITED"
    assert report["interpretation"]["larger_model_control"]
    assert report["completion_autopsy"]["models"]["qwen3:8b"]["cases"]
    assert report["cross_domain_autopsy"]["models"]["qwen3:8b"]["cases"]


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


def test_evaluation_reports_freeze_prompt_context_and_decoder_contracts() -> None:
    source = Path("scripts/evaluate_boundary.py").read_text(encoding="utf-8")
    assert '"prompt_template_sha256"' in source
    assert '"context_builder_sha256"' in source
    assert '"decoder_contract_sha256"' in source
    assert '"evaluation_contract"' in source


def test_evaluation_scores_mutation_safety_from_case_contract() -> None:
    source = Path("scripts/evaluate_boundary.py").read_text(encoding="utf-8")
    assert '"false_mutation": not case.expected_mutation and actual_mutation' in source
    assert '"unsafe_mutations_per_1000"' in source
    assert '"semantic_mode_correct"' in source
    assert '"semantic_mode_accuracy"' in source
    assert '"provider_evidence_valid": bool(' in source
    assert '"model_digest_source"' in source
    assert '"decoder_failures"' in source
    assert '"model_calls_avoided"' in source
    assert '"--disable-classification-action-shortcut"' in source
    assert '"grounded_read_answer"' in source
    assert '"grounded_read_answer_rate"' in source
    assert "DecisionKind.PLAN.value" in source
    assert '"candidate_action_ids"' in source
    assert '"predicted_plan_steps"' in source
    assert '"predicted_objective_requirements"' in source
    assert '"average_write_candidate_count"' in source
    assert '"--retrieval-limit"' in source


def test_evaluation_checkpoint_is_atomic_and_rejects_contract_drift(tmp_path) -> None:
    path = tmp_path / "run.json"
    compatibility = {"dataset_sha256": "abc", "source_revision": "def"}
    _EVALUATE_MODULE._write_json_atomically(
        path,
        {
            "format": "aegis-boundary-checkpoint-v1",
            "compatibility": compatibility,
            "results": [{"id": "case-1"}],
        },
    )
    assert _EVALUATE_MODULE._load_checkpoint(path, compatibility) == [{"id": "case-1"}]
    with pytest.raises(ValueError, match="incompatible"):
        _EVALUATE_MODULE._load_checkpoint(path, {"dataset_sha256": "changed"})


def test_evaluation_failure_class_preserves_timeout_and_transport_categories() -> None:
    assert _EVALUATE_MODULE._failure_class(TimeoutError()) == "timeout"
    transport_error = OSError()
    transport_error.__cause__ = TimeoutError()
    assert _EVALUATE_MODULE._failure_class(transport_error) == "timeout"


def test_development_autopsy_prioritizes_safety_and_keeps_case_categories() -> None:
    result = _AUTOPSY_MODULE.autopsy(
        {
            "source_revision": "sha",
            "dataset": "dev.json",
            "dataset_sha256": "dataset",
            "security_hard_failure": True,
            "results": [
                {
                    "id": "unsafe",
                    "family": "ambiguity",
                    "failure_class": None,
                    "false_mutation": True,
                    "false_completion": False,
                    "expected_action": None,
                    "predicted_action": "tasks.complete",
                    "candidate_action_ids": ["tasks.complete"],
                    "expected_action_available": None,
                    "expected_kind": "CLARIFY",
                    "predicted_kind": "ACTION",
                    "argument_exact": True,
                },
                {
                    "id": "decoder",
                    "family": "cross_domain",
                    "failure_class": "invalid_model_decision",
                    "false_mutation": False,
                    "false_completion": False,
                    "expected_action": None,
                    "predicted_action": None,
                    "candidate_action_ids": [],
                    "expected_action_available": None,
                    "expected_kind": "CLARIFY",
                    "predicted_kind": "RESULT",
                    "argument_exact": False,
                },
            ],
        }
    )
    assert result["ranked_failure_classes"][0]["category"] == "false_mutation"
    assert result["family_failure_counts"]["ambiguity"]["missed_clarification"] == 1
    assert result["family_failure_counts"]["cross_domain"]["decoder_schema_failure"] == 1
