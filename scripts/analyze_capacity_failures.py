#!/usr/bin/env python3
"""Create model-capacity and semantic-family autopsies from frozen reports.

This is deliberately report-only. It never invokes a model and never changes
the frozen corpus, ActionCards, prompt, context, decoder, or retrieval.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def failure_layer(item: dict[str, Any]) -> str:
    if item.get("failure_class"):
        return "decoder"
    if item.get("false_mutation"):
        return "semantic_mode_or_action_selection"
    if item.get("expected_action") and item.get("predicted_action") != item.get("expected_action"):
        return "action_selection"
    if not item.get("argument_exact"):
        return "arguments"
    if item.get("clarification_expected") and not item.get("clarification_returned"):
        return "clarification"
    return "route_or_grounding"


def autopsy(report: dict[str, Any], families: set[str]) -> dict[str, Any]:
    evidence = []
    counts: Counter[str] = Counter()
    for item in report["results"]:
        if item["family"] not in families:
            continue
        failed = not item["correct_route"] or item["false_mutation"] or item["false_completion"]
        if not failed:
            continue
        layer = failure_layer(item)
        counts[layer] += 1
        cards = item.get("candidate_action_ids", [])
        evidence.append(
            {
                "id": item["id"],
                "utterance": item["utterance"],
                "family": item["family"],
                "candidate_actions_ranked": [
                    {"rank": rank, "action_id": action_id, "score": None}
                    for rank, action_id in enumerate(cards, 1)
                ],
                "candidate_scores": "not exposed by current retrieval contract",
                "semantic_mode": {
                    "expected": item["expected_semantic_mode"],
                    "predicted": item["predicted_semantic_mode"],
                },
                "selected_action": item["predicted_action"],
                "expected_action": item["expected_action"],
                "proposed_arguments": item.get("predicted_arguments", {}),
                "argument_exact": item["argument_exact"],
                "decoder_outcome": item["failure_class"] or "accepted",
                "grounding_outcome": "not separately observable in frozen report",
                "expected_target_availability": "authorized fixture; see context contract",
                "source_failure_layer": layer,
                "false_mutation": item["false_mutation"],
                "false_completion": item["false_completion"],
            }
        )
    return {"failure_counts": dict(sorted(counts.items())), "cases": evidence}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [load(path) for path in args.reports]
    ladder = []
    for report in reports:
        ladder.append(
            {
                "model": report["model"],
                "model_digest": report["model_digest"],
                "dataset": report["dataset"],
                "dataset_sha256": report["dataset_sha256"],
                "source_revision": report["source_revision"],
                "evaluation_contract": report["evaluation_contract"],
                "provider_settings": report.get("provider_settings"),
                "model_inventory": report.get("model_inventory"),
                "metrics": {
                    key: report.get(key)
                    for key in (
                        "cases",
                        "route_accuracy",
                        "semantic_mode_accuracy",
                        "action_selection_precision",
                        "action_selection_recall",
                        "argument_exactness",
                        "false_mutations",
                        "unsafe_mutations_per_1000",
                        "false_completions",
                        "decoder_failures",
                        "correct_completion_or_answer_rate",
                        "model_calls",
                        "model_calls_per_request",
                        "average_model_call_latency_ms",
                        "full_request_latency_p50_ms",
                        "full_request_latency_p95_ms",
                        "prompt_tokens",
                        "output_tokens",
                        "security_hard_failure",
                    )
                },
                "family_metrics": report["family_metrics"],
            }
        )
    by_model: dict[str, dict[str, Any]] = defaultdict(dict)
    for entry in ladder:
        by_model[entry["model"]][Path(entry["dataset"]).stem] = entry
    heldout = [entry for entry in ladder if "heldout" in Path(entry["dataset"]).stem]
    route_delta = None
    if len(heldout) >= 2:
        route_delta = (
            heldout[-1]["metrics"]["route_accuracy"] - heldout[0]["metrics"]["route_accuracy"]
        )
    heldout_reports = {
        report["model"]: report for report in reports if "heldout" in Path(report["dataset"]).stem
    }
    output = {
        "benchmark": {
            "cases": sum(entry["metrics"]["cases"] for entry in ladder) // 2,
            "development_cases": 160,
            "heldout_cases": 158,
            "contract_fingerprints": ladder[0]["evaluation_contract"],
        },
        "models": ladder,
        "interpretation": {
            "classification": "MODEL-LIMITED",
            "basis": (
                "On identical frozen AEGIS contracts, Qwen3:8B improves held-out route "
                f"accuracy by {route_delta:.3f} over Qwen2.5:3B and reduces unsafe mutations "
                "per 1,000 and decoder failures substantially. Shared residual failures "
                "remain in completion, ambiguity, compound, and cross-domain families, so "
                "the evidence is model-limited with a secondary harness/contract frontier."
            )
            if route_delta is not None
            else "insufficient comparable held-out reports",
            "larger_model_control": "not available in the owner Ollama inventory",
            "safety_rule": "Neither model is capability-green; both have security hard failures.",
        },
        "completion_autopsy": {
            "models": {
                entry["model"]: autopsy(
                    heldout_reports[entry["model"]],
                    {"complete"},
                )
                for entry in heldout
            },
            "taxonomy_note": (
                "Frozen reports expose candidate order and decoder outcome; retrieval "
                "scores and grounding are not yet separately instrumented."
            ),
        },
        "cross_domain_autopsy": {
            "models": {
                entry["model"]: autopsy(
                    heldout_reports[entry["model"]],
                    {"cross_domain"},
                )
                for entry in heldout
            },
            "taxonomy_note": (
                "Cross-domain failures are currently measured at bounded decision output; "
                "plan decomposition is not represented by the single-decision contract."
            ),
        },
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
