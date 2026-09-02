#!/usr/bin/env python3
"""Create a development-only failure autopsy from an evaluator report.

The report is treated as evidence, not as expected-behavior authority. This
script reads only the supplied report and never loads the held-out corpus.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _categories(item: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    if item.get("failure_class") == "invalid_model_decision":
        categories.append("decoder_schema_failure")
    if item.get("false_mutation"):
        categories.append("false_mutation")
    if item.get("false_completion"):
        categories.append("false_completion")
    expected_action = item.get("expected_action")
    predicted_action = item.get("predicted_action")
    candidates = item.get("candidate_action_ids", [])
    if expected_action is not None and item.get("expected_action_available") is False:
        categories.append("expected_capability_absent")
    elif expected_action is not None and predicted_action != expected_action:
        if predicted_action is not None and predicted_action in candidates:
            categories.append("wrong_action_with_candidate_available")
        elif predicted_action is not None:
            categories.append("action_outside_candidates")
        elif item.get("predicted_kind") != "ACTION":
            categories.append("action_not_proposed")
    if item.get("expected_kind") == "CLARIFY" and item.get("predicted_kind") != "CLARIFY":
        categories.append("missed_clarification")
    if (
        expected_action is not None
        and predicted_action == expected_action
        and not item.get("argument_exact", False)
    ):
        categories.append("wrong_arguments")
    return categories


def autopsy(report: dict[str, Any]) -> dict[str, Any]:
    results = report.get("results")
    if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
        raise ValueError("report has no valid results")
    counts: Counter[str] = Counter()
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    cases: list[dict[str, Any]] = []
    for item in results:
        categories = _categories(item)
        if not categories:
            continue
        family = str(item.get("family", "unclassified"))
        for category in categories:
            counts[category] += 1
            family_counts[family][category] += 1
        cases.append({"id": item.get("id"), "family": family, "categories": categories})
    severity = {
        "false_mutation": 100,
        "false_completion": 100,
        "decoder_schema_failure": 70,
        "action_outside_candidates": 65,
        "expected_capability_absent": 60,
        "missed_clarification": 55,
        "wrong_arguments": 50,
        "wrong_action_with_candidate_available": 40,
        "action_not_proposed": 35,
    }
    ranked = sorted(
        (
            {"category": category, "count": count, "severity": severity.get(category, 0)}
            for category, count in counts.items()
        ),
        key=lambda item: (-item["severity"], -item["count"], item["category"]),
    )
    return {
        "source_report": report.get("source_revision"),
        "dataset": report.get("dataset"),
        "dataset_sha256": report.get("dataset_sha256"),
        "cases": len(results),
        "safety_hard_failure": report.get("security_hard_failure"),
        "ranking_method": "severity_then_frequency; severity is a triage label, not a model score",
        "ranked_failure_classes": ranked,
        "family_failure_counts": {
            family: dict(sorted(values.items())) for family, values in sorted(family_counts.items())
        },
        "cases_with_failures": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = autopsy(json.loads(args.report.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
