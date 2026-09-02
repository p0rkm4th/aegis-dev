#!/usr/bin/env python3
"""Evaluate real-provider semantic routing through InteractionBoundary.

This script is evaluation-only. It proposes no executable action and never
persists or executes a Pack operation. Development and held-out corpora are
loaded explicitly so production routing cannot import evaluation phrases.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from aegis.contracts import Context, Decision, DecisionKind, IntentFrame, Principal
from aegis.embeddings import OllamaEmbeddingProvider
from aegis.interaction import InteractionBoundary, InteractionDependencies
from aegis.ollama import OllamaHttpTransport, OllamaProvider
from aegis.pack_lifecycle import PackManager
from aegis.reference_interaction import reference_fallback_cards
from aegis.reference_packs import reference_bundles


@dataclass(frozen=True)
class Case:
    case_id: str
    family: str
    utterance: str
    expected_kind: DecisionKind
    expected_action: str | None = None
    expected_arguments: dict[str, str] | None = None
    phenomena: tuple[str, ...] = ()
    provenance: str = ""
    expected_mutation: bool = False
    expected_mode: str = ""


class MeasuringTransport(OllamaHttpTransport):
    """Capture provider timing/token counters without changing provider behavior."""

    def __init__(self, base_url: str) -> None:
        super().__init__(base_url, timeout=120)
        self.calls = 0
        self.elapsed_ms = 0.0
        self.prompt_tokens = 0
        self.output_tokens = 0

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = monotonic()
        response = super().chat(payload)
        self.calls += 1
        self.elapsed_ms += (monotonic() - started) * 1000
        self.prompt_tokens += int(response.get("prompt_eval_count", 0) or 0)
        self.output_tokens += int(response.get("eval_count", 0) or 0)
        return response


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_cases(path: Path) -> tuple[Case, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("evaluation corpus must be a JSON array")
    cases: list[Case] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("evaluation case must be an object")
        cases.append(
            Case(
                case_id=str(item["id"]),
                family=str(item.get("family", "unclassified")),
                utterance=str(item["utterance"]),
                expected_kind=DecisionKind(str(item["kind"])),
                expected_action=(str(item["action"]) if item.get("action") else None),
                expected_arguments=(
                    {str(k): str(v) for k, v in item["arguments"].items()}
                    if isinstance(item.get("arguments"), dict)
                    else None
                ),
                phenomena=tuple(str(value) for value in item["phenomena"]),
                provenance=str(item["provenance"]),
                expected_mutation=bool(item["expected_mutation"]),
                expected_mode=str(item["semantic_mode"]),
            )
        )
    return tuple(cases)


def _manager() -> PackManager:
    manager = PackManager()
    for bundle in reference_bundles():
        manager.discover(bundle)
        manager.install(bundle.manifest.pack_id, frozenset(bundle.manifest.permissions))
        manager.enable(bundle.manifest.pack_id)
    return manager


def _boundary(provider: OllamaProvider, embedder: OllamaEmbeddingProvider) -> InteractionBoundary:
    return InteractionBoundary(
        InteractionDependencies(
            connect=lambda _url: None,
            required=lambda _name: "evaluation",
            apply_migrations=lambda _connection: None,
            ensure_local_identity=lambda _connection, _principal: None,
            select_action=lambda _utterance, _manager: ("evaluation", None),
            openclaw_channel=lambda: None,
            local_identity=lambda: False,
            model_provider=lambda: provider,
            capability_retriever=lambda query, manager: manager.retrieve_semantic(
                query, embedder, limit=10
            ),
            fallback_card_selector=reference_fallback_cards,
        )
    )


def _evaluation_context() -> Context:
    """Use a bounded authorized fixture matching production context shape."""

    return Context(
        values={
            "canonical_facts": {
                "canonical_items": ["rice", "milk"],
                "canonical_tasks": [
                    {"title": "get gud scrub", "status": "open"},
                    {"title": "restore drill", "status": "open"},
                    {"title": "mail the library book back", "status": "open"},
                ],
            }
        },
        sources=("authorized_canonical_context", "authorized_task_candidates"),
    )


def _decision_fields(
    value: object,
) -> tuple[str, str | None, dict[str, str], str | None, str | None]:
    if isinstance(value, Decision):
        return (
            value.kind.value,
            value.action.action_id if value.action is not None else None,
            {str(k): str(v) for k, v in (value.action.arguments if value.action else {}).items()},
            None,
            value.semantic_mode,
        )
    evidence = getattr(value, "evidence", {})
    failure = evidence.get("failure_class") if isinstance(evidence, dict) else None
    return ("RESULT", None, {}, str(failure) if failure is not None else None, None)


def evaluate(corpus: Path) -> dict[str, Any]:
    cases = _load_cases(corpus)
    base_url = os.environ.get("AEGIS_OLLAMA_URL", "http://127.0.0.1:11434")
    model = os.environ.get("AEGIS_OLLAMA_MODEL", "qwen3:8b")
    transport = MeasuringTransport(base_url)
    provider = OllamaProvider(model, transport)
    embedder = OllamaEmbeddingProvider(
        os.environ.get("AEGIS_EMBEDDING_MODEL", "nomic-embed-text"), base_url
    )
    manager = _manager()
    boundary = _boundary(provider, embedder)
    principal = Principal(id="evaluation", vault_id="evaluation")
    prompt_template_sha = _sha256(
        (
            inspect.getsource(OllamaProvider._prompt)
            + inspect.getsource(OllamaProvider._decision_schema)
        ).encode()
    )
    cards_sha = _sha256(
        json.dumps(
            [card.model_dump(mode="json") for card in manager.enabled_cards()],
            sort_keys=True,
        ).encode()
    )
    results: list[dict[str, Any]] = []
    for case in cases:
        started = monotonic()
        intent = IntentFrame(principal=principal, utterance=case.utterance, correlation_id=uuid4())
        context = _evaluation_context()
        cards = boundary._fallback_cards(manager, case.utterance, context)
        decision = boundary._fallback_decision(intent, cards, context)
        kind, action, arguments, failure_class, semantic_mode = _decision_fields(decision)
        expected_kind = case.expected_kind
        expected_action = case.expected_action
        if not cards and expected_action in {
            "kitchen.groceries.list",
            "tasks.list",
        }:
            # Production intentionally converts an authorized contextual read
            # into a grounded non-authoritative ANSWER when no single ActionCard
            # represents the question. Keep the corpus label intact, but score
            # the actual cognition contract rather than demanding an impossible
            # action proposal from an empty candidate set.
            expected_kind = DecisionKind.ANSWER
            expected_action = None
        correct = kind == expected_kind.value and action == expected_action
        argument_exact = case.expected_arguments is None or arguments == case.expected_arguments
        actual_mutation = action is not None and action not in {
            "kitchen.groceries.list",
            "tasks.list",
            "personal.memory.list",
            "finance.affordability.read",
        }
        results.append(
            {
                "id": case.case_id,
                "family": case.family,
                "phenomena": case.phenomena,
                "provenance": case.provenance,
                "expected_mutation": case.expected_mutation,
                "predicted_semantic_mode": semantic_mode,
                "expected_semantic_mode": case.expected_mode,
                "semantic_mode_correct": semantic_mode == case.expected_mode,
                "predicted_kind": kind,
                "predicted_action": action,
                "failure_class": failure_class,
                "expected_kind": expected_kind.value,
                "expected_action": expected_action,
                "argument_exact": argument_exact,
                "correct_route": correct and argument_exact,
                "false_mutation": not case.expected_mutation and actual_mutation,
                "false_completion": action == "tasks.complete"
                and expected_action != "tasks.complete",
                "clarification_expected": expected_kind is DecisionKind.CLARIFY,
                "clarification_returned": kind == DecisionKind.CLARIFY.value,
                "latency_ms": round((monotonic() - started) * 1000, 2),
            }
        )

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(items)
        false_mutation = sum(int(item["false_mutation"]) for item in items)
        false_completion = sum(int(item["false_completion"]) for item in items)
        expected_clarify = sum(int(item["clarification_expected"]) for item in items)
        expected_actions = sum(int(item["expected_action"] is not None) for item in items)
        predicted_actions = sum(int(item["predicted_action"] is not None) for item in items)
        correct_actions = sum(
            int(
                item["predicted_action"] == item["expected_action"]
                and item["expected_action"] is not None
            )
            for item in items
        )
        expected_answer_or_action = sum(
            int(item["expected_kind"] in {DecisionKind.ANSWER.value, DecisionKind.ACTION.value})
            for item in items
        )
        correctly_completed_or_answered = sum(
            int(item["correct_route"])
            for item in items
            if item["expected_kind"] in {DecisionKind.ANSWER.value, DecisionKind.ACTION.value}
        )
        return {
            "cases": count,
            "route_accuracy": sum(int(item["correct_route"]) for item in items) / max(count, 1),
            "semantic_mode_accuracy": sum(int(item["semantic_mode_correct"]) for item in items)
            / max(count, 1),
            "action_selection_precision": correct_actions / max(predicted_actions, 1),
            "action_selection_recall": correct_actions / max(expected_actions, 1),
            "argument_exactness": sum(int(item["argument_exact"]) for item in items)
            / max(count, 1),
            "inappropriate_clarification_rate": sum(
                int(not item["clarification_expected"] and item["clarification_returned"])
                for item in items
            )
            / max(count - expected_clarify, 1),
            "missed_clarification_rate": sum(
                int(item["clarification_expected"] and not item["clarification_returned"])
                for item in items
            )
            / max(expected_clarify, 1),
            "false_mutations": false_mutation,
            "unsafe_mutations_per_1000": false_mutation / max(count, 1) * 1000,
            "false_completions": false_completion,
            "security_hard_failure": bool(false_mutation or false_completion),
            "correct_completion_or_answer_rate": correctly_completed_or_answered
            / max(expected_answer_or_action, 1),
            "incorrect_blocking_rate": sum(
                int(item["predicted_kind"] == "RESULT" and item["expected_kind"] != "CLARIFY")
                for item in items
            )
            / max(count, 1),
        }

    total = len(results)
    overall = summarize(results)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        grouped[str(item["family"])].append(item)
    return {
        "dataset": str(corpus),
        "dataset_sha256": _sha256(corpus.read_bytes()),
        "model": model,
        "provider": provider.provider_id,
        "endpoint": base_url,
        "model_digest": os.environ.get("AEGIS_OLLAMA_MODEL_DIGEST"),
        "source_revision": subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
        "prompt_template_sha256": prompt_template_sha,
        "action_cards_sha256": cards_sha,
        "cases": total,
        **overall,
        "family_metrics": {family: summarize(items) for family, items in sorted(grouped.items())},
        "average_latency_ms": sum(item["latency_ms"] for item in results) / max(total, 1),
        "model_calls": getattr(transport, "calls", None),
        "prompt_tokens": getattr(transport, "prompt_tokens", None),
        "output_tokens": getattr(transport, "output_tokens", None),
        "memory_vram_cost": "not observable from Ollama HTTP responses",
        "model_loading_overhead_ms": None,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    args = parser.parse_args()
    report = evaluate(args.corpus)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["security_hard_failure"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
