#!/usr/bin/env python3
"""Compare semantic routing shapes without executing any Pack action.

The corpus is evaluation-only: it is deliberately not imported by production
routing. Architecture A is the current embedding shortlist plus Qwen8 path.
Architecture B is opt-in through AEGIS_ROUTER_MODEL. Architecture C uses the
same embeddings as a lightweight confidence reranker and calls Qwen8 only on
low-margin cases.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from time import monotonic
from typing import Any
from uuid import uuid4

from aegis.contracts import (
    ActionCard,
    Decision,
    DecisionKind,
    IntentFrame,
    ModelRequest,
    Principal,
    WorkingSet,
)
from aegis.decoding import InvalidDecision, StrictDecisionDecoder
from aegis.embeddings import OllamaEmbeddingProvider
from aegis.ollama import OllamaHttpTransport, OllamaProvider
from aegis.reference_packs import reference_bundles


@dataclass(frozen=True)
class Case:
    utterance: str
    expected_kind: DecisionKind
    expected_action: str | None = None


# Evaluation-only unseen family. Do not copy these phrases into production.
CASES = (
    Case("I wrapped up get gud scrub", DecisionKind.ACTION, "tasks.complete"),
    Case(
        "Please put replacing the air filter on my things-to-do list",
        DecisionKind.ACTION,
        "tasks.create",
    ),
    Case("What should I grab for dinner tonight?", DecisionKind.ACTION, "kitchen.groceries.list"),
    Case(
        "Could you tell me what's still on my plate this week?", DecisionKind.ACTION, "tasks.list"
    ),
    Case("Write a short fish story for me", DecisionKind.ANSWER),
    Case("Deal with the backup", DecisionKind.CLARIFY),
    Case("I finished the restore drill", DecisionKind.ACTION, "tasks.complete"),
    Case("Add rice and make a reminder too", DecisionKind.CLARIFY),
)


class MeasuringTransport(OllamaHttpTransport):
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


def cards() -> tuple[ActionCard, ...]:
    return tuple(card for bundle in reference_bundles() for card in bundle.cards)


def request(case: Case, shortlisted: tuple[ActionCard, ...]) -> ModelRequest:
    return ModelRequest(
        working_set=WorkingSet(
            intent=IntentFrame(
                principal=Principal(id="benchmark", vault_id="benchmark"),
                utterance=case.utterance,
                correlation_id=uuid4(),
            )
        ),
        action_cards=shortlisted,
        allow_argument_proposals=True,
        routing_only=True,
    )


def evaluate(
    name: str,
    provider: OllamaProvider | None,
    shortlisted: dict[str, tuple[ActionCard, ...]],
    *,
    lightweight_predictions: dict[str, Decision] | None = None,
) -> dict[str, Any]:
    decoder = StrictDecisionDecoder()
    started = monotonic()
    correct = false_mutation = false_completion = security = 0
    clarifications = 0
    errors: list[str] = []
    for case in CASES:
        if lightweight_predictions and case.utterance in lightweight_predictions:
            decision = lightweight_predictions[case.utterance]
            action_id = decision.action.action_id if decision.action else None
            is_correct = decision.kind is case.expected_kind and action_id == case.expected_action
            correct += int(is_correct)
            clarifications += int(decision.kind is DecisionKind.CLARIFY)
            false_mutation += int(
                case.expected_kind in {DecisionKind.ANSWER, DecisionKind.CLARIFY}
                and action_id is not None
            )
            false_completion += int(
                case.expected_action != "tasks.complete" and action_id == "tasks.complete"
            )
            continue
        if provider is None:
            errors.append("router model unavailable")
            continue
        try:
            decision = decoder.decode(
                provider.decide(request(case, shortlisted[case.utterance])),
                shortlisted[case.utterance],
                allow_argument_proposals=True,
            )
        except (InvalidDecision, ValueError) as exc:
            errors.append(f"{case.utterance}: {type(exc).__name__}")
            security += 1
            continue
        action_id = decision.action.action_id if decision.action else None
        is_correct = decision.kind is case.expected_kind and action_id == case.expected_action
        correct += int(is_correct)
        clarifications += int(decision.kind is DecisionKind.CLARIFY)
        false_mutation += int(
            case.expected_kind in {DecisionKind.ANSWER, DecisionKind.CLARIFY}
            and action_id is not None
        )
        false_completion += int(
            case.expected_action != "tasks.complete" and action_id == "tasks.complete"
        )
    transport = getattr(provider, "transport", None)
    return {
        "architecture": name,
        "cases": len(CASES),
        "correct_routes": correct,
        "route_accuracy": correct / len(CASES),
        "false_mutation_routing": false_mutation,
        "false_completion": false_completion,
        "clarifications": clarifications,
        "security_errors": security,
        "hard_failure": bool(security or false_completion),
        "model_calls": getattr(transport, "calls", 0),
        "model_calls_avoided": len(CASES) - getattr(transport, "calls", 0),
        "latency_ms": (getattr(transport, "elapsed_ms", 0.0) or (monotonic() - started) * 1000),
        "prompt_tokens": getattr(transport, "prompt_tokens", 0),
        "output_tokens": getattr(transport, "output_tokens", 0),
        "errors": errors,
    }


def main() -> int:
    base_url = os.environ.get("AEGIS_OLLAMA_URL", "http://127.0.0.1:11434")
    model = os.environ.get("AEGIS_OLLAMA_MODEL", "qwen3:8b")
    embedder = OllamaEmbeddingProvider(
        os.environ.get("AEGIS_EMBEDDING_MODEL", "nomic-embed-text"), base_url
    )
    all_cards = cards()
    vectors = embedder.embed(
        ("benchmark", *(f"{card.action.action_id}: {card.summary}" for card in all_cards))
    )
    query_vectors = embedder.embed(tuple(case.utterance for case in CASES))
    ranked: dict[str, tuple[ActionCard, ...]] = {}
    scores_by_case: dict[str, tuple[float, float]] = {}
    for case, query in zip(CASES, query_vectors):
        scored = sorted(
            (
                (sum(a * b for a, b in zip(query, vector)), card)
                for card, vector in zip(all_cards, vectors[1:])
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        scores_by_case[case.utterance] = (
            scored[0][0] if scored else 0.0,
            scored[0][0] - scored[1][0] if len(scored) > 1 else 0.0,
        )
        ranked[case.utterance] = tuple(card for _, card in scored[:5])

    a_transport = MeasuringTransport(base_url)
    a = evaluate("A_embedding_qwen8", OllamaProvider(model, a_transport), ranked)
    c_transport = MeasuringTransport(base_url)
    c_provider = OllamaProvider(model, c_transport)
    # C is intentionally conservative: the lightweight reranker only bypasses
    # Qwen for a very clear single-card margin; otherwise it delegates to Qwen.
    c_shortlists: dict[str, tuple[ActionCard, ...]] = {}
    for case in CASES:
        c_shortlists[case.utterance] = ranked[case.utterance]
    c_predictions = {
        case.utterance: Decision(kind=DecisionKind.ACTION, action=ranked[case.utterance][0].action)
        for case in CASES
        if scores_by_case[case.utterance][0] >= 0.70 and scores_by_case[case.utterance][1] >= 0.10
    }
    c = evaluate(
        "C_embedding_confidence_reranker_qwen8",
        c_provider,
        c_shortlists,
        lightweight_predictions=c_predictions,
    )
    c["lightweight_predictions"] = len(c_predictions)
    router_model = os.environ.get("AEGIS_ROUTER_MODEL")
    b = {
        "architecture": "B_embedding_router_qwen8",
        "status": "not_run" if not router_model else "not_implemented_in_benchmark",
        "reason": "no 2B-4B router model is installed/configured"
        if not router_model
        else "router adapter requires explicit evaluation contract",
        "model": router_model,
        "hard_failure": False,
    }
    print(json.dumps({"corpus_cases": len(CASES), "results": [a, b, c]}, indent=2))
    return 0 if not any(result.get("hard_failure") for result in (a, c)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
