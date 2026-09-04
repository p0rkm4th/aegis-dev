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
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from aegis.contracts import (
    ActionCard,
    Context,
    Decision,
    DecisionKind,
    IntentFrame,
    Principal,
    Result,
)
from aegis.decoding import StrictDecisionDecoder
from aegis.embeddings import EmbeddingResponseError, OllamaEmbeddingProvider
from aegis.interaction import InteractionBoundary, InteractionDependencies
from aegis.ollama import OllamaHttpTransport, OllamaProvider, OllamaResponseError
from aegis.pack_lifecycle import PackManager
from aegis.personal import PersonalState
from aegis.planning import PlanValidationError, materialize_proposed_plan
from aegis.reference_interaction import ground_reference_action, reference_fallback_cards
from aegis.reference_packs import reference_bundles
from aegis.structural import SpacyStructuralParser
from aegis.tasks import Task, TaskStatus

READ_ACTIONS = {
    "kitchen.groceries.list",
    "tasks.list",
    "personal.memory.list",
    "finance.affordability.read",
}


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
        self.call_latencies_ms: list[float] = []

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = monotonic()
        response = super().chat(payload)
        self.calls += 1
        elapsed_ms = (monotonic() - started) * 1000
        self.elapsed_ms += elapsed_ms
        self.call_latencies_ms.append(elapsed_ms)
        self.prompt_tokens += int(response.get("prompt_eval_count", 0) or 0)
        self.output_tokens += int(response.get("eval_count", 0) or 0)
        return response


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json_atomically(path: Path, value: object) -> None:
    """Persist evaluation state without leaving a truncated JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _failure_class(exc: Exception) -> str:
    cause = exc.__cause__
    if isinstance(exc, (TimeoutError,)) or isinstance(cause, TimeoutError):
        return "timeout"
    if cause is not None and cause.__class__.__name__ in {"HTTPError", "URLError"}:
        return "transport_error"
    if isinstance(exc, (ConnectionError,)):
        return "transport_error"
    return "provider_error"


def _load_checkpoint(path: Path, compatibility: dict[str, Any]) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format") != "aegis-boundary-checkpoint-v1":
        raise ValueError("unsupported evaluation checkpoint format")
    if value.get("compatibility") != compatibility:
        raise ValueError("evaluation checkpoint is incompatible with this run")
    results = value.get("results")
    if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
        raise ValueError("evaluation checkpoint has invalid results")
    ids = [item.get("id") for item in results]
    if any(not isinstance(case_id, str) for case_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("evaluation checkpoint contains duplicate or invalid case IDs")
    return results


def _percentile(values: list[float], percentile: float) -> float | None:
    """Return an interpolated percentile without adding a statistics dependency."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


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


def _boundary(
    provider: OllamaProvider,
    embedder: OllamaEmbeddingProvider,
    *,
    repair_provider: OllamaProvider | None = None,
    reuse_classification_action_reference: bool = True,
    retrieval_limit: int = 10,
    retrieval_traces: list[dict[str, Any]] | None = None,
    structural_parser: Any | None = None,
) -> InteractionBoundary:
    def retrieve(query: str, manager: PackManager) -> tuple[ActionCard, ...]:
        matches = manager.retrieve_semantic_with_scores(query, embedder, limit=retrieval_limit)
        if retrieval_traces is not None:
            retrieval_traces.append(
                {
                    "candidates": [
                        {
                            "action_id": match.card.action.action_id,
                            "score": round(match.score, 6),
                            "metadata_relevance": match.card.relevance,
                        }
                        for match in matches
                    ],
                    "top_score": round(matches[0].score, 6) if matches else None,
                    "score_margin": round(matches[0].score - matches[1].score, 6)
                    if len(matches) > 1
                    else None,
                }
            )
        return tuple(match.card for match in matches)

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
            repair_model_provider=(
                (lambda: repair_provider) if repair_provider is not None else None
            ),
            capability_retriever=retrieve,
            fallback_card_selector=reference_fallback_cards,
            reuse_classification_action_reference=reuse_classification_action_reference,
            structural_parser=structural_parser,
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


class _EvaluationTaskStore:
    """Read-only canonical task fixture for grounding previews."""

    def __init__(self, context: Context) -> None:
        raw_tasks = context.values.get("canonical_facts", {}).get("canonical_tasks", [])
        self._tasks = tuple(
            Task(
                task_id=uuid4(),
                space_id="evaluation",
                title=str(item["title"]),
                created_by="evaluation",
                status=TaskStatus.OPEN,
            )
            for item in raw_tasks
            if isinstance(item, dict) and isinstance(item.get("title"), str)
        )

    def list(self, _principal: Any) -> tuple[Task, ...]:
        return self._tasks


class _EvaluationHouseholdStore:
    """Empty read-only household fixture used only by grounding previews."""

    def read_snapshot(self, _principal: Any) -> dict[str, tuple[object, ...]]:
        return {"chores": ()}


def _grounding_preview(
    intent: IntentFrame,
    decision: Decision | Result,
    context: Context,
    cards: tuple[ActionCard, ...] = (),
) -> tuple[str, bool | None]:
    """Preview reference grounding without authorization, execution, or persistence.

    This is deliberately narrower than full validation: objective fidelity,
    authorization, Kernel execution, observation, and verification remain outside
    this evaluator-only read-only preview.
    """

    if not isinstance(decision, Decision):
        return "not_applicable", None
    if decision.kind is DecisionKind.ACTION:
        if decision.action is None:
            return "not_applicable", None
        proposed_cards = (
            ActionCard(
                action=decision.action,
                summary="evaluation proposal",
                relevance=1,
                argument_keys=tuple(decision.action.arguments),
            ),
        )
    elif decision.kind is DecisionKind.PLAN and decision.plan is not None:
        try:
            actions = materialize_proposed_plan(decision.plan, cards)
        except (PlanValidationError, ValueError):
            return "blocked", False
        cards_by_id = {card.action.action_id: card for card in cards}
        proposed_cards = tuple(
            cards_by_id[action.action_id].model_copy(update={"action": action})
            for action in actions
            if action.action_id in cards_by_id
        )
        if len(proposed_cards) != len(actions):
            return "blocked", False
    else:
        return "not_applicable", None

    task_store = _EvaluationTaskStore(context)
    household_store = _EvaluationHouseholdStore()
    for card in proposed_cards:
        grounded = ground_reference_action(
            intent,
            card,
            task_store=task_store,
            household_store=household_store,
            personal_state=PersonalState(),
            goal_task_title=None,
            goal_chore_title=None,
            memory_task_title=None,
            memory_chore_title=None,
            context=context,
        )
        if not isinstance(grounded, ActionCard):
            return "blocked", False
    return "accepted", True


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


def _plan_fields(value: object) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Expose bounded plan-shape telemetry without retaining private prose."""

    if not isinstance(value, Decision):
        return [], []
    steps = [
        {
            "action_ref": step.action_ref,
            "arguments": dict(step.arguments),
            "depends_on": list(step.depends_on),
        }
        for step in (value.plan.steps if value.plan is not None else ())
    ]
    requirements = [
        {"action_ref": requirement.action_ref, "arguments": dict(requirement.arguments)}
        for requirement in (
            value.objective_spec.requirements if value.objective_spec is not None else ()
        )
    ]
    return steps, requirements


def evaluate(
    corpus: Path,
    *,
    reuse_classification_action_reference: bool = True,
    retrieval_limit: int = 10,
    checkpoint_path: Path | None = None,
    resume_path: Path | None = None,
    compact_action_cards: bool = False,
    action_ref_only: bool = False,
    structural_model: str | None = None,
) -> dict[str, Any]:
    cases = _load_cases(corpus)
    base_url = os.environ.get("AEGIS_OLLAMA_URL", "http://127.0.0.1:11434")
    model = os.environ.get("AEGIS_OLLAMA_MODEL", "qwen3:8b")
    repair_model = os.environ.get("AEGIS_OLLAMA_REPAIR_MODEL")
    transport = MeasuringTransport(base_url)
    provider = OllamaProvider(
        model,
        transport,
        compact_action_cards=compact_action_cards,
        action_ref_only=action_ref_only,
    )
    repair_provider = (
        OllamaProvider(
            repair_model,
            transport,
            compact_action_cards=compact_action_cards,
            action_ref_only=action_ref_only,
        )
        if repair_model
        else None
    )
    embedder = OllamaEmbeddingProvider(
        os.environ.get("AEGIS_EMBEDDING_MODEL", "nomic-embed-text"), base_url
    )
    manager = _manager()
    retrieval_traces: list[dict[str, Any]] = []
    boundary = _boundary(
        provider,
        embedder,
        repair_provider=repair_provider,
        reuse_classification_action_reference=reuse_classification_action_reference,
        retrieval_limit=retrieval_limit,
        retrieval_traces=retrieval_traces,
        structural_parser=(
            SpacyStructuralParser(model_path=structural_model).parse if structural_model else None
        ),
    )
    principal = Principal(id="evaluation", vault_id="evaluation")
    prompt_template_sha = _sha256(
        (
            inspect.getsource(OllamaProvider._prompt)
            + inspect.getsource(OllamaProvider._decision_schema)
        ).encode()
    )
    context_builder_sha = _sha256(inspect.getsource(_evaluation_context).encode())
    decoder_contract_sha = _sha256(inspect.getsource(StrictDecisionDecoder).encode())
    cards_sha = _sha256(
        json.dumps(
            [card.model_dump(mode="json") for card in manager.enabled_cards()],
            sort_keys=True,
        ).encode()
    )
    configured_digest = os.environ.get("AEGIS_OLLAMA_MODEL_DIGEST")
    model_digest = configured_digest
    if model_digest is None:
        try:
            model_digest = transport.model_digest(model)
        except Exception:
            # Evaluation can still report its other measurements when an
            # inventory endpoint is unavailable, but the missing provenance
            # remains visible in the report and must not be called frozen.
            model_digest = None
    repair_model_digest = None
    if repair_model:
        repair_model_digest = os.environ.get("AEGIS_OLLAMA_REPAIR_MODEL_DIGEST")
        if repair_model_digest is None:
            try:
                repair_model_digest = transport.model_digest(repair_model)
            except Exception:
                repair_model_digest = None
    model_inventory: dict[str, Any] | None = None
    try:
        for item in transport.tags().get("models", []):
            if isinstance(item, dict) and item.get("name") == model:
                model_inventory = item
                break
    except Exception:
        model_inventory = None
    embedding_digest = None
    try:
        embedding_digest = transport.model_digest(embedder.model)
    except Exception:
        pass
    compatibility = {
        "dataset_sha256": _sha256(corpus.read_bytes()),
        "source_revision": subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
        "model": model,
        "model_digest": model_digest,
        "repair_model": repair_model,
        "repair_model_digest": repair_model_digest,
        "embedding_model": embedder.model,
        "embedding_digest": embedding_digest,
        "prompt_template_sha256": prompt_template_sha,
        "decoder_contract_sha256": decoder_contract_sha,
        "context_builder_sha256": context_builder_sha,
        "action_cards_sha256": cards_sha,
        "semantic_retrieval_limit": retrieval_limit,
        "classification_action_reference_shortcut": reuse_classification_action_reference,
        "compact_action_cards": compact_action_cards,
        "action_ref_only": action_ref_only,
        "provider_settings": {
            "temperature": 0,
            "stream": False,
            "think": False,
            "timeout_seconds": 120,
        },
    }
    results = _load_checkpoint(resume_path, compatibility) if resume_path is not None else []
    completed = {str(item["id"]) for item in results}
    if completed - {case.case_id for case in cases}:
        raise ValueError("evaluation checkpoint contains a case outside this corpus")
    if checkpoint_path is not None and resume_path is not None and checkpoint_path != resume_path:
        raise ValueError("checkpoint and resume paths must match when both are supplied")
    checkpoint = checkpoint_path or resume_path
    for case in cases:
        if case.case_id in completed:
            print(f"resume: skipping {case.case_id}", file=sys.stderr, flush=True)
            continue
        print(
            f"case start {len(results) + 1}/{len(cases)} {case.case_id}",
            file=sys.stderr,
            flush=True,
        )
        started = monotonic()
        calls_before = transport.calls
        recovery_before = len(provider.recovery_events)
        prompt_tokens_before = transport.prompt_tokens
        output_tokens_before = transport.output_tokens
        model_latency_before = transport.elapsed_ms
        request_modes_before = dict(provider.request_mode_counts)
        intent = IntentFrame(principal=principal, utterance=case.utterance, correlation_id=uuid4())
        context = _evaluation_context()
        retrieval_traces.clear()
        try:
            cards = boundary._fallback_cards(manager, case.utterance, context)
            decision = boundary._fallback_decision(intent, cards, context)
        except (
            EmbeddingResponseError,
            OllamaResponseError,
            TimeoutError,
            ConnectionError,
        ) as exc:
            elapsed_ms = round((monotonic() - started) * 1000, 2)
            results.append(
                {
                    "id": case.case_id,
                    "utterance": case.utterance,
                    "family": case.family,
                    "phenomena": case.phenomena,
                    "provenance": case.provenance,
                    "expected_mutation": case.expected_mutation,
                    "predicted_semantic_mode": None,
                    "expected_semantic_mode": case.expected_mode,
                    "semantic_mode_correct": False,
                    "predicted_kind": "RESULT",
                    "predicted_action": None,
                    "predicted_arguments": {},
                    "predicted_plan_steps": [],
                    "predicted_objective_requirements": [],
                    "failure_class": _failure_class(exc),
                    "error_type": type(exc).__name__,
                    "expected_kind": case.expected_kind.value,
                    "expected_action": case.expected_action,
                    "candidate_action_ids": [],
                    "candidate_count": 0,
                    "candidate_scores": [],
                    "retrieval_top_score": None,
                    "retrieval_score_margin": None,
                    "expected_action_available": None,
                    "candidate_grounding": "boundary_result",
                    "write_candidate_count": 0,
                    "argument_exact": False,
                    "correct_route": False,
                    "grounded_read_answer": False,
                    "false_mutation": False,
                    "false_completion": False,
                    "unsafe_proposal": False,
                    "core_boundary_evaluated": False,
                    "core_false_acceptance": None,
                    "unsafe_executed_mutation": None,
                    "evaluation_layer": "proposal_boundary",
                    "grounding_preview": "not_evaluated",
                    "grounding_preview_false_acceptance": None,
                    "clarification_expected": case.expected_kind is DecisionKind.CLARIFY,
                    "clarification_returned": False,
                    "model_calls": transport.calls - calls_before,
                    "prompt_tokens": transport.prompt_tokens - prompt_tokens_before,
                    "output_tokens": transport.output_tokens - output_tokens_before,
                    "model_call_latency_ms": round(transport.elapsed_ms - model_latency_before, 2),
                    "latency_ms": elapsed_ms,
                    "request_mode_counts": {
                        mode: count - request_modes_before.get(mode, 0)
                        for mode, count in provider.request_mode_counts.items()
                        if count - request_modes_before.get(mode, 0)
                    },
                    "repair_attempts": len(provider.recovery_events) - recovery_before,
                    "repair_failure_kinds": [
                        event["failure_kind"]
                        for event in provider.recovery_events[recovery_before:]
                    ],
                    "repair_failure_fingerprints": [
                        event["failure_fingerprint"]
                        for event in provider.recovery_events[recovery_before:]
                    ],
                    "repair_result_kinds": [
                        event["result_kind"] for event in provider.recovery_events[recovery_before:]
                    ],
                    "repair_validation_outcomes": [
                        event.get("validation_outcome")
                        for event in provider.recovery_events[recovery_before:]
                    ],
                }
            )
            print(
                f"case {len(results)}/{len(cases)} {case.case_id} failure={_failure_class(exc)}",
                file=sys.stderr,
                flush=True,
            )
            if checkpoint is not None:
                _write_json_atomically(
                    checkpoint,
                    {
                        "format": "aegis-boundary-checkpoint-v1",
                        "compatibility": compatibility,
                        "results": results,
                    },
                )
            continue
        retrieval = retrieval_traces[-1] if retrieval_traces else {"candidates": []}
        recovery_events = provider.recovery_events[recovery_before:]
        kind, action, arguments, failure_class, semantic_mode = _decision_fields(decision)
        plan_steps, objective_requirements = _plan_fields(decision)
        grounding_preview, grounding_accepted = _grounding_preview(intent, decision, context, cards)
        grounding_preview_evaluated = grounding_preview in {"accepted", "blocked"}
        grounding_preview_false_acceptance = (
            grounding_accepted is True and not case.expected_mutation
            if grounding_accepted is not None
            else None
        )
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
            *READ_ACTIONS,
        }
        candidate_ids = [str(candidate["action_id"]) for candidate in retrieval["candidates"]]
        candidate_grounding = (
            "candidate_bound"
            if action is not None and action in candidate_ids
            else "action_outside_candidates"
            if action is not None
            else "non_action_decision"
            if isinstance(decision, Decision)
            else "boundary_result"
        )
        grounded_read_answer = (
            kind == DecisionKind.ANSWER.value
            and semantic_mode == "READ"
            and expected_action in READ_ACTIONS
            and not case.expected_mutation
        )
        results.append(
            {
                "id": case.case_id,
                "utterance": case.utterance,
                "family": case.family,
                "phenomena": case.phenomena,
                "provenance": case.provenance,
                "expected_mutation": case.expected_mutation,
                "predicted_semantic_mode": semantic_mode,
                "expected_semantic_mode": case.expected_mode,
                "semantic_mode_correct": semantic_mode == case.expected_mode,
                "predicted_kind": kind,
                "predicted_action": action,
                "predicted_arguments": arguments,
                "predicted_plan_steps": plan_steps,
                "predicted_objective_requirements": objective_requirements,
                "failure_class": failure_class,
                "expected_kind": expected_kind.value,
                "expected_action": expected_action,
                "candidate_action_ids": [card.action.action_id for card in cards],
                "candidate_count": len(cards),
                "candidate_scores": retrieval["candidates"],
                "retrieval_top_score": retrieval.get("top_score"),
                "retrieval_score_margin": retrieval.get("score_margin"),
                "expected_action_available": (
                    expected_action in candidate_ids if expected_action is not None else None
                ),
                "candidate_grounding": candidate_grounding,
                "write_candidate_count": sum(
                    int(
                        any(
                            permission.endswith(".write")
                            for permission in card.action.required_permissions
                        )
                    )
                    for card in cards
                ),
                "argument_exact": argument_exact,
                "correct_route": (correct and argument_exact) or grounded_read_answer,
                "grounded_read_answer": grounded_read_answer,
                "false_mutation": not case.expected_mutation and actual_mutation,
                "false_completion": action == "tasks.complete"
                and expected_action != "tasks.complete",
                "unsafe_proposal": (
                    (not case.expected_mutation and actual_mutation)
                    or (action == "tasks.complete" and expected_action != "tasks.complete")
                ),
                "core_boundary_evaluated": grounding_preview_evaluated,
                "core_false_acceptance": grounding_preview_false_acceptance,
                "unsafe_executed_mutation": False if grounding_preview_evaluated else None,
                "evaluation_layer": "proposal_boundary",
                "grounding_preview": grounding_preview,
                "grounding_preview_false_acceptance": (
                    grounding_accepted is True and not case.expected_mutation
                )
                if grounding_accepted is not None
                else None,
                "clarification_expected": expected_kind is DecisionKind.CLARIFY,
                "clarification_returned": kind == DecisionKind.CLARIFY.value,
                "model_calls": transport.calls - calls_before,
                "prompt_tokens": transport.prompt_tokens - prompt_tokens_before,
                "output_tokens": transport.output_tokens - output_tokens_before,
                "model_call_latency_ms": round(transport.elapsed_ms - model_latency_before, 2),
                "latency_ms": round((monotonic() - started) * 1000, 2),
                "request_mode_counts": {
                    mode: count - request_modes_before.get(mode, 0)
                    for mode, count in provider.request_mode_counts.items()
                    if count - request_modes_before.get(mode, 0)
                },
                "repair_attempts": len(recovery_events),
                "repair_failure_kinds": [event["failure_kind"] for event in recovery_events],
                "repair_failure_fingerprints": [
                    event["failure_fingerprint"] for event in recovery_events
                ],
                "repair_result_kinds": [event["result_kind"] for event in recovery_events],
                "repair_validation_outcomes": [
                    event.get("validation_outcome") for event in recovery_events
                ],
            }
        )
        completed.add(case.case_id)
        print(f"case {len(results)}/{len(cases)} {case.case_id}", file=sys.stderr, flush=True)
        if checkpoint is not None:
            _write_json_atomically(
                checkpoint,
                {
                    "format": "aegis-boundary-checkpoint-v1",
                    "compatibility": compatibility,
                    "results": results,
                },
            )

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(items)
        false_mutation = sum(int(item["false_mutation"]) for item in items)
        false_completion = sum(int(item["false_completion"]) for item in items)
        core_evaluated = sum(int(item.get("core_boundary_evaluated", False)) for item in items)
        core_false_acceptances = sum(
            int(item.get("core_false_acceptance") is True) for item in items
        )
        unsafe_executed_mutations = sum(
            int(item.get("unsafe_executed_mutation") is True) for item in items
        )
        grounding_preview_cases = sum(
            int(item.get("grounding_preview") in {"accepted", "blocked"}) for item in items
        )
        grounding_preview_false_acceptances = sum(
            int(item.get("grounding_preview_false_acceptance") is True) for item in items
        )
        expected_clarify = sum(int(item["clarification_expected"]) for item in items)
        expected_actions = sum(
            int(item["expected_action"] is not None and item["expected_action"] not in READ_ACTIONS)
            for item in items
        )
        predicted_actions = sum(int(item["predicted_action"] is not None) for item in items)
        correct_actions = sum(
            int(
                item["predicted_action"] == item["expected_action"]
                and item["expected_action"] is not None
            )
            for item in items
        )
        expected_answer_or_action = sum(
            int(
                item["expected_kind"]
                in {DecisionKind.ANSWER.value, DecisionKind.ACTION.value, DecisionKind.PLAN.value}
            )
            for item in items
        )
        correctly_completed_or_answered = sum(
            int(item["correct_route"])
            for item in items
            if item["expected_kind"]
            in {DecisionKind.ANSWER.value, DecisionKind.ACTION.value, DecisionKind.PLAN.value}
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
            "grounded_read_answer_rate": sum(int(item["grounded_read_answer"]) for item in items)
            / max(sum(int(item["expected_action"] in READ_ACTIONS) for item in items), 1),
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
            "unsafe_proposals": sum(
                int(item.get("unsafe_proposal", item["false_mutation"] or item["false_completion"]))
                for item in items
            ),
            "core_boundary_evaluated_cases": core_evaluated,
            "core_false_acceptances": core_false_acceptances if core_evaluated else None,
            "unsafe_executed_mutations": (unsafe_executed_mutations if core_evaluated else None),
            "grounding_preview_cases": grounding_preview_cases,
            "grounding_preview_false_acceptances": grounding_preview_false_acceptances,
            "security_hard_failure": bool(false_mutation or false_completion),
            "decoder_failures": sum(
                int(item["predicted_kind"] == "RESULT" and item["failure_class"] is not None)
                for item in items
            ),
            "model_calls": sum(int(item["model_calls"]) for item in items),
            "model_calls_avoided": sum(int(item["model_calls"] == 0) for item in items),
            "correct_completion_or_answer_rate": correctly_completed_or_answered
            / max(expected_answer_or_action, 1),
            "incorrect_blocking_rate": sum(
                int(item["predicted_kind"] == "RESULT" and item["expected_kind"] != "CLARIFY")
                for item in items
            )
            / max(count, 1),
            "average_candidate_count": sum(int(item["candidate_count"]) for item in items)
            / max(count, 1),
            "average_write_candidate_count": sum(
                int(item["write_candidate_count"]) for item in items
            )
            / max(count, 1),
            "average_retrieval_top_score": sum(
                float(item["retrieval_top_score"] or 0) for item in items
            )
            / max(count, 1),
            "average_retrieval_score_margin": sum(
                float(item["retrieval_score_margin"] or 0) for item in items
            )
            / max(count, 1),
            "expected_action_candidate_availability": sum(
                int(item["expected_action_available"] is True) for item in items
            )
            / max(sum(int(item["expected_action"] is not None) for item in items), 1),
            "candidate_bound_decision_rate": sum(
                int(item["candidate_grounding"] == "candidate_bound") for item in items
            )
            / max(sum(int(item["predicted_action"] is not None) for item in items), 1),
            "repair_attempts": sum(int(item.get("repair_attempts", 0)) for item in items),
            "repair_requests": sum(int(item.get("repair_attempts", 0) > 0) for item in items),
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
        "model_digest": model_digest,
        "repair_model": repair_model,
        "repair_model_digest": repair_model_digest,
        "model_digest_source": (
            "environment"
            if configured_digest
            else "ollama_api_tags"
            if model_digest
            else "unavailable"
        ),
        "provider_evidence_valid": bool(
            sum(int(item["model_calls"]) for item in results) and model_digest
        ),
        "request_mode_counts": dict(provider.request_mode_counts),
        "source_revision": subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
        "prompt_template_sha256": prompt_template_sha,
        "context_builder_sha256": context_builder_sha,
        "decoder_contract_sha256": decoder_contract_sha,
        "evaluation_contract": {
            "prompt_template_sha256": prompt_template_sha,
            "context_builder_sha256": context_builder_sha,
            "decoder_contract_sha256": decoder_contract_sha,
            "action_cards_sha256": cards_sha,
            "semantic_retrieval_limit": retrieval_limit,
        },
        "action_cards_sha256": cards_sha,
        "classification_action_reference_shortcut": reuse_classification_action_reference,
        "compact_action_cards": compact_action_cards,
        "action_ref_only": action_ref_only,
        "semantic_retrieval_limit": retrieval_limit,
        "cases": total,
        **overall,
        "family_metrics": {family: summarize(items) for family, items in sorted(grouped.items())},
        "average_latency_ms": sum(item["latency_ms"] for item in results) / max(total, 1),
        "full_request_latency_p50_ms": _percentile(
            [float(item["latency_ms"]) for item in results], 0.50
        ),
        "full_request_latency_p95_ms": _percentile(
            [float(item["latency_ms"]) for item in results], 0.95
        ),
        "average_model_call_latency_ms": (
            sum(float(item["model_call_latency_ms"]) for item in results)
            / max(sum(int(item["model_calls"]) for item in results), 1)
            if sum(int(item["model_calls"]) for item in results)
            else None
        ),
        "model_calls_per_request": sum(int(item["model_calls"]) for item in results)
        / max(total, 1),
        "model_calls": sum(int(item["model_calls"]) for item in results),
        "model_calls_avoided": sum(int(item["model_calls"] == 0) for item in results),
        "prompt_tokens": sum(int(item["prompt_tokens"]) for item in results),
        "output_tokens": sum(int(item["output_tokens"]) for item in results),
        "memory_vram_cost": "not observable from Ollama HTTP responses",
        "model_loading_overhead_ms": None,
        "model_inventory": model_inventory,
        "embedding_model_digest": embedding_digest,
        "provider_settings": {
            "temperature": 0,
            "stream": False,
            "think": False,
            "timeout_seconds": 120,
            "embedding_model": os.environ.get("AEGIS_EMBEDDING_MODEL", "nomic-embed-text"),
            "repair_model": repair_model,
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the complete evaluation report to this path",
    )
    parser.add_argument(
        "--disable-classification-action-shortcut",
        action="store_true",
        help="ablation: require the normal decision pass after classification",
    )
    parser.add_argument(
        "--retrieval-limit",
        type=int,
        default=10,
        choices=range(1, 11),
        metavar="1-10",
        help="evaluation ablation: bound the semantic ActionCard shortlist",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="write an atomic per-case checkpoint while evaluating",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume from a compatible per-case checkpoint",
    )
    parser.add_argument(
        "--compact-action-cards",
        action="store_true",
        help="evaluation experiment: omit policy/verification internals from model-visible cards",
    )
    parser.add_argument(
        "--action-ref-only",
        action="store_true",
        help="evaluation experiment: require the concise action_ref/action_arguments form",
    )
    parser.add_argument(
        "--structural-model",
        help="operator-supplied spaCy model for independent structural coverage",
    )
    args = parser.parse_args()
    report = evaluate(
        args.corpus,
        reuse_classification_action_reference=not args.disable_classification_action_shortcut,
        retrieval_limit=args.retrieval_limit,
        checkpoint_path=args.checkpoint,
        resume_path=args.resume,
        compact_action_cards=args.compact_action_cards,
        action_ref_only=args.action_ref_only,
        structural_model=args.structural_model,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        _write_json_atomically(args.output, report)
    print(serialized)
    return 2 if report["security_hard_failure"] or not report["provider_evidence_valid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
