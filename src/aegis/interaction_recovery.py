"""Bounded recovery for malformed cognition proposals."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar
from uuid import uuid4

from .contracts import (
    ActionCard,
    ActionSpec,
    ClarificationAmbiguityType,
    ClarificationRecoveryOutcome,
    ClarificationRecoveryProposal,
    Context,
    Decision,
    DecisionKind,
    IntentFrame,
    ModelRequest,
    ObjectiveState,
    ProposalFailureEvidence,
    ProposalFailureKind,
    Result,
    WorkingSet,
)
from .decoding import InvalidDecision, StrictDecisionDecoder
from .interaction_context import grounded_context_answer
from .utterance import is_mutation_request, is_question_request

T = TypeVar("T")


@dataclass(frozen=True)
class ValidationResult:
    """The result of the real validator for one untrusted proposal."""

    valid: bool
    failure: ProposalFailureEvidence | None = None

    def __post_init__(self) -> None:
        if self.valid and self.failure is not None:
            raise ValueError("valid proposal cannot carry failure evidence")
        if not self.valid and self.failure is None:
            raise ValueError("invalid proposal requires failure evidence")


@dataclass(frozen=True)
class ProposalRepairEvent:
    """Bounded runtime telemetry for one repair attempt."""

    attempt: int
    input_failure_kind: ProposalFailureKind
    input_failure_fingerprint: str
    decode_outcome: str
    validator_stage: str
    validation_outcome: str
    output_failure_kind: ProposalFailureKind | None = None
    output_failure_fingerprint: str | None = None
    stop_reason: str | None = None


def proposal_repair_event_record(event: ProposalRepairEvent) -> dict[str, Any]:
    """Expose rich recovery telemetry without breaking evaluator consumers."""

    return {
        "attempt": event.attempt,
        "failure_kind": event.input_failure_kind.value,
        "failure_fingerprint": event.input_failure_fingerprint,
        "result_kind": event.output_failure_kind.value if event.output_failure_kind else None,
        "validation_outcome": event.validation_outcome,
        "validator_stage": event.validator_stage,
        "decode_outcome": event.decode_outcome,
        "output_failure_kind": (
            event.output_failure_kind.value if event.output_failure_kind else None
        ),
        "output_failure_fingerprint": event.output_failure_fingerprint,
        "stop_reason": event.stop_reason,
    }


@dataclass(frozen=True)
class ProposalRepairResult(Generic[T]):
    proposal: T | None
    events: tuple[ProposalRepairEvent, ...]
    stop_reason: str


def bounded_proposal_repair(
    initial: T,
    failure: ProposalFailureEvidence,
    repair: Callable[[T, ProposalFailureEvidence], tuple[T | None, ProposalFailureEvidence]],
    validate: Callable[[T], ValidationResult],
    *,
    validator_stage: str,
    max_attempts: int = 2,
) -> ProposalRepairResult[T]:
    """Run a finite repair loop where every candidate returns to the validator."""

    if max_attempts < 0 or max_attempts > 2:
        raise ValueError("proposal repair budget must be between zero and two")
    current = initial
    current_failure = failure
    seen = {proposal_failure_fingerprint(failure)}
    events: list[ProposalRepairEvent] = []
    for attempt in range(1, max_attempts + 1):
        candidate, decode_failure = repair(current, current_failure)
        next_failure: ProposalFailureEvidence | None
        if candidate is None:
            next_failure = decode_failure
            event = ProposalRepairEvent(
                attempt=attempt,
                input_failure_kind=current_failure.kind,
                input_failure_fingerprint=proposal_failure_fingerprint(current_failure),
                decode_outcome="rejected",
                validator_stage=validator_stage,
                validation_outcome="not_run",
                output_failure_kind=next_failure.kind,
                output_failure_fingerprint=proposal_failure_fingerprint(next_failure),
            )
        else:
            result = validate(candidate)
            next_failure = result.failure
            event = ProposalRepairEvent(
                attempt=attempt,
                input_failure_kind=current_failure.kind,
                input_failure_fingerprint=proposal_failure_fingerprint(current_failure),
                decode_outcome="decoded",
                validator_stage=validator_stage,
                validation_outcome="valid" if result.valid else "invalid",
                output_failure_kind=next_failure.kind if next_failure else None,
                output_failure_fingerprint=(
                    proposal_failure_fingerprint(next_failure) if next_failure else None
                ),
            )
            if result.valid:
                events.append(event)
                return ProposalRepairResult(candidate, tuple(events), "VALIDATED")
        events.append(event)
        if next_failure is None:
            raise ValueError("invalid repair must produce failure evidence")
        fingerprint = proposal_failure_fingerprint(next_failure)
        if fingerprint in seen:
            return ProposalRepairResult(None, tuple(events), "REPEATED_FAILURE")
        seen.add(fingerprint)
        current = candidate if candidate is not None else current
        current_failure = next_failure
    return ProposalRepairResult(None, tuple(events), "BUDGET_EXHAUSTED")


def proposal_failure_evidence(error: Exception) -> ProposalFailureEvidence:
    """Map validator failures to a bounded repair diagnosis."""

    message = str(error).casefold()
    mappings = (
        (("more than one", "change"), ProposalFailureKind.UNACCOUNTED_STRUCTURAL_ANCHOR),
        (("complete", "verified", "plan"), ProposalFailureKind.MISSING_EFFECT),
        (("coverage", "requirement", "missing"), ProposalFailureKind.MISSING_EFFECT),
        (("extra", "coverage"), ProposalFailureKind.EXTRA_EFFECT),
        (("negat",), ProposalFailureKind.NEGATED_EFFECT_ACTIVE),
        (("supersed",), ProposalFailureKind.SUPERSEDED_EFFECT_ACTIVE),
        (("span", "ground"), ProposalFailureKind.BAD_SOURCE_SPAN),
        (("argument", "missing"), ProposalFailureKind.MISSING_ARGUMENT),
        (("argument",), ProposalFailureKind.INVALID_ARGUMENT),
        (("not an authorized",), ProposalFailureKind.CAPABILITY_MISMATCH),
        (("capability", "unavailable"), ProposalFailureKind.CAPABILITY_UNAVAILABLE),
        (("ambiguous",), ProposalFailureKind.AMBIGUOUS_ENTITY),
        (("unknown", "entity"), ProposalFailureKind.UNKNOWN_ENTITY),
        (("unsupported",), ProposalFailureKind.UNSUPPORTED_REQUIREMENT),
    )
    kind = ProposalFailureKind.DECODER_SCHEMA_FAILURE
    for needles, candidate in mappings:
        if all(needle in message for needle in needles):
            kind = candidate
            break
    return ProposalFailureEvidence(kind=kind, detail=str(error)[:240] or None)


def proposal_failure_fingerprint(evidence: ProposalFailureEvidence) -> str:
    """Return a stable, privacy-minimal identity for repeated repair failures."""

    payload = evidence.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def repair_invalid_decision_once(
    provider: Any,
    intent: IntentFrame,
    context: Context,
    cards: tuple[ActionCard, ...],
    raw: dict[str, Any] | None,
    error: InvalidDecision,
    evidence: ProposalFailureEvidence | None = None,
    validator_stage: str = "decision_decoder",
) -> Decision | None:
    """Ask for one bounded repair; the ordinary decoder remains the gate."""

    repaired, _failure, _raw = repair_invalid_decision_once_with_evidence(
        provider, intent, context, cards, raw, error, evidence, validator_stage
    )
    return repaired


def repair_invalid_decision_once_with_evidence(
    provider: Any,
    intent: IntentFrame,
    context: Context,
    cards: tuple[ActionCard, ...],
    raw: dict[str, Any] | None,
    error: InvalidDecision,
    evidence: ProposalFailureEvidence | None = None,
    validator_stage: str = "decision_decoder",
) -> tuple[Decision | None, ProposalFailureEvidence, dict[str, Any] | None]:
    """Repair once while retaining the next typed failure and failed proposal."""

    evidence = proposal_failure_evidence(error) if evidence is None else evidence
    response = provider.decide(
        ModelRequest(
            working_set=WorkingSet(intent=intent, context=context),
            action_cards=cards,
            allow_argument_proposals=True,
            allow_plan_proposals=True,
            proposal_repair_only=True,
            repair_validator_stage=validator_stage,
            proposal_failure=evidence,
            current_proposal=raw,
        )
    )
    next_evidence = evidence
    try:
        repaired = StrictDecisionDecoder().decode(response, cards, allow_argument_proposals=True)
    except InvalidDecision as next_error:
        repaired = None
        next_evidence = proposal_failure_evidence(next_error)
    events = getattr(provider, "recovery_events", None)
    if isinstance(events, list):
        events.append(
            {
                "attempt": 1,
                "failure_kind": evidence.kind.value,
                "failure_fingerprint": proposal_failure_fingerprint(evidence),
                "result_kind": response.raw.get("kind") if isinstance(response.raw, dict) else None,
                "validation_outcome": "decoded" if repaired is not None else "rejected",
                "validator_stage": validator_stage,
                "decode_outcome": "decoded" if repaired is not None else "rejected",
                "output_failure_kind": next_evidence.kind.value if repaired is None else None,
                "output_failure_fingerprint": (
                    proposal_failure_fingerprint(next_evidence) if repaired is None else None
                ),
                "stop_reason": None,
            }
        )
    return repaired, next_evidence, response.raw if isinstance(response.raw, dict) else None


@dataclass(frozen=True)
class ClarificationRecoveryEvaluationCase:
    """Development-only case for the deterministic recovery safety boundary."""

    name: str
    proposal: ClarificationRecoveryProposal
    cards: tuple[ActionCard, ...]
    context: Context
    expected_resolved: bool


@dataclass(frozen=True)
class ClarificationRecoveryEvaluationMetrics:
    """Core safety metrics; provider latency/model quality stay with the caller."""

    cases: int
    expected_resolutions: int
    accepted_resolutions: int
    unsafe_acceptances: int
    expected_rejections: int
    rejected_cases: int


def validate_clarification_recovery(
    proposal: ClarificationRecoveryProposal,
    cards: tuple[ActionCard, ...],
    context: Context,
) -> bool:
    """Validate a recovery proposal without turning it into executable authority."""

    if proposal.outcome is not ClarificationRecoveryOutcome.RESOLVED:
        return False
    card = next((card for card in cards if card.action.action_id == proposal.action_ref), None)
    if card is None:
        return False
    if not set(proposal.arguments).issubset(card.argument_keys):
        return False
    if (
        proposal.ambiguity_type is ClarificationAmbiguityType.REFERENT
        and proposal.referent_ref is None
    ):
        return False
    if proposal.referent_ref is None:
        return True
    referents = context.values.get("referents")
    those = referents.get("those") if isinstance(referents, dict) else None
    candidates = those.get("candidates") if isinstance(those, dict) else None
    if not isinstance(candidates, list):
        return False
    matches = []
    label_matches = []
    proposed_title = proposal.arguments.get("title")
    for candidate in candidates:
        if isinstance(candidate, dict):
            identity = candidate.get("id") or candidate.get("task_id") or candidate.get("chore_id")
            if identity == proposal.referent_ref:
                matches.append(candidate)
            if isinstance(proposed_title, str) and candidate.get("title") == proposed_title:
                label_matches.append(candidate)
        elif candidate == proposal.referent_ref:
            matches.append(candidate)
    return len(matches) == 1 and (not proposed_title or len(label_matches) == 1)


def development_clarification_recovery_cases() -> tuple[ClarificationRecoveryEvaluationCase, ...]:
    """Return varied, execution-free recovery cases for the bounded spike."""

    card = ActionCard(
        action=ActionSpec(action_id="example.complete", capability="example.complete"),
        summary="Complete a named task",
        relevance=1,
        argument_keys=("title",),
    )

    def context(*items: dict[str, str]) -> Context:
        return Context(
            values={
                "referents": {"those": {"fact_key": "canonical_tasks", "candidates": list(items)}}
            },
            sources=("authorized_canonical_result",),
        )

    def proposal(
        outcome: ClarificationRecoveryOutcome,
        *,
        referent_ref: str | None = "task-1",
        title: str | None = "Replace porch bulb",
        ambiguity_type: ClarificationAmbiguityType = ClarificationAmbiguityType.REFERENT,
    ) -> ClarificationRecoveryProposal:
        return ClarificationRecoveryProposal(
            outcome=outcome,
            ambiguity_type=ambiguity_type,
            action_ref="example.complete",
            referent_ref=referent_ref,
            arguments={"title": title} if title is not None else {},
        )

    return (
        ClarificationRecoveryEvaluationCase(
            "unique-referent-pronoun-typo",
            proposal(ClarificationRecoveryOutcome.RESOLVED),
            (card,),
            context({"task_id": "task-1", "title": "Replace porch bulb"}),
            True,
        ),
        ClarificationRecoveryEvaluationCase(
            "duplicate-partial-name",
            proposal(ClarificationRecoveryOutcome.RESOLVED, title="Call dentist"),
            (card,),
            context(
                {"task_id": "task-1", "title": "Call dentist"},
                {"task_id": "task-2", "title": "Call dentist"},
            ),
            False,
        ),
        ClarificationRecoveryEvaluationCase(
            "stale-referent",
            proposal(ClarificationRecoveryOutcome.RESOLVED, referent_ref="gone"),
            (card,),
            context({"task_id": "task-1", "title": "Replace porch bulb"}),
            False,
        ),
        ClarificationRecoveryEvaluationCase(
            "missing-argument",
            proposal(ClarificationRecoveryOutcome.RESOLVED).model_copy(
                update={"arguments": {"task_id": "task-1"}}
            ),
            (card,),
            context({"task_id": "task-1", "title": "Replace porch bulb"}),
            False,
        ),
        ClarificationRecoveryEvaluationCase(
            "unsupported-capability",
            proposal(
                ClarificationRecoveryOutcome.UNSUPPORTED,
                ambiguity_type=ClarificationAmbiguityType.CAPABILITY,
            ),
            (),
            Context(values={}, sources=()),
            False,
        ),
        ClarificationRecoveryEvaluationCase(
            "ambiguous-read-write",
            proposal(ClarificationRecoveryOutcome.NEED_USER),
            (card,),
            context({"task_id": "task-1", "title": "Replace porch bulb"}),
            False,
        ),
    )


def evaluate_clarification_recovery_cases(
    cases: tuple[ClarificationRecoveryEvaluationCase, ...] | None = None,
) -> ClarificationRecoveryEvaluationMetrics:
    """Measure Core recovery safety without invoking a provider or Kernel."""

    cases = development_clarification_recovery_cases() if cases is None else cases
    accepted = sum(
        validate_clarification_recovery(case.proposal, case.cards, case.context) for case in cases
    )
    expected = sum(case.expected_resolved for case in cases)
    unsafe = sum(
        int(
            validate_clarification_recovery(case.proposal, case.cards, case.context)
            and not case.expected_resolved
        )
        for case in cases
    )
    return ClarificationRecoveryEvaluationMetrics(
        cases=len(cases),
        expected_resolutions=expected,
        accepted_resolutions=accepted,
        unsafe_acceptances=unsafe,
        expected_rejections=len(cases) - expected,
        rejected_cases=len(cases) - accepted,
    )


def request_clarification_recovery(
    provider: Any,
    intent: IntentFrame,
    context: Context,
    cards: tuple[ActionCard, ...],
    reason: str,
) -> ClarificationRecoveryProposal | None:
    """Run the isolated recovery mode and return only its validated-shaped proposal."""

    if not context.values.get("referents") and not context.values.get("canonical_facts"):
        return None
    response = provider.decide(
        ModelRequest(
            working_set=WorkingSet(intent=intent, context=context),
            action_cards=cards,
            clarification_recovery_only=True,
            clarification_reason=reason,
        )
    )
    try:
        proposal = ClarificationRecoveryProposal.model_validate(response.raw)
    except Exception:
        return None
    return proposal if validate_clarification_recovery(proposal, cards, context) else None


def recover_invalid_model_decision(
    dependencies: Any,
    intent: IntentFrame,
    context: Context,
    focused_raw: dict[str, Any] | None,
    error: InvalidDecision,
) -> Decision | Result:
    """Recover only to grounded answers or a truthful retryable failure.

    Recovery has no action-capable path.  A malformed proposal can therefore
    never become execution merely because a best-effort retry succeeds.
    """
    if focused_raw is not None:
        grounded = grounded_context_answer(context, focused_raw)
        if grounded is not None:
            return grounded
    if is_question_request(intent.utterance) and not is_mutation_request(intent.utterance):
        try:
            provider_factory = dependencies.model_provider
            if provider_factory is None:
                raise RuntimeError("model provider unavailable")
            recovered = StrictDecisionDecoder().decode(
                provider_factory().decide(
                    ModelRequest(
                        working_set=WorkingSet(intent=intent, context=context), action_cards=()
                    )
                ),
                (),
                allow_argument_proposals=False,
            )
            if recovered.kind is DecisionKind.ANSWER:
                return recovered
        except Exception:
            # Recovery is deliberately best-effort and answer-only; retain the
            # original bounded failure if it cannot produce a valid answer.
            pass
    return Result(
        objective_id=uuid4(),
        state=ObjectiveState.BLOCKED,
        message="I could not safely interpret that request. Please rephrase it.",
        evidence={
            "provenance": "model_boundary",
            "authoritative": False,
            "failure_class": "invalid_model_decision",
            "failure_reason": str(error),
        },
        correlation_id=intent.correlation_id,
        retryable=True,
    )
