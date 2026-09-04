from aegis.contracts import (
    ActionCard,
    ActionSpec,
    ClarificationAmbiguityType,
    ClarificationRecoveryOutcome,
    ClarificationRecoveryProposal,
    Context,
    IntentFrame,
    Principal,
    ProposalFailureEvidence,
    ProposalFailureKind,
)
from aegis.decoding import InvalidDecision
from aegis.interaction_recovery import (
    ValidationResult,
    bounded_proposal_repair,
    evaluate_clarification_recovery_cases,
    proposal_failure_evidence,
    proposal_failure_fingerprint,
    proposal_repair_event_record,
    repair_invalid_decision_once,
    repair_invalid_decision_once_with_evidence,
    request_clarification_recovery,
    validate_clarification_recovery,
)


def card() -> ActionCard:
    return ActionCard(
        action=ActionSpec(action_id="tasks.complete", capability="tasks.complete"),
        summary="Complete a named task",
        relevance=1,
        argument_keys=("title",),
    )


def resolved(referent_ref: str | None = "task-1") -> ClarificationRecoveryProposal:
    return ClarificationRecoveryProposal(
        outcome=ClarificationRecoveryOutcome.RESOLVED,
        ambiguity_type=ClarificationAmbiguityType.REFERENT,
        action_ref="tasks.complete",
        referent_ref=referent_ref,
        arguments={"title": "Replace porch bulb"},
    )


def context(*candidates: dict[str, str]) -> Context:
    return Context(
        values={
            "referents": {"those": {"fact_key": "canonical_tasks", "candidates": list(candidates)}}
        },
        sources=("authorized_canonical_result",),
    )


def test_recovery_accepts_one_current_authorized_referent() -> None:
    assert validate_clarification_recovery(
        resolved(),
        (card(),),
        context({"task_id": "task-1", "title": "Replace porch bulb"}),
    )


def test_recovery_rejects_duplicate_referent_identity() -> None:
    assert not validate_clarification_recovery(
        resolved(),
        (card(),),
        context(
            {"task_id": "task-1", "title": "Call dentist"},
            {"task_id": "task-2", "title": "Call dentist"},
        ),
    )


def test_recovery_rejects_stale_or_unauthorized_referent() -> None:
    assert not validate_clarification_recovery(
        resolved("missing"),
        (card(),),
        context({"task_id": "task-1", "title": "Replace porch bulb"}),
    )


def test_recovery_referent_cannot_fabricate_a_title_without_identity() -> None:
    assert not validate_clarification_recovery(
        resolved(None),
        (card(),),
        context({"task_id": "task-1", "title": "Replace porch bulb"}),
    )


def test_recovery_rejects_nonresolved_proposal_and_undeclared_arguments() -> None:
    proposal = resolved()
    unsupported = proposal.model_copy(update={"outcome": ClarificationRecoveryOutcome.UNSUPPORTED})
    assert not validate_clarification_recovery(
        unsupported, (card(),), context({"task_id": "task-1", "title": "Replace porch bulb"})
    )
    invalid_args = proposal.model_copy(update={"arguments": {"task_id": "task-1"}})
    assert not validate_clarification_recovery(
        invalid_args, (card(),), context({"task_id": "task-1", "title": "Replace porch bulb"})
    )


def test_recovery_request_isolated_mode_returns_no_direct_action() -> None:
    class Provider:
        recovery_events: list[dict[str, object]] = []

        def decide(self, request: object) -> object:
            assert getattr(request, "clarification_recovery_only") is True
            assert getattr(request, "clarification_reason") == "ambiguous target"
            return type(
                "Response",
                (),
                {"raw": resolved().model_dump(mode="json")},
            )()

    proposal = request_clarification_recovery(
        Provider(),
        IntentFrame(
            principal=Principal(id="alice", vault_id="vault"),
            utterance="finish that bulb thing",
        ),
        context({"task_id": "task-1", "title": "Replace porch bulb"}),
        (card(),),
        "ambiguous target",
    )
    assert proposal is not None
    assert not hasattr(proposal, "action")


def test_development_recovery_corpus_has_zero_unsafe_acceptances() -> None:
    metrics = evaluate_clarification_recovery_cases()

    assert metrics.cases == 6
    assert metrics.expected_resolutions == 1
    assert metrics.accepted_resolutions == 1
    assert metrics.unsafe_acceptances == 0
    assert metrics.rejected_cases == 5


def test_proposal_failure_evidence_is_bounded_and_fingerprint_stable() -> None:
    evidence = proposal_failure_evidence(ValueError("plan has missing requirement coverage"))
    assert evidence.kind is ProposalFailureKind.MISSING_EFFECT
    assert proposal_failure_fingerprint(evidence) == proposal_failure_fingerprint(
        ProposalFailureEvidence(kind=ProposalFailureKind.MISSING_EFFECT, detail=evidence.detail)
    )
    assert len(proposal_failure_fingerprint(evidence)) == 16
    structural = proposal_failure_evidence(
        ValueError("This request contains more than one change, but the plan is incomplete")
    )
    assert structural.kind is ProposalFailureKind.UNACCOUNTED_STRUCTURAL_ANCHOR


def test_repair_contract_is_not_an_execution_contract() -> None:
    from aegis.contracts import IntentFrame, ModelRequest, WorkingSet

    request = ModelRequest(
        working_set=WorkingSet(
            intent=IntentFrame(utterance="add milk", principal=Principal(id="a", vault_id="v")),
            context=Context(),
        ),
        action_cards=(card(),),
        proposal_repair_only=True,
        proposal_failure=ProposalFailureEvidence(kind=ProposalFailureKind.MISSING_ARGUMENT),
        current_proposal={"kind": "ACTION"},
    )
    assert request.proposal_repair_only is True
    assert request.proposal_failure is not None
    assert not hasattr(request.proposal_failure, "action")


def test_invalid_proposal_repair_is_decoded_against_supplied_cards() -> None:
    from aegis.decoding import InvalidDecision

    class Provider:
        recovery_events: list[dict[str, object]] = []

        def decide(self, request: object) -> object:
            assert getattr(request, "proposal_repair_only") is True
            return type(
                "Response",
                (),
                {
                    "raw": {
                        "kind": "ACTION",
                        "semantic_mode": "ACTION",
                        "knowledge_source": "general_model_knowledge",
                        "action": {
                            **card().action.model_dump(mode="json"),
                            "arguments": {"title": "Replace porch bulb"},
                        },
                    }
                },
            )()

    provider = Provider()
    repaired = repair_invalid_decision_once(
        provider,
        IntentFrame(utterance="finish that bulb thing", principal=Principal(id="a", vault_id="v")),
        Context(),
        (card(),),
        {"kind": "ACTION", "action_ref": "tasks.complete"},
        InvalidDecision("missing argument"),
    )
    assert repaired is not None
    assert repaired.kind.value == "ACTION"
    assert repaired.action is not None
    assert repaired.action.action_id == "tasks.complete"
    assert provider.recovery_events[0]["attempt"] == 1
    assert provider.recovery_events[0]["validator_stage"] == "decision_decoder"
    assert provider.recovery_events[0]["decode_outcome"] == "decoded"


def test_bounded_repair_revalidates_and_uses_new_failure_for_attempt_two() -> None:
    f1 = ProposalFailureEvidence(kind=ProposalFailureKind.MISSING_EFFECT)
    f2 = ProposalFailureEvidence(kind=ProposalFailureKind.INVALID_ARGUMENT)
    calls: list[tuple[str, str]] = []

    def repair(value: str, failure: ProposalFailureEvidence) -> tuple[str, ProposalFailureEvidence]:
        calls.append((value, failure.kind.value))
        return ("p1", f2) if len(calls) == 1 else ("p2", f2)

    def validate(value: str) -> ValidationResult:
        return ValidationResult(valid=value == "p2", failure=None if value == "p2" else f2)

    result = bounded_proposal_repair(
        "p0", f1, repair, validate, validator_stage="test", max_attempts=2
    )
    assert result.proposal == "p2"
    assert result.stop_reason == "VALIDATED"
    assert calls == [("p0", "MISSING_EFFECT"), ("p1", "INVALID_ARGUMENT")]
    assert result.events[0].validation_outcome == "invalid"
    assert result.events[1].validation_outcome == "valid"


def test_bounded_repair_stops_repeated_failure_without_second_attempt() -> None:
    failure = ProposalFailureEvidence(kind=ProposalFailureKind.MISSING_EFFECT)
    calls = 0

    def repair(
        value: str, _failure: ProposalFailureEvidence
    ) -> tuple[str, ProposalFailureEvidence]:
        nonlocal calls
        calls += 1
        return value, failure

    result = bounded_proposal_repair(
        "p0",
        failure,
        repair,
        lambda _value: ValidationResult(valid=False, failure=failure),
        validator_stage="test",
        max_attempts=2,
    )
    assert result.stop_reason == "REPEATED_FAILURE"
    assert calls == 1


def test_repair_event_record_preserves_legacy_and_rich_telemetry() -> None:
    failure = ProposalFailureEvidence(kind=ProposalFailureKind.MISSING_EFFECT)
    result = bounded_proposal_repair(
        "p0",
        failure,
        lambda _value, _failure: ("p1", failure),
        lambda _value: ValidationResult(
            valid=False, failure=ProposalFailureEvidence(kind=ProposalFailureKind.INVALID_ARGUMENT)
        ),
        validator_stage="requested_effect_structural_coverage",
        max_attempts=1,
    )

    record = proposal_repair_event_record(result.events[0])
    assert record["failure_kind"] == "MISSING_EFFECT"
    assert record["result_kind"] == "INVALID_ARGUMENT"
    assert record["validator_stage"] == "requested_effect_structural_coverage"
    assert record["attempt"] == 1


def test_decoder_repair_preserves_failed_proposal_for_second_attempt() -> None:
    responses = iter(
        [
            {"kind": "ACTION"},
            {
                "kind": "ACTION",
                "semantic_mode": "ACTION",
                "knowledge_source": "general_model_knowledge",
                "action_ref": "tasks.complete",
                "action_arguments": {"title": "Replace porch bulb"},
            },
        ]
    )

    class Provider:
        def decide(self, _request: object) -> object:
            return type("Response", (), {"raw": next(responses)})()

    error = InvalidDecision("initial decision is malformed")
    first, failure, failed_raw = repair_invalid_decision_once_with_evidence(
        Provider(),
        IntentFrame(utterance="finish that bulb thing", principal=Principal(id="a", vault_id="v")),
        Context(),
        (card(),),
        {"kind": "CLARIFY"},
        error,
    )
    assert first is None
    assert failed_raw == {"kind": "ACTION"}
    second, _next_failure, _next_raw = repair_invalid_decision_once_with_evidence(
        Provider(),
        IntentFrame(utterance="finish that bulb thing", principal=Principal(id="a", vault_id="v")),
        Context(),
        (card(),),
        failed_raw,
        InvalidDecision(failure.detail or "decoder failure"),
        failure,
    )
    assert second is not None
    assert second.action is not None
    assert second.action.arguments["title"] == "Replace porch bulb"
