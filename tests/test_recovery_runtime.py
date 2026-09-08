from __future__ import annotations

from uuid import uuid4

from aegis.contracts import (
    ExternalEffectAssurance,
    IntentFrame,
    Objective,
    ObjectiveState,
    Principal,
    RecoveryDisposition,
    RecoveryReason,
    RecoveryState,
    Result,
)
from aegis.interaction_recovery import advance_recovery, classify_recovery
from aegis.store import SqliteObjectiveStore


def _result(*, evidence: dict, state: ObjectiveState = ObjectiveState.BLOCKED) -> Result:
    return Result(
        objective_id=uuid4(),
        state=state,
        message="not complete",
        evidence=evidence,
        correlation_id=uuid4(),
    )


def test_capability_block_is_internal_until_owner_input_is_required() -> None:
    internal = _result(
        evidence={
            "authoritative": False,
            "objective_open": True,
            "capability_needs": [{"need_id": "need-1", "status": "open"}],
        }
    )
    classified = classify_recovery(internal)
    assert classified.disposition is RecoveryDisposition.INTERNAL_BLOCKED
    assert classified.reason is RecoveryReason.CAPABILITY_UNAVAILABLE

    owner = _result(
        evidence={
            "authoritative": False,
            "objective_open": True,
            "capability_needs": [{"need_id": "need-1", "status": "owner_input_required"}],
        }
    )
    classified = classify_recovery(owner)
    assert classified.disposition is RecoveryDisposition.OWNER_BLOCKED
    assert classified.owner_actionable is True


def test_recovery_budget_stops_repeated_identical_evidence() -> None:
    result = _result(
        evidence={
            "authoritative": False,
            "objective_open": True,
            "capability_needs": [{"need_id": "need-1", "status": "open"}],
        }
    )
    classification = classify_recovery(result)
    first = advance_recovery(RecoveryState(), classification)
    stopped = advance_recovery(first, classification)
    assert first.steps_consumed == 1
    assert stopped.steps_consumed == 1
    assert stopped.disposition is RecoveryDisposition.NONE
    assert stopped.reason is RecoveryReason.RECOVERY_BUDGET_EXHAUSTED


def test_unknown_external_outcome_never_becomes_retry_recovery() -> None:
    result = _result(
        state=ObjectiveState.FAILED,
        evidence={"assurance": ExternalEffectAssurance.OUTCOME_UNKNOWN.value},
    )
    classified = classify_recovery(result)
    assert classified.disposition is RecoveryDisposition.NONE
    assert advance_recovery(RecoveryState(), classified).steps_consumed == 0


def test_recovery_state_round_trips_and_legacy_objective_defaults() -> None:
    principal = Principal(id="alice", vault_id="vault", space_ids=("space",))
    objective = Objective(
        intent=IntentFrame(principal=principal, utterance="find the missing capability"),
        correlation_id=uuid4(),
        state=ObjectiveState.BLOCKED,
        recovery=RecoveryState(
            disposition=RecoveryDisposition.INTERNAL_BLOCKED,
            reason=RecoveryReason.CAPABILITY_UNAVAILABLE,
            steps_consumed=1,
            capability_investigations_consumed=1,
            last_failure_fingerprint="fingerprint",
        ),
    )
    store = SqliteObjectiveStore(":memory:")
    store.save_objective(objective)
    restored = store.get_objective(objective.id)
    assert restored is not None
    assert restored.recovery == objective.recovery

    legacy = Objective.model_validate(objective.model_dump(exclude={"recovery"}))
    assert legacy.recovery.disposition is RecoveryDisposition.NONE
