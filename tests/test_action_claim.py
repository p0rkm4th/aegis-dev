from __future__ import annotations

import os
from threading import Barrier, Event, Lock, Thread
from uuid import uuid4

import pytest

from aegis.contracts import (
    ActionSpec,
    ExecutionRequest,
    IntentFrame,
    ModelResponse,
    Objective,
    ObjectiveState,
    Observation,
    PolicyDecision,
    Principal,
    VerificationContract,
    VerificationResult,
)
from aegis.decoding import StrictDecisionDecoder
from aegis.kernel import Kernel
from aegis.store import InMemoryObjectiveStore, PostgresObjectiveStore, SqliteObjectiveStore


def request(key: str, objective_id=None) -> ExecutionRequest:
    return ExecutionRequest(
        objective_id=objective_id or uuid4(),
        action_id=uuid4(),
        action=ActionSpec(action_id="write", capability="test.write"),
        idempotency_key=key,
    )


@pytest.mark.parametrize(
    "store_factory", [InMemoryObjectiveStore, lambda: SqliteObjectiveStore(":memory:")]
)
def test_action_claim_contract_is_atomic_and_canonical(store_factory):
    store = store_factory()
    first = request("same-key")
    second = request("same-key")

    first_claim = store.claim_action(first)
    second_claim = store.claim_action(second)
    independent = store.claim_action(request("other-key"))

    assert first_claim.acquired is True
    assert second_claim.acquired is False
    assert first_claim.request == second_claim.request == first
    assert independent.acquired is True
    if hasattr(store, "close"):
        store.close()


def test_kernel_loser_returns_transient_claimed_state_without_execution():
    store = InMemoryObjectiveStore()
    principal = Principal(id="alice", vault_id="alice-vault")
    correlation_id = uuid4()
    action = ActionSpec(
        action_id="write",
        capability="test.write",
        verification=VerificationContract(kind="readback"),
    )
    objective = Objective(
        intent=IntentFrame(principal=principal, utterance="do it", correlation_id=correlation_id),
        correlation_id=correlation_id,
        action=action,
        state=ObjectiveState.VALIDATED,
    )
    store.save_objective(objective)
    key = f"{correlation_id}:write"
    canonical = request(key, objective.id).model_copy(update={"action": action})
    assert store.claim_action(canonical).acquired is True

    class Executor:
        calls = 0

        def execute(self, _request):
            self.calls += 1
            raise AssertionError("loser crossed the executor boundary")

    class Model:
        def decide(self, _request):
            return ModelResponse(raw={"kind": "ACTION", "action": action.model_dump(mode="json")})

    class Policy:
        def authorize(self, _request):
            return PolicyDecision(allowed=True, reason="ok")

    class Verifier:
        def verify(self, _observation, _contract):
            return VerificationResult(verified=True, evidence={}, reason="verified")

    executor = Executor()
    result = Kernel(
        Model(), StrictDecisionDecoder(), Policy(), executor, Verifier(), store=store
    ).run(IntentFrame(principal=principal, utterance="do it again", correlation_id=correlation_id))

    assert result.state is ObjectiveState.EXECUTING
    assert result.retryable is False
    assert result.evidence == {"execution_claimed": True, "observation": "pending"}
    assert executor.calls == 0
    assert store.get_observation(key) is None
    assert store.get_result(key) is None


@pytest.mark.skipif(
    not os.environ.get("AEGIS_TEST_DATABASE_URL"), reason="requires disposable PostgreSQL"
)
def test_postgres_two_connection_kernel_claim_has_one_executor_owner():
    import psycopg

    url = os.environ["AEGIS_TEST_DATABASE_URL"]
    principal = Principal(id=f"claim-{uuid4()}", vault_id=f"vault-{uuid4()}")
    correlation_id = uuid4()
    action = ActionSpec(
        action_id="write",
        capability="test.write",
        verification=VerificationContract(kind="readback"),
    )
    objective = Objective(
        intent=IntentFrame(principal=principal, utterance="do it", correlation_id=correlation_id),
        correlation_id=correlation_id,
        action=action,
        state=ObjectiveState.VALIDATED,
    )
    setup = psycopg.connect(url)
    setup.execute(
        "INSERT INTO aegis_principals (id, external_subject) VALUES (%s, %s)",
        (principal.id, principal.id),
    )
    setup.execute(
        "INSERT INTO vaults (id, owner_principal_id) VALUES (%s, %s)",
        (principal.vault_id, principal.id),
    )
    PostgresObjectiveStore(setup).save_objective(objective)
    setup.close()

    class Model:
        def decide(self, _request):
            return ModelResponse(raw={"kind": "ACTION", "action": action.model_dump(mode="json")})

    class Policy:
        def authorize(self, _request):
            return PolicyDecision(allowed=True, reason="ok")

    class Verifier:
        def verify(self, _observation, _contract):
            return VerificationResult(verified=True, evidence={"readback": True}, reason="verified")

    class SharedExecutor:
        def __init__(self):
            self.calls = 0
            self.lock = Lock()
            self.started = Event()
            self.release = Event()

        def execute(self, request):
            with self.lock:
                self.calls += 1
                self.started.set()
            assert self.release.wait(60)
            return Observation(
                execution_id=request.action_id,
                evidence={"accepted": True},
                command_succeeded=True,
            )

    executor = SharedExecutor()
    start = Barrier(2)
    loser_claimed = Event()
    results = []
    errors = []

    class GatedStore(PostgresObjectiveStore):
        def claim_action(self, request, state=ObjectiveState.EXECUTING):
            claim = super().claim_action(request, state)
            if not claim.acquired:
                loser_claimed.set()
            return claim

    def contender():
        connection = psycopg.connect(url)
        try:
            start.wait(60)
            result = Kernel(
                Model(),
                StrictDecisionDecoder(),
                Policy(),
                executor,
                Verifier(),
                store=GatedStore(connection),
            ).run(
                IntentFrame(principal=principal, utterance="do it", correlation_id=correlation_id)
            )
            results.append(result)
        except BaseException as exc:
            errors.append(exc)
        finally:
            connection.close()

    threads = [Thread(target=contender) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert executor.started.wait(60)
    assert loser_claimed.wait(60)
    executor.release.set()
    for thread in threads:
        thread.join(60)

    assert not errors
    assert executor.calls == 1
    # The ownership proof is established before release above: the loser
    # observed a claimed action while the sole executor was still blocked.
    # Once the winner commits, the loser may legitimately converge to the
    # canonical completed Result instead of returning its transient state.
    assert len(results) == 2
    assert all(
        result.state in {ObjectiveState.COMPLETED, ObjectiveState.EXECUTING} for result in results
    )
    assert any(result.state is ObjectiveState.COMPLETED for result in results)

    check = psycopg.connect(url)
    key = f"{correlation_id}:write"
    action_count = check.execute(
        "SELECT count(*) FROM actions WHERE idempotency_key = %s", (key,)
    ).fetchone()[0]
    observation_count = check.execute(
        "SELECT count(*) FROM observations o JOIN actions a ON a.id = o.action_id "
        "WHERE a.idempotency_key = %s",
        (key,),
    ).fetchone()[0]
    assert action_count == 1
    assert observation_count == 1
    replay = Kernel(
        Model(),
        StrictDecisionDecoder(),
        Policy(),
        SharedExecutor(),
        Verifier(),
        store=PostgresObjectiveStore(check),
    ).run(IntentFrame(principal=principal, utterance="later retry", correlation_id=correlation_id))
    assert replay.state is ObjectiveState.COMPLETED
    check.close()
