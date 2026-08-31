"""Small semantic pipeline with authority and completion outside the model."""

from __future__ import annotations

from uuid import UUID, uuid4

from .audit import AuditLog
from .contracts import (
    ActionCard,
    AuthorizationRequest,
    DecisionKind,
    ExecutionRequest,
    IntentFrame,
    ModelRequest,
    Objective,
    ObjectiveState,
    Result,
    WorkingSet,
)
from .decoding import InvalidDecision
from .fastpath import DeterministicFastPath, NoopFastPath
from .ports import DecisionDecoder, Executor, ModelRouter, Policy, Verifier
from .store import InMemoryObjectiveStore, ObjectiveStore


class Kernel:
    def __init__(
        self,
        model: ModelRouter,
        decoder: DecisionDecoder,
        policy: Policy,
        executor: Executor,
        verifier: Verifier,
        fast_path: DeterministicFastPath | None = None,
        store: ObjectiveStore | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self.model = model
        self.decoder = decoder
        self.policy = policy
        self.executor = executor
        self.verifier = verifier
        self.fast_path = fast_path or NoopFastPath()
        self.store: ObjectiveStore = store or InMemoryObjectiveStore()
        self.audit = audit or AuditLog()
        self.objectives: dict[UUID, Objective] = {}
        self._executed: set[str] = set()
        self._results: dict[str, Result] = {}

    def run(self, intent: IntentFrame, cards: tuple[ActionCard, ...] = ()) -> Result:
        if len(cards) > 5:
            raise ValueError("model-facing action cards must be bounded to five")
        fast_result = self.fast_path.resolve(intent)
        if fast_result is not None:
            return fast_result
        objective = Objective(intent=intent, correlation_id=intent.correlation_id)
        self.objectives[objective.id] = objective
        self.store.save_objective(objective)
        self.audit.append(
            "objective.created",
            intent.principal.id,
            {"objective_id": str(objective.id), "correlation_id": str(intent.correlation_id)},
            objective_id=objective.id,
        )
        try:
            decision = self.decoder.decode(
                self.model.decide(
                    ModelRequest(
                        working_set=WorkingSet(intent=intent),
                        action_cards=cards,
                    )
                ),
                cards,
            )
        except InvalidDecision as exc:
            result = Result(
                objective_id=objective.id,
                state=ObjectiveState.BLOCKED,
                message=f"Invalid model decision: {exc}",
                correlation_id=intent.correlation_id,
            )
            self.store.save_result(f"decision:{objective.id}", result)
            self.audit.append(
                "decision.rejected",
                intent.principal.id,
                {"reason": str(exc)},
                objective_id=objective.id,
            )
            return result
        if decision.kind is not DecisionKind.ACTION:
            state = {
                DecisionKind.ANSWER: ObjectiveState.COMPLETED,
                DecisionKind.NEED_CONTEXT: ObjectiveState.BLOCKED,
                DecisionKind.CLARIFY: ObjectiveState.BLOCKED,
                DecisionKind.BLOCKED: ObjectiveState.BLOCKED,
            }[decision.kind]
            self.audit.append(
                "decision.terminal",
                intent.principal.id,
                {"kind": decision.kind.value, "state": state.value},
                objective_id=objective.id,
            )
            return Result(
                objective_id=objective.id,
                state=state,
                message=decision.answer or decision.reason or decision.kind,
                correlation_id=intent.correlation_id,
            )
        if decision.action is None:
            return Result(
                objective_id=objective.id,
                state=ObjectiveState.BLOCKED,
                message="ACTION decision did not contain an action",
                correlation_id=intent.correlation_id,
            )
        objective = objective.model_copy(
            update={"action": decision.action, "state": ObjectiveState.VALIDATED}
        )
        self.objectives[objective.id] = objective
        policy = self.policy.authorize(
            AuthorizationRequest(
                principal=intent.principal, objective_id=objective.id, action=decision.action
            )
        )
        if not policy.allowed:
            self.objectives[objective.id] = objective.model_copy(
                update={"state": ObjectiveState.BLOCKED}
            )
            self.audit.append(
                "policy.denied",
                intent.principal.id,
                {"capability": decision.action.capability, "reason": policy.reason},
                objective_id=objective.id,
            )
            return Result(
                objective_id=objective.id,
                state=ObjectiveState.BLOCKED,
                message=policy.reason,
                correlation_id=intent.correlation_id,
            )
        if policy.approval_required:
            self.audit.append(
                "approval.required",
                intent.principal.id,
                {"capability": decision.action.capability, "reason": policy.reason},
                objective_id=objective.id,
            )
            return Result(
                objective_id=objective.id,
                state=ObjectiveState.APPROVAL_REQUIRED,
                message=policy.reason,
                correlation_id=intent.correlation_id,
            )
        # Correlation is stable across a recovered/replayed turn; objective IDs
        # are intentionally not used as the idempotency key.
        key = f"{intent.correlation_id}:{decision.action.action_id}"
        prior = self._results.get(key) or self.store.get_result(key)
        if prior is not None:
            self._results[key] = prior
            self._executed.add(key)
            return prior
        if key in self._executed:
            self.audit.append(
                "action.replay_suppressed",
                intent.principal.id,
                {"capability": decision.action.capability},
                objective_id=objective.id,
            )
            return Result(
                objective_id=objective.id,
                state=ObjectiveState.OBSERVED,
                message="Execution already recorded; verification is required",
                correlation_id=intent.correlation_id,
            )
        self._executed.add(key)
        execution_id = uuid4()
        observation = self.executor.execute(
            ExecutionRequest(
                objective_id=objective.id,
                action_id=execution_id,
                action=decision.action,
                idempotency_key=key,
            )
        )
        self.objectives[objective.id] = objective.model_copy(
            update={"state": ObjectiveState.OBSERVED}
        )
        self.audit.append(
            "action.observed",
            intent.principal.id,
            {
                "capability": decision.action.capability,
                "command_succeeded": observation.command_succeeded,
            },
            objective_id=objective.id,
            action_id=execution_id,
        )
        if decision.action.verification is None:
            result = Result(
                objective_id=objective.id,
                state=ObjectiveState.FAILED,
                message="Action executed but has no verification contract",
                evidence=observation.evidence,
                correlation_id=intent.correlation_id,
            )
            self._results[key] = result
            self.store.save_result(key, result)
            self.audit.append(
                "result.failed",
                intent.principal.id,
                {"state": ObjectiveState.FAILED.value, "verified": False},
                objective_id=objective.id,
                action_id=execution_id,
            )
            return result
        verified = self.verifier.verify(observation, decision.action.verification)
        state = ObjectiveState.COMPLETED if verified.verified else ObjectiveState.FAILED
        self.objectives[objective.id] = objective.model_copy(update={"state": state})
        result = Result(
            objective_id=objective.id,
            state=state,
            message=verified.reason,
            evidence=verified.evidence,
            correlation_id=intent.correlation_id,
        )
        self._results[key] = result
        self.store.save_result(key, result)
        self.audit.append(
            "result.verified" if verified.verified else "result.failed",
            intent.principal.id,
            {"state": state.value, "verified": verified.verified},
            objective_id=objective.id,
            action_id=execution_id,
        )
        return result
