"""Small semantic pipeline with authority and completion outside the model."""

from __future__ import annotations

from uuid import UUID, uuid4, uuid5

from .audit import AuditLog
from .contracts import (
    ActionCard,
    ActionSpec,
    AuthorizationRequest,
    Context,
    Decision,
    DecisionKind,
    ExecutionRequest,
    IntentFrame,
    ModelRequest,
    ModelResponse,
    Objective,
    ObjectiveState,
    Observation,
    Result,
    VerificationResult,
    WorkingSet,
)
from .decoding import InvalidDecision, StrictDecisionDecoder
from .fastpath import DeterministicFastPath, NoopFastPath
from .ports import DecisionDecoder, Executor, ModelRouter, Policy, Verifier
from .store import InMemoryObjectiveStore, ObjectiveStore


class _FixedActionModel:
    """Turn a preselected plan step into a proposal for the normal decoder."""

    def __init__(self, action: ActionSpec) -> None:
        self.action = action

    def decide(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            raw={"kind": DecisionKind.ACTION.value, "action": self.action.model_dump(mode="json")}
        )


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

    def run(
        self,
        intent: IntentFrame,
        cards: tuple[ActionCard, ...] = (),
        context: Context | None = None,
    ) -> Result:
        if len(cards) > 5:
            raise ValueError("model-facing action cards must be bounded to five")
        fast_result = self.fast_path.resolve(intent)
        if fast_result is not None:
            return fast_result
        recovered = self.store.get_objective_by_correlation(intent.correlation_id, intent.principal)
        if recovered is not None and recovered.action is not None:
            objective = recovered
            self.objectives[objective.id] = objective
            decision = Decision(kind=DecisionKind.ACTION, action=recovered.action)
        else:
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
                            working_set=WorkingSet(intent=intent, context=context or Context()),
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
                    retryable=True,
                )
                self.store.save_result(f"decision:{intent.correlation_id}", result)
                self.audit.append(
                    "decision.rejected",
                    intent.principal.id,
                    {"reason": str(exc)},
                    objective_id=objective.id,
                )
                return result
            except Exception as exc:
                objective = objective.model_copy(update={"state": ObjectiveState.FAILED})
                self.objectives[objective.id] = objective
                self.store.save_objective(objective)
                result = Result(
                    objective_id=objective.id,
                    state=ObjectiveState.FAILED,
                    message="Model unavailable; request can be retried",
                    evidence={"model": "unavailable", "error_type": type(exc).__name__},
                    correlation_id=intent.correlation_id,
                    retryable=True,
                )
                self.store.save_result(f"decision:{intent.correlation_id}", result)
                self.audit.append(
                    "model.failed",
                    intent.principal.id,
                    {"error_type": type(exc).__name__},
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
        self.store.save_objective(objective)
        policy = self.policy.authorize(
            AuthorizationRequest(
                principal=intent.principal, objective_id=objective.id, action=decision.action
            )
        )
        if not policy.allowed:
            self.objectives[objective.id] = objective.model_copy(
                update={"state": ObjectiveState.BLOCKED}
            )
            self.store.save_objective(self.objectives[objective.id])
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
            self.objectives[objective.id] = objective.model_copy(
                update={"state": ObjectiveState.APPROVAL_REQUIRED}
            )
            self.store.save_objective(self.objectives[objective.id])
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
            prior_observation = self.store.get_observation(key)
            if not (
                prior.state is ObjectiveState.FAILED
                and prior_observation is not None
                and prior_observation.command_succeeded
            ):
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
        existing_action = self.store.get_action(key)
        execution_request = existing_action or ExecutionRequest(
            objective_id=objective.id,
            action_id=uuid4(),
            action=decision.action,
            idempotency_key=key,
        )
        self.store.save_action(execution_request, ObjectiveState.EXECUTING)
        self._executed.add(key)
        observation = self.store.get_observation(key)
        if observation is None:
            try:
                observation = self.executor.execute(execution_request)
            except Exception as exc:
                # An adapter may have crossed an external boundary before
                # failing. Persist an ambiguous outcome so recovery never
                # mistakes an exception for permission to replay blindly.
                observation = Observation(
                    execution_id=execution_request.action_id,
                    evidence={
                        "executor": "unavailable",
                        "outcome": "unknown",
                        "error_type": type(exc).__name__,
                        "idempotency_key": key,
                    },
                    command_succeeded=False,
                )
            self.store.save_observation(key, observation)
        self.store.update_action_state(key, ObjectiveState.OBSERVED)
        execution_id = execution_request.action_id
        self.objectives[objective.id] = objective.model_copy(
            update={"state": ObjectiveState.OBSERVED}
        )
        self.store.save_objective(self.objectives[objective.id])
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
            self.store.update_action_state(key, ObjectiveState.FAILED)
            self.audit.append(
                "result.failed",
                intent.principal.id,
                {"state": ObjectiveState.FAILED.value, "verified": False},
                objective_id=objective.id,
                action_id=execution_id,
            )
            return result
        if not observation.command_succeeded:
            outcome_unknown = observation.evidence.get("outcome") == "unknown"
            verified = VerificationResult(
                verified=False,
                evidence=observation.evidence,
                reason=(
                    "execution outcome is unknown; external state must be checked before retrying"
                    if outcome_unknown
                    else "execution failed"
                ),
            )
        else:
            try:
                verified = self.verifier.verify(observation, decision.action.verification)
            except Exception as exc:
                verified = VerificationResult(
                    verified=False,
                    evidence={
                        **observation.evidence,
                        "verification": "unavailable",
                        "error_type": type(exc).__name__,
                    },
                    reason=(
                        "verification unavailable; external state must be checked before retrying"
                    ),
                )
        state = ObjectiveState.COMPLETED if verified.verified else ObjectiveState.FAILED
        self.objectives[objective.id] = objective.model_copy(update={"state": state})
        self.store.save_objective(self.objectives[objective.id])
        result = Result(
            objective_id=objective.id,
            state=state,
            message=verified.reason,
            evidence=verified.evidence,
            correlation_id=intent.correlation_id,
            retryable=(state is not ObjectiveState.COMPLETED and observation.command_succeeded),
        )
        self._results[key] = result
        self.store.save_result(key, result)
        self.store.update_action_state(key, state)
        self.audit.append(
            "result.verified" if verified.verified else "result.failed",
            intent.principal.id,
            {"state": state.value, "verified": verified.verified},
            objective_id=objective.id,
            action_id=execution_id,
        )
        return result

    def run_sequence(
        self,
        intent: IntentFrame,
        actions: tuple[ActionSpec, ...],
        context: Context | None = None,
    ) -> Result:
        """Execute a bounded durable sequence through the ordinary Core path.

        The sequence is a persisted proposal, not a second authority layer.
        Each step gets its own correlation/idempotency key and re-enters Kernel
        so policy, execution, observation, and verification remain independent.
        """
        if not actions or len(actions) > 5:
            raise ValueError("plans must contain between one and five actions")
        plan_key = f"plan:{intent.correlation_id}"
        objective = self.store.get_objective_by_correlation(intent.correlation_id, intent.principal)
        prior = self.store.get_result(plan_key)
        if objective is None and prior is not None:
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message="plan is unavailable for this identity",
                correlation_id=intent.correlation_id,
            )
        if objective is not None:
            if objective.steps != actions:
                return Result(
                    objective_id=objective.id,
                    state=ObjectiveState.BLOCKED,
                    message="plan correlation is already bound to a different action sequence",
                    correlation_id=intent.correlation_id,
                )
            if (
                prior is not None
                and prior.state is ObjectiveState.COMPLETED
                and not prior.retryable
            ):
                return prior
        else:
            if prior is not None:
                return Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.BLOCKED,
                    message="plan is unavailable for this identity",
                    correlation_id=intent.correlation_id,
                )
            objective = Objective(
                intent=intent,
                correlation_id=intent.correlation_id,
                steps=actions,
            )
            self.store.save_objective(objective)
        step_results: list[dict[str, object]] = []
        for index, action in enumerate(actions):
            step_correlation = uuid5(
                intent.correlation_id, f"aegis-plan-step:{index}:{action.action_id}"
            )
            step_intent = intent.model_copy(update={"correlation_id": step_correlation})
            card = ActionCard(
                action=action,
                summary=f"Plan step {index + 1}: {action.action_id}",
                relevance=1,
            )
            step_kernel = Kernel(
                _FixedActionModel(action),
                StrictDecisionDecoder(),
                self.policy,
                self.executor,
                self.verifier,
                store=self.store,
                audit=self.audit,
            )
            result = step_kernel.run(step_intent, (card,), context=context)
            step_results.append(
                {
                    "index": index,
                    "action_id": action.action_id,
                    "state": result.state.value,
                    "objective_id": str(result.objective_id),
                    "correlation_id": str(result.correlation_id),
                    "message": result.message,
                    "evidence": result.evidence,
                }
            )
            if result.state is not ObjectiveState.COMPLETED:
                objective = objective.model_copy(update={"state": result.state})
                self.store.save_objective(objective)
                aggregate = Result(
                    objective_id=objective.id,
                    state=result.state,
                    message=f"Plan stopped at step {index + 1} of {len(actions)}: {result.message}",
                    evidence={"steps": step_results},
                    correlation_id=intent.correlation_id,
                    retryable=result.retryable,
                )
                self.store.save_result(plan_key, aggregate)
                return aggregate
        objective = objective.model_copy(update={"state": ObjectiveState.COMPLETED})
        self.store.save_objective(objective)
        aggregate = Result(
            objective_id=objective.id,
            state=ObjectiveState.COMPLETED,
            message=f"Completed all {len(actions)} plan steps",
            evidence={"steps": step_results},
            correlation_id=intent.correlation_id,
        )
        self.store.save_result(plan_key, aggregate)
        return aggregate
