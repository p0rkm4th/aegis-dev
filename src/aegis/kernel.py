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
    ObjectiveRequirement,
    ObjectiveSpec,
    ObjectiveSpecProposal,
    ObjectiveState,
    Observation,
    ProposedPlan,
    Result,
    ValidatedPlan,
    VerificationResult,
    WorkingSet,
)
from .decoding import InvalidDecision, StrictDecisionDecoder
from .fastpath import DeterministicFastPath, NoopFastPath
from .planning import (
    materialize_proposed_plan,
    materialize_validated_plan,
    objective_requirements_satisfied,
)
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
        self._results: dict[str, Result] = {}

    def run(
        self,
        intent: IntentFrame,
        cards: tuple[ActionCard, ...] = (),
        context: Context | None = None,
    ) -> Result:
        if len(cards) > 10:
            raise ValueError("model-facing action cards must be bounded to ten")
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
                return prior
        proposed_request = ExecutionRequest(
            objective_id=objective.id,
            action_id=uuid4(),
            action=decision.action,
            idempotency_key=key,
        )
        claim = self.store.claim_action(proposed_request)
        execution_request = claim.request
        canonical_objective = self.store.get_objective(execution_request.objective_id)
        if canonical_objective is not None:
            objective = canonical_objective
            self.objectives[objective.id] = objective
        action = execution_request.action
        if not claim.acquired:
            canonical_result = self._results.get(key) or self.store.get_result(key)
            canonical_observation = self.store.get_observation(key)
            if canonical_result is not None and not (
                canonical_result.state is ObjectiveState.FAILED
                and canonical_observation is not None
                and canonical_observation.command_succeeded
            ):
                self._results[key] = canonical_result
                return canonical_result
            if canonical_observation is None:
                return Result(
                    objective_id=execution_request.objective_id,
                    state=ObjectiveState.EXECUTING,
                    message=(
                        "Execution has been claimed; canonical observation is not yet available"
                    ),
                    evidence={"execution_claimed": True, "observation": "pending"},
                    correlation_id=intent.correlation_id,
                    retryable=False,
                )
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
                "capability": action.capability,
                "command_succeeded": observation.command_succeeded,
            },
            objective_id=objective.id,
            action_id=execution_id,
        )
        if action.verification is None:
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
                verified = self.verifier.verify(observation, action.verification)
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
        objective_spec: ObjectiveSpec | None = None,
        validated_plan: ValidatedPlan | None = None,
        objective_id: UUID | None = None,
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
            if objective.objective_spec is not None and (
                objective_spec is None or validated_plan is None
            ):
                return Result(
                    objective_id=objective.id,
                    state=ObjectiveState.BLOCKED,
                    message="objective requires its persisted validated plan for recovery",
                    correlation_id=intent.correlation_id,
                )
            if (
                objective.steps != actions
                or objective_spec is not None
                and objective.objective_spec != objective_spec
                or validated_plan is not None
                and objective.validated_plan != validated_plan
            ):
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
                id=objective_id or uuid4(),
                intent=intent,
                correlation_id=intent.correlation_id,
                steps=actions,
                objective_spec=objective_spec,
                validated_plan=validated_plan,
            )
            self.store.save_objective(objective)
        step_results: list[dict[str, object]] = []
        satisfied_requirement_ids: set[UUID] = set()

        def aggregate_evidence() -> dict[str, object]:
            evidence: dict[str, object] = {"steps": step_results}
            if objective_spec is not None and validated_plan is not None:
                states = {
                    str(step["requirement_id"]): step["state"]
                    for step in step_results
                    if "requirement_id" in step and "state" in step
                }
                evidence["objective_requirements"] = [
                    {
                        "requirement_id": str(requirement.requirement_id),
                        "state": states.get(str(requirement.requirement_id), "pending"),
                    }
                    for requirement in objective_spec.requirements
                ]
            return evidence

        completed_requirement_ids: set[UUID] = set()
        had_failure = False
        for index, action in enumerate(actions):
            if validated_plan is not None:
                validated_step = validated_plan.steps[index]
                dependency_states: list[ObjectiveState] = []
                for dependency_id in validated_step.depends_on:
                    dependency = next(
                        (
                            candidate
                            for candidate in validated_plan.steps
                            if candidate.step_id == dependency_id
                        ),
                        None,
                    )
                    if dependency is None:
                        dependency_states.append(ObjectiveState.BLOCKED)
                        continue
                    dependency_index = validated_plan.steps.index(dependency)
                    dependency_correlation = uuid5(
                        intent.correlation_id,
                        f"aegis-plan-step:{dependency_index}:{dependency.action.action_id}",
                    )
                    dependency_key = f"{dependency_correlation}:{dependency.action.action_id}"
                    dependency_result = self.store.get_result(dependency_key)
                    if dependency_result is not None:
                        dependency_states.append(dependency_result.state)
                    elif dependency.requirement_id in completed_requirement_ids:
                        dependency_states.append(ObjectiveState.COMPLETED)
                    else:
                        dependency_states.append(ObjectiveState.BLOCKED)
                if any(state is not ObjectiveState.COMPLETED for state in dependency_states):
                    step_results.append(
                        {
                            "index": index,
                            "action_id": action.action_id,
                            "state": ObjectiveState.BLOCKED.value,
                            "objective_id": str(objective.id),
                            "correlation_id": str(intent.correlation_id),
                            "message": "A verified prerequisite is not available",
                            "requirement_id": str(validated_step.requirement_id),
                        }
                    )
                    continue
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
                    **(
                        {"requirement_id": str(validated_plan.steps[index].requirement_id)}
                        if validated_plan is not None
                        else {}
                    ),
                }
            )
            if result.state is ObjectiveState.COMPLETED and validated_plan is not None:
                completed_requirement_ids.add(validated_plan.steps[index].requirement_id)
                satisfied_requirement_ids.add(validated_plan.steps[index].requirement_id)
            if result.state is ObjectiveState.FAILED:
                had_failure = True
            if result.state is not ObjectiveState.COMPLETED and validated_plan is None:
                objective = objective.model_copy(update={"state": result.state})
                self.store.save_objective(objective)
                aggregate = Result(
                    objective_id=objective.id,
                    state=result.state,
                    message=f"Plan stopped at step {index + 1} of {len(actions)}: {result.message}",
                    evidence=aggregate_evidence(),
                    correlation_id=intent.correlation_id,
                    retryable=result.retryable,
                )
                self.store.save_result(plan_key, aggregate)
                return aggregate
        if objective_spec is not None and not objective_requirements_satisfied(
            objective_spec, satisfied_requirement_ids
        ):
            incomplete_state = ObjectiveState.FAILED if had_failure else ObjectiveState.BLOCKED
            objective = objective.model_copy(update={"state": incomplete_state})
            self.store.save_objective(objective)
            return Result(
                objective_id=objective.id,
                state=incomplete_state,
                message=(
                    "Objective remains incomplete because a requested requirement is unsatisfied"
                ),
                evidence=aggregate_evidence(),
                correlation_id=intent.correlation_id,
            )
        objective = objective.model_copy(update={"state": ObjectiveState.COMPLETED})
        self.store.save_objective(objective)
        aggregate = Result(
            objective_id=objective.id,
            state=ObjectiveState.COMPLETED,
            message=f"Completed all {len(actions)} plan steps",
            evidence=aggregate_evidence(),
            correlation_id=intent.correlation_id,
        )
        self.store.save_result(plan_key, aggregate)
        return aggregate

    def run_proposed_plan(
        self,
        intent: IntentFrame,
        proposal: ProposedPlan,
        cards: tuple[ActionCard, ...],
        context: Context | None = None,
        objective_spec: ObjectiveSpec | ObjectiveSpecProposal | None = None,
    ) -> Result:
        """Validate a proposal against candidates, then reuse durable sequence execution."""

        if objective_spec is not None:
            objective_id = uuid5(intent.correlation_id, "objective-completeness")
            if isinstance(objective_spec, ObjectiveSpecProposal):
                objective_spec = ObjectiveSpec(
                    requirements=tuple(
                        ObjectiveRequirement(
                            requirement_id=uuid5(objective_id, f"objective-requirement:{index}"),
                            action_ref=requirement.action_ref,
                            arguments=requirement.arguments,
                        )
                        for index, requirement in enumerate(objective_spec.requirements)
                    )
                )
            validated = materialize_validated_plan(objective_id, objective_spec, proposal, cards)
            return self.run_sequence(
                intent,
                tuple(step.action for step in validated.steps),
                context=context,
                objective_spec=objective_spec,
                validated_plan=validated,
                objective_id=objective_id,
            )
        actions = materialize_proposed_plan(proposal, cards)
        return self.run_sequence(intent, actions, context=context)
