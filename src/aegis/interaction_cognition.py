"""Bounded model cognition coordination for the interaction boundary."""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, cast
from uuid import uuid4

from .contracts import (
    ActionCard,
    Context,
    Decision,
    DecisionKind,
    IntentFrame,
    ModelRequest,
    ObjectiveFidelityVerdict,
    ObjectiveRequirementProposal,
    ObjectiveSpecProposal,
    ObjectiveState,
    ProposalFailureEvidence,
    ProposalFailureKind,
    ProposedPlan,
    ProposedPlanStep,
    Result,
    WorkingSet,
)
from .decoding import InvalidDecision, StrictDecisionDecoder
from .interaction_recovery import (
    ValidationResult,
    bounded_proposal_repair,
    proposal_failure_evidence,
    proposal_failure_fingerprint,
    proposal_repair_event_record,
    recover_invalid_model_decision,
    repair_invalid_decision_once,
    repair_invalid_decision_once_with_evidence,
)
from .objective_fidelity import (
    RequestedEffectProposal,
    compare_objective_proposals,
    effects_to_proposal,
    fidelity_message,
    materialize_requested_effects,
    validate_structural_coverage,
)
from .utterance import is_question_request


def _scope_plan_by_capability(
    provider: Any,
    decoder: StrictDecisionDecoder,
    intent: IntentFrame,
    cards: tuple[ActionCard, ...],
    context: Context,
    proposed: Decision,
) -> Decision:
    """Reinterpret a multi-action objective one authorized capability at a time.

    A single overloaded plan response can omit an independent operation.  This
    bounded pass asks the same provider to interpret each already-retrieved
    write card independently, then preserves only candidate-bound ACTIONs.  It
    is decomposition, not a completeness reviewer: no result is trusted until
    normal plan validation, policy, execution ownership, observation, and
    verification run below this boundary.
    """

    if proposed.kind is not DecisionKind.PLAN or len(cards) < 3:
        return proposed
    # A normal two-operation plan commonly contains one conjunction. Extra
    # capability-scoped calls are justified only when multiple structural
    # conjunction boundaries suggest a third clause.
    if len(re.findall(r"\b(?:and|then|plus)\b", intent.utterance.casefold())) < 2:
        return proposed

    def blocked() -> Decision:
        return Decision(
            kind=DecisionKind.CLARIFY,
            clarification=(
                "I found several independent changes in that request, but I could not "
                "safely account for each one. Please separate the changes or clarify them."
            ),
            semantic_mode="CLARIFY",
        )

    scoped: dict[str, Decision] = {}
    proposed_arguments = (
        {step.action_ref: dict(step.arguments) for step in proposed.plan.steps}
        if proposed.plan is not None
        else {}
    )
    try:
        for card in cards:
            if not any(
                permission.endswith(".write") for permission in card.action.required_permissions
            ):
                continue
            request = ModelRequest(
                working_set=WorkingSet(intent=intent, context=context),
                action_cards=(card,),
                allow_argument_proposals=True,
                capability_scoped=True,
            )
            decision = decoder.decode(
                provider.decide(request), (card,), allow_argument_proposals=True
            )
            action = decision.action
            if decision.kind is DecisionKind.ACTION and action is not None:
                scoped[card.action.action_id] = decision
    except Exception:
        return blocked()
    if len(scoped) < 2:
        return blocked()
    original_order = [step.action_ref for step in proposed.plan.steps] if proposed.plan else []
    ordered_ids = [action_id for action_id in original_order if action_id in scoped] + [
        action_id for action_id in scoped if action_id not in original_order
    ]
    steps_list: list[ProposedPlanStep] = []
    for index, action_id in enumerate(ordered_ids):
        scoped_decision = scoped.get(action_id)
        if scoped_decision is None or scoped_decision.action is None:
            continue
        arguments = dict(scoped_decision.action.arguments)
        if action_id in proposed_arguments:
            # A capability-scoped pass must not add optional arguments that the
            # original bounded proposal did not assign to that operation. This
            # prevents details such as a date intended for one step from
            # leaking into sibling steps; Core still performs final schema and
            # objective validation below this boundary.
            arguments = {
                key: value
                for key, value in arguments.items()
                if key in proposed_arguments[action_id]
            }
        steps_list.append(
            ProposedPlanStep(
                action_ref=action_id,
                arguments=arguments,
                depends_on=((index - 1,) if index else ()),
            )
        )
    steps = tuple(steps_list)
    if len(steps) < 2:
        return proposed
    return Decision(
        kind=DecisionKind.PLAN,
        plan=ProposedPlan(steps=steps),
        objective_spec=ObjectiveSpecProposal(
            requirements=tuple(
                ObjectiveRequirementProposal(action_ref=step.action_ref, arguments=step.arguments)
                for step in steps
            )
        ),
        semantic_mode="ACTION",
    )


def _structural_write_failure(
    dependencies: Any,
    intent: IntentFrame,
    decision: Decision,
    cards: tuple[ActionCard, ...],
) -> str | None:
    """Return a safe clarification reason for an unaccounted-for write."""

    if decision.kind is not DecisionKind.ACTION or decision.action is None:
        return None
    is_write = any(
        permission.endswith(".write")
        for card in cards
        if card.action.action_id == decision.action.action_id
        for permission in card.action.required_permissions
    )
    if not is_write or not hasattr(dependencies, "structural_parser"):
        return None
    structural_parser = getattr(dependencies, "structural_parser", None)
    if structural_parser is None:
        return (
            "I could not independently verify the requested change; please clarify the objective."
        )
    try:
        structural_signal = structural_parser(intent.utterance)
    except Exception as exc:
        raise InvalidDecision("structural coverage unavailable") from exc
    if structural_signal.negation_spans:
        return (
            "I noticed a negated or contrasting change in that request, but I could not "
            "safely determine which change you want. Please clarify it."
        )
    if len(structural_signal.anchors) != 1:
        return (
            "This request contains more than one change, but I could not form a complete "
            "verified objective. Please clarify it."
        )
    return None


def _has_structural_plurality(dependencies: Any, utterance: str) -> bool:
    """Return whether the optional parser found multiple structural anchors."""

    structural_parser = getattr(dependencies, "structural_parser", None)
    if structural_parser is None:
        return False
    try:
        signal = structural_parser(utterance)
    except Exception:
        return False
    return len(signal.anchors) > 1


def _repair_clarification(
    provider: Any,
    intent: IntentFrame,
    context: Context,
    cards: tuple[ActionCard, ...],
    failed_proposal: Decision,
    clarification: str,
    failure_evidence: ProposalFailureEvidence,
    validator: Callable[[Decision], ValidationResult] | None = None,
    *,
    plans_only: bool = False,
) -> Decision:
    """Give fidelity-generated clarification one bounded repair opportunity."""

    decision = failed_proposal

    def repair(
        candidate: Decision, evidence: ProposalFailureEvidence
    ) -> tuple[Decision | None, ProposalFailureEvidence]:
        repair_error = InvalidDecision(evidence.detail or evidence.kind.value)
        repaired = repair_invalid_decision_once(
            provider,
            intent,
            context,
            cards,
            candidate.model_dump(mode="json"),
            repair_error,
            evidence,
        )
        return repaired, evidence if repaired is None else evidence

    def validate(candidate: Decision) -> ValidationResult:
        if plans_only and candidate.kind is not DecisionKind.PLAN:
            return ValidationResult(
                valid=False,
                failure=ProposalFailureEvidence(kind=ProposalFailureKind.CAPABILITY_MISMATCH),
            )
        if validator is not None:
            return validator(candidate)
        return ValidationResult(valid=True)

    result = bounded_proposal_repair(
        decision,
        failure_evidence,
        repair,
        validate,
        validator_stage="proposal_repair",
        max_attempts=2,
    )
    events = getattr(provider, "recovery_events", None)
    if isinstance(events, list):
        events.extend(proposal_repair_event_record(event) for event in result.events)
    return result.proposal if result.proposal is not None else decision


def decide_fallback(
    dependencies: Any, intent: IntentFrame, cards: tuple[ActionCard, ...], context: Context
) -> Decision | Result | None:
    if dependencies.model_provider is None:
        return None
    last_raw: dict[str, Any] | None = None
    focused_raw: dict[str, Any] | None = None
    try:
        provider = dependencies.model_provider()
        if cards:
            classification_values = {
                key: value for key, value in context.values.items() if key != "canonical_facts"
            }
            classification_request = ModelRequest(
                working_set=WorkingSet(
                    intent=intent,
                    context=context.model_copy(update={"values": classification_values}),
                ),
                # Classification may choose only a semantic mode, but it
                # needs the bounded candidate vocabulary to distinguish a
                # list destination from a request to create or complete.
                # Core still decodes the later action proposal against the
                # same cards and retains all authority gates.
                action_cards=cards,
                classification_only=True,
            )
            classification_response = provider.decide(classification_request)
            last_raw = (
                classification_response.raw
                if isinstance(classification_response.raw, dict)
                else None
            )
            if last_raw is not None and last_raw.get("context_focus") is not None:
                focused_raw = last_raw
            semantic_mode = (
                classification_response.raw.get("semantic_mode")
                if isinstance(classification_response.raw, dict)
                else None
            )
            if (
                dependencies.reuse_classification_action_reference
                and semantic_mode == "ACTION"
                and isinstance(classification_response.raw, dict)
                and len(cards) == 1
            ):
                # Some providers return a bounded action reference even
                # during the mode pass. Reuse it only when it exactly
                # names a supplied ActionCard and its arguments are a
                # declared subset. This prevents a second cognition pass
                # from drifting a clear mutation into a nearby read;
                # grounding, policy, execution, and verification remain
                # below this proposal boundary.
                action_ref = classification_response.raw.get("action_ref")
                action_arguments = classification_response.raw.get("action_arguments", {})
                selected_card = next(
                    (card for card in cards if card.action.action_id == action_ref), None
                )
                if selected_card is not None and isinstance(action_arguments, dict):
                    declared = set(selected_card.argument_keys)
                    if set(action_arguments).issubset(declared):
                        classification_decision = Decision(
                            kind=DecisionKind.ACTION,
                            action=selected_card.action.model_copy(
                                update={"arguments": dict(action_arguments)}
                            ),
                            semantic_mode="ACTION",
                        )
                        structural_failure = _structural_write_failure(
                            dependencies, intent, classification_decision, cards
                        )
                        if structural_failure is not None:
                            return Decision(
                                kind=DecisionKind.CLARIFY,
                                clarification=structural_failure,
                                semantic_mode="CLARIFY",
                            )
                        if dependencies.decision_rewriter is not None:
                            rewritten = dependencies.decision_rewriter(
                                intent, classification_decision, cards, context
                            )
                            if isinstance(rewritten, Result):
                                return rewritten
                            if isinstance(rewritten, Decision):
                                return rewritten
                        return classification_decision
            if semantic_mode in {"GENERATION", "READ"}:
                knowledge_source = (
                    classification_response.raw.get("knowledge_source")
                    if isinstance(classification_response.raw, dict)
                    else None
                )
                research_answer = getattr(dependencies, "research_answer", None)
                if research_answer is not None:
                    source_response = provider.decide(
                        ModelRequest(
                            working_set=WorkingSet(
                                intent=intent,
                                context=context.model_copy(
                                    update={
                                        "values": {
                                            key: value
                                            for key, value in context.values.items()
                                            if key != "canonical_facts"
                                        }
                                    }
                                ),
                            ),
                            action_cards=(),
                            source_selection_only=True,
                        )
                    )
                    source_raw = source_response.raw
                    if isinstance(source_raw, dict):
                        selected_source = source_raw.get("knowledge_source")
                        if selected_source in {
                            "general_model_knowledge",
                            "external_evidence",
                            "mixed_evidence",
                        }:
                            knowledge_source = selected_source
                if knowledge_source in {"external_evidence", "mixed_evidence"}:
                    if research_answer is None:
                        return Result(
                            objective_id=uuid4(),
                            state=ObjectiveState.FAILED,
                            message="I couldn't verify current information right now.",
                            evidence={"source_kind": knowledge_source, "authoritative": False},
                            correlation_id=intent.correlation_id,
                            retryable=True,
                        )
                    researched = cast(
                        Result | None, research_answer(intent, context, knowledge_source)
                    )
                    if researched is not None:
                        return researched
                    return Result(
                        objective_id=uuid4(),
                        state=ObjectiveState.FAILED,
                        message="I couldn't verify current information right now.",
                        evidence={"source_kind": knowledge_source, "authoritative": False},
                        correlation_id=intent.correlation_id,
                        retryable=True,
                    )
                if semantic_mode == "GENERATION":
                    generated_answer = (
                        classification_response.raw.get("answer")
                        if isinstance(classification_response.raw, dict)
                        else None
                    )
                    if isinstance(generated_answer, str) and generated_answer.strip():
                        return Decision(
                            kind=DecisionKind.ANSWER,
                            answer=generated_answer,
                            semantic_mode="GENERATION",
                            knowledge_source="general_model_knowledge",
                        )
                answer_context = context
                if semantic_mode == "GENERATION":
                    answer_context = context.model_copy(
                        update={
                            "values": {
                                key: value
                                for key, value in context.values.items()
                                if key != "canonical_facts"
                            }
                        }
                    )
                answer_request = ModelRequest(
                    working_set=WorkingSet(intent=intent, context=answer_context),
                    action_cards=(),
                )
                answer_response = provider.decide(answer_request)
                last_raw = answer_response.raw if isinstance(answer_response.raw, dict) else None
                if last_raw is not None and last_raw.get("context_focus") is not None:
                    focused_raw = last_raw
                answer = StrictDecisionDecoder().decode(
                    answer_response, (), allow_argument_proposals=False
                )
                if answer.kind is DecisionKind.ANSWER:
                    return answer.model_copy(
                        update={
                            "semantic_mode": semantic_mode,
                            "knowledge_source": knowledge_source or "general_model_knowledge",
                        }
                    )
        structural_plurality = _has_structural_plurality(dependencies, intent.utterance)
        routing_only = (
            any(
                permission.endswith(".write")
                for card in cards
                for permission in card.action.required_permissions
            )
            and not structural_plurality
        )
        routing_context = context
        if routing_only:
            routing_values = {
                key: value for key, value in context.values.items() if key != "canonical_facts"
            }
            routing_context = context.model_copy(update={"values": routing_values})
        request = ModelRequest(
            working_set=WorkingSet(intent=intent, context=routing_context),
            action_cards=cards,
            allow_argument_proposals=True,
            routing_only=routing_only,
            allow_plan_proposals=True,
        )
        decoder = StrictDecisionDecoder()
        decision: Decision | None = None
        response = provider.decide(request)
        last_raw = response.raw if isinstance(response.raw, dict) else None
        if last_raw is not None and last_raw.get("context_focus") is not None:
            focused_raw = last_raw
        try:
            decision = decoder.decode(response, request.action_cards, allow_argument_proposals=True)
        except InvalidDecision as error:
            # A malformed or rejected proposal may receive at most two
            # validation-guided repairs. Every repair is decoded against the
            # original candidate set before this function continues.
            current_raw = last_raw
            current_failure = proposal_failure_evidence(error)
            seen_failures = {proposal_failure_fingerprint(current_failure)}
            for _ in range(2):
                repaired, next_failure, repaired_raw = repair_invalid_decision_once_with_evidence(
                    provider,
                    intent,
                    context,
                    cards,
                    current_raw,
                    error,
                    current_failure,
                )
                if repaired is not None:
                    decision = repaired
                    break
                fingerprint = proposal_failure_fingerprint(next_failure)
                if fingerprint in seen_failures:
                    break
                seen_failures.add(fingerprint)
                current_raw = repaired_raw or current_raw
                current_failure = next_failure
                error = InvalidDecision(next_failure.detail or next_failure.kind.value)
            if decision is None:
                raise error
        if decision is None and routing_only:
            # Providers can apply the routing-only instruction inconsistently.
            # Give one final-pass opportunity with the same bounded cards and
            # full authorized context before reporting a model-boundary
            # failure. Strict decoding remains the acceptance gate.
            final_request = request.model_copy(
                update={
                    "working_set": WorkingSet(intent=intent, context=context),
                    "routing_only": False,
                    "action_cards": cards,
                }
            )
            try:
                decision = decoder.decode(
                    provider.decide(final_request),
                    final_request.action_cards,
                    allow_argument_proposals=True,
                )
            except InvalidDecision as error:
                raise error
        if decision is None:
            raise InvalidDecision("model answer repair did not produce a decision")
        if (
            decision.kind in {DecisionKind.CLARIFY, DecisionKind.NEED_CONTEXT}
            and any(
                permission.endswith(".write")
                for card in cards
                for permission in card.action.required_permissions
            )
            and (context.values.get("referents") or context.values.get("canonical_facts"))
        ):
            # Clarification recovery is deliberately narrow: only a mutation
            # with bounded authorized evidence may get two proposal repairs.
            # A repaired proposal still passes the ordinary structural,
            # objective, authorization, Kernel, and verification boundaries.
            clarification_fingerprint: str | None = None
            repair_error = InvalidDecision(
                decision.clarification or decision.reason or "proposal requires clarification"
            )
            for _ in range(1):
                repaired = repair_invalid_decision_once(
                    provider,
                    intent,
                    context,
                    cards,
                    decision.model_dump(mode="json"),
                    repair_error,
                )
                if repaired is not None and repaired.kind not in {
                    DecisionKind.CLARIFY,
                    DecisionKind.NEED_CONTEXT,
                }:
                    decision = repaired
                    break
                fingerprint = proposal_failure_fingerprint(proposal_failure_evidence(repair_error))
                if fingerprint == clarification_fingerprint:
                    break
                clarification_fingerprint = fingerprint
        if decision.kind is DecisionKind.ANSWER and decision.knowledge_source in {
            "external_evidence",
            "mixed_evidence",
        }:
            research_answer = getattr(dependencies, "research_answer", None)
            if research_answer is None:
                return Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.FAILED,
                    message="I couldn't verify current information right now.",
                    evidence={
                        "source_kind": decision.knowledge_source,
                        "authoritative": False,
                    },
                    correlation_id=intent.correlation_id,
                    retryable=True,
                )
            researched = cast(
                Result | None, research_answer(intent, context, decision.knowledge_source)
            )
            if researched is not None:
                return researched
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.FAILED,
                message="I couldn't verify current information right now.",
                evidence={
                    "source_kind": decision.knowledge_source,
                    "authoritative": False,
                },
                correlation_id=intent.correlation_id,
                retryable=True,
            )
        if decision.kind is DecisionKind.PLAN and decision.plan is not None:
            fidelity_response = provider.decide(
                request.model_copy(
                    update={
                        "objective_interpretation_only": False,
                        "objective_fidelity_only": False,
                        "objective_effect_only": True,
                        "allow_plan_proposals": False,
                        "allow_argument_proposals": False,
                        "objective_spec_proposal": None,
                    }
                )
            )
            effect_raw = fidelity_response.raw if isinstance(fidelity_response.raw, dict) else {}
            raw_effects = effect_raw.get("effects")
            try:
                effects = (
                    tuple(RequestedEffectProposal.model_validate(item) for item in raw_effects)
                    if isinstance(raw_effects, list)
                    else ()
                )
            except Exception:
                effects = ()
            structural_parser = getattr(dependencies, "structural_parser", None)
            if structural_parser is None:
                return Decision(
                    kind=DecisionKind.CLARIFY,
                    clarification=(
                        "I could not independently account for every requested change; "
                        "please clarify the objective."
                    ),
                    semantic_mode="CLARIFY",
                )
            try:
                structural_signal = structural_parser(intent.utterance)
                materialized_effects = materialize_requested_effects(intent.utterance, effects)
            except Exception as exc:
                raise InvalidDecision("objective structural coverage failed") from exc

            def validate_effect_proposal(candidate: dict[str, Any]) -> ValidationResult:
                raw_candidate = candidate.get("effects")
                if not isinstance(raw_candidate, list):
                    return ValidationResult(
                        valid=False,
                        failure=ProposalFailureEvidence(
                            kind=ProposalFailureKind.DECODER_SCHEMA_FAILURE,
                            detail="objective effects must be a list",
                        ),
                    )
                try:
                    candidate_effects = tuple(
                        RequestedEffectProposal.model_validate(item) for item in raw_candidate
                    )
                except Exception:
                    return ValidationResult(
                        valid=False,
                        failure=ProposalFailureEvidence(
                            kind=ProposalFailureKind.DECODER_SCHEMA_FAILURE,
                            detail="objective effect schema is invalid",
                        ),
                    )
                candidate_materialized = materialize_requested_effects(
                    intent.utterance, candidate_effects
                )
                if candidate_materialized is None:
                    return ValidationResult(
                        valid=False,
                        failure=ProposalFailureEvidence(kind=ProposalFailureKind.BAD_SOURCE_SPAN),
                    )
                for effect in candidate_materialized:
                    if effect.polarity == "NEGATED":
                        kind = ProposalFailureKind.NEGATED_EFFECT_ACTIVE
                    elif effect.polarity == "SUPERSEDED":
                        kind = ProposalFailureKind.SUPERSEDED_EFFECT_ACTIVE
                    else:
                        continue
                    return ValidationResult(valid=False, failure=ProposalFailureEvidence(kind=kind))
                if not validate_structural_coverage(
                    intent.utterance, candidate_materialized, structural_signal
                ):
                    return ValidationResult(
                        valid=False,
                        failure=ProposalFailureEvidence(
                            kind=ProposalFailureKind.UNACCOUNTED_STRUCTURAL_ANCHOR
                        ),
                    )
                return ValidationResult(valid=True)

            effect_failure = validate_effect_proposal(effect_raw).failure
            if effect_failure is not None:

                def repair_effects(
                    current: dict[str, Any], evidence: ProposalFailureEvidence
                ) -> tuple[dict[str, Any] | None, ProposalFailureEvidence]:
                    response = provider.decide(
                        request.model_copy(
                            update={
                                "objective_effect_only": True,
                                "proposal_repair_only": True,
                                "proposal_failure": evidence,
                                "current_proposal": current,
                                "allow_plan_proposals": False,
                                "allow_argument_proposals": False,
                            }
                        )
                    )
                    if not isinstance(response.raw, dict):
                        return None, ProposalFailureEvidence(
                            kind=ProposalFailureKind.DECODER_SCHEMA_FAILURE,
                            detail="objective effect repair must be an object",
                        )
                    result = validate_effect_proposal(response.raw)
                    return response.raw, result.failure or evidence

                repaired_effects = bounded_proposal_repair(
                    effect_raw,
                    effect_failure,
                    repair_effects,
                    validate_effect_proposal,
                    validator_stage="requested_effect_structural_coverage",
                    max_attempts=2,
                )
                recovery_events = getattr(provider, "recovery_events", None)
                if isinstance(recovery_events, list):
                    recovery_events.extend(
                        proposal_repair_event_record(event) for event in repaired_effects.events
                    )
                if repaired_effects.proposal is None:
                    return Decision(
                        kind=DecisionKind.CLARIFY,
                        clarification=(
                            "I could not independently account for every requested change; "
                            "please clarify the objective."
                        ),
                        semantic_mode="CLARIFY",
                    )
                repaired_raw = repaired_effects.proposal.get("effects")
                if not isinstance(repaired_raw, list):
                    raise InvalidDecision("objective effect repair returned no effects")
                effects = tuple(
                    RequestedEffectProposal.model_validate(item) for item in repaired_raw
                )
                materialized_effects = materialize_requested_effects(intent.utterance, effects)
                if materialized_effects is None:
                    raise InvalidDecision("objective effect repair was not grounded")
            independent_spec = effects_to_proposal(intent.utterance, effects)
            if independent_spec is None:
                return Decision(
                    kind=DecisionKind.CLARIFY,
                    clarification=(
                        "I could not establish the complete requested objective safely; "
                        "please clarify the changes."
                    ),
                    semantic_mode="CLARIFY",
                )
            if decision.objective_spec is None:
                return Decision(
                    kind=DecisionKind.CLARIFY,
                    clarification=(
                        "I could not establish the complete requested objective safely; "
                        "please clarify the changes."
                    ),
                    semantic_mode="CLARIFY",
                )
            assert decision.objective_spec is not None
            verdict = compare_objective_proposals(decision.objective_spec, independent_spec)
            if verdict is not ObjectiveFidelityVerdict.COMPLETE:

                def validate_fidelity(candidate: Decision) -> ValidationResult:
                    if candidate.kind is not DecisionKind.PLAN or candidate.objective_spec is None:
                        return ValidationResult(
                            valid=False,
                            failure=ProposalFailureEvidence(
                                kind=ProposalFailureKind.CAPABILITY_MISMATCH
                            ),
                        )
                    next_verdict = compare_objective_proposals(
                        candidate.objective_spec, independent_spec
                    )
                    if next_verdict is ObjectiveFidelityVerdict.COMPLETE:
                        return ValidationResult(valid=True)
                    return ValidationResult(
                        valid=False,
                        failure=ProposalFailureEvidence(
                            kind=(
                                ProposalFailureKind.MISSING_EFFECT
                                if next_verdict is ObjectiveFidelityVerdict.MISSING_REQUIREMENT
                                else ProposalFailureKind.EXTRA_EFFECT
                                if next_verdict is ObjectiveFidelityVerdict.EXTRA_REQUIREMENT
                                else ProposalFailureKind.CANONICAL_CONTRADICTION
                            )
                        ),
                    )

                decision = _repair_clarification(
                    provider,
                    intent,
                    context,
                    cards,
                    decision,
                    fidelity_message(verdict),
                    ProposalFailureEvidence(
                        kind=(
                            ProposalFailureKind.MISSING_EFFECT
                            if verdict is ObjectiveFidelityVerdict.MISSING_REQUIREMENT
                            else ProposalFailureKind.EXTRA_EFFECT
                            if verdict is ObjectiveFidelityVerdict.EXTRA_REQUIREMENT
                            else ProposalFailureKind.CANONICAL_CONTRADICTION
                        )
                    ),
                    validator=validate_fidelity,
                    plans_only=True,
                )
                if decision.kind is DecisionKind.CLARIFY:
                    return decision
            decision = decision.model_copy(update={"objective_spec": decision.objective_spec})
        if routing_only and decision.kind is DecisionKind.ANSWER:
            reconsidered = decoder.decode(
                provider.decide(request), request.action_cards, allow_argument_proposals=True
            )
            if reconsidered.kind is not decision.kind:
                return Decision(
                    kind=DecisionKind.CLARIFY,
                    clarification=(
                        "I am not confident whether you want to change something or "
                        "ask about it. Please clarify the intended action."
                    ),
                )
            if reconsidered.kind is DecisionKind.ANSWER:
                decision = reconsidered
            else:
                decision = reconsidered
        if routing_only and decision.kind is DecisionKind.ANSWER:
            request = request.model_copy(
                update={
                    "working_set": WorkingSet(intent=intent, context=context),
                    "routing_only": False,
                }
            )
            decision = decoder.decode(
                provider.decide(request), request.action_cards, allow_argument_proposals=True
            )
        if dependencies.decision_rewriter is not None:
            try:
                inspect.signature(dependencies.decision_rewriter).bind(
                    intent, decision, cards, context
                )
            except (TypeError, ValueError):
                rewritten = dependencies.decision_rewriter(intent, decision, cards)
            else:
                rewritten = dependencies.decision_rewriter(intent, decision, cards, context)
            if isinstance(rewritten, Result):
                return rewritten
            if rewritten is not None:
                decision = rewritten
        if decision.kind is DecisionKind.ACTION and decision.action is not None:
            structural_failure = _structural_write_failure(dependencies, intent, decision, cards)
            if structural_failure is not None:
                return Decision(
                    kind=DecisionKind.CLARIFY,
                    clarification=structural_failure,
                    semantic_mode="CLARIFY",
                )
        if (
            decision.kind is DecisionKind.ACTION
            and decision.action is not None
            and " and " in f" {intent.utterance.casefold()} "
            and any(
                permission.endswith(".write")
                for card in cards
                if card.action.action_id == decision.action.action_id
                for permission in card.action.required_permissions
            )
        ):
            # Durable structural safety rule: a conjunction in a consequential
            # request cannot be silently reduced to one ACTION.  Compound
            # interpretation must produce a complete PLAN or clarify; this is
            # intentionally not a vocabulary of English action phrases.
            return Decision(
                kind=DecisionKind.CLARIFY,
                clarification=(
                    "This request contains more than one change, but I could not "
                    "form a complete verified plan. Please clarify the changes."
                ),
                semantic_mode="CLARIFY",
            )
        if decision.kind is DecisionKind.ACTION and decision.action is not None:
            action = decision.action
            if decision.kind is not DecisionKind.ACTION or action is None:
                return decision
            selected_card = next(
                (card for card in cards if card.action.action_id == action.action_id),
                None,
            )
            if (
                is_question_request(intent.utterance)
                and selected_card is not None
                and any(
                    permission.endswith(".write")
                    for permission in selected_card.action.required_permissions
                )
            ):
                read_cards = tuple(
                    card
                    for card in cards
                    if not any(
                        permission.endswith(".write")
                        for permission in card.action.required_permissions
                    )
                )
                if read_cards:
                    read_request = ModelRequest(
                        working_set=WorkingSet(intent=intent, context=context),
                        action_cards=read_cards,
                    )
                    read_decision = decoder.decode(
                        provider.decide(read_request),
                        read_cards,
                        allow_argument_proposals=False,
                    )
                    if (
                        read_decision.kind is not DecisionKind.ACTION
                        or read_decision.action is not None
                    ):
                        return read_decision
            action = decision.action
            if action is None:
                return decision
            card = next(
                (card for card in cards if card.action.action_id == action.action_id),
                None,
            )
            if card is not None and card.argument_keys and not action.arguments:
                # A small model can identify the capability but omit its
                # object argument. Re-ask with the already-selected card;
                # this remains bounded cognition, not phrase extraction.
                focused = request.model_copy(update={"action_cards": (card,)})
                decision = decoder.decode(
                    provider.decide(focused), (card,), allow_argument_proposals=True
                )
        return decision
    except InvalidDecision as exc:
        return recover_invalid_model_decision(dependencies, intent, context, focused_raw, exc)
    except Exception as exc:
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.FAILED,
            message="Model unavailable; request can be retried",
            evidence={
                "provenance": "model_boundary",
                "authoritative": False,
                "error_type": type(exc).__name__,
            },
            correlation_id=intent.correlation_id,
            retryable=True,
        )
