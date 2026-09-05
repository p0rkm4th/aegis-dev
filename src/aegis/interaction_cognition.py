"""Bounded model cognition coordination for the interaction boundary."""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, cast
from uuid import uuid4

from .contracts import (
    ActionCard,
    CapabilityInvestigationState,
    CapabilityNeed,
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
    RequestedEffectResolution,
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
    plan_covers_objective,
    validate_structural_coverage,
)
from .utterance import is_question_request


def _requires_external_research(utterance: str) -> bool:
    """Require provenance for explicit current/public research requests."""

    folded = " ".join(utterance.casefold().split())
    if "research" in folded or "sources" in folded or "source" in folded:
        return True
    current_markers = ("current", "currently", "latest", "recent", "today", "now")
    public_markers = ("public", "recommended", "documented", "official", "according to")
    return any(marker in folded for marker in current_markers) and any(
        marker in folded for marker in public_markers
    )


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


def _structural_plurality_failure(dependencies: Any, utterance: str) -> ProposalFailureEvidence:
    """Expose bounded parser spans when a clarification enters repair."""

    structural_parser = getattr(dependencies, "structural_parser", None)
    if structural_parser is not None:
        try:
            signal = structural_parser(utterance)
            spans = [anchor.source_span for anchor in signal.anchors]
            return ProposalFailureEvidence(
                kind=ProposalFailureKind.UNACCOUNTED_STRUCTURAL_ANCHOR,
                detail=(
                    "structural plurality requires one plan step per requested change; "
                    f"anchor_count={len(spans)} anchor_spans={spans}"
                ),
                related_source_spans=tuple(spans[:5]),
            )
        except Exception:
            pass
    return ProposalFailureEvidence(
        kind=ProposalFailureKind.UNACCOUNTED_STRUCTURAL_ANCHOR,
        detail="structural plurality requires one plan step per requested change",
    )


def _effect_first_repair_context(
    provider: Any,
    dependencies: Any,
    intent: IntentFrame,
    context: Context,
    cards: tuple[ActionCard, ...],
) -> Context | None:
    """Collect bounded effect evidence before repairing a compound clarification.

    This is deliberately evidence-only. The resulting context does not grant
    capability or execution authority; the normal PLAN path re-runs effect,
    fidelity, policy, and Kernel validation after repair.
    """

    structural_parser = getattr(dependencies, "structural_parser", None)
    if structural_parser is None:
        return None
    try:
        signal = structural_parser(intent.utterance)
        response = provider.decide(
            ModelRequest(
                working_set=WorkingSet(intent=intent, context=context),
                action_cards=cards,
                objective_effect_only=True,
                allow_plan_proposals=False,
                allow_argument_proposals=False,
            )
        )
        raw_effects = response.raw.get("effects") if isinstance(response.raw, dict) else None
        if not isinstance(raw_effects, list):
            return None
        effects = tuple(RequestedEffectProposal.model_validate(item) for item in raw_effects)
        materialized = materialize_requested_effects(intent.utterance, effects)
        if materialized is None or not validate_structural_coverage(
            intent.utterance, materialized, signal
        ):
            return None
    except Exception:
        return None
    values = dict(context.values)
    values["grounded_requested_effects"] = [
        {
            "effect_text": effect.effect_text,
            "source_span": list(effect.source_span),
            "polarity": effect.polarity,
        }
        for effect in effects
    ]
    return context.model_copy(update={"values": values})


def _bounded_decision_repair(
    provider: Any,
    intent: IntentFrame,
    context: Context,
    cards: tuple[ActionCard, ...],
    raw: dict[str, Any] | None,
    error: InvalidDecision,
) -> Decision | None:
    """Repair a decoded action proposal within the shared two-attempt budget."""

    current_raw = raw
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
            return repaired
        fingerprint = proposal_failure_fingerprint(next_failure)
        if fingerprint in seen_failures:
            return None
        seen_failures.add(fingerprint)
        current_raw = repaired_raw or current_raw
        current_failure = next_failure
        error = InvalidDecision(next_failure.detail or next_failure.kind.value)
    return None


def _unresolved_investigation_result(
    dependencies: Any, intent: IntentFrame, context: Context, effects: tuple[Any, ...]
) -> Result | None:
    """Accept only an explicit non-authoritative, non-completing investigation result."""

    investigator = getattr(dependencies, "unresolved_requirement_investigator", None)
    if investigator is None:
        return None
    try:
        investigated = investigator(intent, context, effects)
    except Exception:
        return None
    if (
        isinstance(investigated, Result)
        and investigated.state in {ObjectiveState.BLOCKED, ObjectiveState.FAILED}
        and investigated.evidence.get("authoritative") is False
    ):
        evidence = dict(investigated.evidence)
        evidence["objective_open"] = True
        evidence["unsatisfied_requirements"] = [
            {
                "effect_id": str(effect.effect_id),
                "normalized_effect": effect.normalized_effect,
                "source_spans": [list(span) for span in effect.source_spans],
                "resolution": RequestedEffectResolution.UNSUPPORTED.value,
            }
            for effect in effects
        ]
        evidence["capability_needs"] = [
            CapabilityNeed(
                requirement_id=effect.effect_id,
                requested_effect=effect.normalized_effect,
                reason="No enabled ActionCard currently satisfies this requested effect.",
                permitted_scope=(
                    "installed_capabilities",
                    "authorized_canonical_state",
                    "public_research",
                ),
                investigation=CapabilityInvestigationState.COMPLETE,
                candidate_resolutions=tuple(
                    item
                    for item in (
                        {
                            "kind": "available_action_ids",
                            "action_ids": investigated.evidence.get("available_action_ids", []),
                        },
                    )
                    if isinstance(item, dict)
                ),
                parent_objective_id=investigated.objective_id,
            ).model_dump(mode="json")
            for effect in effects
        ]
        return investigated.model_copy(update={"evidence": evidence})
    return None


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
    max_attempts: int = 2,
) -> Decision:
    """Give fidelity-generated clarification bounded repair opportunities."""

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
            "proposal_repair",
        )
        return repaired, evidence if repaired is None else evidence

    def validate(candidate: Decision) -> ValidationResult:
        if plans_only and candidate.kind is not DecisionKind.PLAN:
            return ValidationResult(
                valid=False,
                failure=ProposalFailureEvidence(
                    kind=ProposalFailureKind.UNACCOUNTED_STRUCTURAL_ANCHOR,
                    detail=(
                        "compound repair must return a PLAN with one independently "
                        "grounded step per requested change"
                    ),
                ),
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
        max_attempts=max_attempts,
        telemetry=lambda: dict(getattr(provider, "last_response_metrics", {})),
    )
    return result.proposal if result.proposal is not None else decision


def _bounded_repair_provider(dependencies: Any, resident_provider: Any) -> Any:
    """Select optional stronger cognition only for bounded proposal repair."""

    factory = getattr(dependencies, "repair_model_provider", None)
    if factory is None:
        return resident_provider
    repair_provider = factory()
    # Recovery telemetry belongs to the runtime, not to whichever model was
    # selected. Share the existing sinks so evaluators cannot lose or split
    # repair events when escalation is enabled.
    if hasattr(resident_provider, "recovery_events") and hasattr(
        repair_provider, "recovery_events"
    ):
        repair_provider.recovery_events = resident_provider.recovery_events
    if hasattr(resident_provider, "request_mode_counts") and hasattr(
        repair_provider, "request_mode_counts"
    ):
        repair_provider.request_mode_counts = resident_provider.request_mode_counts
    return repair_provider


def decide_fallback(
    dependencies: Any, intent: IntentFrame, cards: tuple[ActionCard, ...], context: Context
) -> Decision | Result | None:
    if dependencies.model_provider is None:
        return None
    last_raw: dict[str, Any] | None = None
    focused_raw: dict[str, Any] | None = None
    try:
        provider = dependencies.model_provider()
        repair_provider = _bounded_repair_provider(dependencies, provider)
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
                if _requires_external_research(intent.utterance):
                    knowledge_source = "external_evidence"
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
            decision = _bounded_decision_repair(
                repair_provider, intent, context, cards, last_raw, error
            )
            if decision is None:
                raise error
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
        if decision.kind is DecisionKind.CLARIFY and _has_structural_plurality(
            dependencies, intent.utterance
        ):
            # A compound-shaped utterance that the ordinary cognition pass
            # conservatively clarified has two bounded opportunities to produce
            # a complete PLAN. The resulting proposal immediately re-enters
            # requested-effect coverage and objective-fidelity validation.
            effect_first_context = _effect_first_repair_context(
                repair_provider, dependencies, intent, context, cards
            )
            decision = _repair_clarification(
                repair_provider,
                intent,
                effect_first_context or context,
                cards,
                decision,
                "the request contains multiple independently requested changes",
                _structural_plurality_failure(dependencies, intent.utterance),
                plans_only=True,
                max_attempts=1 if effect_first_context is not None else 2,
            )
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
                    repair_provider,
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
        if decision.kind is DecisionKind.ACTION and decision.action is not None:
            structural_failure = _structural_write_failure(dependencies, intent, decision, cards)
            conjunction_failure = " and " in f" {intent.utterance.casefold()} " and any(
                permission.endswith(".write")
                for card in cards
                if card.action.action_id == decision.action.action_id
                for permission in card.action.required_permissions
            )
            has_contrast_evidence = False
            structural_parser = getattr(dependencies, "structural_parser", None)
            if structural_parser is not None:
                try:
                    has_contrast_evidence = bool(structural_parser(intent.utterance).negation_spans)
                except Exception:
                    has_contrast_evidence = True
            if (
                structural_failure is not None or conjunction_failure
            ) and not has_contrast_evidence:
                # Route the incomplete single-action proposal through the same
                # bounded PLAN repair before the fidelity validator runs. A
                # late clarification would otherwise discard the repair seam
                # and leave supported compound requests unrecoverable.
                decision = _repair_clarification(
                    repair_provider,
                    intent,
                    context,
                    cards,
                    decision,
                    structural_failure
                    or "the request contains multiple independently requested changes",
                    ProposalFailureEvidence(
                        kind=ProposalFailureKind.UNACCOUNTED_STRUCTURAL_ANCHOR,
                        detail=structural_failure
                        or "structural plurality requires one plan step per requested change",
                    ),
                    plans_only=True,
                    max_attempts=2,
                )
        if decision.kind is DecisionKind.ANSWER and _requires_external_research(intent.utterance):
            decision = decision.model_copy(update={"knowledge_source": "external_evidence"})
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
                        failure=ProposalFailureEvidence(
                            kind=ProposalFailureKind.BAD_SOURCE_SPAN,
                            detail=(
                                "requested effect spans did not ground uniquely; "
                                f"effect_count={len(candidate_effects)}"
                            ),
                        ),
                    )
                for effect in candidate_materialized:
                    if effect.polarity == "NEGATED":
                        kind = ProposalFailureKind.NEGATED_EFFECT_ACTIVE
                    elif effect.polarity == "SUPERSEDED":
                        kind = ProposalFailureKind.SUPERSEDED_EFFECT_ACTIVE
                    else:
                        continue
                    return ValidationResult(valid=False, failure=ProposalFailureEvidence(kind=kind))
                available_actions = {card.action.action_id for card in cards}
                if any(
                    effect.action_ref is not None and effect.action_ref not in available_actions
                    for effect in candidate_effects
                ):
                    return ValidationResult(
                        valid=False,
                        failure=ProposalFailureEvidence(
                            kind=ProposalFailureKind.CAPABILITY_UNAVAILABLE,
                            detail="requested effect has no available ActionCard",
                        ),
                    )
                if not validate_structural_coverage(
                    intent.utterance, candidate_materialized, structural_signal
                ):
                    anchor_spans = [anchor.source_span for anchor in structural_signal.anchors]
                    effect_spans = [
                        span for effect in candidate_materialized for span in effect.source_spans
                    ]

                    def overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
                        return left[0] < right[1] and right[0] < left[1]

                    unmatched_anchors = [
                        span
                        for span in anchor_spans
                        if not any(overlaps(span, effect_span) for effect_span in effect_spans)
                    ]
                    unmatched_effects = [
                        span
                        for span in effect_spans
                        if not any(overlaps(span, anchor_span) for anchor_span in anchor_spans)
                    ]
                    return ValidationResult(
                        valid=False,
                        failure=ProposalFailureEvidence(
                            kind=ProposalFailureKind.UNACCOUNTED_STRUCTURAL_ANCHOR,
                            detail=(
                                "structural/effect coverage mismatch; "
                                f"anchor_count={len(structural_signal.anchors)} "
                                f"effect_count={len(candidate_materialized)} "
                                f"unmatched_anchors={unmatched_anchors[:5]} "
                                f"unmatched_effects={unmatched_effects[:5]}"
                            ),
                            related_source_spans=tuple(
                                unmatched_anchors[:5] + unmatched_effects[:5]
                            )[:5],
                        ),
                    )
                return ValidationResult(valid=True)

            effect_failure = validate_effect_proposal(effect_raw).failure
            if effect_failure is not None:

                def repair_effects(
                    current: dict[str, Any], evidence: ProposalFailureEvidence
                ) -> tuple[dict[str, Any] | None, ProposalFailureEvidence]:
                    response = repair_provider.decide(
                        request.model_copy(
                            update={
                                "objective_effect_only": True,
                                "proposal_repair_only": True,
                                "repair_validator_stage": "requested_effect_structural_coverage",
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
                    telemetry=lambda: dict(getattr(repair_provider, "last_response_metrics", {})),
                )
                recovery_events = getattr(provider, "recovery_events", None)
                if isinstance(recovery_events, list):
                    recovery_events.extend(
                        proposal_repair_event_record(event) for event in repaired_effects.events
                    )
                if repaired_effects.proposal is None:
                    if (
                        effect_failure.kind
                        in {
                            ProposalFailureKind.CAPABILITY_UNAVAILABLE,
                            ProposalFailureKind.UNSUPPORTED_REQUIREMENT,
                        }
                        and materialized_effects is not None
                    ):
                        investigated = _unresolved_investigation_result(
                            dependencies, intent, context, tuple(materialized_effects)
                        )
                        if investigated is not None:
                            return investigated
                        return Result(
                            objective_id=uuid4(),
                            state=ObjectiveState.BLOCKED,
                            message=(
                                "I recognized a requested change that is not currently "
                                "supported. I kept it open rather than dropping it."
                            ),
                            evidence={
                                "authoritative": False,
                                "provenance": "requested_effect_resolution",
                                "objective_open": True,
                                "unsatisfied_requirements": [
                                    {
                                        "effect_id": str(effect.effect_id),
                                        "normalized_effect": effect.normalized_effect,
                                        "source_spans": [
                                            list(span) for span in effect.source_spans
                                        ],
                                        "resolution": RequestedEffectResolution.UNSUPPORTED.value,
                                    }
                                    for effect in materialized_effects
                                ],
                            },
                            correlation_id=intent.correlation_id,
                        )
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
            independent_spec: ObjectiveSpecProposal | None
            # Keep effect segmentation and capability mapping independent even
            # when the effect model volunteers action references. Those refs
            # remain untrusted; every grounded effect goes through the existing
            # candidate-bound mapping validator before fidelity comparison.
            if effects:
                mapping_values = dict(context.values)
                mapping_values["grounded_requested_effects"] = [
                    {
                        "effect_text": effect.effect_text,
                        "source_span": list(effect.source_span),
                        "polarity": effect.polarity,
                    }
                    for effect in effects
                ]
                mapping_request = request.model_copy(
                    update={
                        "working_set": WorkingSet(
                            intent=intent,
                            context=context.model_copy(update={"values": mapping_values}),
                        ),
                        "objective_effect_only": False,
                        "objective_interpretation_only": True,
                        "objective_fidelity_only": False,
                        "allow_plan_proposals": False,
                        "allow_argument_proposals": False,
                    }
                )
                mapping_response = provider.decide(mapping_request)
                mapping_raw = mapping_response.raw if isinstance(mapping_response.raw, dict) else {}

                def validate_mapping(candidate: dict[str, Any]) -> ValidationResult:
                    try:
                        candidate_spec = ObjectiveSpecProposal.model_validate(candidate)
                    except Exception:
                        return ValidationResult(
                            valid=False,
                            failure=ProposalFailureEvidence(
                                kind=ProposalFailureKind.DECODER_SCHEMA_FAILURE,
                                detail="objective capability mapping schema is invalid",
                            ),
                        )
                    if len(candidate_spec.requirements) != len(effects):
                        return ValidationResult(
                            valid=False,
                            failure=ProposalFailureEvidence(
                                kind=ProposalFailureKind.MISSING_EFFECT,
                                detail=(
                                    "objective capability mapping must cover every grounded "
                                    f"effect; effect_count={len(effects)} "
                                    f"requirement_count={len(candidate_spec.requirements)}"
                                ),
                            ),
                        )
                    for requirement in candidate_spec.requirements:
                        card = next(
                            (
                                card
                                for card in cards
                                if card.action.action_id == requirement.action_ref
                            ),
                            None,
                        )
                        if card is None:
                            return ValidationResult(
                                valid=False,
                                failure=ProposalFailureEvidence(
                                    kind=ProposalFailureKind.CAPABILITY_UNAVAILABLE,
                                    detail="objective mapping selected an unavailable ActionCard",
                                ),
                            )
                        if not set(requirement.arguments).issubset(set(card.argument_keys)):
                            return ValidationResult(
                                valid=False,
                                failure=ProposalFailureEvidence(
                                    kind=ProposalFailureKind.INVALID_ARGUMENT,
                                    detail="objective mapping proposed an undeclared argument",
                                ),
                            )
                    return ValidationResult(valid=True)

                mapping_failure = validate_mapping(mapping_raw).failure
                if mapping_failure is not None:

                    def repair_mapping(
                        current: dict[str, Any], evidence: ProposalFailureEvidence
                    ) -> tuple[dict[str, Any] | None, ProposalFailureEvidence]:
                        response = repair_provider.decide(
                            mapping_request.model_copy(
                                update={
                                    "proposal_repair_only": True,
                                    "repair_validator_stage": "objective_capability_mapping",
                                    "proposal_failure": evidence,
                                    "current_proposal": current,
                                }
                            )
                        )
                        candidate = response.raw if isinstance(response.raw, dict) else None
                        if candidate is None:
                            return None, ProposalFailureEvidence(
                                kind=ProposalFailureKind.DECODER_SCHEMA_FAILURE,
                                detail="objective capability repair must be an object",
                            )
                        result = validate_mapping(candidate)
                        return candidate, result.failure or evidence

                    repaired_mapping = bounded_proposal_repair(
                        mapping_raw,
                        mapping_failure,
                        repair_mapping,
                        validate_mapping,
                        validator_stage="objective_capability_mapping",
                        max_attempts=2,
                        telemetry=lambda: dict(
                            getattr(repair_provider, "last_response_metrics", {})
                        ),
                    )
                    recovery_events = getattr(provider, "recovery_events", None)
                    if isinstance(recovery_events, list):
                        recovery_events.extend(
                            proposal_repair_event_record(event) for event in repaired_mapping.events
                        )
                    if repaired_mapping.proposal is None:
                        return Decision(
                            kind=DecisionKind.CLARIFY,
                            clarification=(
                                "I grounded the requested changes but could not safely map "
                                "each one to an available capability."
                            ),
                            semantic_mode="CLARIFY",
                        )
                    mapping_raw = repaired_mapping.proposal
                mapped_spec = ObjectiveSpecProposal.model_validate(mapping_raw)
                independent_spec = mapped_spec
            else:
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
                    if candidate.plan is None or not plan_covers_objective(
                        candidate.plan, candidate.objective_spec
                    ):
                        return ValidationResult(
                            valid=False,
                            failure=ProposalFailureEvidence(
                                kind=ProposalFailureKind.MISSING_EFFECT,
                                detail="repaired plan does not cover its objective requirements",
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
                    repair_provider,
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
            action = decision.action
            structural_failure = _structural_write_failure(dependencies, intent, decision, cards)
            if structural_failure is not None:
                return Decision(
                    kind=DecisionKind.CLARIFY,
                    clarification=structural_failure,
                    semantic_mode="CLARIFY",
                )
            if " and " in f" {intent.utterance.casefold()} " and any(
                permission.endswith(".write")
                for card in cards
                if card.action.action_id == action.action_id
                for permission in card.action.required_permissions
            ):
                # A failed bounded repair must not restore the original
                # single-action proposal across a compound request.
                return Decision(
                    kind=DecisionKind.CLARIFY,
                    clarification=(
                        "This request contains more than one change, but I could not "
                        "form a complete verified plan. Please clarify the changes."
                    ),
                    semantic_mode="CLARIFY",
                )
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
                focused_response = provider.decide(focused)
                focused_raw = (
                    focused_response.raw if isinstance(focused_response.raw, dict) else None
                )
                try:
                    decision = decoder.decode(
                        focused_response, (card,), allow_argument_proposals=True
                    )
                except InvalidDecision as error:
                    decision = _bounded_decision_repair(
                        repair_provider, intent, context, (card,), focused_raw, error
                    )
                    if decision is None:
                        raise error
                if decision.action is not None and not decision.action.arguments:
                    missing_argument_error = InvalidDecision("required action argument is missing")
                    repaired = _bounded_decision_repair(
                        repair_provider,
                        intent,
                        context,
                        (card,),
                        focused_raw,
                        missing_argument_error,
                    )
                    if repaired is None:
                        raise missing_argument_error
                    decision = repaired
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
