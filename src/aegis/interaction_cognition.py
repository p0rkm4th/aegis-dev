"""Bounded model cognition coordination for the interaction boundary."""

from __future__ import annotations

import inspect
import re
from typing import Any, cast
from uuid import uuid4

from .contracts import (
    ActionCard,
    Context,
    Decision,
    DecisionKind,
    IntentFrame,
    ModelRequest,
    ObjectiveRequirementProposal,
    ObjectiveSpecProposal,
    ObjectiveState,
    ProposedPlan,
    ProposedPlanStep,
    Result,
    WorkingSet,
)
from .decoding import InvalidDecision, StrictDecisionDecoder
from .interaction_recovery import recover_invalid_model_decision
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
    # A normal two-operation plan commonly contains one conjunction.  Extra
    # capability-scoped calls are justified only when the request has multiple
    # structural conjunction boundaries, which is evidence of a third clause.
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
                if knowledge_source in {"external_evidence", "mixed_evidence"}:
                    research_answer = getattr(dependencies, "research_answer", None)
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
        routing_only = any(
            permission.endswith(".write")
            for card in cards
            for permission in card.action.required_permissions
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
        last_invalid: InvalidDecision | None = None
        for attempt in range(2):
            response = provider.decide(request)
            last_raw = response.raw if isinstance(response.raw, dict) else None
            if last_raw is not None and last_raw.get("context_focus") is not None:
                focused_raw = last_raw
            try:
                decision = decoder.decode(
                    response, request.action_cards, allow_argument_proposals=True
                )
                break
            except InvalidDecision as error:
                last_invalid = error
                # Repair only an empty benign answer with no capability
                # cards. This remains bounded cognition and cannot turn a
                # malformed action into an executable proposal.
                raw = response.raw
                if not (
                    attempt == 0
                    and isinstance(raw, dict)
                    and raw.get("kind") == DecisionKind.ANSWER.value
                    and not isinstance(raw.get("answer"), str)
                ):
                    raise error
                request = request.model_copy(update={"action_cards": ()})
        if decision is None and routing_only and last_invalid is not None:
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
            except InvalidDecision:
                raise last_invalid
        if decision is None:
            raise InvalidDecision("model answer repair did not produce a decision")
        if decision.kind is DecisionKind.PLAN and decision.plan is not None:
            if decision.objective_spec is None:
                interpretation_request = request.model_copy(
                    update={"objective_interpretation_only": True}
                )
                interpretation_response = provider.decide(interpretation_request)
                if not isinstance(interpretation_response.raw, dict):
                    raise InvalidDecision("objective interpretation must be an object")
                try:
                    objective_spec = ObjectiveSpecProposal.model_validate(
                        interpretation_response.raw
                    )
                except Exception as exc:
                    raise InvalidDecision(
                        "objective interpretation failed its strict schema"
                    ) from exc
                decision = decision.model_copy(update={"objective_spec": objective_spec})
            decision = _scope_plan_by_capability(
                provider, decoder, intent, cards, context, decision
            )
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
