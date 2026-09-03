"""Strict model response decoding; model output is never trusted as authority."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .contracts import ActionCard, ActionSpec, Decision, DecisionKind, ModelResponse
from .planning import PlanValidationError, materialize_proposed_plan


class InvalidDecision(ValueError):
    """Raised when a model response cannot be safely converted to a decision."""


class StrictDecisionDecoder:
    def decode(
        self,
        response: ModelResponse,
        cards: tuple[ActionCard, ...],
        *,
        allow_argument_proposals: bool = False,
    ) -> Decision:
        if not isinstance(response.raw, dict):
            raise InvalidDecision("model response must be an object")
        raw: dict[str, Any] = response.raw
        try:
            decision = Decision.model_validate(raw)
        except ValidationError as exc:
            raise InvalidDecision("model response failed the decision schema") from exc
        if decision.kind is DecisionKind.ACTION:
            if decision.objective_spec is not None:
                raise InvalidDecision("objective requirements are only valid for PLAN")
            if decision.semantic_mode not in {None, "ACTION"}:
                raise InvalidDecision("ACTION decision must use semantic_mode ACTION")
            proposed = decision.action
            action_ref = decision.action_ref
            if proposed is None:
                if not isinstance(action_ref, str) or not action_ref:
                    raise InvalidDecision("ACTION requires an action reference")
                card = next((c for c in cards if c.action.action_id == action_ref), None)
                if card is None:
                    raise InvalidDecision("action reference is not a retrieved ActionCard")
                if not allow_argument_proposals and decision.action_arguments:
                    raise InvalidDecision("action arguments require bounded proposal mode")
                if not set(decision.action_arguments).issubset(card.argument_keys):
                    raise InvalidDecision("action proposal exceeds the ActionCard contract")
                return decision.model_copy(
                    update={
                        "action": card.action.model_copy(
                            update={"arguments": decision.action_arguments}
                        )
                    }
                )
            card = next((c for c in cards if c.action.action_id == proposed.action_id), None)
            if card is None:
                raise InvalidDecision("action is not an exact match for a retrieved ActionCard")
            if action_ref is not None and action_ref != proposed.action_id:
                raise InvalidDecision("action reference does not match the proposed action")
            if decision.action_arguments:
                raise InvalidDecision("action arguments must be inside the legacy action object")
            if proposed != card.action:
                if allow_argument_proposals:
                    canonical = card.action
                    if (
                        proposed.action_id != canonical.action_id
                        or proposed.capability != canonical.capability
                        or proposed.required_permissions != canonical.required_permissions
                        or proposed.verification != canonical.verification
                        or not set(proposed.arguments).issubset(card.argument_keys)
                    ):
                        raise InvalidDecision("action proposal exceeds the ActionCard contract")
                    return decision.model_copy(
                        update={
                            "action": canonical.model_copy(update={"arguments": proposed.arguments})
                        }
                    )
                if len(cards) == 1:
                    # A small model may copy the card ID while dropping defaults or
                    # changing arguments. The retrieved card remains authoritative;
                    # canonicalize the proposal before policy or execution.
                    return decision.model_copy(update={"action": card.action})
                raise InvalidDecision("action is not an exact match for a retrieved ActionCard")
        elif decision.kind is DecisionKind.PLAN:
            if not allow_argument_proposals:
                raise InvalidDecision("plan proposals require bounded proposal mode")
            if decision.semantic_mode not in {None, "ACTION"}:
                raise InvalidDecision("PLAN decision must use semantic_mode ACTION")
            if decision.plan is None or len(decision.plan.steps) < 2:
                raise InvalidDecision("PLAN requires at least two steps")
            try:
                materialize_proposed_plan(decision.plan, cards)
            except PlanValidationError as exc:
                raise InvalidDecision(str(exc)) from exc
            if (
                decision.action is not None
                or decision.action_ref is not None
                or decision.action_arguments
                or decision.answer is not None
                or decision.clarification is not None
                or decision.context_focus is not None
            ):
                raise InvalidDecision("PLAN cannot contain fields for another decision kind")
        elif decision.kind is DecisionKind.ANSWER:
            if decision.objective_spec is not None:
                raise InvalidDecision("objective requirements are only valid for PLAN")
            if decision.semantic_mode not in {None, "READ", "GENERATION"}:
                raise InvalidDecision("ANSWER decision must use semantic_mode READ or GENERATION")
            if not decision.answer or not decision.answer.strip():
                raise InvalidDecision("ANSWER requires non-empty answer content")
        elif decision.kind is DecisionKind.CLARIFY:
            if decision.objective_spec is not None:
                raise InvalidDecision("objective requirements are only valid for PLAN")
            if decision.semantic_mode not in {None, "CLARIFY"}:
                raise InvalidDecision("CLARIFY decision must use semantic_mode CLARIFY")
            if not decision.clarification or not decision.clarification.strip():
                raise InvalidDecision("CLARIFY requires a clarification question")
        elif (
            decision.action is not None
            or decision.action_ref is not None
            or decision.action_arguments
            or decision.context_focus is not None
            or decision.objective_spec is not None
        ):
            raise InvalidDecision("only ACTION decisions may contain an action proposal")
        return decision


def action_from_card(card: ActionCard) -> ActionSpec:
    """Return a copy boundary for callers that need a typed action proposal."""
    return card.action.model_copy(deep=True)
