"""Strict model response decoding; model output is never trusted as authority."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .contracts import ActionCard, ActionSpec, Decision, DecisionKind, ModelResponse


class InvalidDecision(ValueError):
    """Raised when a model response cannot be safely converted to a decision."""


class StrictDecisionDecoder:
    def decode(self, response: ModelResponse, cards: tuple[ActionCard, ...]) -> Decision:
        if not isinstance(response.raw, dict):
            raise InvalidDecision("model response must be an object")
        raw: dict[str, Any] = response.raw
        try:
            decision = Decision.model_validate(raw)
        except ValidationError as exc:
            raise InvalidDecision("model response failed the decision schema") from exc
        if decision.kind is DecisionKind.ACTION:
            if decision.action is None:
                raise InvalidDecision("ACTION requires an action")
            card = next((c for c in cards if c.action.action_id == decision.action.action_id), None)
            if card is None:
                raise InvalidDecision("action is not an exact match for a retrieved ActionCard")
            if decision.action != card.action:
                if len(cards) == 1:
                    # A small model may copy the card ID while dropping defaults or
                    # changing arguments. The retrieved card remains authoritative;
                    # canonicalize the proposal before policy or execution.
                    return decision.model_copy(update={"action": card.action})
                raise InvalidDecision("action is not an exact match for a retrieved ActionCard")
        elif decision.kind is DecisionKind.CLARIFY:
            if not decision.clarification or not decision.clarification.strip():
                raise InvalidDecision("CLARIFY requires a clarification question")
        elif decision.action is not None:
            raise InvalidDecision("only ACTION decisions may contain an action")
        return decision


def action_from_card(card: ActionCard) -> ActionSpec:
    """Return a copy boundary for callers that need a typed action proposal."""
    return card.action.model_copy(deep=True)
