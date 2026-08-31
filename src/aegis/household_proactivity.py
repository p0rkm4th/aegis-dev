"""Deterministic household signals to ambient suggestion composition."""

from __future__ import annotations

from pydantic import Field

from .ambient import AmbientSuggestion
from .contracts import ActionSpec, StrictModel


class HouseholdSignals(StrictModel):
    """Already-authorized shared signals; private Vault data is not accepted."""

    space_id: str = Field(min_length=1)
    low_groceries: tuple[str, ...] = ()
    expiring_ingredients: tuple[str, ...] = ()
    shared_day: str | None = None


class HouseholdProactivity:
    @staticmethod
    def suggest_meal(signals: HouseholdSignals) -> AmbientSuggestion | None:
        if not signals.low_groceries or not signals.expiring_ingredients or not signals.shared_day:
            return None
        grocery = signals.low_groceries[0]
        expiring = signals.expiring_ingredients[0]
        return AmbientSuggestion(
            reason=f"{expiring} expires soon and {grocery} is low",
            text=(
                f"Everyone is home {signals.shared_day}; I can plan a meal around "
                f"{expiring} and add {grocery} to groceries."
            ),
            proposed_action=ActionSpec(
                action_id="kitchen.groceries.add",
                capability="kitchen.groceries.write",
                arguments={"item": grocery},
            ),
        )
