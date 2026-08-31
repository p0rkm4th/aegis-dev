"""Capability registry with bounded, relevance-ranked retrieval."""

from __future__ import annotations

from .contracts import ActionCard


class CapabilityRegistry:
    def __init__(self, cards: tuple[ActionCard, ...] = ()) -> None:
        self._cards: dict[str, ActionCard] = {card.action.action_id: card for card in cards}

    def register(self, card: ActionCard) -> None:
        if card.action.action_id in self._cards:
            raise ValueError(f"duplicate action id: {card.action.action_id}")
        self._cards[card.action.action_id] = card

    def retrieve(self, domain: str, limit: int = 5) -> tuple[ActionCard, ...]:
        if not 1 <= limit <= 5:
            raise ValueError("capability retrieval limit must be between one and five")
        ranked = sorted(
            (card for card in self._cards.values() if domain in card.action.capability),
            key=lambda card: card.relevance,
            reverse=True,
        )
        return tuple(ranked[:limit])
