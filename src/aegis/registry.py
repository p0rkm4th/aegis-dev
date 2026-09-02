"""Capability registry with bounded, relevance-ranked retrieval."""

from __future__ import annotations

import math
from typing import NamedTuple, Protocol

from .contracts import ActionCard


class CapabilityEmbedder(Protocol):
    """Minimal embedding seam; vectors are retrieval hints, never canonical state."""

    def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]: ...


class CapabilityMatch(NamedTuple):
    card: ActionCard
    score: float


class CapabilityRegistry:
    def __init__(self, cards: tuple[ActionCard, ...] = ()) -> None:
        self._cards: dict[str, ActionCard] = {card.action.action_id: card for card in cards}

    def register(self, card: ActionCard) -> None:
        if card.action.action_id in self._cards:
            raise ValueError(f"duplicate action id: {card.action.action_id}")
        self._cards[card.action.action_id] = card

    def retrieve(self, domain: str, limit: int = 5) -> tuple[ActionCard, ...]:
        if not 1 <= limit <= 10:
            raise ValueError("capability retrieval limit must be between one and ten")
        ranked = sorted(
            (card for card in self._cards.values() if domain in card.action.capability),
            key=lambda card: card.relevance,
            reverse=True,
        )
        return tuple(ranked[:limit])

    def retrieve_semantic_with_scores(
        self, query: str, embedder: CapabilityEmbedder, limit: int = 5
    ) -> tuple[CapabilityMatch, ...]:
        """Rank enabled cards and expose retrieval scores for diagnostics."""

        if not query.strip():
            raise ValueError("capability query must be non-empty")
        if not 1 <= limit <= 10:
            raise ValueError("capability retrieval limit must be between one and ten")
        cards = tuple(self._cards.values())
        if not cards:
            return ()
        descriptions = tuple(
            f"{card.action.action_id}: {card.action.capability}. {card.summary}. "
            f"Arguments: {'; '.join(card.argument_descriptions.values())}"
            for card in cards
        )
        vectors = embedder.embed((query, *descriptions))
        if len(vectors) != len(descriptions) + 1:
            raise ValueError("capability embedder returned an invalid vector count")
        query_vector = vectors[0]

        def cosine(vector: tuple[float, ...]) -> float:
            if len(vector) != len(query_vector):
                raise ValueError("capability embedder returned inconsistent dimensions")
            query_norm = math.sqrt(sum(value * value for value in query_vector))
            vector_norm = math.sqrt(sum(value * value for value in vector))
            if query_norm == 0 or vector_norm == 0:
                return 0.0
            return sum(left * right for left, right in zip(query_vector, vector)) / (
                query_norm * vector_norm
            )

        ranked = sorted(
            zip(cards, vectors[1:]),
            key=lambda item: (cosine(item[1]), item[0].relevance),
            reverse=True,
        )
        return tuple(
            CapabilityMatch(card=card, score=cosine(vector)) for card, vector in ranked[:limit]
        )

    def retrieve_semantic(
        self, query: str, embedder: CapabilityEmbedder, limit: int = 5
    ) -> tuple[ActionCard, ...]:
        """Rank enabled cards by semantic description without widening authority."""

        return tuple(
            match.card for match in self.retrieve_semantic_with_scores(query, embedder, limit)
        )
