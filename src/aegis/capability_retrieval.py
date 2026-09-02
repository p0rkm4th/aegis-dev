"""Bounded Pack capability retrieval for the shared interaction boundary."""

from __future__ import annotations

from typing import Any, cast

from .contracts import ActionCard, Context
from .pack_lifecycle import PackManager


def retrieve_action_cards(
    dependencies: Any,
    manager: PackManager,
    utterance: str,
    context: Context | None = None,
) -> tuple[ActionCard, ...]:
    """Return a bounded authorized candidate vocabulary for cognition.

    Retrieval is an optimization. It may narrow the model working set, but it
    never grants authority and a failure falls back to Pack-provided metadata.
    """

    if dependencies.capability_retriever is not None:
        try:
            semantic_cards = dependencies.capability_retriever(utterance, manager)
        except Exception:
            # Retrieval outage must not bypass the bounded model/decoder or
            # change the authority path.
            semantic_cards = ()
        if semantic_cards:
            write_cards = tuple(
                card
                for card in semantic_cards
                if any(
                    permission.endswith(".write") for permission in card.action.required_permissions
                )
            )
            if write_cards:
                namespace = write_cards[0].action.action_id.split(".", 1)[0]
                scoped_cards = tuple(
                    card
                    for card in semantic_cards
                    if card.action.action_id.split(".", 1)[0] == namespace
                )
                if scoped_cards:
                    return scoped_cards[:10]
            return tuple(semantic_cards)[:10]
    if dependencies.fallback_card_selector is not None:
        return cast(
            tuple[ActionCard, ...],
            dependencies.fallback_card_selector(manager, utterance, context),
        )
    return tuple(manager.enabled_cards())[:10]
