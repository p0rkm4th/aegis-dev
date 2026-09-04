"""Bounded Pack capability retrieval for the shared interaction boundary."""

from __future__ import annotations

from typing import Any, cast

from .contracts import ActionCard, Context
from .pack_lifecycle import PackManager


def _structural_anchor_count(dependencies: Any, utterance: str) -> int:
    """Use optional structural evidence to avoid narrowing a compound request.

    This is only candidate-set shaping.  The parser remains non-authoritative,
    and failures leave ordinary semantic retrieval unchanged; Core still
    validates every proposal and structural fidelity before execution.
    """

    parser = getattr(dependencies, "structural_parser", None)
    if parser is None:
        return 0
    try:
        signal = parser(utterance)
    except Exception:
        return 0
    return len(signal.anchors)


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
            anchor_count = _structural_anchor_count(dependencies, utterance)
            if anchor_count > 1:
                enabled_cards = getattr(manager, "enabled_cards", lambda: ())()
                write_cards = tuple(
                    card
                    for card in enabled_cards
                    if any(
                        permission.endswith(".write")
                        for permission in card.action.required_permissions
                    )
                )
                if len(write_cards) > 1:
                    # Structural plurality is evidence that a narrow semantic
                    # shortlist may have hidden a second capability.  Widen
                    # only to the bounded installed write vocabulary; this
                    # grants no authority and keeps the model candidate-bound.
                    semantic_writes = tuple(
                        card
                        for card in semantic_cards
                        if card.action.action_id in {item.action.action_id for item in write_cards}
                    )
                    by_id = {card.action.action_id: card for card in semantic_writes}
                    by_id.update({card.action.action_id: card for card in write_cards})
                    return tuple(by_id.values())[: min(5, max(2, anchor_count))]
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
                    # When retrieval has surfaced several write capabilities,
                    # read-only cards in the same namespace add decision noise
                    # without helping a mutation proposal. Keep the full
                    # namespace for a single-write case so ordinary read/write
                    # ambiguity still has its bounded vocabulary.
                    if len(write_cards) > 1:
                        scoped_write_cards = tuple(
                            card
                            for card in scoped_cards
                            if any(
                                permission.endswith(".write")
                                for permission in card.action.required_permissions
                            )
                        )
                        if scoped_write_cards:
                            return scoped_write_cards[:10]
                    return scoped_cards[:10]
            return tuple(semantic_cards)[:10]
    if dependencies.fallback_card_selector is not None:
        return cast(
            tuple[ActionCard, ...],
            dependencies.fallback_card_selector(manager, utterance, context),
        )
    return tuple(manager.enabled_cards())[:10]
