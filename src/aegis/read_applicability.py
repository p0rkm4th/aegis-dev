"""Small semantic-scope guard for canonical read projections."""

from __future__ import annotations

import re
from enum import StrEnum


class ReadApplicability(StrEnum):
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    CLARIFY = "CLARIFY"


def assess_read_applicability(utterance: str, semantic_scope: str | None) -> ReadApplicability:
    """Check whether a declared read scope answers the requested concept.

    This is deliberately conservative and bounded.  It is a semantic-scope
    safety check, not an action router: an unsupported adjacent concept is
    refused rather than substituted with whatever canonical collection exists.
    Unknown scopes remain applicable so existing Pack-specific semantics are
    not changed by this generic seam.
    """

    if semantic_scope != "kitchen.shopping_list":
        return ReadApplicability.MATCH
    text = " ".join(utterance.casefold().split())
    inventory_concept = (
        "pantry" in text
        or "on hand" in text
        or "in stock" in text
        or "how much" in text
        or "how many" in text
        or re.search(r"\bdo we have\b", text) is not None
    )
    shopping_concept = (
        "shopping list" in text
        or "need" in text
        or "to buy" in text
        or "to pick up" in text
        or "to get" in text
    )
    if inventory_concept and shopping_concept:
        return ReadApplicability.CLARIFY
    if inventory_concept:
        return ReadApplicability.NO_MATCH
    return ReadApplicability.MATCH
