"""Small shared lexical guards for client-facing intent fast paths."""

from __future__ import annotations

import re

_MUTATION_PREFIX = re.compile(
    r"^(?:(?:please|kindly)\s+|(?:(?:could|would|can)\s+you\s+)|(?:i\s+need\s+you\s+to\s+))*"
    r"(?:(?:i\s+want|i\s+would\s+like|i'd\s+like)\s+to\s+)*"
    r"(?:add|create|update|complete|remove|put|place)\b"
)
_MARK_DONE = re.compile(
    r"^mark\s+(?:the\s+)?(?:task|chore)\s+.+?\s+as\s+"
    r"(?:done|complete|completed)[.!?]?$"
)


def is_mutation_request(utterance: str) -> bool:
    """Recognize polite imperative prefixes before read fast paths inspect terms."""

    normalized = " ".join(utterance.casefold().split())
    return (
        _MUTATION_PREFIX.match(normalized) is not None or _MARK_DONE.match(normalized) is not None
    )


def is_question_request(utterance: str) -> bool:
    """Recognize question shape for the read/write safety boundary."""

    normalized = " ".join(utterance.casefold().split())
    return normalized.endswith("?") or normalized.startswith(
        ("what ", "which ", "who ", "when ", "where ", "how ", "anything ", "can i ", "can we ")
    )
