"""Small shared lexical guards for client-facing intent fast paths."""

from __future__ import annotations

import re

_MUTATION_PREFIX = re.compile(
    r"^(?:(?:please|kindly)\s+|(?:(?:could|would|can)\s+you\s+)|(?:i\s+need\s+you\s+to\s+))*"
    r"(?:add|create|update|complete|remove|put|place)\b"
)


def is_mutation_request(utterance: str) -> bool:
    """Recognize polite imperative prefixes before read fast paths inspect terms."""

    return _MUTATION_PREFIX.match(" ".join(utterance.casefold().split())) is not None
