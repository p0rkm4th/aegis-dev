"""Small shared lexical guards for client-facing intent fast paths."""

from __future__ import annotations

import re

_MUTATION_PREFIX = re.compile(
    r"^(?:(?:please|kindly)\s+|(?:(?:could|would|can)\s+you\s+)|(?:i\s+need\s+you\s+to\s+))*"
    r"(?:(?:i\s+want|i\s+would\s+like|i'd\s+like)\s+to\s+)*"
    r"(?:add|create|update|complete|remove|put|place|jot\s+down)\b"
)
_MARK_DONE = re.compile(
    r"^mark\s+(?:the\s+)?(?:task|chore)\s+.+?\s+as\s+"
    r"(?:done|complete|completed)[.!?]?$"
)
_CONTEXT_RESET_PREFIX = re.compile(
    r"^(?:actually[,:;\s]+)?(?:never\s+mind|forget\s+that|scratch\s+that)[,:;\s\-—–]+"
)


def strip_context_reset(utterance: str) -> str:
    """Remove a conversational reset before matching the owner's new objective."""

    normalized = " ".join(utterance.casefold().split())
    return _CONTEXT_RESET_PREFIX.sub("", normalized, count=1)


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


def has_multiple_question_clauses(utterance: str) -> bool:
    """Detect compound questions before a single-result fast path.

    This is a structural safety boundary, not a natural-language vocabulary
    parser. A deterministic read must not claim completion when the user asks
    for two independent interrogative objectives joined by ``and``.
    """

    normalized = " ".join(utterance.casefold().split())
    clauses = re.split(r"\s+and\s+", normalized)
    question_starts = (
        "what ",
        "which ",
        "who ",
        "when ",
        "where ",
        "how ",
        "can ",
        "could ",
        "should ",
        "would ",
        "is ",
        "are ",
        "do ",
        "does ",
    )
    question_clause_count = 0
    for clause in clauses:
        if clause.startswith(question_starts):
            question_clause_count += 1
    return question_clause_count >= 2


def is_task_destination_request(utterance: str) -> bool:
    """Recognize an explicit task-list destination for action conflict checks."""

    normalized = " ".join(utterance.casefold().split())
    return any(term in normalized for term in ("todo", "to-do", "task list", "things to do"))
