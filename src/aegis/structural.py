"""Optional dependency-parser evidence for requested-effect coverage.

The parser is deliberately outside Core's semantic and authority boundaries.
It exposes conservative structural anchors; Core still validates effect spans,
capability mapping, authorization, execution, verification, and completion.
"""

from __future__ import annotations

import importlib
import re
from typing import Any

from .contracts import StructuralAnchor, StructuralCoverageSignal


class StructuralParserUnavailable(RuntimeError):
    """The optional parser or its operator-supplied model is unavailable."""


class SpacyStructuralParser:
    """Adapt a configured spaCy dependency parser into bounded Core evidence."""

    def __init__(self, model: Any | None = None, model_path: str | None = None) -> None:
        if model is not None:
            self._model = model
            return
        try:
            spacy = importlib.import_module("spacy")
        except ImportError as exc:
            raise StructuralParserUnavailable("spaCy is not installed") from exc
        try:
            self._model = spacy.load(model_path or "en_core_web_sm")
        except Exception as exc:
            raise StructuralParserUnavailable("configured spaCy model is unavailable") from exc

    def parse(self, utterance: str) -> StructuralCoverageSignal:
        doc = self._model(utterance)
        predicates = [token for token in doc if token.pos_ in {"VERB", "AUX"}]
        coordinated_predicates = [token for token in predicates if self._is_effect_predicate(token)]
        anchors: list[StructuralAnchor]
        if len(coordinated_predicates) > 1:
            anchors = [
                StructuralAnchor(
                    source_span=(token.idx, token.idx + len(token.text)), kind="predicate"
                )
                for token in coordinated_predicates
            ]
        else:
            objects = [
                token for token in doc if token.dep_ in {"dobj", "obj", "obl", "iobj", "conj"}
            ]
            coordinated_objects = [token for token in objects if token.dep_ == "conj"]
            if coordinated_objects and objects:
                heads = [objects[0], *coordinated_objects]
                anchors = [
                    StructuralAnchor(
                        source_span=(token.idx, token.idx + len(token.text)), kind="object"
                    )
                    for token in heads
                ]
            else:
                roots = [token for token in doc if token.dep_ == "ROOT"]
                if not roots:
                    raise StructuralParserUnavailable("parser produced no structural root")
                root = roots[0]
                anchors = [
                    StructuralAnchor(
                        source_span=(root.idx, root.idx + len(root.text)), kind="predicate"
                    )
                ]
        negation_spans = [
            (token.idx, token.idx + len(token.text)) for token in doc if token.dep_ == "neg"
        ]
        # Dependency models can attach a contrastive correction such as
        # "wait, I meant ..." to one root and omit the superseded clause.
        # Preserve that deterministic safety signal so a single-action fast
        # path cannot silently mutate the wrong referent.
        for match in re.finditer(
            r"\b(?:wait\s*,?\s*i\s+meant|actually\s*,?\s+i\s+meant|no\s*,?\s+i\s+meant)\b",
            utterance,
            flags=re.IGNORECASE,
        ):
            negation_spans.append(match.span())
        return StructuralCoverageSignal(
            anchors=tuple(anchors), negation_spans=tuple(negation_spans)
        )

    @staticmethod
    def _is_effect_predicate(token: Any) -> bool:
        """Select bounded independent predicates without treating framing as effects."""

        if token.dep_ in {"ROOT", "conj"}:
            return True
        if token.dep_ not in {"advcl", "ccomp", "dep", "relcl", "xcomp"}:
            return False
        children = tuple(getattr(token, "children", ()))
        if any(child.dep_ == "cc" for child in children):
            return False
        if any(child.dep_ in {"dobj", "obj", "obl", "iobj"} for child in children):
            return True
        return sum(child.dep_ in {"nsubj", "csubj"} for child in children) > 1
