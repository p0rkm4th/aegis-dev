"""Optional dependency-parser evidence for requested-effect coverage.

The parser is deliberately outside Core's semantic and authority boundaries.
It exposes conservative structural anchors; Core still validates effect spans,
capability mapping, authorization, execution, verification, and completion.
"""

from __future__ import annotations

import importlib
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
        coordinated_predicates = [token for token in predicates if token.dep_ in {"ROOT", "conj"}]
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
        return StructuralCoverageSignal(anchors=tuple(anchors))
