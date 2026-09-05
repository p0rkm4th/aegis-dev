"""Provider-neutral authorized document reads for bounded owner workflows."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    text: str
    source: str = "authorized_document"


class DocumentProvider(Protocol):
    def list_documents(self) -> tuple[Document, ...]: ...


@dataclass(frozen=True)
class FixtureDocumentProvider:
    documents: tuple[Document, ...] = ()

    def list_documents(self) -> tuple[Document, ...]:
        return self.documents


def configured_document_provider() -> FixtureDocumentProvider:
    """Load only an explicit bounded fixture; absence means no documents."""

    raw = os.environ.get("AEGIS_DOCUMENT_FIXTURE_JSON")
    if not raw:
        return FixtureDocumentProvider()
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("document fixture configuration is invalid") from exc
    if not isinstance(values, list) or len(values) > 50:
        raise ValueError("document fixture must be a list of at most 50 documents")
    documents: list[Document] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("document fixture entries must be objects")
        document = Document(
            document_id=str(value.get("document_id", "")),
            title=str(value.get("title", "")),
            text=str(value.get("text", ""))[:20_000],
            source=str(value.get("source", "configured_document_fixture")),
        )
        documents.append(document)
    return FixtureDocumentProvider(tuple(documents))


def documents_evidence(documents: tuple[Document, ...]) -> dict[str, object]:
    return {
        "source": "authorized_document_fixture",
        "documents": [
            {
                "document_id": document.document_id,
                "title": document.title,
                "text": document.text[:20_000],
                "source": document.source,
            }
            for document in documents[:50]
        ],
    }
