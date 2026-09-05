"""Provider-neutral authorized document reads for bounded owner workflows."""

from __future__ import annotations

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
