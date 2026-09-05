"""Bounded cross-capability compositions owned by Core-facing orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from .documents import DocumentProvider
from .workspace import WorkspaceManager


@dataclass(frozen=True)
class DocumentWorkspaceResult:
    """Provenance-preserving result for a document-to-artifact composition."""

    correlation_id: UUID
    document_id: str
    target_path: str
    files: tuple[str, ...]
    validated: bool
    source: str


def document_to_workspace(
    provider: DocumentProvider,
    workspaces: WorkspaceManager,
    *,
    principal_id: str,
    objective_id: UUID,
    document_id: str,
    target_path: str,
    correlation_id: UUID,
) -> DocumentWorkspaceResult:
    """Export one authorized document into an isolated verified workspace.

    This composition is intentionally typed and single-document: the caller
    must supply the target path and objective scope, and document text is never
    treated as an instruction or as deployment authority.
    """

    document = next(
        (item for item in provider.list_documents() if item.document_id == document_id), None
    )
    if document is None:
        raise ValueError("authorized document is unavailable")
    workspace = workspaces.for_objective(principal_id, objective_id)
    content = f"# {document.title}\n\n{document.text}"
    artifact = workspace.write_artifact(
        {target_path: content},
        correlation_id,
        lambda current: (
            None if current.read(target_path) == content else "document export readback mismatch"
        ),
    )
    return DocumentWorkspaceResult(
        correlation_id=correlation_id,
        document_id=document.document_id,
        target_path=target_path,
        files=artifact.files,
        validated=artifact.validated,
        source=document.source,
    )
