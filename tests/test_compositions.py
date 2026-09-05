from uuid import uuid4

from aegis.compositions import document_to_workspace
from aegis.documents import Document, FixtureDocumentProvider
from aegis.workspace import WorkspaceManager


def test_authorized_document_can_be_exported_to_verified_scoped_workspace(tmp_path) -> None:
    objective_id = uuid4()
    correlation_id = uuid4()
    result = document_to_workspace(
        FixtureDocumentProvider((Document("doc-1", "Starter", "Use bounded settings."),)),
        WorkspaceManager(tmp_path),
        principal_id="alice",
        objective_id=objective_id,
        document_id="doc-1",
        target_path="starter.md",
        correlation_id=correlation_id,
    )
    assert result.validated is True
    assert result.correlation_id == correlation_id
    assert result.files == ("starter.md",)
