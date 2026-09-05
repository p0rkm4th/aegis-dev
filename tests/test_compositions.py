from datetime import datetime, timezone
from uuid import uuid4

from aegis.compositions import document_to_workspace, research_to_workspace
from aegis.documents import Document, FixtureDocumentProvider
from aegis.research import Evidence, EvidenceSet, KnowledgeSource, ResearchAnswer
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


def test_research_answer_can_be_preserved_as_non_authoritative_workspace_notes(tmp_path) -> None:
    answer = ResearchAnswer(
        text="Use the documented setting.",
        source_kind=KnowledgeSource.EXTERNAL,
        evidence=EvidenceSet(
            query="setting",
            provider_id="fixture",
            evidence=(
                Evidence(
                    source_id="source-1",
                    final_url="https://example.test/docs",
                    title="Documented setting",
                    text="Use the documented setting.",
                    retrieved_at=datetime.now(timezone.utc),
                ),
            ),
            retrieved_at=datetime.now(timezone.utc),
        ),
    )
    result = research_to_workspace(
        answer,
        WorkspaceManager(tmp_path),
        principal_id="alice",
        objective_id=uuid4(),
        target_path="research.md",
        correlation_id=uuid4(),
    )
    assert result.validated is True
    assert result.authoritative is False
    assert result.source_ids == ("source-1",)
