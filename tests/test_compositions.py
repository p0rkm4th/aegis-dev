from datetime import datetime, timezone
from uuid import uuid4

from aegis.calendar import CalendarEvent
from aegis.compositions import (
    available_compositions,
    calendar_to_task_attention,
    document_to_workspace,
    research_to_workspace,
)
from aegis.documents import Document, FixtureDocumentProvider
from aegis.research import Evidence, EvidenceSet, KnowledgeSource, ResearchAnswer
from aegis.web import CompositionProjection
from aegis.workspace import WorkspaceManager


def test_homelab_health_workspace_composition_is_owner_visible() -> None:
    composition = next(
        item for item in available_compositions() if item["id"] == "homelab-health-to-workspace"
    )
    assert composition["surfaces"] == ("Systems", "Workspace")
    assert "no restart authority" in composition["authority"]


def test_tasks_to_communication_composition_is_owner_visible() -> None:
    composition = next(
        item for item in available_compositions() if item["id"] == "tasks-to-communication"
    )
    assert composition["surfaces"] == ("Tasks", "Communications")
    assert "approved destination" in composition["authority"]


def test_today_workspace_communication_chain_is_explicitly_non_authoritative() -> None:
    composition = next(
        item
        for item in available_compositions()
        if item["id"] == "today-workspace-to-communication"
    )
    assert composition["surfaces"] == ("Today", "Workspace", "Communications")
    assert "each step requires its own Core authorization" in composition["authority"]
    assert "does not imply delivery" in composition["authority"]


def test_composition_projection_accepts_current_bounded_catalog() -> None:
    projection = CompositionProjection.model_validate({"compositions": available_compositions()})
    assert len(projection.compositions) >= 20


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


def test_calendar_and_tasks_compose_into_read_only_attention() -> None:
    start = datetime(2026, 9, 10, tzinfo=timezone.utc)
    result = calendar_to_task_attention(
        (CalendarEvent("event-1", "Leave town", start),),
        (
            {"title": "Pack bags", "due_at": datetime(2026, 9, 9, tzinfo=timezone.utc)},
            {"title": "Later task", "due_at": datetime(2026, 9, 12, tzinfo=timezone.utc)},
        ),
        until=start,
    )
    assert result[0].task_titles == ("Pack bags",)
