"""Bounded cross-capability compositions owned by Core-facing orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from .calendar import CalendarEvent
from .documents import DocumentProvider
from .research import ResearchAnswer
from .workspace import WorkspaceManager


def available_compositions() -> tuple[dict[str, object], ...]:
    """Owner-facing workflow metadata; this declares no authority or execution."""

    return (
        {
            "id": "documents-to-workspace",
            "label": "Document → Workspace",
            "description": "Export one authorized document into a verified scoped artifact.",
            "surfaces": ("Documents", "Workspace"),
            "authority": "read document + write scoped workspace; Core authorization required",
        },
        {
            "id": "document-search-to-workspace",
            "label": "Document search → Workspace",
            "description": "Save bounded matches from authorized documents as a verified artifact.",
            "surfaces": ("Documents", "Workspace"),
            "authority": "document read + scoped workspace write; Core authorization required",
        },
        {
            "id": "research-to-workspace",
            "label": "Research → Workspace",
            "description": "Preserve sourced public research as non-authoritative workspace notes.",
            "surfaces": ("Research", "Workspace"),
            "authority": "external evidence remains non-canonical; Core authorization required",
        },
        {
            "id": "calendar-task-attention",
            "label": "Calendar + Tasks attention",
            "description": (
                "Find canonical tasks due before an agenda event without mutating either source."
            ),
            "surfaces": ("Calendar", "Tasks"),
            "authority": "read-only narrowing; no task mutation or authorization",
        },
        {
            "id": "weather-to-task-followup",
            "label": "Weather → Task follow-up",
            "description": "Turn a public weather check into an explicit owner task request.",
            "surfaces": ("Weather", "Tasks"),
            "authority": (
                "public evidence informs a proposal; normal task authorization remains required"
            ),
        },
        {
            "id": "air-quality-to-task-followup",
            "label": "Air quality → Task follow-up",
            "description": "Turn a public air-quality check into an explicit owner task request.",
            "surfaces": ("Air quality", "Tasks"),
            "authority": (
                "public evidence informs a proposal; normal task authorization remains required"
            ),
        },
        {
            "id": "calendar-holidays-to-today",
            "label": "Calendar + Public holidays → Today",
            "description": (
                "Bring live public-holiday context into the owner's daily planning view."
            ),
            "surfaces": ("Calendar", "Today"),
            "authority": "external dates are read-only context; no personal event is created",
        },
        {
            "id": "weather-to-today",
            "label": "Weather → Today",
            "description": "Bring a bounded public forecast into the owner's daily planning view.",
            "surfaces": ("Weather", "Today"),
            "authority": (
                "public evidence is read-only context; no canonical personal state is changed"
            ),
        },
        {
            "id": "weather-to-workspace",
            "label": "Weather → Workspace",
            "description": "Save a bounded public forecast as a verified scoped report.",
            "surfaces": ("Weather", "Workspace"),
            "authority": (
                "public evidence remains non-canonical; Core authorization and Workspace "
                "verification are required"
            ),
        },
        {
            "id": "public-holiday-to-task",
            "label": "Public holiday → Task",
            "description": "Prepare an explicit task for an authorized public-holiday date.",
            "surfaces": ("Calendar", "Tasks"),
            "authority": (
                "external date informs a proposal; normal task authorization remains required"
            ),
        },
        {
            "id": "calendar-to-workspace",
            "label": "Calendar → Workspace",
            "description": (
                "Save an authorized calendar snapshot as an independently verified artifact."
            ),
            "surfaces": ("Calendar", "Workspace"),
            "authority": "calendar read + scoped workspace write; Core authorization required",
        },
        {
            "id": "calendar-to-communication-draft",
            "label": "Calendar → Communication draft",
            "description": "Prepare an unsent calendar snapshot for an explicit recipient.",
            "surfaces": ("Calendar", "Communications", "Workspace"),
            "authority": "calendar read + draft + scoped workspace write; no send authority",
        },
        {
            "id": "device-to-workspace",
            "label": "Device → Workspace",
            "description": (
                "Preserve an authorized device-state snapshot in a verified scoped artifact."
            ),
            "surfaces": ("Devices", "Workspace"),
            "authority": "device read + scoped workspace write; Core authorization required",
        },
        {
            "id": "homelab-to-workspace",
            "label": "Homelab → Workspace",
            "description": "Render authorized Homelab inventory as a verified static page.",
            "surfaces": ("Systems", "Workspace"),
            "authority": "Homelab read + scoped workspace write; Core authorization required",
        },
        {
            "id": "homelab-health-to-workspace",
            "label": "Homelab health → Workspace",
            "description": (
                "Preserve bounded service health observations in a verified scoped report."
            ),
            "surfaces": ("Systems", "Workspace"),
            "authority": "Homelab health read + scoped workspace write; no restart authority",
        },
    )


@dataclass(frozen=True)
class DocumentWorkspaceResult:
    """Provenance-preserving result for a document-to-artifact composition."""

    correlation_id: UUID
    document_id: str
    target_path: str
    files: tuple[str, ...]
    validated: bool
    source: str


@dataclass(frozen=True)
class ResearchWorkspaceResult:
    correlation_id: UUID
    target_path: str
    files: tuple[str, ...]
    validated: bool
    source_ids: tuple[str, ...]
    authoritative: bool = False


@dataclass(frozen=True)
class CalendarTaskAttention:
    event_id: str
    event_title: str
    task_titles: tuple[str, ...]


def calendar_to_task_attention(
    events: tuple[CalendarEvent, ...], tasks: tuple[dict[str, object], ...], *, until: datetime
) -> tuple[CalendarTaskAttention, ...]:
    """Join external agenda windows with canonical task deadlines read-only.

    This narrows attention only; it never creates, completes, or authorizes a task.
    """

    attention: list[CalendarTaskAttention] = []
    for event in events:
        if event.starts_at > until:
            continue
        titles: list[str] = []
        for task in tasks:
            title = task.get("title")
            due_at = task.get("due_at")
            if (
                isinstance(title, str)
                and isinstance(due_at, datetime)
                and due_at <= event.starts_at
            ):
                titles.append(title)
        attention.append(CalendarTaskAttention(event.event_id, event.title, tuple(titles)))
    return tuple(attention)


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


def document_summary_to_workspace(
    provider: DocumentProvider,
    workspaces: WorkspaceManager,
    *,
    principal_id: str,
    objective_id: UUID,
    document_id: str,
    target_path: str,
    correlation_id: UUID,
) -> DocumentWorkspaceResult:
    """Write a bounded deterministic summary of one authorized document."""

    document = next(
        (item for item in provider.list_documents() if item.document_id == document_id), None
    )
    if document is None:
        raise ValueError("authorized document is unavailable")
    summary = " ".join(document.text.split())[:500]
    content = f"# Summary: {document.title}\n\n{summary}\n"
    workspace = workspaces.for_objective(principal_id, objective_id)
    artifact = workspace.write_artifact(
        {target_path: content},
        correlation_id,
        lambda current: (
            None if current.read(target_path) == content else "document summary readback mismatch"
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


def document_search_to_workspace(
    provider: DocumentProvider,
    workspaces: WorkspaceManager,
    *,
    principal_id: str,
    objective_id: UUID,
    query: str,
    target_path: str,
    correlation_id: UUID,
) -> DocumentWorkspaceResult:
    """Save bounded authorized document matches into an isolated artifact."""

    needle = query.strip().casefold()
    if not needle:
        raise ValueError("document search query is required")
    matches = [
        document
        for document in provider.list_documents()
        if needle in f"{document.title}\n{document.text}".casefold()
    ][:20]
    lines = [f"# Document search: {query.strip()[:500]}", ""]
    for document in matches:
        lines.extend(
            (
                f"## {document.title} ({document.document_id})",
                "",
                document.text[:500],
                "",
            )
        )
    if not matches:
        lines.append("No authorized documents matched this query.")
    content = "\n".join(lines)
    workspace = workspaces.for_objective(principal_id, objective_id)
    artifact = workspace.write_artifact(
        {target_path: content},
        correlation_id,
        lambda current: (
            None if current.read(target_path) == content else "document search readback mismatch"
        ),
    )
    return DocumentWorkspaceResult(
        correlation_id=correlation_id,
        document_id=f"search:{query.strip()[:100]}",
        target_path=target_path,
        files=artifact.files,
        validated=artifact.validated,
        source="authorized_document_search",
    )


def research_to_workspace(
    answer: ResearchAnswer,
    workspaces: WorkspaceManager,
    *,
    principal_id: str,
    objective_id: UUID,
    target_path: str,
    correlation_id: UUID,
) -> ResearchWorkspaceResult:
    """Preserve a bounded sourced answer as a non-authoritative artifact."""

    sources = "\n".join(f"- [{item.title}]({item.final_url})" for item in answer.evidence.evidence)
    content = f"# Research notes\n\n{answer.text}\n\n## Sources\n{sources}\n"
    workspace = workspaces.for_objective(principal_id, objective_id)
    artifact = workspace.write_artifact(
        {target_path: content},
        correlation_id,
        lambda current: (
            None if current.read(target_path) == content else "research artifact readback mismatch"
        ),
    )
    return ResearchWorkspaceResult(
        correlation_id=correlation_id,
        target_path=target_path,
        files=artifact.files,
        validated=artifact.validated,
        source_ids=tuple(item.source_id for item in answer.evidence.evidence),
    )
