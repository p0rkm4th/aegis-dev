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
            "id": "calendar-conflict-inspection",
            "label": "Calendar conflict inspection",
            "description": "Inspect overlapping external calendar events without changing them.",
            "surfaces": ("Calendar",),
            "authority": "calendar.read only; no event mutation or authorization",
        },
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
            "id": "document-search-to-communication",
            "label": "Document search → Communication",
            "description": (
                "Send bounded matches from authorized documents to one approved destination."
            ),
            "surfaces": ("Documents", "Communications"),
            "authority": (
                "authorized document search is fixed before communications send; explicit approved "
                "destination and Core authorization are required, and provider delivery "
                "remains distinct"
            ),
        },
        {
            "id": "research-to-workspace",
            "label": "Research → Workspace",
            "description": "Preserve sourced public research as non-authoritative workspace notes.",
            "surfaces": ("Research", "Workspace"),
            "authority": "external evidence remains non-canonical; Core authorization required",
        },
        {
            "id": "capability-need-to-research-workspace",
            "label": "CapabilityNeed → Research → Workspace",
            "description": (
                "Research a bounded candidate path for an unresolved requirement and preserve "
                "the sourced result as non-authoritative Workspace notes."
            ),
            "surfaces": ("Objectives", "Research", "Workspace"),
            "authority": (
                "investigation is owner-selected and read-only; public evidence remains external, "
                "and no installation, enablement, approval, or execution authority is granted"
            ),
        },
        {
            "id": "capability-need-research-workspace-to-communication",
            "label": "CapabilityNeed → Research → Workspace → Communication",
            "description": (
                "Send the owner-selected, sourced candidate notes from a verified Workspace "
                "artifact to the approved communication destination."
            ),
            "surfaces": ("Objectives", "Research", "Workspace", "Communications"),
            "authority": (
                "each read/write/send step requires separate Core authorization; the original "
                "Need remains open and provider acceptance does not imply delivery"
            ),
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
            "id": "calendar-task-attention-to-workspace",
            "label": "Calendar + Tasks attention → Workspace",
            "description": "Save bounded Calendar + Tasks attention as a verified scoped report.",
            "surfaces": ("Calendar", "Tasks", "Workspace"),
            "authority": (
                "calendar/task reads + scoped workspace write; no task or calendar mutation; "
                "Core authorization and independent Workspace verification required"
            ),
        },
        {
            "id": "workspace-artifact-copy",
            "label": "Workspace artifact copy",
            "description": (
                "Copy an authorized Workspace file into a new independently verified artifact."
            ),
            "surfaces": ("Workspace",),
            "authority": (
                "Principal-scoped Workspace read + scoped Workspace write; source content is "
                "fixed before mutation and independently verified"
            ),
        },
        {
            "id": "workspace-artifact-append",
            "label": "Workspace artifact append",
            "description": "Append explicit content to an existing scoped Workspace artifact.",
            "surfaces": ("Workspace",),
            "authority": (
                "Principal-scoped Workspace read + write; content and path remain explicitly "
                "grounded, and Core independently verifies the resulting file"
            ),
        },
        {
            "id": "workspace-append-to-communication",
            "label": "Workspace append → Communication",
            "description": "Update a scoped artifact, then send its exact resulting content.",
            "surfaces": ("Workspace", "Communications"),
            "authority": (
                "Principal-scoped Workspace read/write + communications send; append and send "
                "retain separate Core authorization, and provider acceptance does not imply "
                "delivery"
            ),
        },
        {
            "id": "homelab-health-to-research",
            "label": "Homelab health → Research",
            "description": "Research likely causes for an authorized observed service condition.",
            "surfaces": ("Systems", "Research"),
            "authority": (
                "authorized health observation + bounded public evidence; "
                "research is non-canonical "
                "and no restart or mutation authority is granted"
            ),
        },
        {
            "id": "homelab-health-to-task",
            "label": "Homelab health → Task",
            "description": (
                "Turn an authorized unhealthy service observation into an explicit "
                "investigation task request."
            ),
            "surfaces": ("Systems", "Tasks"),
            "authority": (
                "health is read-only context; task creation still requires explicit owner intent, "
                "normal Core authorization, and canonical task readback"
            ),
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
            "id": "public-holidays-to-communication",
            "label": "Public holidays → Communication",
            "description": "Send configured public-holiday evidence to one approved destination.",
            "surfaces": ("Calendar", "Communications"),
            "authority": (
                "holiday data remains external and non-canonical; configured country/year scope, "
                "Core authorization, and provider acceptance are distinct from delivery"
            ),
        },
        {
            "id": "public-holidays-to-workspace",
            "label": "Public holidays → Workspace",
            "description": "Preserve bounded public-holiday evidence as a verified scoped report.",
            "surfaces": ("Calendar", "Workspace"),
            "authority": (
                "public calendar evidence + scoped Workspace write; evidence remains external "
                "and non-canonical, with Core authorization and independent verification required"
            ),
        },
        {
            "id": "air-quality-to-workspace",
            "label": "Air quality → Workspace",
            "description": (
                "Preserve bounded public air-quality evidence as a verified scoped report."
            ),
            "surfaces": ("Air quality", "Workspace"),
            "authority": (
                "public air-quality evidence + scoped Workspace write; evidence remains external "
                "and non-canonical, with Core authorization and independent verification required"
            ),
        },
        {
            "id": "air-quality-workspace-to-communication",
            "label": "Air quality → Workspace → Communication",
            "description": (
                "Send a verified air-quality Workspace artifact to the explicitly approved "
                "communication destination."
            ),
            "surfaces": ("Air quality", "Workspace", "Communications"),
            "authority": (
                "air-quality evidence remains external; Workspace read and communications send "
                "require separate Core authorization and provider delivery is not implied"
            ),
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
            "id": "weather-to-communication",
            "label": "Weather → Communication",
            "description": "Send a bounded public forecast to one approved destination.",
            "surfaces": ("Weather", "Communications"),
            "authority": (
                "public weather read + communications send; forecast remains non-canonical, "
                "explicit approved destination and Core authorization required"
            ),
        },
        {
            "id": "document-to-communication",
            "label": "Document → Communication",
            "description": "Send one authorized document to one approved destination.",
            "surfaces": ("Documents", "Communications"),
            "authority": (
                "authorized document read + communications send; explicit approved destination "
                "and Core authorization required"
            ),
        },
        {
            "id": "homelab-health-to-communication",
            "label": "Homelab health → Communication",
            "description": (
                "Send one bounded authorized service-health observation to one approved "
                "destination."
            ),
            "surfaces": ("Systems", "Communications"),
            "authority": (
                "authorized Homelab health read + communications send; explicit approved "
                "destination "
                "and Core authorization required; no restart authority"
            ),
        },
        {
            "id": "workspace-artifact-to-communication",
            "label": "Workspace artifact → Communication",
            "description": (
                "Send one Principal-scoped Workspace artifact to one approved destination."
            ),
            "surfaces": ("Workspace", "Communications"),
            "authority": (
                "Principal-scoped Workspace read + communications send; explicit approved "
                "destination and Core authorization required"
            ),
        },
        {
            "id": "device-state-to-communication",
            "label": "Device state → Communication",
            "description": (
                "Send a bounded authorized device-state snapshot to one approved destination."
            ),
            "surfaces": ("Devices", "Communications"),
            "authority": (
                "authorized device read + communications send; explicit approved destination "
                "and Core authorization required; no device mutation authority"
            ),
        },
        {
            "id": "tasks-to-communication",
            "label": "Tasks → Communication",
            "description": (
                "Send a bounded canonical open-task snapshot to one approved destination."
            ),
            "surfaces": ("Tasks", "Communications"),
            "authority": (
                "authorized task read + communications send; explicit approved destination "
                "and Core authorization required; provider delivery remains distinct"
            ),
        },
        {
            "id": "household-chores-to-communication",
            "label": "Household chores → Communication",
            "description": "Send the canonical open chore snapshot to one approved destination.",
            "surfaces": ("Household", "Communications"),
            "authority": (
                "canonical chore state remains read-only; Core fixes the authorized snapshot "
                "before send and provider acceptance remains distinct from delivery"
            ),
        },
        {
            "id": "household-obligations-to-communication",
            "label": "Household obligations → Communication",
            "description": (
                "Send the canonical open obligation snapshot to one approved destination."
            ),
            "surfaces": ("Household", "Communications"),
            "authority": (
                "canonical obligation state remains read-only; Core fixes the authorized snapshot "
                "before send and provider acceptance remains distinct from delivery"
            ),
        },
        {
            "id": "completed-tasks-to-workspace",
            "label": "Completed tasks → Workspace",
            "description": (
                "Preserve canonical completed-task history as a verified scoped artifact."
            ),
            "surfaces": ("Tasks", "Workspace"),
            "authority": (
                "completed task read + scoped Workspace write; no task mutation, Core "
                "authorization and independent verification required"
            ),
        },
        {
            "id": "household-chores-to-workspace",
            "label": "Household chores → Workspace",
            "description": (
                "Preserve canonical open household chores as a verified scoped artifact."
            ),
            "surfaces": ("Household", "Workspace"),
            "authority": (
                "household read + scoped Workspace write; no chore mutation, Core authorization "
                "and independent verification required"
            ),
        },
        {
            "id": "groceries-to-workspace",
            "label": "Groceries → Workspace",
            "description": "Preserve the canonical grocery list as a verified scoped artifact.",
            "surfaces": ("Food", "Workspace"),
            "authority": (
                "grocery read + scoped Workspace write; no grocery mutation, Core authorization "
                "and independent verification required"
            ),
        },
        {
            "id": "calendar-task-attention-to-communication",
            "label": "Calendar + Tasks attention → Communication",
            "description": (
                "Send bounded task attention before calendar events to one approved destination."
            ),
            "surfaces": ("Calendar", "Tasks", "Communications"),
            "authority": (
                "calendar/task reads + communications send; explicit approved destination "
                "and Core authorization required; no task or calendar mutation"
            ),
        },
        {
            "id": "today-to-communication",
            "label": "Today → Communication",
            "description": (
                "Send a bounded canonical personal/household brief to one approved destination."
            ),
            "surfaces": ("Today", "Communications"),
            "authority": (
                "canonical task/household reads + communications send; explicit approved "
                "destination and Core authorization required; external evidence is excluded"
            ),
        },
        {
            "id": "today-to-workspace",
            "label": "Today → Workspace",
            "description": (
                "Save canonical personal and household Today state as a verified report."
            ),
            "surfaces": ("Today", "Workspace"),
            "authority": (
                "canonical task/household reads + scoped Workspace write; Core authorization "
                "and independent verification required; external evidence is excluded"
            ),
        },
        {
            "id": "today-workspace-to-communication",
            "label": "Today → Workspace → Communication",
            "description": (
                "Preserve the canonical Today brief as a verified artifact, then send that "
                "artifact to one approved destination."
            ),
            "surfaces": ("Today", "Workspace", "Communications"),
            "authority": (
                "canonical task/household reads + scoped Workspace write/read + communications "
                "send; each step requires its own Core authorization, and provider acceptance "
                "does not imply delivery"
            ),
        },
        {
            "id": "groceries-workspace-to-communication",
            "label": "Groceries → Workspace → Communication",
            "description": (
                "Preserve the canonical grocery list as a verified artifact, then send that "
                "artifact to one approved destination."
            ),
            "surfaces": ("Food", "Workspace", "Communications"),
            "authority": (
                "canonical grocery read + scoped Workspace write/read + communications send; "
                "each step requires its own Core authorization, and provider acceptance does "
                "not imply delivery"
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
            "id": "calendar-to-communication-send",
            "label": "Calendar → Communication",
            "description": "Send an authorized calendar snapshot to one approved destination.",
            "surfaces": ("Calendar", "Communications"),
            "authority": (
                "calendar read + communications send; explicit approved destination and Core "
                "authorization required; provider delivery remains distinct"
            ),
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
