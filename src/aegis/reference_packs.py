"""Small reference Packs used to prove generic Core semantics."""

from __future__ import annotations

import json
import os
import shlex
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from .calendar import (
    CalendarEvent,
    CalendarWriteProvider,
    calendar_events_evidence,
    calendar_snapshot_content,
    configured_calendar_provider,
)
from .communications import (
    CommunicationSendProvider,
    FixtureCommunicationProvider,
    OutboundMessage,
    SendStatus,
    communications_evidence,
)
from .compositions import (
    document_search_to_workspace,
    document_summary_to_workspace,
    document_to_workspace,
    research_to_workspace,
)
from .contracts import (
    ActionCard,
    ActionSpec,
    ArgumentGroundingRule,
    ArgumentProvenanceKind,
    ExecutionRequest,
    Observation,
    Principal,
    VerificationContract,
    VerificationResult,
)
from .devices import (
    DeviceCommand,
    FixtureDeviceGateway,
    HomeAssistantAdapter,
    HomeAssistantRestControlGateway,
    HomeAssistantRestGateway,
    device_states_evidence,
)
from .documents import configured_document_provider, documents_evidence
from .gateway_rpc import (
    CorrelatedRpcClient,
    OpenClawGatewayRpc,
    OpenClawWebSocketChannel,
    RpcProtocolError,
)
from .holidays import configured_holiday_provider, holidays_evidence
from .homelab import PostgresHomelabStore
from .household import PostgresHouseholdStore
from .pack_lifecycle import PackBundle, PackManifest, PackUI
from .research import (
    KnowledgeSource,
    ResearchAnswer,
    ResearchUnavailable,
    SearchRequest,
    configured_research_service,
)
from .weather import configured_weather_provider, weather_evidence
from .workspace import WorkspaceManager, workspace_expected_postcondition


def prepare_reference_action(
    action: ActionSpec, principal: Principal, objective_id: Any, connection: Any = None
) -> ActionSpec:
    """Bind invocation-specific postconditions before a consequential write."""
    args = action.arguments
    files: dict[str, str] | None = None
    if action.action_id == "workspace.artifact.create":
        candidate = args.get("files")
        if isinstance(candidate, dict):
            files = candidate
        elif isinstance(args.get("path"), str) and isinstance(args.get("content"), str):
            files = {args["path"]: args["content"]}
    elif action.action_id == "communication-drafts.messages.draft":
        if args.get("body_source") == "bounded.research":
            values = (
                args.get("recipient"),
                args.get("subject"),
                args.get("query"),
                args.get("target_path"),
            )
            if all(isinstance(value, str) and value.strip() for value in values):
                recipient, subject, query, target_path = cast(tuple[str, str, str, str], values)
                try:
                    evidence = configured_research_service().collect(SearchRequest(query))
                except ResearchUnavailable as exc:
                    raise ValueError(f"bounded research is unavailable: {exc}") from exc
                excerpts = "\n\n".join(
                    f"## {item.title}\n{item.text[:1_200]}\nSource: {item.final_url}"
                    for item in evidence.evidence
                )
                if not excerpts:
                    raise ValueError("bounded research returned no usable evidence")
                body = f"Research notes for: {query}\n\n{excerpts}"
                args = {**args, "body": body}
                files = {
                    target_path: f"# Draft message\n\nTo: {recipient}\n"
                    f"Subject: {subject}\n\n{body}\n"
                }
        values = (
            args.get("recipient"),
            args.get("subject"),
            args.get("body"),
            args.get("target_path"),
        )
        if files is None and all(isinstance(value, str) and value.strip() for value in values):
            recipient, subject, body, target_path = cast(tuple[str, str, str, str], values)
            files = {
                str(target_path): (
                    f"# Draft message\n\nTo: {recipient}\nSubject: {subject}\n\n{body}\n"
                )
            }
    elif action.action_id == "communications.messages.send":
        if args.get("body_source") == "canonical.groceries" and connection is not None:
            items = PostgresHouseholdStore(connection).list_groceries(principal)
            body = "Grocery list:\n" + "\n".join(f"- {item}" for item in items)
            args = {**args, "body": body}
        target, message_body = args.get("target"), args.get("body")
        channel, account = args.get("channel", "default"), args.get("account")
        if (
            isinstance(target, str)
            and isinstance(message_body, str)
            and isinstance(channel, str)
            and (account is None or isinstance(account, str))
        ):
            verification = action.verification or VerificationContract(kind="custom")
            return action.model_copy(
                update={
                    "arguments": args,
                    "verification": verification.model_copy(
                        update={
                            "expected": {
                                "version": 1,
                                "principal_id": principal.id,
                                "objective_id": str(objective_id),
                                "target": target,
                                "channel": channel,
                                "account": account,
                                "body": message_body,
                            }
                        }
                    ),
                }
            )
    elif action.action_id == "calendar.events.create":
        title, starts_at, ends_at = (
            args.get("title"),
            args.get("starts_at"),
            args.get("ends_at"),
        )
        if (
            isinstance(title, str)
            and isinstance(starts_at, str)
            and (ends_at is None or isinstance(ends_at, str))
        ):
            verification = action.verification or VerificationContract(kind="custom")
            return action.model_copy(
                update={
                    "verification": verification.model_copy(
                        update={
                            "expected": {
                                "version": 1,
                                "principal_id": principal.id,
                                "objective_id": str(objective_id),
                                "title": title,
                                "starts_at": starts_at,
                                "ends_at": ends_at,
                            }
                        }
                    )
                }
            )
    elif action.action_id == "calendar.events.cancel":
        event_id = args.get("event_id")
        if isinstance(event_id, str) and event_id.strip():
            verification = action.verification or VerificationContract(kind="custom")
            return action.model_copy(
                update={
                    "verification": verification.model_copy(
                        update={
                            "expected": {
                                "version": 1,
                                "principal_id": principal.id,
                                "objective_id": str(objective_id),
                                "event_id": event_id,
                                "exists": False,
                            }
                        }
                    )
                }
            )
    elif action.action_id == "calendar.events.update":
        event_id, title, starts_at, ends_at = (
            args.get("event_id"),
            args.get("title"),
            args.get("starts_at"),
            args.get("ends_at"),
        )
        if (
            isinstance(event_id, str)
            and event_id.strip()
            and isinstance(title, str)
            and title.strip()
            and isinstance(starts_at, str)
            and starts_at.strip()
            and (ends_at is None or isinstance(ends_at, str))
        ):
            verification = action.verification or VerificationContract(kind="custom")
            return action.model_copy(
                update={
                    "verification": verification.model_copy(
                        update={
                            "expected": {
                                "version": 1,
                                "principal_id": principal.id,
                                "objective_id": str(objective_id),
                                "event_id": event_id,
                                "title": title,
                                "starts_at": starts_at,
                                "ends_at": ends_at,
                            }
                        }
                    )
                }
            )
    elif action.action_id == "device-controls.devices.command.execute":
        entity_id = args.get("entity_id")
        expected_state = args.get("expected_state")
        if isinstance(entity_id, str) and isinstance(expected_state, str):
            verification = action.verification or VerificationContract(kind="custom")
            return action.model_copy(
                update={
                    "verification": verification.model_copy(
                        update={
                            "expected": {
                                "version": 1,
                                "principal_id": principal.id,
                                "objective_id": str(objective_id),
                                "entity_id": entity_id,
                                "expected_state": expected_state,
                            }
                        }
                    )
                }
            )
    elif action.action_id == "device-reports.devices.snapshot_to_workspace":
        snapshot_path = args.get("target_path")
        if isinstance(snapshot_path, str) and snapshot_path.strip():
            snapshot = (
                DeviceStatesExecutor()
                .execute(
                    ExecutionRequest(
                        objective_id=objective_id,
                        action_id=uuid4(),
                        action=action,
                        idempotency_key=f"prepare-device-snapshot-{objective_id}",
                    )
                )
                .evidence.get("states", [])
            )
            content = _device_snapshot_content(snapshot)
            verification = action.verification or VerificationContract(kind="custom")
            return action.model_copy(
                update={
                    "verification": verification.model_copy(
                        update={
                            "expected": workspace_expected_postcondition(
                                principal.id, objective_id, {snapshot_path: content}
                            )
                        }
                    )
                }
            )
    elif action.action_id == "calendar-reports.events.snapshot_to_workspace":
        snapshot_path = args.get("target_path")
        if isinstance(snapshot_path, str) and snapshot_path.strip():
            content = calendar_snapshot_content(configured_calendar_provider().list_events())
            verification = action.verification or VerificationContract(kind="custom")
            return action.model_copy(
                update={
                    "verification": verification.model_copy(
                        update={
                            "expected": workspace_expected_postcondition(
                                principal.id, objective_id, {snapshot_path: content}
                            )
                        }
                    )
                }
            )
    elif action.action_id == "homelab-reports.inventory.to_workspace":
        report_target = args.get("target_path")
        if isinstance(report_target, str) and report_target.strip() and connection is not None:
            page = _homelab_page_files(connection, principal)
            files = {report_target: page["index.html"], "style.css": page["style.css"]}
    elif action.action_id == "homelab-reports.health.to_workspace":
        report_target = args.get("target_path")
        if isinstance(report_target, str) and report_target.strip() and connection is not None:
            page = _homelab_health_report_files(connection, principal)
            files = {report_target: page["index.html"], "style.css": page["style.css"]}
    elif action.action_id == "calendar-communications.events.draft":
        draft_recipient, draft_target = args.get("recipient"), args.get("target_path")
        if (
            isinstance(draft_recipient, str)
            and draft_recipient.strip()
            and isinstance(draft_target, str)
            and draft_target.strip()
        ):
            body = calendar_snapshot_content(configured_calendar_provider().list_events())
            files = {
                draft_target: (
                    f"# Draft message\n\nTo: {draft_recipient}\n"
                    f"Subject: Calendar snapshot\n\n{body}"
                )
            }
    elif action.action_id == "documents.export_to_workspace":
        document_id = args.get("document_id")
        document_target_path = args.get("target_path")
        if isinstance(document_id, str) and isinstance(document_target_path, str):
            document = next(
                (
                    item
                    for item in configured_document_provider().list_documents()
                    if item.document_id == document_id
                ),
                None,
            )
            if document is not None:
                files = {document_target_path: f"# {document.title}\n\n{document.text}"}
    elif action.action_id == "documents.summarize_to_workspace":
        document_id = args.get("document_id")
        document_target_path = args.get("target_path")
        if isinstance(document_id, str) and isinstance(document_target_path, str):
            document = next(
                (
                    item
                    for item in configured_document_provider().list_documents()
                    if item.document_id == document_id
                ),
                None,
            )
            if document is not None:
                summary = " ".join(document.text.split())[:500]
                files = {document_target_path: f"# Summary: {document.title}\n\n{summary}\n"}
    elif action.action_id == "documents.search_to_workspace":
        search_query, search_target_path = args.get("query"), args.get("target_path")
        if isinstance(search_query, str) and isinstance(search_target_path, str):
            matches = [
                item
                for item in configured_document_provider().list_documents()
                if search_query.strip().casefold() in f"{item.title}\n{item.text}".casefold()
            ][:20]
            lines = [f"# Document search: {search_query.strip()[:500]}", ""]
            for document in matches:
                lines.extend(
                    (f"## {document.title} ({document.document_id})", "", document.text[:500], "")
                )
            if not matches:
                lines.append("No authorized documents matched this query.")
            files = {search_target_path: "\n".join(lines)}
    if files is None:
        return action
    expectation = workspace_expected_postcondition(principal.id, objective_id, files)
    verification = action.verification or VerificationContract(kind="custom")
    return action.model_copy(
        update={
            "arguments": args,
            "verification": verification.model_copy(update={"expected": expectation}),
        }
    )


def _verify_workspace_expectation(
    principal: Principal, contract: VerificationContract
) -> tuple[bool, str]:
    expected = contract.expected
    if not isinstance(expected, dict):
        return False, "workspace postcondition is missing"
    if expected.get("principal_id") != principal.id:
        return False, "workspace principal scope does not match"
    try:
        objective_id = UUID(str(expected["objective_id"]))
        files = expected["files"]
        if not isinstance(files, dict) or not all(
            isinstance(path, str) and isinstance(digest, str) for path, digest in files.items()
        ):
            return False, "workspace postcondition files are invalid"
        workspace = WorkspaceManager(
            Path(os.environ.get("AEGIS_WORKSPACE_ROOT", "/tmp/aegis-owner-workspaces"))
        ).for_objective(principal.id, objective_id)
        return workspace.verify_expected_files(cast(dict[str, str], files))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return False, f"workspace postcondition unavailable: {exc}"


@dataclass(frozen=True)
class _ReferencePackSpec:
    pack_id: str
    version: str
    cards: tuple[ActionCard, ...]


def _wait_for_terminal_ready(channel: OpenClawWebSocketChannel) -> None:
    """Drain OpenClaw shell initialization before sending PTY input.

    ``terminal.open`` acknowledges session creation before the shell has
    finished its startup output.  Sending input during that window can be
    echoed and then discarded.  OpenClaw's shell integration emits the OSC
    prompt-start marker when the interactive prompt is ready; this remains a
    transport concern and is shared by all Gateway-backed Packs.
    """
    for _ in range(32):
        event = channel.receive_event("terminal.data")
        data = str(event.get("data", ""))
        if "\x1b]133;B" in data:
            return
    raise RpcProtocolError("OpenClaw terminal did not become ready")


def _unknown_gateway_observation(request: ExecutionRequest) -> Observation:
    """Record an ambiguous Gateway outcome without permitting blind replay."""

    return Observation(
        execution_id=request.action_id,
        evidence={
            "gateway": "openclaw",
            "outcome": "unknown",
            "idempotency_key": request.idempotency_key,
        },
        command_succeeded=False,
    )


def _reference_pack_specs() -> tuple[_ReferencePackSpec, ...]:
    return (
        _ReferencePackSpec(
            "tasks",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="tasks.create",
                        capability="tasks.create",
                        required_permissions=("tasks.write",),
                        verification=VerificationContract(kind="readback"),
                    ),
                    summary=(
                        "Personal task creation: add work to the user's own to-do list or "
                        "reminders; this is a task, not a shared household chore"
                    ),
                    relevance=1,
                    argument_keys=("title", "due_at"),
                    argument_descriptions={
                        "title": (
                            "the complete user-described task or reminder, excluding list "
                            "destination wording"
                        ),
                        "due_at": "a user-supplied deadline, if one was clearly stated",
                    },
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="tasks.complete",
                        capability="tasks.complete",
                        required_permissions=("tasks.write",),
                        verification=VerificationContract(kind="readback"),
                    ),
                    summary="Complete, finish, or close a named task",
                    relevance=1,
                    argument_keys=("title",),
                    argument_descriptions={
                        "title": "the exact or uniquely matching existing task title"
                    },
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="tasks.chores.create",
                        capability="tasks.chores.create",
                        required_permissions=("tasks.write",),
                        verification=VerificationContract(kind="readback"),
                    ),
                    summary="Create a shared household chore someone needs to do",
                    relevance=1,
                    argument_keys=("title",),
                    argument_descriptions={"title": "the complete chore description"},
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="tasks.chores.complete",
                        capability="tasks.chores.complete",
                        required_permissions=("tasks.write",),
                        verification=VerificationContract(kind="readback"),
                    ),
                    summary="Complete, finish, or close a shared household chore",
                    relevance=1,
                    argument_keys=("title",),
                    argument_descriptions={
                        "title": "the exact or uniquely matching existing chore title"
                    },
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="tasks.events.create",
                        capability="tasks.events.create",
                        required_permissions=("tasks.write",),
                        verification=VerificationContract(kind="readback"),
                    ),
                    summary=(
                        "Schedule or book a shared household event or appointment with a "
                        "title and time"
                    ),
                    relevance=1,
                    argument_keys=("title", "starts_at"),
                    argument_descriptions={
                        "title": "the event or appointment description",
                        "starts_at": "the user-supplied event start time",
                    },
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="tasks.list",
                        capability="tasks.read",
                        required_permissions=("tasks.read",),
                        verification=VerificationContract(kind="readback"),
                    ),
                    summary="Show tasks",
                    relevance=1,
                ),
            ),
        ),
        _ReferencePackSpec(
            "kitchen",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="kitchen.groceries.add",
                        capability="kitchen.groceries.write",
                        required_permissions=("kitchen.write",),
                        verification=VerificationContract(kind="readback"),
                    ),
                    summary="Add an item to groceries",
                    relevance=1,
                    argument_keys=("item",),
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="kitchen.groceries.list",
                        capability="kitchen.groceries.read",
                        required_permissions=("kitchen.read",),
                        verification=VerificationContract(kind="readback"),
                    ),
                    summary=(
                        "Read the authorized grocery shopping list: what to buy, pick up, "
                        "bring home, or get at the store"
                    ),
                    relevance=1,
                    semantic_scope="kitchen.shopping_list",
                ),
            ),
        ),
        _ReferencePackSpec(
            "homelab",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="homelab.service.restart",
                        capability="homelab.service.restart",
                        required_permissions=("homelab.service.restart",),
                        verification=VerificationContract(kind="health"),
                    ),
                    summary="Restart a service and verify health",
                    relevance=1,
                    argument_keys=("service",),
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="homelab.service.health",
                        capability="homelab.service.health",
                        required_permissions=("homelab.read",),
                        verification=VerificationContract(kind="health"),
                    ),
                    summary="Read and independently verify an authorized service health endpoint",
                    relevance=1,
                    argument_keys=("service",),
                    argument_grounding={
                        "service": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        )
                    },
                ),
            ),
        ),
        _ReferencePackSpec(
            "homelab-reports",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="homelab-reports.inventory.to_workspace",
                        capability="homelab-reports.inventory.to_workspace",
                        required_permissions=("homelab.read", "workspace.write"),
                        verification=VerificationContract(kind="custom"),
                    ),
                    summary="Create a verified static page from authorized Homelab inventory",
                    relevance=1,
                    argument_keys=("target_path",),
                    argument_grounding={
                        "target_path": ArgumentGroundingRule(
                            permitted_provenance=(
                                ArgumentProvenanceKind.EXPLICIT_UTTERANCE,
                                ArgumentProvenanceKind.DETERMINISTIC_DERIVATION,
                            ),
                            approved_derivations=("reference.homelab_page_target.v1",),
                        )
                    },
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="homelab-reports.health.to_workspace",
                        capability="homelab-reports.health.to_workspace",
                        required_permissions=("homelab.read", "workspace.write"),
                        verification=VerificationContract(kind="custom"),
                    ),
                    summary="Create a verified Workspace health snapshot from authorized services",
                    relevance=1,
                    argument_keys=("target_path",),
                    argument_grounding={
                        "target_path": ArgumentGroundingRule(
                            permitted_provenance=(
                                ArgumentProvenanceKind.EXPLICIT_UTTERANCE,
                                ArgumentProvenanceKind.DETERMINISTIC_DERIVATION,
                            ),
                            approved_derivations=("reference.homelab_page_target.v1",),
                        )
                    },
                ),
            ),
        ),
        _ReferencePackSpec(
            "network",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="network.probe",
                        capability="network.probe",
                        required_permissions=("network.read",),
                        verification=VerificationContract(kind="health"),
                    ),
                    summary="Probe an authorized network target",
                    relevance=1,
                    argument_keys=("address", "scope_id", "port"),
                    argument_grounding={
                        "address": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        ),
                        "scope_id": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        ),
                        "port": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        ),
                    },
                ),
            ),
        ),
        _ReferencePackSpec(
            "communications",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="communications.messages.list",
                        capability="communications.messages.list",
                        required_permissions=("communications.read",),
                        verification=VerificationContract(kind="readback"),
                    ),
                    summary="Read authorized messages without sending or mutating communications",
                    relevance=1,
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="communications.messages.send",
                        capability="communications.messages.send",
                        required_permissions=("communications.send",),
                        verification=VerificationContract(kind="custom"),
                    ),
                    summary="Send an explicitly addressed message through an authorized provider",
                    relevance=1,
                    argument_keys=("target", "body", "channel", "account", "body_source"),
                    argument_grounding={
                        key: ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        )
                        for key in ("target", "body", "channel", "account")
                    }
                    | {
                        "body_source": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.DETERMINISTIC_DERIVATION,),
                            approved_derivations=(
                                "reference.communication_body_from_groceries.v1",
                            ),
                        )
                    },
                ),
            ),
        ),
        _ReferencePackSpec(
            "communication-drafts",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="communication-drafts.messages.draft",
                        capability="communication-drafts.messages.draft",
                        required_permissions=("communications.draft", "workspace.write"),
                        verification=VerificationContract(kind="custom"),
                    ),
                    summary="Draft a message into a scoped workspace artifact without sending it",
                    relevance=1,
                    argument_keys=(
                        "recipient",
                        "subject",
                        "body",
                        "target_path",
                        "body_source",
                        "query",
                    ),
                    argument_grounding={
                        key: ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        )
                        for key in ("recipient", "subject", "body", "target_path")
                    }
                    | {
                        "body_source": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.DETERMINISTIC_DERIVATION,),
                            approved_derivations=("reference.communication_body_from_research.v1",),
                        ),
                        "query": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,),
                        ),
                    },
                ),
            ),
        ),
        _ReferencePackSpec(
            "devices",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="devices.states.list",
                        capability="devices.states.list",
                        required_permissions=("devices.read",),
                        verification=VerificationContract(kind="readback"),
                    ),
                    summary="Read authorized Home Assistant entity states",
                    relevance=1,
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="devices.states.research",
                        capability="devices.states.research",
                        required_permissions=("devices.read",),
                        verification=VerificationContract(kind="custom"),
                    ),
                    summary="Explain an authorized device state with bounded public evidence",
                    relevance=1,
                    argument_keys=("entity_id", "query"),
                    argument_grounding={
                        "entity_id": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        ),
                        "query": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        ),
                    },
                ),
            ),
        ),
        _ReferencePackSpec(
            "device-controls",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="device-controls.devices.command.execute",
                        capability="device-controls.devices.command.execute",
                        required_permissions=("devices.control",),
                        verification=VerificationContract(kind="custom"),
                    ),
                    summary="Run one bounded low-risk Home Assistant command and verify readback",
                    relevance=1,
                    argument_keys=("entity_id", "service", "expected_state"),
                    argument_grounding={
                        "entity_id": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        ),
                        "service": ArgumentGroundingRule(
                            permitted_provenance=(
                                ArgumentProvenanceKind.EXPLICIT_UTTERANCE,
                                ArgumentProvenanceKind.DETERMINISTIC_DERIVATION,
                            ),
                            approved_derivations=("reference.device_service.v1",),
                        ),
                        "expected_state": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        ),
                    },
                ),
            ),
        ),
        _ReferencePackSpec(
            "device-reports",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="device-reports.devices.snapshot_to_workspace",
                        capability="device-reports.devices.snapshot_to_workspace",
                        required_permissions=("devices.read", "workspace.write"),
                        verification=VerificationContract(kind="custom"),
                    ),
                    summary=(
                        "Save an authorized device-state snapshot into a scoped Workspace artifact"
                    ),
                    relevance=1,
                    argument_keys=("target_path",),
                    argument_grounding={
                        "target_path": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        )
                    },
                ),
            ),
        ),
        _ReferencePackSpec(
            "documents",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="documents.list",
                        capability="documents.list",
                        required_permissions=("documents.read",),
                        verification=VerificationContract(kind="readback"),
                    ),
                    summary="Read authorized documents and their bounded text",
                    relevance=1,
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="documents.search",
                        capability="documents.search",
                        required_permissions=("documents.read",),
                        verification=VerificationContract(kind="readback"),
                    ),
                    summary="Search authorized document titles and bounded text",
                    relevance=1,
                    argument_keys=("query",),
                    argument_grounding={
                        "query": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        )
                    },
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="documents.export_to_workspace",
                        capability="documents.export_to_workspace",
                        required_permissions=("documents.read", "workspace.write"),
                        verification=VerificationContract(kind="custom"),
                    ),
                    summary=(
                        "Export one authorized document into a verified scoped workspace artifact"
                    ),
                    relevance=1,
                    argument_keys=("document_id", "target_path"),
                    argument_grounding={
                        "document_id": ArgumentGroundingRule(
                            permitted_provenance=(
                                ArgumentProvenanceKind.AUTHORIZED_CANONICAL_REFERENT,
                            ),
                            canonical_source="authorized_documents",
                        ),
                        "target_path": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        ),
                    },
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="documents.search_to_workspace",
                        capability="documents.search_to_workspace",
                        required_permissions=("documents.read", "workspace.write"),
                        verification=VerificationContract(kind="custom"),
                    ),
                    summary=(
                        "Save bounded authorized document search results as a verified artifact"
                    ),
                    relevance=1,
                    argument_keys=("query", "target_path"),
                    argument_grounding={
                        "query": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        ),
                        "target_path": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        ),
                    },
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="documents.summarize_to_workspace",
                        capability="documents.summarize_to_workspace",
                        required_permissions=("documents.read", "workspace.write"),
                        verification=VerificationContract(kind="custom"),
                    ),
                    summary=(
                        "Create a bounded summary of one authorized document in a verified artifact"
                    ),
                    relevance=1,
                    argument_keys=("document_id", "target_path"),
                    argument_grounding={
                        "document_id": ArgumentGroundingRule(
                            permitted_provenance=(
                                ArgumentProvenanceKind.AUTHORIZED_CANONICAL_REFERENT,
                            ),
                            canonical_source="authorized_documents",
                        ),
                        "target_path": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        ),
                    },
                ),
            ),
        ),
        _ReferencePackSpec(
            "calendar",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="calendar.events.list",
                        capability="calendar.events.list",
                        required_permissions=("calendar.read",),
                        verification=VerificationContract(kind="readback"),
                    ),
                    summary="Read events from the connected external calendar",
                    relevance=1,
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="calendar.events.create",
                        capability="calendar.events.create",
                        required_permissions=("calendar.write",),
                        verification=VerificationContract(kind="custom"),
                    ),
                    summary="Create an explicitly timed calendar event and read it back",
                    relevance=1,
                    argument_keys=("title", "starts_at", "ends_at"),
                    argument_grounding={
                        key: ArgumentGroundingRule(
                            permitted_provenance=(
                                ArgumentProvenanceKind.EXPLICIT_UTTERANCE,
                                ArgumentProvenanceKind.DETERMINISTIC_DERIVATION,
                            ),
                            approved_derivations=("reference.temporal_grounding.v1",)
                            if key in {"starts_at", "ends_at"}
                            else (),
                        )
                        for key in ("title", "starts_at", "ends_at")
                    },
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="calendar.events.cancel",
                        capability="calendar.events.cancel",
                        required_permissions=("calendar.write",),
                        verification=VerificationContract(kind="custom"),
                    ),
                    summary=(
                        "Cancel an explicitly identified calendar event and read back its absence"
                    ),
                    relevance=1,
                    argument_keys=("event_id",),
                    argument_grounding={
                        "event_id": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        )
                    },
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="calendar.events.update",
                        capability="calendar.events.update",
                        required_permissions=("calendar.write",),
                        verification=VerificationContract(kind="custom"),
                    ),
                    summary=(
                        "Update an explicitly identified calendar event and read back "
                        "the changed fields"
                    ),
                    relevance=1,
                    argument_keys=("event_id", "title", "starts_at", "ends_at"),
                    argument_grounding={
                        key: ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        )
                        for key in ("event_id", "title", "starts_at", "ends_at")
                    },
                ),
            ),
        ),
        _ReferencePackSpec(
            "holidays",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="holidays.public_holidays.list",
                        capability="holidays.public_holidays.list",
                        required_permissions=("calendar.read",),
                        verification=VerificationContract(kind="readback"),
                    ),
                    summary="Read public holidays for an explicit country and year",
                    relevance=1,
                    argument_keys=("country_code", "year"),
                    argument_grounding={
                        "country_code": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        ),
                        "year": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        ),
                    },
                ),
            ),
        ),
        _ReferencePackSpec(
            "weather",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="weather.current.read",
                        capability="weather.current.read",
                        required_permissions=("weather.read",),
                        verification=VerificationContract(kind="readback"),
                    ),
                    summary="Read current public weather for explicit coordinates",
                    relevance=1,
                    argument_keys=("latitude", "longitude"),
                    argument_descriptions={
                        "latitude": "explicit latitude in decimal degrees",
                        "longitude": "explicit longitude in decimal degrees",
                    },
                    argument_grounding={
                        key: ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        )
                        for key in ("latitude", "longitude")
                    },
                ),
            ),
        ),
        _ReferencePackSpec(
            "calendar-reports",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="calendar-reports.events.snapshot_to_workspace",
                        capability="calendar-reports.events.snapshot_to_workspace",
                        required_permissions=("calendar.read", "workspace.write"),
                        verification=VerificationContract(kind="custom"),
                    ),
                    summary="Save an authorized calendar snapshot into a verified Workspace report",
                    relevance=1,
                    argument_keys=("target_path",),
                    argument_grounding={
                        "target_path": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        )
                    },
                ),
            ),
        ),
        _ReferencePackSpec(
            "calendar-communications",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="calendar-communications.events.draft",
                        capability="calendar-communications.events.draft",
                        required_permissions=(
                            "calendar.read",
                            "communications.draft",
                            "workspace.write",
                        ),
                        verification=VerificationContract(kind="custom"),
                    ),
                    summary="Draft an unsent calendar snapshot for an explicit recipient",
                    relevance=1,
                    argument_keys=("recipient", "target_path"),
                    argument_grounding={
                        key: ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        )
                        for key in ("recipient", "target_path")
                    },
                ),
            ),
        ),
        _ReferencePackSpec(
            "workspace",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="workspace.artifact.create",
                        capability="workspace.artifact.create",
                        required_permissions=("workspace.write",),
                        verification=VerificationContract(kind="custom"),
                    ),
                    summary="Create files in a bounded owner-scoped workspace and verify them",
                    relevance=1,
                    argument_keys=("path", "content", "files"),
                    argument_grounding={
                        "path": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        ),
                        "content": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        ),
                        "files": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        ),
                    },
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="workspace.research_notes.create",
                        capability="workspace.research_notes.create",
                        required_permissions=("workspace.write",),
                        verification=VerificationContract(kind="custom"),
                    ),
                    summary=(
                        "Research a public question with bounded evidence and save "
                        "non-authoritative "
                        "sourced notes into a scoped workspace artifact"
                    ),
                    relevance=1,
                    argument_keys=("query", "target_path"),
                    argument_grounding={
                        "query": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        ),
                        "target_path": ArgumentGroundingRule(
                            permitted_provenance=(ArgumentProvenanceKind.EXPLICIT_UTTERANCE,)
                        ),
                    },
                ),
            ),
        ),
    )


def reference_packs() -> tuple[PackBundle, ...]:
    """Return first-party Packs through the same generic lifecycle contract."""
    permissions = {
        "calendar": ("calendar.read", "calendar.write"),
        "weather": ("weather.read",),
        "holidays": ("calendar.read",),
        "calendar-reports": ("calendar.read", "workspace.write"),
        "calendar-communications": ("calendar.read", "communications.draft", "workspace.write"),
        "communications": ("communications.read", "communications.send"),
        "communication-drafts": ("communications.draft", "workspace.write"),
        "devices": ("devices.read",),
        "device-controls": ("devices.control",),
        "device-reports": ("devices.read", "workspace.write"),
        "documents": ("documents.read", "workspace.write"),
        "tasks": ("tasks.write", "tasks.read"),
        "kitchen": ("kitchen.write", "kitchen.read"),
        "homelab": ("homelab.service.restart", "homelab.read"),
        "homelab-reports": ("homelab.read", "workspace.write"),
        "network": ("network.read",),
        "workspace": ("workspace.write",),
    }
    return tuple(
        PackBundle(
            manifest=PackManifest(
                pack_id=pack.pack_id,
                version=pack.version,
                permissions=permissions[pack.pack_id],
                ui=PackUI(
                    label=pack.pack_id.replace("-", " ").title(),
                    category="domain",
                    detail_view="list",
                ),
            ),
            cards=pack.cards,
        )
        for pack in _reference_pack_specs()
    )


def reference_bundles() -> tuple[PackBundle, ...]:
    """Compatibility alias for the generic first-party Pack bundles."""

    return reference_packs()


@dataclass
class ReferenceWorld:
    tasks: list[dict[str, Any]] = field(default_factory=list)
    groceries: list[str] = field(default_factory=list)
    services: dict[str, str] = field(default_factory=lambda: {"test-service": "healthy"})


class CalendarEventsExecutor:
    """Read-only external calendar adapter with a bounded configured seam."""

    def execute(self, request: ExecutionRequest) -> Observation:
        del request
        events = configured_calendar_provider().list_events()
        return Observation(
            execution_id=uuid4(),
            evidence=calendar_events_evidence(events),
            command_succeeded=True,
        )


class _NoopHomelabRuntime:
    def restart(self, _service: Any) -> bool:
        return False

    def health(self, _service: Any) -> bool:
        return False


def _canonical_homelab_service(connection: Any, principal: Principal, service_id: str) -> Any:
    pack = PostgresHomelabStore(connection).load(principal, _NoopHomelabRuntime())
    try:
        return pack.services[service_id]
    except KeyError as exc:
        raise ValueError("authorized Homelab service is unavailable") from exc


def _homelab_page_files(connection: Any, principal: Principal) -> dict[str, str]:
    """Render a bounded owner page from the authorized canonical Homelab inventory."""

    pack = PostgresHomelabStore(connection).load(principal, _NoopHomelabRuntime())
    hosts = sorted(pack.hosts.values(), key=lambda item: item.host_id)
    services = sorted(pack.services.values(), key=lambda item: item.service_id)
    host_rows = (
        "".join(
            f"<li><strong>{host.hostname}</strong> <code>{host.address}</code></li>"
            for host in hosts
        )
        or "<li>No authorized hosts are configured.</li>"
    )
    service_rows = (
        "".join(
            f"<li><strong>{service.name}</strong> <code>{service.service_id}</code></li>"
            for service in services
        )
        or "<li>No authorized services are configured.</li>"
    )
    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Homelab</title><link rel='stylesheet' href='style.css'></head>"
        "<body><main><p class='eyebrow'>AEGIS · authorized inventory</p>"
        "<h1>Homelab</h1><section><h2>Hosts</h2><ul>"
        f"{host_rows}</ul></section><section><h2>Services</h2><ul>"
        f"{service_rows}</ul></section></main></body></html>"
    )
    css = (
        ":root{color-scheme:dark;font:16px system-ui,sans-serif;background:#10141c;"
        "color:#edf2f7}body{margin:0;padding:3rem}main{max-width:52rem;margin:auto}"
        ".eyebrow{color:#8cc8ff;letter-spacing:.12em;text-transform:uppercase;font-size:.75rem}"
        "section{border:1px solid #334155;border-radius:12px;padding:1rem 1.25rem;"
        "margin:1rem 0;background:#172033}li{margin:.55rem 0}code{color:#9fddae}"
    )
    return {"index.html": html, "style.css": css}


def _homelab_health_report_files(connection: Any, principal: Principal) -> dict[str, str]:
    """Render a bounded health snapshot from authorized canonical services."""

    pack = PostgresHomelabStore(connection).load(principal, _NoopHomelabRuntime())
    services = sorted(pack.services.values(), key=lambda item: item.service_id)
    rows = []
    for service in services:
        healthy, status = _health_read(service.health_endpoint)
        state = "healthy" if healthy else status
        rows.append(
            f"<li><strong>{escape(service.name)}</strong> "
            f"<code>{escape(service.service_id)}</code> "
            f"<span>{escape(state)}</span></li>"
        )
    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Homelab health</title><link rel='stylesheet' href='style.css'></head>"
        "<body><main><p class='eyebrow'>AEGIS · authorized observation</p>"
        "<h1>Homelab health</h1><p>Bounded service health snapshot.</p><ul>"
        f"{''.join(rows) or '<li>No authorized services are configured.</li>'}"
        "</ul></main></body></html>"
    )
    css = (
        ":root{color-scheme:dark;font:16px system-ui,sans-serif;background:#10141c;"
        "color:#edf2f7}body{margin:0;padding:3rem}main{max-width:52rem;margin:auto}"
        ".eyebrow{color:#8cc8ff;letter-spacing:.12em;text-transform:uppercase;font-size:.75rem}"
        "li{margin:.7rem 0}code{color:#9fddae}span{margin-left:.5rem;color:#f5c97a}"
    )
    return {"index.html": html, "style.css": css}


def _health_read(endpoint: str) -> tuple[bool, str]:
    try:
        parsed = urllib.parse.urlsplit(endpoint)
    except ValueError:
        return False, "invalid_endpoint"
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False, "invalid_endpoint"
    try:
        with urllib.request.urlopen(endpoint, timeout=2.0) as response:
            response.read(4_096)
            return response.status == 200, f"http_{response.status}"
    except urllib.error.HTTPError as exc:
        return False, f"http_{exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError):
        return False, "unavailable"


class HomelabHealthExecutor:
    def __init__(self, connection: Any, principal: Principal) -> None:
        self.connection = connection
        self.principal = principal

    def execute(self, request: ExecutionRequest) -> Observation:
        service_id = request.action.arguments.get("service")
        if not isinstance(service_id, str) or not service_id.strip():
            return Observation(
                execution_id=uuid4(),
                evidence={"homelab_health": "invalid_service"},
                command_succeeded=False,
            )
        try:
            service = _canonical_homelab_service(self.connection, self.principal, service_id)
            healthy, status = _health_read(service.health_endpoint)
        except (PermissionError, ValueError) as exc:
            return Observation(
                execution_id=uuid4(),
                evidence={"homelab_health": "rejected", "reason": str(exc)},
                command_succeeded=False,
            )
        return Observation(
            execution_id=uuid4(),
            evidence={
                "homelab_health": {
                    "service": service.service_id,
                    "attempt_status": status,
                    "attempt_healthy": healthy,
                }
            },
            command_succeeded=True,
        )


class HomelabHealthVerifier:
    def __init__(self, connection: Any, principal: Principal) -> None:
        self.connection = connection
        self.principal = principal

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        evidence = observation.evidence.get("homelab_health")
        if not observation.command_succeeded or not isinstance(evidence, dict):
            return VerificationResult(
                verified=False, evidence=observation.evidence, reason="Homelab health read failed"
            )
        service_id = evidence.get("service")
        if not isinstance(service_id, str):
            return VerificationResult(
                verified=False,
                evidence=observation.evidence,
                reason="Homelab service identity missing",
            )
        try:
            service = _canonical_homelab_service(self.connection, self.principal, service_id)
            healthy, status = _health_read(service.health_endpoint)
        except (PermissionError, ValueError) as exc:
            return VerificationResult(
                verified=False,
                evidence={**observation.evidence, "reason": str(exc)},
                reason="Homelab service scope failed",
            )
        verified = healthy and status == "http_200"
        return VerificationResult(
            verified=verified,
            evidence={
                **observation.evidence,
                "independent_status": status,
                "independent_healthy": healthy,
            },
            reason="Homelab health independently verified"
            if verified
            else "Homelab health readback failed",
        )


class HomelabWorkspaceExecutor:
    """Create a small static inventory page from authorized canonical state."""

    def __init__(self, connection: Any, principal: Principal) -> None:
        self.connection = connection
        self.principal = principal

    def execute(self, request: ExecutionRequest) -> Observation:
        target_path = request.action.arguments.get("target_path")
        if not isinstance(target_path, str) or not target_path.strip():
            return Observation(
                execution_id=uuid4(),
                evidence={"homelab_workspace": "invalid_arguments"},
                command_succeeded=False,
            )
        try:
            page = _homelab_page_files(self.connection, self.principal)
            files = {target_path: page["index.html"], "style.css": page["style.css"]}
            workspace = WorkspaceManager(
                Path(os.environ.get("AEGIS_WORKSPACE_ROOT", "/tmp/aegis-owner-workspaces"))
            ).for_objective(self.principal.id, request.objective_id)
            artifact = workspace.write_artifact(files, request.action_id, lambda current: None)
        except (PermissionError, ValueError, OSError) as exc:
            return Observation(
                execution_id=uuid4(),
                evidence={"homelab_workspace": "rejected", "reason": str(exc)},
                command_succeeded=False,
            )
        return Observation(
            execution_id=uuid4(),
            evidence={
                "homelab_workspace": "artifact",
                "files": list(artifact.files),
                "executor_local_validated": artifact.validated,
            },
            command_succeeded=True,
        )


class HomelabWorkspaceVerifier:
    def __init__(self, principal: Principal) -> None:
        self.principal = principal

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        verified, detail = _verify_workspace_expectation(self.principal, contract)
        verified = observation.command_succeeded and verified
        return VerificationResult(
            verified=verified,
            evidence={"homelab_workspace_verified": verified, "postcondition": detail},
            reason="Homelab page independently verified"
            if verified
            else f"Homelab page verification failed: {detail}",
        )


class HomelabHealthWorkspaceExecutor(HomelabWorkspaceExecutor):
    """Write a pre-execution-fixed authorized health snapshot to Workspace."""

    def execute(self, request: ExecutionRequest) -> Observation:
        target_path = request.action.arguments.get("target_path")
        if not isinstance(target_path, str) or not target_path.strip():
            return Observation(
                execution_id=uuid4(),
                evidence={"homelab_health_workspace": "invalid_arguments"},
                command_succeeded=False,
            )
        try:
            page = _homelab_health_report_files(self.connection, self.principal)
            workspace = WorkspaceManager(
                Path(os.environ.get("AEGIS_WORKSPACE_ROOT", "/tmp/aegis-owner-workspaces"))
            ).for_objective(self.principal.id, request.objective_id)
            artifact = workspace.write_artifact(
                {target_path: page["index.html"], "style.css": page["style.css"]},
                request.action_id,
                lambda current: None,
            )
        except (PermissionError, ValueError, OSError) as exc:
            return Observation(
                execution_id=uuid4(),
                evidence={"homelab_health_workspace": "rejected", "reason": str(exc)},
                command_succeeded=False,
            )
        return Observation(
            execution_id=uuid4(),
            evidence={
                "homelab_health_workspace": "artifact",
                "files": list(artifact.files),
                "executor_local_validated": artifact.validated,
            },
            command_succeeded=True,
        )


class CalendarCreateExecutor:
    def __init__(self, provider: CalendarWriteProvider) -> None:
        self.provider = provider

    def execute(self, request: ExecutionRequest) -> Observation:
        args = request.action.arguments
        title, starts_at, ends_at = args.get("title"), args.get("starts_at"), args.get("ends_at")
        if not (
            isinstance(title, str)
            and title.strip()
            and isinstance(starts_at, str)
            and starts_at.strip()
            and (ends_at is None or isinstance(ends_at, str))
        ):
            return Observation(
                execution_id=uuid4(),
                evidence={"calendar_create": "invalid_arguments"},
                command_succeeded=False,
            )
        try:
            start = datetime.fromisoformat(starts_at)
            end = datetime.fromisoformat(ends_at) if ends_at else None
        except ValueError:
            return Observation(
                execution_id=uuid4(),
                evidence={"calendar_create": "invalid_timestamp"},
                command_succeeded=False,
            )
        created = self.provider.create_event(
            CalendarEvent("pending", title, start, end), request.idempotency_key
        )
        return Observation(
            execution_id=uuid4(),
            evidence={
                "calendar_create": {
                    "event_id": created.event_id,
                    "title": created.title,
                    "starts_at": created.starts_at.isoformat(),
                    "ends_at": created.ends_at.isoformat() if created.ends_at else None,
                }
            },
            command_succeeded=True,
        )


class CalendarCreateVerifier:
    def __init__(self, provider: CalendarWriteProvider) -> None:
        self.provider = provider

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        evidence = observation.evidence.get("calendar_create")
        if not observation.command_succeeded or not isinstance(evidence, dict):
            return VerificationResult(
                verified=False, evidence={}, reason="calendar create did not execute"
            )
        event_id = evidence.get("event_id")
        actual = self.provider.get_event(event_id) if isinstance(event_id, str) else None
        expected = contract.expected
        verified = (
            actual is not None
            and actual.title == expected.get("title")
            and actual.starts_at.isoformat() == expected.get("starts_at")
            and (actual.ends_at.isoformat() if actual.ends_at else None) == expected.get("ends_at")
        )
        return VerificationResult(
            verified=verified,
            evidence={"calendar_event_id": event_id, "provider_readback": actual is not None},
            reason="calendar event independently read back"
            if verified
            else "calendar event readback did not match the expected event",
        )


class CalendarCancelExecutor:
    def __init__(self, provider: CalendarWriteProvider) -> None:
        self.provider = provider

    def execute(self, request: ExecutionRequest) -> Observation:
        event_id = request.action.arguments.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            return Observation(
                execution_id=uuid4(),
                evidence={"calendar_cancel": "invalid_event_id"},
                command_succeeded=False,
            )
        try:
            self.provider.delete_event(event_id)
        except (RuntimeError, ValueError) as exc:
            return Observation(
                execution_id=uuid4(),
                evidence={"calendar_cancel": {"event_id": event_id, "error": str(exc)}},
                command_succeeded=False,
            )
        return Observation(
            execution_id=uuid4(),
            evidence={"calendar_cancel": {"event_id": event_id}},
            command_succeeded=True,
        )


class CalendarCancelVerifier:
    def __init__(self, provider: CalendarWriteProvider) -> None:
        self.provider = provider

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        evidence = observation.evidence.get("calendar_cancel")
        if not observation.command_succeeded or not isinstance(evidence, dict):
            return VerificationResult(
                verified=False, evidence={}, reason="calendar cancel did not execute"
            )
        event_id = contract.expected.get("event_id")
        actual = self.provider.get_event(event_id) if isinstance(event_id, str) else None
        verified = actual is None and contract.expected.get("exists") is False
        return VerificationResult(
            verified=verified,
            evidence={"calendar_event_id": event_id, "provider_readback_absent": actual is None},
            reason="calendar event absence independently read back"
            if verified
            else "calendar event remained present after cancellation",
        )


class CalendarUpdateExecutor:
    def __init__(self, provider: CalendarWriteProvider) -> None:
        self.provider = provider

    def execute(self, request: ExecutionRequest) -> Observation:
        args = request.action.arguments
        event_id, title, starts_at, ends_at = (
            args.get("event_id"),
            args.get("title"),
            args.get("starts_at"),
            args.get("ends_at"),
        )
        if not (
            isinstance(event_id, str)
            and event_id.strip()
            and isinstance(title, str)
            and title.strip()
            and isinstance(starts_at, str)
            and starts_at.strip()
            and (ends_at is None or isinstance(ends_at, str))
        ):
            return Observation(
                execution_id=uuid4(),
                evidence={"calendar_update": "invalid_arguments"},
                command_succeeded=False,
            )
        try:
            event = CalendarEvent(
                event_id,
                title,
                datetime.fromisoformat(starts_at),
                datetime.fromisoformat(ends_at) if ends_at else None,
            )
            updated = self.provider.update_event(event_id, event)
        except (ValueError, RuntimeError) as exc:
            return Observation(
                execution_id=uuid4(),
                evidence={"calendar_update": {"event_id": event_id, "error": str(exc)}},
                command_succeeded=False,
            )
        return Observation(
            execution_id=uuid4(),
            evidence={
                "calendar_update": {
                    "event_id": updated.event_id,
                    "title": updated.title,
                    "starts_at": updated.starts_at.isoformat(),
                    "ends_at": updated.ends_at.isoformat() if updated.ends_at else None,
                }
            },
            command_succeeded=True,
        )


class CalendarUpdateVerifier:
    def __init__(self, provider: CalendarWriteProvider) -> None:
        self.provider = provider

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        evidence = observation.evidence.get("calendar_update")
        if not observation.command_succeeded or not isinstance(evidence, dict):
            return VerificationResult(
                verified=False, evidence={}, reason="calendar update did not execute"
            )
        event_id = contract.expected.get("event_id")
        actual = self.provider.get_event(event_id) if isinstance(event_id, str) else None
        verified = (
            actual is not None
            and actual.title == contract.expected.get("title")
            and actual.starts_at.isoformat() == contract.expected.get("starts_at")
            and (actual.ends_at.isoformat() if actual.ends_at else None)
            == contract.expected.get("ends_at")
        )
        return VerificationResult(
            verified=verified,
            evidence={"calendar_event_id": event_id, "provider_readback": actual is not None},
            reason="calendar event update independently read back"
            if verified
            else "calendar event update readback did not match the expected event",
        )


class CommunicationsExecutor:
    def execute(self, request: ExecutionRequest) -> Observation:
        del request
        return Observation(
            execution_id=uuid4(),
            evidence=communications_evidence(FixtureCommunicationProvider().list_messages()),
            command_succeeded=True,
        )


class CommunicationsVerifier:
    def verify(
        self, observation: Observation, _contract: VerificationContract
    ) -> VerificationResult:
        messages = observation.evidence.get("messages")
        verified = observation.command_succeeded and isinstance(messages, list)
        return VerificationResult(
            verified=verified,
            evidence={"message_count": len(cast(list[Any], messages)) if verified else 0},
            reason="communications readback is structurally valid"
            if verified
            else "communications read failed",
        )


class CommunicationsSendExecutor:
    """Send an explicitly grounded message through a replaceable provider."""

    def __init__(
        self,
        provider: CommunicationSendProvider,
        approved_targets: frozenset[tuple[str, str, str | None]] | None = None,
    ) -> None:
        self.provider = provider
        self.approved_targets = approved_targets

    def execute(self, request: ExecutionRequest) -> Observation:
        args = request.action.arguments
        target, body = args.get("target"), args.get("body")
        channel, account = args.get("channel", "default"), args.get("account")
        if not (
            isinstance(target, str)
            and target.strip()
            and isinstance(body, str)
            and body.strip()
            and isinstance(channel, str)
            and channel.strip()
            and (account is None or isinstance(account, str))
        ):
            return Observation(
                execution_id=uuid4(),
                evidence={"communication_send": "invalid_arguments"},
                command_succeeded=False,
            )
        if (
            self.approved_targets is not None
            and (
                target,
                channel,
                account,
            )
            not in self.approved_targets
        ):
            return Observation(
                execution_id=uuid4(),
                evidence={
                    "communication_send": "target_not_approved",
                    "target": target,
                    "channel": channel,
                },
                command_succeeded=False,
            )
        result = self.provider.send(
            OutboundMessage(target=target, body=body, channel=channel, account=account),
            request.idempotency_key,
        )
        return Observation(
            execution_id=uuid4(),
            evidence={
                "communication_send": {
                    "status": result.status.value,
                    "provider_message_id": result.provider_message_id,
                    "detail": result.detail,
                    "target": target,
                    "channel": channel,
                    "account": account,
                }
            },
            command_succeeded=result.status in {SendStatus.PROVIDER_ACCEPTED, SendStatus.DELIVERED},
        )


class CommunicationsSendVerifier:
    """Verify acceptance and optionally exact provider readback, never delivery."""

    def __init__(self, provider: Any | None = None) -> None:
        self.provider = provider

    def verify(
        self, observation: Observation, _contract: VerificationContract
    ) -> VerificationResult:
        evidence = observation.evidence.get("communication_send")
        verified = (
            observation.command_succeeded
            and isinstance(evidence, dict)
            and evidence.get("status")
            in {SendStatus.PROVIDER_ACCEPTED.value, SendStatus.DELIVERED.value}
            and isinstance(evidence.get("provider_message_id"), str)
        )
        status = evidence.get("status") if isinstance(evidence, dict) else None
        readback = False
        expected = _contract.expected
        send_evidence = evidence if isinstance(evidence, dict) else {}
        if verified and self.provider is not None:
            getter = getattr(self.provider, "get_sent_message", None)
            expected_body = _expected_message_body(_contract)
            if callable(getter) and expected_body is not None:
                try:
                    actual = getter(send_evidence["provider_message_id"])
                except (OSError, RuntimeError, ValueError):
                    actual = None
                verified = (
                    actual is not None
                    and actual.target == send_evidence.get("target")
                    and actual.channel == send_evidence.get("channel")
                    and actual.account == expected.get("account")
                    and actual.body == expected_body
                )
                readback = verified
        if verified and readback:
            reason = (
                "communication provider accepted and read back the exact message; "
                "delivery is not independently proven"
            )
        elif verified:
            reason = (
                "communication provider accepted the message; delivery is not independently proven"
            )
        else:
            reason = "communication provider did not accept the message"
        return VerificationResult(
            verified=verified,
            evidence={
                "communication_send_status": status,
                "independent_provider_readback": readback,
                "independent_delivery": False,
            },
            reason=reason,
        )


def _expected_message_body(contract: VerificationContract) -> str | None:
    """Read the grounded body from the generic contract when available."""

    expected = contract.expected
    body = expected.get("body") if isinstance(expected, dict) else None
    return body if isinstance(body, str) else None


class DeviceStatesExecutor:
    """Read-only device provider adapter; no service call is exposed here."""

    def execute(self, request: ExecutionRequest) -> Observation:
        del request
        gateway = (
            HomeAssistantRestGateway(
                os.environ["AEGIS_HOME_ASSISTANT_URL"],
                os.environ["AEGIS_HOME_ASSISTANT_TOKEN"],
            )
            if os.environ.get("AEGIS_HOME_ASSISTANT_URL")
            and os.environ.get("AEGIS_HOME_ASSISTANT_TOKEN")
            else FixtureDeviceGateway(
                {"homeassistant.status": {"state": "ready", "attributes": {}}}
            )
        )
        adapter = HomeAssistantAdapter(gateway, policy=_ReadOnlyDevicePolicy())
        states = adapter.read_states(datetime.now(timezone.utc))
        return Observation(
            execution_id=uuid4(),
            evidence=device_states_evidence(states),
            command_succeeded=True,
        )


class DeviceControlExecutor:
    """Execute only the typed low-risk device command contract."""

    def __init__(self, gateway: Any | None = None) -> None:
        self.gateway = gateway

    def execute(self, request: ExecutionRequest) -> Observation:
        args = request.action.arguments
        entity_id = args.get("entity_id")
        service = args.get("service")
        expected_state = args.get("expected_state")
        if (
            not isinstance(entity_id, str)
            or not isinstance(service, str)
            or not isinstance(expected_state, str)
        ):
            raise ValueError("device control requires explicit entity, service, and postcondition")
        if service not in {"turn_on", "turn_off"}:
            raise PermissionError("device service is not in the bounded control allowlist")
        authorized_entities = {
            item.strip()
            for item in os.environ.get("AEGIS_AUTHORIZED_DEVICE_ENTITIES", "").split(",")
            if item.strip()
        }
        if entity_id not in authorized_entities or len(authorized_entities) > 50:
            return Observation(
                execution_id=uuid4(),
                evidence={
                    "failure_reason": "device entity is outside the authorized device scope",
                    "device_execution": {
                        "accepted": False,
                        "verified": False,
                        "reason": "device entity is outside the authorized device scope",
                        "entity_id": entity_id,
                    },
                },
                command_succeeded=False,
            )
        gateway = self.gateway or (
            HomeAssistantRestControlGateway(
                os.environ["AEGIS_HOME_ASSISTANT_URL"],
                os.environ["AEGIS_HOME_ASSISTANT_TOKEN"],
            )
            if os.environ.get("AEGIS_HOME_ASSISTANT_URL")
            and os.environ.get("AEGIS_HOME_ASSISTANT_TOKEN")
            else FixtureDeviceGateway({entity_id: {"state": "off", "attributes": {}}})
        )
        command = DeviceCommand(entity_id=entity_id, service=service, expected_state=expected_state)
        if not _LowRiskDevicePolicy().allow_command(command):
            raise PermissionError("Home Assistant command denied by policy")
        gateway.call_service(command.model_dump(mode="json"))
        return Observation(
            execution_id=uuid4(),
            evidence={
                "device_execution": {
                    "accepted": True,
                    "entity_id": entity_id,
                    "service": service,
                    "expected_state": expected_state,
                    "readback": "deferred_to_independent_verifier",
                }
            },
            command_succeeded=True,
        )


class _LowRiskDevicePolicy:
    def allow_command(self, command: Any) -> bool:
        return (
            command.entity_id.startswith(("light.", "switch.", "input_boolean."))
            and command.service in {"turn_on", "turn_off"}
            and command.expected_state in {"on", "off"}
        )


class DeviceControlVerifier:
    def __init__(self, gateway: Any | None = None, principal: Principal | None = None) -> None:
        self.gateway = gateway
        self.principal = principal

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        expected = contract.expected
        execution = observation.evidence.get("device_execution")
        verified = False
        reason = "device postcondition is unavailable"
        if (
            observation.command_succeeded
            and isinstance(execution, dict)
            and isinstance(expected, dict)
            and (self.principal is None or expected.get("principal_id") == self.principal.id)
            and isinstance(expected.get("entity_id"), str)
            and isinstance(expected.get("expected_state"), str)
        ):
            gateway = self.gateway or FixtureDeviceGateway({})
            actual = gateway.get_state(expected["entity_id"])
            verified = actual.get("state") == expected["expected_state"]
            reason = "device state independently verified" if verified else "device state mismatch"
        return VerificationResult(
            verified=verified,
            evidence={"readback_verified": verified, "verification": "independent_provider_read"},
            reason=reason,
        )


class DeviceSnapshotWorkspaceExecutor:
    """Compose authorized device observation into a scoped Workspace artifact."""

    def __init__(self, principal: Principal) -> None:
        self.principal = principal

    def execute(self, request: ExecutionRequest) -> Observation:
        target_path = request.action.arguments.get("target_path")
        if not isinstance(target_path, str) or not target_path.strip():
            return Observation(
                execution_id=uuid4(),
                evidence={"device_workspace": "invalid_arguments"},
                command_succeeded=False,
            )
        states = DeviceStatesExecutor().execute(request).evidence.get("states", [])
        content = _device_snapshot_content(states)
        workspace = WorkspaceManager(
            Path(os.environ.get("AEGIS_WORKSPACE_ROOT", "/tmp/aegis-owner-workspaces"))
        ).for_objective(self.principal.id, request.objective_id)
        try:
            artifact = workspace.write_artifact(
                {target_path: content},
                request.action_id,
                lambda current: (
                    None
                    if current.read(target_path) == content
                    else "device snapshot readback mismatch"
                ),
            )
        except (ValueError, OSError) as exc:
            return Observation(
                execution_id=uuid4(),
                evidence={"device_workspace": "rejected", "reason": str(exc)},
                command_succeeded=False,
            )
        return Observation(
            execution_id=uuid4(),
            evidence={
                "composition": "device_snapshot_to_workspace",
                "device_count": len(states) if isinstance(states, list) else 0,
                "files": list(artifact.files),
                "executor_local_validated": artifact.validated,
            },
            command_succeeded=True,
        )


def _device_snapshot_content(states: object) -> str:
    """Serialize stable device facts; observation timestamps are not artifact input."""

    if not isinstance(states, list):
        states = []
    stable = [
        {key: value for key, value in item.items() if key != "observed_at"}
        for item in states
        if isinstance(item, dict)
    ]
    return "# Authorized device snapshot\n\n" + json.dumps(stable, indent=2) + "\n"


class DeviceSnapshotWorkspaceVerifier:
    def __init__(self, principal: Principal) -> None:
        self.principal = principal

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        verified, detail = _verify_workspace_expectation(self.principal, contract)
        verified = observation.command_succeeded and verified
        return VerificationResult(
            verified=verified,
            evidence={
                "device_snapshot_workspace_verified": verified,
                "postcondition": detail,
            },
            reason="device snapshot independently verified"
            if verified
            else f"device snapshot verification failed: {detail}",
        )


class _ReadOnlyDevicePolicy:
    def allow_command(self, command: Any) -> bool:
        del command
        return False


class DeviceStatesVerifier:
    def verify(
        self, observation: Observation, _contract: VerificationContract
    ) -> VerificationResult:
        states = observation.evidence.get("states")
        verified = observation.command_succeeded and isinstance(states, list)
        return VerificationResult(
            verified=verified,
            evidence={"entity_count": len(cast(list[Any], states)) if verified else 0},
            reason="device state readback is structurally valid"
            if verified
            else "device state read failed",
        )


class DeviceResearchExecutor:
    """Read one authorized device and collect bounded non-authoritative evidence."""

    def execute(self, request: ExecutionRequest) -> Observation:
        entity_id = request.action.arguments.get("entity_id")
        query = request.action.arguments.get("query")
        if not isinstance(entity_id, str) or not isinstance(query, str) or not query.strip():
            return Observation(
                execution_id=uuid4(),
                evidence={"device_research": "invalid_arguments"},
                command_succeeded=False,
            )
        gateway = (
            HomeAssistantRestGateway(
                os.environ["AEGIS_HOME_ASSISTANT_URL"],
                os.environ["AEGIS_HOME_ASSISTANT_TOKEN"],
            )
            if os.environ.get("AEGIS_HOME_ASSISTANT_URL")
            and os.environ.get("AEGIS_HOME_ASSISTANT_TOKEN")
            else FixtureDeviceGateway({entity_id: {"state": "off", "attributes": {}}})
        )
        try:
            state = gateway.get_state(entity_id)
            public_query = query.replace(entity_id, "the smart-home device")
            evidence = configured_research_service().collect(SearchRequest(public_query))
        except (OSError, ResearchUnavailable, ValueError) as exc:
            return Observation(
                execution_id=uuid4(),
                evidence={"device_research": "unavailable", "reason": str(exc)},
                command_succeeded=False,
            )
        return Observation(
            execution_id=uuid4(),
            evidence={
                "device_research": {
                    "entity_id": entity_id,
                    "state": state.get("state", "unknown"),
                    "query": public_query,
                    "sources": [
                        {
                            "source_id": item.source_id,
                            "title": item.title,
                            "url": item.final_url,
                            "retrieved_at": item.retrieved_at.isoformat(),
                        }
                        for item in evidence.evidence
                    ],
                    "authoritative": False,
                }
            },
            command_succeeded=True,
        )


class DeviceResearchVerifier:
    """Independently reread the device provider before accepting the composition."""

    def verify(
        self, observation: Observation, _contract: VerificationContract
    ) -> VerificationResult:
        evidence = observation.evidence.get("device_research")
        if not observation.command_succeeded or not isinstance(evidence, dict):
            return VerificationResult(
                verified=False,
                evidence=observation.evidence,
                reason="device research did not produce evidence",
            )
        entity_id, expected_state, sources = (
            evidence.get("entity_id"),
            evidence.get("state"),
            evidence.get("sources"),
        )
        if not isinstance(entity_id, str) or not isinstance(expected_state, str):
            return VerificationResult(
                verified=False, evidence=observation.evidence, reason="device identity is invalid"
            )
        gateway = (
            HomeAssistantRestGateway(
                os.environ["AEGIS_HOME_ASSISTANT_URL"],
                os.environ["AEGIS_HOME_ASSISTANT_TOKEN"],
            )
            if os.environ.get("AEGIS_HOME_ASSISTANT_URL")
            and os.environ.get("AEGIS_HOME_ASSISTANT_TOKEN")
            else FixtureDeviceGateway({entity_id: {"state": "off", "attributes": {}}})
        )
        try:
            actual = gateway.get_state(entity_id)
        except (OSError, ValueError) as exc:
            return VerificationResult(
                verified=False,
                evidence={**observation.evidence, "readback_error": str(exc)},
                reason="device provider readback failed",
            )
        verified = (
            actual.get("state") == expected_state and isinstance(sources, list) and bool(sources)
        )
        return VerificationResult(
            verified=verified,
            evidence={
                "device_state": actual.get("state"),
                "independent_provider_read": True,
                "source_count": len(sources) if isinstance(sources, list) else 0,
            },
            reason=(
                f"{entity_id} is {expected_state}; bounded public evidence collected"
                if verified
                else "device state or research evidence mismatch"
            ),
        )


class DocumentsExecutor:
    def execute(self, request: ExecutionRequest) -> Observation:
        query = request.action.arguments.get("query")
        documents = configured_document_provider().list_documents()
        if request.action.action_id == "documents.search":
            if not isinstance(query, str) or not query.strip():
                return Observation(
                    execution_id=uuid4(),
                    evidence={"documents_search": "invalid_query"},
                    command_succeeded=False,
                )
            needle = query.strip().casefold()
            matches = []
            for document in documents:
                haystack = f"{document.title}\n{document.text}".casefold()
                if needle not in haystack:
                    continue
                matches.append(
                    {
                        "document_id": document.document_id,
                        "title": document.title,
                        "source": document.source,
                        "snippet": document.text[:500],
                    }
                )
            evidence: dict[str, object] = {
                "query": query.strip()[:500],
                "documents": matches[:20],
                "source": "authorized_document_fixture",
            }
        else:
            evidence = documents_evidence(documents)
        return Observation(
            execution_id=uuid4(),
            evidence=evidence,
            command_succeeded=True,
        )


class DocumentWorkspaceExecutor:
    def __init__(self, principal: Principal, provider: Any | None = None) -> None:
        self.principal = principal
        self.provider = provider or configured_document_provider()

    def execute(self, request: ExecutionRequest) -> Observation:
        document_id = request.action.arguments.get("document_id")
        target_path = request.action.arguments.get("target_path")
        if not isinstance(document_id, str) or not isinstance(target_path, str):
            return Observation(
                execution_id=uuid4(),
                evidence={"documents": "invalid_arguments"},
                command_succeeded=False,
            )
        root = Path(os.environ.get("AEGIS_WORKSPACE_ROOT", "/tmp/aegis-owner-workspaces"))
        try:
            result = document_to_workspace(
                self.provider,
                WorkspaceManager(root),
                principal_id=self.principal.id,
                objective_id=request.objective_id,
                document_id=document_id,
                target_path=target_path,
                correlation_id=request.action_id,
            )
        except (ValueError, OSError) as exc:
            return Observation(
                execution_id=uuid4(),
                evidence={"documents": "export_rejected", "reason": str(exc)},
                command_succeeded=False,
            )
        return Observation(
            execution_id=uuid4(),
            evidence={
                "composition": "document_to_workspace",
                "document_id": result.document_id,
                "target_path": result.target_path,
                "files": list(result.files),
                "executor_local_validated": result.validated,
                "source": result.source,
            },
            command_succeeded=True,
        )


class DocumentSummaryWorkspaceExecutor(DocumentWorkspaceExecutor):
    """Create a deterministic summary artifact from an authorized document."""

    def execute(self, request: ExecutionRequest) -> Observation:
        document_id = request.action.arguments.get("document_id")
        target_path = request.action.arguments.get("target_path")
        if not isinstance(document_id, str) or not isinstance(target_path, str):
            return Observation(
                execution_id=uuid4(),
                evidence={"documents": "invalid_arguments"},
                command_succeeded=False,
            )
        root = Path(os.environ.get("AEGIS_WORKSPACE_ROOT", "/tmp/aegis-owner-workspaces"))
        try:
            result = document_summary_to_workspace(
                self.provider,
                WorkspaceManager(root),
                principal_id=self.principal.id,
                objective_id=request.objective_id,
                document_id=document_id,
                target_path=target_path,
                correlation_id=request.action_id,
            )
        except (ValueError, OSError) as exc:
            return Observation(
                execution_id=uuid4(),
                evidence={"documents": "summary_rejected", "reason": str(exc)},
                command_succeeded=False,
            )
        return Observation(
            execution_id=uuid4(),
            evidence={
                "composition": "document_summary_to_workspace",
                "document_id": result.document_id,
                "target_path": result.target_path,
                "files": list(result.files),
                "executor_local_validated": result.validated,
                "source": result.source,
            },
            command_succeeded=True,
        )


class DocumentSearchWorkspaceExecutor(DocumentWorkspaceExecutor):
    """Create a bounded search-results artifact from authorized documents."""

    def execute(self, request: ExecutionRequest) -> Observation:
        query = request.action.arguments.get("query")
        target_path = request.action.arguments.get("target_path")
        if not isinstance(query, str) or not isinstance(target_path, str):
            return Observation(
                execution_id=uuid4(),
                evidence={"documents": "invalid_arguments"},
                command_succeeded=False,
            )
        root = Path(os.environ.get("AEGIS_WORKSPACE_ROOT", "/tmp/aegis-owner-workspaces"))
        try:
            result = document_search_to_workspace(
                self.provider,
                WorkspaceManager(root),
                principal_id=self.principal.id,
                objective_id=request.objective_id,
                query=query,
                target_path=target_path,
                correlation_id=request.action_id,
            )
        except (ValueError, OSError) as exc:
            return Observation(
                execution_id=uuid4(),
                evidence={"documents": "search_rejected", "reason": str(exc)},
                command_succeeded=False,
            )
        return Observation(
            execution_id=uuid4(),
            evidence={
                "composition": "document_search_to_workspace",
                "query": query,
                "target_path": result.target_path,
                "files": list(result.files),
                "executor_local_validated": result.validated,
                "source": result.source,
            },
            command_succeeded=True,
        )


class DocumentWorkspaceVerifier:
    def __init__(self, principal: Principal) -> None:
        self.principal = principal

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        verified, detail = _verify_workspace_expectation(self.principal, contract)
        verified = observation.command_succeeded and verified
        composition = observation.evidence.get("composition")
        operation = (
            "document summary"
            if composition == "document_summary_to_workspace"
            else "document search"
            if composition == "document_search_to_workspace"
            else "document export"
        )
        return VerificationResult(
            verified=verified,
            evidence={"composition_verified": verified, "postcondition": detail},
            reason=f"{operation} independently verified"
            if verified
            else f"{operation} verification failed: {detail}",
        )


class DocumentsVerifier:
    def verify(
        self, observation: Observation, _contract: VerificationContract
    ) -> VerificationResult:
        documents = observation.evidence.get("documents")
        verified = observation.command_succeeded and isinstance(documents, list)
        document_rows = cast(list[Any], documents) if verified else []
        return VerificationResult(
            verified=verified,
            evidence={"document_count": len(document_rows)},
            reason=(
                "document readback is structurally valid" if verified else "document read failed"
            ),
        )


class CalendarEventsVerifier:
    def verify(
        self, observation: Observation, _contract: VerificationContract
    ) -> VerificationResult:
        evidence = observation.evidence
        verified = observation.command_succeeded and isinstance(evidence.get("events"), list)
        return VerificationResult(
            verified=verified,
            evidence={"event_count": len(evidence.get("events", [])) if verified else 0},
            reason=(
                "calendar readback is structurally valid" if verified else "calendar read failed"
            ),
        )


class WeatherExecutor:
    """Read current conditions from the bounded configured weather provider."""

    def execute(self, request: ExecutionRequest) -> Observation:
        try:
            latitude = float(request.action.arguments["latitude"])
            longitude = float(request.action.arguments["longitude"])
            if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                raise ValueError("weather coordinates are out of range")
            reading = configured_weather_provider().current(latitude, longitude)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return Observation(
                execution_id=uuid4(),
                evidence={"weather": "unavailable", "reason": str(exc)},
                command_succeeded=False,
            )
        return Observation(
            execution_id=uuid4(),
            evidence={"weather": weather_evidence(reading)},
            command_succeeded=True,
        )


class WeatherVerifier:
    def verify(
        self, observation: Observation, _contract: VerificationContract
    ) -> VerificationResult:
        evidence = observation.evidence.get("weather")
        verified = observation.command_succeeded and isinstance(evidence, dict)
        return VerificationResult(
            verified=verified,
            evidence={"weather": evidence} if verified else observation.evidence,
            reason="weather readback is structurally valid" if verified else "weather read failed",
        )


class PublicHolidayExecutor:
    def execute(self, request: ExecutionRequest) -> Observation:
        try:
            country_code = str(request.action.arguments["country_code"]).upper()
            year = int(request.action.arguments["year"])
            if len(country_code) != 2 or not 1900 <= year <= 2200:
                raise ValueError("holiday country or year is invalid")
            holidays = configured_holiday_provider().list_holidays(country_code, year)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return Observation(
                execution_id=uuid4(),
                evidence={"holidays": "unavailable", "reason": str(exc)},
                command_succeeded=False,
            )
        return Observation(
            execution_id=uuid4(), evidence=holidays_evidence(holidays), command_succeeded=True
        )


class PublicHolidayVerifier:
    def verify(
        self, observation: Observation, _contract: VerificationContract
    ) -> VerificationResult:
        holidays = observation.evidence.get("holidays")
        verified = observation.command_succeeded and isinstance(holidays, list)
        holiday_rows = holidays if isinstance(holidays, list) else []
        return VerificationResult(
            verified=verified,
            evidence={"holiday_count": len(holiday_rows) if verified else 0},
            reason="public holiday readback is structurally valid"
            if verified
            else "public holiday read failed",
        )


class CalendarSnapshotWorkspaceExecutor:
    """Write a pre-authorized calendar read into an isolated Workspace."""

    def __init__(self, principal: Principal) -> None:
        self.principal = principal

    def execute(self, request: ExecutionRequest) -> Observation:
        target_path = request.action.arguments.get("target_path")
        if not isinstance(target_path, str) or not target_path.strip():
            return Observation(
                execution_id=uuid4(),
                evidence={"calendar_workspace": "invalid_arguments"},
                command_succeeded=False,
            )
        content = calendar_snapshot_content(configured_calendar_provider().list_events())
        workspace = WorkspaceManager(
            Path(os.environ.get("AEGIS_WORKSPACE_ROOT", "/tmp/aegis-owner-workspaces"))
        ).for_objective(self.principal.id, request.objective_id)
        try:
            artifact = workspace.write_artifact(
                {target_path: content}, request.action_id, lambda current: None
            )
        except (ValueError, OSError) as exc:
            return Observation(
                execution_id=uuid4(),
                evidence={"calendar_workspace": "rejected", "reason": str(exc)},
                command_succeeded=False,
            )
        return Observation(
            execution_id=uuid4(),
            evidence={
                "composition": "calendar_snapshot_to_workspace",
                "files": list(artifact.files),
                "executor_local_validated": artifact.validated,
            },
            command_succeeded=True,
        )


class CalendarSnapshotWorkspaceVerifier:
    def __init__(self, principal: Principal) -> None:
        self.principal = principal

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        verified, detail = _verify_workspace_expectation(self.principal, contract)
        verified = observation.command_succeeded and verified
        return VerificationResult(
            verified=verified,
            evidence={"calendar_snapshot_workspace_verified": verified, "postcondition": detail},
            reason="calendar snapshot independently verified"
            if verified
            else f"calendar snapshot verification failed: {detail}",
        )


class CalendarCommunicationDraftExecutor:
    def __init__(self, principal: Principal) -> None:
        self.principal = principal

    def execute(self, request: ExecutionRequest) -> Observation:
        recipient = request.action.arguments.get("recipient")
        target_path = request.action.arguments.get("target_path")
        if not all(isinstance(value, str) and value.strip() for value in (recipient, target_path)):
            return Observation(
                execution_id=uuid4(),
                evidence={"calendar_communication": "invalid_arguments"},
                command_succeeded=False,
            )
        assert isinstance(recipient, str)
        assert isinstance(target_path, str)
        body = calendar_snapshot_content(configured_calendar_provider().list_events())
        content = f"# Draft message\n\nTo: {recipient}\nSubject: Calendar snapshot\n\n{body}"
        workspace = WorkspaceManager(
            Path(os.environ.get("AEGIS_WORKSPACE_ROOT", "/tmp/aegis-owner-workspaces"))
        ).for_objective(self.principal.id, request.objective_id)
        try:
            artifact = workspace.write_artifact(
                {target_path: content}, request.action_id, lambda current: None
            )
        except (ValueError, OSError) as exc:
            return Observation(
                execution_id=uuid4(),
                evidence={"calendar_communication": "rejected", "reason": str(exc)},
                command_succeeded=False,
            )
        return Observation(
            execution_id=uuid4(),
            evidence={
                "composition": "calendar_to_communication_draft",
                "files": list(artifact.files),
                "executor_local_validated": artifact.validated,
                "sent": False,
            },
            command_succeeded=True,
        )


class CalendarCommunicationDraftVerifier:
    def __init__(self, principal: Principal) -> None:
        self.principal = principal

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        verified, detail = _verify_workspace_expectation(self.principal, contract)
        verified = (
            observation.command_succeeded and verified and observation.evidence.get("sent") is False
        )
        return VerificationResult(
            verified=verified,
            evidence={"calendar_communication_verified": verified, "postcondition": detail},
            reason="calendar communication draft independently verified"
            if verified
            else f"calendar communication draft verification failed: {detail}",
        )


class WorkspaceArtifactExecutor:
    def __init__(self, principal: Principal) -> None:
        self.principal = principal

    def execute(self, request: ExecutionRequest) -> Observation:
        args = request.action.arguments
        path, content = args.get("path"), args.get("content")
        files = args.get("files")
        if files is not None:
            if (
                not isinstance(files, dict)
                or not files
                or len(files) > 50
                or any(
                    not isinstance(item, str) or not isinstance(value, str)
                    for item, value in files.items()
                )
            ):
                return Observation(
                    execution_id=uuid4(),
                    evidence={"workspace": "invalid_arguments"},
                    command_succeeded=False,
                )
            artifact_files = files
        elif isinstance(path, str) and isinstance(content, str):
            artifact_files = {path: content}
        else:
            return Observation(
                execution_id=uuid4(),
                evidence={"workspace": "invalid_arguments"},
                command_succeeded=False,
            )
        root = Path(os.environ.get("AEGIS_WORKSPACE_ROOT", "/tmp/aegis-owner-workspaces"))
        workspace = WorkspaceManager(root).for_objective(self.principal.id, request.objective_id)
        try:
            artifact = workspace.write_artifact(
                artifact_files,
                request.action_id,
                lambda current: (
                    None
                    if all(current.read(item) == value for item, value in artifact_files.items())
                    else "readback mismatch"
                ),
            )
        except (ValueError, OSError) as exc:
            return Observation(
                execution_id=uuid4(),
                evidence={"workspace": "rejected", "reason": str(exc)},
                command_succeeded=False,
            )
        return Observation(
            execution_id=uuid4(),
            evidence={
                "workspace": "artifact",
                "files": list(artifact.files),
                "executor_local_validated": artifact.validated,
            },
            command_succeeded=True,
        )


class ResearchWorkspaceExecutor:
    """Compose bounded answer-only research with scoped artifact creation."""

    def __init__(self, principal: Principal) -> None:
        self.principal = principal

    def execute(self, request: ExecutionRequest) -> Observation:
        query = request.action.arguments.get("query")
        target_path = request.action.arguments.get("target_path")
        if not isinstance(query, str) or not isinstance(target_path, str):
            return Observation(
                execution_id=uuid4(),
                evidence={"research_workspace": "invalid_arguments"},
                command_succeeded=False,
            )
        try:
            evidence = configured_research_service().collect(SearchRequest(query))
            answer = ResearchAnswer(
                text="\n\n".join(item.text[:4_000] for item in evidence.evidence),
                source_kind=KnowledgeSource.EXTERNAL,
                evidence=evidence,
            )
            result = research_to_workspace(
                answer,
                WorkspaceManager(
                    Path(os.environ.get("AEGIS_WORKSPACE_ROOT", "/tmp/aegis-owner-workspaces"))
                ),
                principal_id=self.principal.id,
                objective_id=request.objective_id,
                target_path=target_path,
                correlation_id=request.action_id,
            )
        except (ResearchUnavailable, ValueError, OSError) as exc:
            return Observation(
                execution_id=uuid4(),
                evidence={"research_workspace": "rejected", "reason": str(exc)},
                command_succeeded=False,
            )
        return Observation(
            execution_id=uuid4(),
            evidence={
                "composition": "research_to_workspace",
                "query": evidence.query,
                "provider_id": evidence.provider_id,
                "source_ids": list(result.source_ids),
                "files": list(result.files),
                "executor_local_validated": result.validated,
                "authoritative": result.authoritative,
            },
            command_succeeded=True,
        )


class ResearchWorkspaceVerifier:
    def verify(
        self, observation: Observation, _contract: VerificationContract
    ) -> VerificationResult:
        # Research output is produced during execution, so its final bytes
        # are not a trusted pre-execution expectation.
        verified = False
        return VerificationResult(
            verified=verified,
            evidence={"research_workspace_verified": verified, "postcondition": "not established"},
            reason="research notes verification is pending a fixed-input workspace step",
        )


class CommunicationDraftExecutor:
    """Create an unsent, owner-scoped message draft with readback."""

    def __init__(self, principal: Principal) -> None:
        self.principal = principal

    def execute(self, request: ExecutionRequest) -> Observation:
        args = request.action.arguments
        recipient, subject, body, target_path = (
            args.get("recipient"),
            args.get("subject"),
            args.get("body"),
            args.get("target_path"),
        )
        if not all(
            isinstance(value, str) and value.strip()
            for value in (recipient, subject, body, target_path)
        ):
            return Observation(
                execution_id=uuid4(),
                evidence={"communication_draft": "invalid_arguments"},
                command_succeeded=False,
            )
        assert isinstance(recipient, str)
        assert isinstance(subject, str)
        assert isinstance(body, str)
        assert isinstance(target_path, str)
        content = f"# Draft message\n\nTo: {recipient}\nSubject: {subject}\n\n{body}\n"
        workspace = WorkspaceManager(
            Path(os.environ.get("AEGIS_WORKSPACE_ROOT", "/tmp/aegis-owner-workspaces"))
        ).for_objective(self.principal.id, request.objective_id)
        try:
            artifact = workspace.write_artifact(
                {target_path: content},
                request.action_id,
                lambda current: (
                    None
                    if current.read(target_path) == content
                    else "communication draft readback mismatch"
                ),
            )
        except (ValueError, OSError) as exc:
            return Observation(
                execution_id=uuid4(),
                evidence={"communication_draft": "rejected", "reason": str(exc)},
                command_succeeded=False,
            )
        return Observation(
            execution_id=uuid4(),
            evidence={
                "composition": "communication_draft",
                "files": list(artifact.files),
                "executor_local_validated": artifact.validated,
                "sent": False,
            },
            command_succeeded=True,
        )


class CommunicationDraftVerifier:
    def __init__(self, principal: Principal) -> None:
        self.principal = principal

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        verified, detail = _verify_workspace_expectation(self.principal, contract)
        verified = (
            observation.command_succeeded and verified and observation.evidence.get("sent") is False
        )
        return VerificationResult(
            verified=verified,
            evidence={"communication_draft_verified": verified, "postcondition": detail},
            reason="unsent communication draft independently verified"
            if verified
            else f"communication draft verification failed: {detail}",
        )


class WorkspaceArtifactVerifier:
    def __init__(self, principal: Principal) -> None:
        self.principal = principal

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        verified, detail = _verify_workspace_expectation(self.principal, contract)
        verified = observation.command_succeeded and verified
        return VerificationResult(
            verified=verified,
            evidence={"workspace_artifact_verified": verified, "postcondition": detail},
            reason="workspace artifact independently verified"
            if verified
            else f"workspace artifact verification failed: {detail}",
        )


class ReferenceExecutor:
    """Deterministic fake adapter with readback evidence, not fake success."""

    def __init__(self, world: ReferenceWorld) -> None:
        self.world = world
        self._completed_keys: set[str] = set()

    def execute(self, request: ExecutionRequest) -> Observation:
        if request.idempotency_key in self._completed_keys:
            return Observation(
                execution_id=uuid4(),
                evidence={"replayed": True, "idempotency_key": request.idempotency_key},
                command_succeeded=True,
            )
        self._completed_keys.add(request.idempotency_key)
        args = request.action.arguments
        if request.action.action_id == "tasks.create":
            title = str(args["title"])
            self.world.tasks.append({"title": title, "status": "open"})
            evidence = {"collection": "tasks", "title": title}
        elif request.action.action_id == "kitchen.groceries.add":
            item = str(args["item"])
            self.world.groceries.append(item)
            evidence = {"collection": "groceries", "item": item}
        elif request.action.action_id == "homelab.service.restart":
            service = str(args["service"])
            self.world.services[service] = "healthy"
            evidence = {"service": service, "health": self.world.services[service]}
        else:
            return Observation(
                execution_id=uuid4(), evidence={"unknown_action": True}, command_succeeded=False
            )
        return Observation(execution_id=uuid4(), evidence=evidence, command_succeeded=True)


class ReferenceVerifier:
    def __init__(self, world: ReferenceWorld) -> None:
        self.world = world

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        evidence = observation.evidence
        if not observation.command_succeeded:
            return VerificationResult(verified=False, evidence=evidence, reason="adapter failed")
        if contract.kind == "readback" and evidence.get("collection") == "tasks":
            ok = any(t["title"] == evidence["title"] for t in self.world.tasks)
        elif contract.kind == "readback" and evidence.get("collection") == "groceries":
            ok = evidence["item"] in self.world.groceries
        elif contract.kind == "health":
            ok = self.world.services.get(str(evidence.get("service"))) == "healthy"
        else:
            ok = False
        return VerificationResult(
            verified=ok,
            evidence=evidence,
            reason="canonical readback verified" if ok else "canonical readback failed",
        )


class OpenClawGroceryExecutor:
    """Execute the existing grocery action through a paired OpenClaw Gateway.

    The acceptance store is an external newline-delimited record. The shell
    command is idempotency-guarded, and the command marker is only transport
    evidence; the verifier independently reads the record afterward.
    """

    def __init__(
        self,
        channel: OpenClawWebSocketChannel,
        state_path: str,
        canonical_store: PostgresHouseholdStore | None = None,
        principal: Any | None = None,
    ) -> None:
        if not channel.persistent:
            raise ValueError("OpenClaw grocery execution requires a persistent channel")
        self.channel = channel
        self.gateway = OpenClawGatewayRpc(CorrelatedRpcClient(channel))
        self.state_path = str(Path(state_path))
        self.canonical_store = canonical_store
        self.principal = principal

    def execute(self, request: ExecutionRequest) -> Observation:
        if request.action.action_id != "kitchen.groceries.add":
            return Observation(
                execution_id=request.action_id,
                evidence={"unknown_action": request.action.action_id},
                command_succeeded=False,
            )
        item = request.action.arguments.get("item")
        if not isinstance(item, str) or not item.strip():
            return Observation(
                execution_id=request.action_id,
                evidence={"invalid_item": True},
                command_succeeded=False,
            )
        marker = f"AEGIS_GROCERY_DONE_{uuid4().hex}"
        # Avoid control characters in PTY input: fish interprets a literal tab
        # as completion input rather than passing it to the command.  LF is
        # required to submit the line; CR only leaves it echoed in this PTY.
        record = f"{request.idempotency_key}|{item.strip()}"
        script = (
            f"touch {shlex.quote(self.state_path)}; "
            f"grep -Fqx -- {shlex.quote(record)} {shlex.quote(self.state_path)} "
            f"|| printf '%s\\n' {shlex.quote(record)} >> {shlex.quote(self.state_path)}; "
            f"printf '%s\\n' {shlex.quote(marker)}"
        )
        try:
            opened = self.gateway.terminal_open({"rows": 40, "cols": 120})
            session_id = opened["sessionId"]
            _wait_for_terminal_ready(self.channel)
            self.gateway.terminal_input(
                {"sessionId": session_id, "data": f"sh -c {shlex.quote(script)}"}
            )
            self.gateway.terminal_input({"sessionId": session_id, "data": "\n"})
            saw_marker = False
            terminal_output = ""
            for _ in range(32):
                event = self.channel.receive_event("terminal.data")
                terminal_output += str(event.get("data", ""))
                if any(line.strip() == marker for line in terminal_output.splitlines()):
                    saw_marker = True
                    break
            self.gateway.terminal_close({"sessionId": session_id})
        except (KeyError, RpcProtocolError):
            return _unknown_gateway_observation(request)
        if saw_marker and self.canonical_store is not None and self.principal is not None:
            try:
                self.canonical_store.add_grocery(
                    self.principal, item.strip(), request.idempotency_key
                )
            except (PermissionError, ValueError) as exc:
                return Observation(
                    execution_id=request.action_id,
                    evidence={
                        "gateway": "openclaw",
                        "external_state_path": self.state_path,
                        "idempotency_key": request.idempotency_key,
                        "canonical_persistence_error": str(exc),
                    },
                    command_succeeded=False,
                )
        return Observation(
            execution_id=request.action_id,
            evidence={
                "gateway": "openclaw",
                "external_state_path": self.state_path,
                "idempotency_key": request.idempotency_key,
                "terminal_marker": marker,
                "terminal_marker_observed": saw_marker,
                "terminal_output_bytes": len(terminal_output),
            },
            command_succeeded=saw_marker,
        )


class OpenClawGroceryVerifier:
    """Independently read external grocery state after Gateway execution."""

    def __init__(
        self,
        canonical_store: PostgresHouseholdStore | None = None,
        principal: Any | None = None,
    ) -> None:
        self.canonical_store = canonical_store
        self.principal = principal

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        if contract.kind != "readback" or not observation.command_succeeded:
            return VerificationResult(
                verified=False,
                evidence=observation.evidence,
                reason="Gateway execution or verification contract failed",
            )
        path = observation.evidence.get("external_state_path")
        key = observation.evidence.get("idempotency_key")
        if not isinstance(path, str) or not isinstance(key, str):
            return VerificationResult(
                verified=False,
                evidence=observation.evidence,
                reason="missing external readback identity",
            )
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            return VerificationResult(
                verified=False,
                evidence={**observation.evidence, "readback_error": str(exc)},
                reason="external readback failed",
            )
        matches = [line for line in lines if line.startswith(f"{key}|")]
        verified = len(matches) == 1
        canonical_verified = True
        if verified and self.canonical_store is not None and self.principal is not None:
            try:
                canonical_verified = self.canonical_store.grocery_recorded(
                    self.principal, matches[0].split("|", 1)[1], key
                )
            except (PermissionError, ValueError):
                canonical_verified = False
        verified = verified and canonical_verified
        return VerificationResult(
            verified=verified,
            evidence={
                **observation.evidence,
                "external_records_for_key": len(matches),
                "canonical_grocery_verified": canonical_verified,
            },
            reason="external and canonical grocery readback verified"
            if verified
            else "external grocery postcondition failed",
        )


class OpenClawHomelabExecutor:
    """Run a preconfigured service restart through OpenClaw terminal execution.

    Commands are supplied by the trusted adapter configuration, never by model
    arguments. The model can select a known service identifier only.
    """

    def __init__(
        self,
        channel: OpenClawWebSocketChannel,
        services: dict[str, str],
    ) -> None:
        if not channel.persistent:
            raise ValueError("Homelab execution requires a persistent channel")
        self.channel = channel
        self.gateway = OpenClawGatewayRpc(CorrelatedRpcClient(channel))
        self.services = services

    def execute(self, request: ExecutionRequest) -> Observation:
        if request.action.action_id != "homelab.service.restart":
            return Observation(
                execution_id=request.action_id,
                evidence={"unknown_action": request.action.action_id},
                command_succeeded=False,
            )
        service = request.action.arguments.get("service")
        command = self.services.get(str(service))
        if command is None:
            return Observation(
                execution_id=request.action_id,
                evidence={"unknown_service": service},
                command_succeeded=False,
            )
        marker = f"AEGIS_HOMELAB_DONE_{uuid4().hex}"
        script = f"{command}; printf '%s\\n' {shlex.quote(marker)}"
        terminal_output = ""
        try:
            opened = self.gateway.terminal_open({"rows": 40, "cols": 120})
            session_id = opened["sessionId"]
            _wait_for_terminal_ready(self.channel)
            self.gateway.terminal_input(
                {"sessionId": session_id, "data": f"sh -c {shlex.quote(script)}"}
            )
            self.gateway.terminal_input({"sessionId": session_id, "data": "\n"})
            saw_marker = False
            for _ in range(32):
                event = self.channel.receive_event("terminal.data")
                terminal_output += str(event.get("data", ""))
                if any(line.strip() == marker for line in terminal_output.splitlines()):
                    saw_marker = True
                    break
            self.gateway.terminal_close({"sessionId": session_id})
        except (KeyError, RpcProtocolError):
            return _unknown_gateway_observation(request)
        return Observation(
            execution_id=request.action_id,
            evidence={
                "gateway": "openclaw",
                "service": str(service),
                "terminal_marker": marker,
                "terminal_marker_observed": saw_marker,
                "terminal_output_bytes": len(terminal_output),
            },
            command_succeeded=saw_marker,
        )


class OpenClawHomelabVerifier:
    """Verify service health with an independent HTTP read, not Gateway output."""

    def __init__(self, health_endpoints: dict[str, str], timeout: float = 5.0) -> None:
        self.health_endpoints = health_endpoints
        self.timeout = timeout

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        service = observation.evidence.get("service")
        endpoint = self.health_endpoints.get(str(service))
        if contract.kind != "health" or not observation.command_succeeded or endpoint is None:
            return VerificationResult(
                verified=False,
                evidence=observation.evidence,
                reason="Homelab restart or health contract failed",
            )
        try:
            with urllib.request.urlopen(endpoint, timeout=self.timeout) as response:
                status = response.status
                body_bytes = len(response.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return VerificationResult(
                verified=False,
                evidence={**observation.evidence, "health_error": str(exc)},
                reason="independent service health read failed",
            )
        verified = status == 200
        return VerificationResult(
            verified=verified,
            evidence={
                **observation.evidence,
                "health_endpoint": endpoint,
                "health_status": status,
                "health_body_bytes": body_bytes,
            },
            reason="independent service health verified" if verified else "service health failed",
        )


class OpenClawNetworkProbeExecutor:
    """Run a bounded ping through OpenClaw after Core scope authorization."""

    def __init__(self, channel: OpenClawWebSocketChannel) -> None:
        if not channel.persistent:
            raise ValueError("Network probing requires a persistent channel")
        self.channel = channel
        self.gateway = OpenClawGatewayRpc(CorrelatedRpcClient(channel))

    def execute(self, request: ExecutionRequest) -> Observation:
        if request.action.action_id != "network.probe":
            return Observation(
                execution_id=request.action_id,
                evidence={"unknown_action": request.action.action_id},
                command_succeeded=False,
            )
        address = request.action.arguments.get("address")
        port = request.action.arguments.get("port")
        if not isinstance(address, str) or not isinstance(port, int) or not 1 <= port <= 65535:
            return Observation(
                execution_id=request.action_id,
                evidence={"invalid_endpoint": True},
                command_succeeded=False,
            )
        marker = f"AEGIS_NETWORK_DONE_{uuid4().hex}"
        receipt_path = Path(f"/tmp/{marker}.receipt")
        script = (
            f"curl --fail --silent --show-error --max-time 3 "
            f"-o /dev/null -w HTTP_%{{http_code}}\\n "
            f"{shlex.quote(f'http://{address}:{port}/')} 2>&1; code=$?; "
            f'printf "%s %s\\n" {shlex.quote(marker)} $code > {shlex.quote(str(receipt_path))}'
        )
        terminal_output = ""
        try:
            opened = self.gateway.terminal_open({"rows": 40, "cols": 120})
            session_id = opened["sessionId"]
            _wait_for_terminal_ready(self.channel)
            self.gateway.terminal_input(
                {"sessionId": session_id, "data": f"sh -c {shlex.quote(script)}"}
            )
            self.gateway.terminal_input({"sessionId": session_id, "data": "\n"})
            for _ in range(50):
                if receipt_path.is_file():
                    break
                time.sleep(0.1)
            for _ in range(32):
                event = self.channel.receive_event("terminal.data")
                terminal_output += str(event.get("data", ""))
                if receipt_path.is_file() or marker in terminal_output:
                    break
            self.gateway.terminal_close({"sessionId": session_id})
        except (KeyError, RpcProtocolError):
            return _unknown_gateway_observation(request)
        receipt = receipt_path.read_text() if receipt_path.is_file() else ""
        receipt_path.unlink(missing_ok=True)
        success = any(line.strip().endswith(" 0") for line in receipt.splitlines())
        return Observation(
            execution_id=request.action_id,
            evidence={
                "gateway": "openclaw",
                "address": address,
                "port": request.action.arguments.get("port"),
                "terminal_marker": marker,
                "terminal_output_bytes": len(terminal_output),
                "terminal_output_tail": terminal_output[-500:],
                "gateway_receipt": receipt.strip(),
            },
            command_succeeded=success,
        )


class OpenClawNetworkProbeVerifier:
    """Independently verify target reachability with a TCP connection."""

    def __init__(self, timeout: float = 3.0) -> None:
        self.timeout = timeout

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        address = observation.evidence.get("address")
        port = observation.evidence.get("port")
        if contract.kind != "health" or not observation.command_succeeded:
            return VerificationResult(
                verified=False,
                evidence=observation.evidence,
                reason="network probe failed",
            )
        if not isinstance(address, str) or not isinstance(port, int):
            return VerificationResult(
                verified=False,
                evidence=observation.evidence,
                reason="network endpoint is invalid",
            )
        try:
            with socket.create_connection((address, port), timeout=self.timeout):
                pass
        except OSError as exc:
            return VerificationResult(
                verified=False,
                evidence={**observation.evidence, "readback_error": str(exc)},
                reason="independent network readback failed",
            )
        return VerificationResult(
            verified=True,
            evidence={**observation.evidence, "tcp_readback": "connected"},
            reason="independent network reachability verified",
        )


class PostgresGroceryListExecutor:
    """Adapt the canonical Kitchen grocery read to the generic Executor port."""

    def __init__(self, store: PostgresHouseholdStore, principal: Any) -> None:
        self.store = store
        self.principal = principal

    def execute(self, request: ExecutionRequest) -> Observation:
        if request.action.action_id != "kitchen.groceries.list":
            return Observation(
                execution_id=request.action_id,
                evidence={"unknown_action": request.action.action_id},
                command_succeeded=False,
            )
        return Observation(
            execution_id=request.action_id,
            evidence={
                "collection": "groceries",
                "items": list(self.store.list_groceries(self.principal)),
            },
            command_succeeded=True,
        )


class PostgresGroceryListVerifier:
    """Independently compare a grocery list observation with canonical state."""

    def __init__(self, store: PostgresHouseholdStore, principal: Any) -> None:
        self.store = store
        self.principal = principal

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        if contract.kind != "readback" or not observation.command_succeeded:
            return VerificationResult(
                verified=False, evidence=observation.evidence, reason="grocery list read failed"
            )
        expected = observation.evidence.get("items")
        actual = list(self.store.list_groceries(self.principal))
        verified = expected == actual
        return VerificationResult(
            verified=verified,
            evidence={**observation.evidence, "canonical_items": actual},
            reason=(
                "canonical grocery list verified" if verified else "canonical grocery list changed"
            ),
        )
