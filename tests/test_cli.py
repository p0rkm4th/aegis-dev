import hashlib
import json
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from aegis import InteractionBoundary, InteractionDependencies, InteractionInputError
from aegis.cli import _deterministic_composition_action, _domain_and_action, _ensure_local_identity
from aegis.contracts import (
    ActionSpec,
    Context,
    Decision,
    DecisionKind,
    IntentFrame,
    Objective,
    ObjectiveState,
    Principal,
    Result,
    StructuralAnchor,
    StructuralCoverageSignal,
    VerificationContract,
)
from aegis.interaction import (
    _argument_provenance_error,
    _authorized_context_evidence,
    _context_from_prior_result,
    _with_continuation_context,
)
from aegis.pack_lifecycle import PackManager
from aegis.reference_interaction import build_reference_fallback_context, reference_fallback_cards
from aegis.reference_packs import reference_bundles, reference_packs
from aegis.tasks import (
    Task,
    TaskIntentClarificationFastPath,
    TaskPriorityFastPath,
    TaskReadFastPath,
    TaskStatus,
    ground_task_due_at,
)
from aegis.web import BrowserApp


def test_interaction_boundary_is_public_without_live_runtime():
    assert InteractionBoundary.__module__ == "aegis.interaction"
    assert InteractionDependencies.__module__ == "aegis.interaction"


def test_deterministic_research_workspace_action_requires_enabled_pack():
    manager = PackManager()
    workspace = next(
        bundle for bundle in reference_bundles() if bundle.manifest.pack_id == "workspace"
    )
    manager.discover(workspace)
    manager.install("workspace", frozenset({"workspace.write"}))
    manager.enable("workspace")
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="vault"),
        utterance="Research the guide and save sourced notes to workspace as notes.md",
    )
    card = _deterministic_composition_action(intent, manager, Context())
    assert card is not None
    assert card.action.action_id == "workspace.research_notes.create"
    assert card.action.arguments["target_path"] == "notes.md"


def test_deterministic_workspace_artifact_action_preserves_explicit_file_content():
    manager = PackManager()
    bundle = next(
        bundle for bundle in reference_bundles() if bundle.manifest.pack_id == "workspace"
    )
    manager.discover(bundle)
    manager.install("workspace", frozenset({"workspace.write"}))
    manager.enable("workspace")
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="vault"),
        utterance="Create a workspace artifact at owner-proof.html with content owner proof",
    )
    card = _deterministic_composition_action(intent, manager, Context())
    assert card is not None
    assert card.action.action_id == "workspace.artifact.create"
    assert card.action.arguments == {"path": "owner-proof.html", "content": "owner proof"}


def test_deterministic_calendar_workspace_report_action_uses_pack_metadata():
    manager = PackManager()
    bundle = next(
        bundle for bundle in reference_bundles() if bundle.manifest.pack_id == "calendar-reports"
    )
    manager.discover(bundle)
    manager.install("calendar-reports", frozenset({"calendar.read", "workspace.write"}))
    manager.enable("calendar-reports")
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="vault"),
        utterance="Save my calendar snapshot to workspace as agenda.md",
    )
    card = _deterministic_composition_action(intent, manager, Context())
    assert card is not None
    assert card.action.action_id == "calendar-reports.events.snapshot_to_workspace"
    assert card.action.arguments["target_path"] == "agenda.md"


def test_deterministic_calendar_communication_draft_requires_recipient_and_path():
    manager = PackManager()
    bundle = next(
        bundle
        for bundle in reference_bundles()
        if bundle.manifest.pack_id == "calendar-communications"
    )
    manager.discover(bundle)
    manager.install(
        "calendar-communications",
        frozenset({"calendar.read", "communications.draft", "workspace.write"}),
    )
    manager.enable("calendar-communications")
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="vault"),
        utterance="Draft my calendar for alice as schedule.md",
    )
    card = _deterministic_composition_action(intent, manager, Context())
    assert card is not None
    assert card.action.action_id == "calendar-communications.events.draft"
    assert card.action.arguments == {"recipient": "alice", "target_path": "schedule.md"}


def test_deterministic_calendar_create_action_requires_explicit_times():
    manager = PackManager()
    bundle = next(bundle for bundle in reference_bundles() if bundle.manifest.pack_id == "calendar")
    manager.discover(bundle)
    manager.install("calendar", frozenset({"calendar.read", "calendar.write"}))
    manager.enable("calendar")
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="vault"),
        utterance=(
            "Create a calendar event titled Dinner from "
            "2026-09-07T19:00:00+00:00 to 2026-09-07T20:00:00+00:00 at 7:00 pm"
        ),
    )
    card = _deterministic_composition_action(intent, manager, Context())
    assert card is not None
    assert card.action.action_id == "calendar.events.create"
    assert card.action.arguments["title"] == "Dinner"


def test_deterministic_calendar_create_accepts_bounded_natural_timed_request():
    manager = PackManager()
    bundle = next(bundle for bundle in reference_bundles() if bundle.manifest.pack_id == "calendar")
    manager.discover(bundle)
    manager.install("calendar", frozenset({"calendar.read", "calendar.write"}))
    manager.enable("calendar")
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="vault"),
        utterance="Put dinner on my calendar tomorrow at 7 pm",
    )
    card = _deterministic_composition_action(intent, manager, Context())
    assert card is not None
    assert card.action.action_id == "calendar.events.create"
    assert card.action.arguments["title"] == "dinner"
    assert card.action.arguments["starts_at"].endswith("T19:00:00+00:00")


def test_deterministic_homelab_health_action_uses_explicit_service():
    manager = PackManager()
    bundle = next(bundle for bundle in reference_bundles() if bundle.manifest.pack_id == "homelab")
    manager.discover(bundle)
    manager.install("homelab", frozenset({"homelab.service.restart", "homelab.read"}))
    manager.enable("homelab")
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="vault"),
        utterance="Check health of service acceptance-plex",
    )
    card = _deterministic_composition_action(intent, manager, Context())
    assert card is not None
    assert card.action.action_id == "homelab.service.health"
    assert card.action.arguments == {"service": "acceptance-plex"}


def test_deterministic_network_probe_preserves_explicit_endpoint_and_scope():
    manager = PackManager()
    bundle = next(bundle for bundle in reference_bundles() if bundle.manifest.pack_id == "network")
    manager.discover(bundle)
    manager.install("network", frozenset({"network.read"}))
    manager.enable("network")
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="vault"),
        utterance="Probe 10.0.0.5 in scope alpha-lab on port 443",
    )

    card = _deterministic_composition_action(intent, manager, Context())

    assert card is not None
    assert card.action.action_id == "network.probe"
    assert card.action.arguments == {
        "address": "10.0.0.5",
        "scope_id": "alpha-lab",
        "port": 443,
    }


def test_deterministic_device_research_action_preserves_explicit_entity_and_query():
    manager = PackManager()
    bundle = next(bundle for bundle in reference_bundles() if bundle.manifest.pack_id == "devices")
    manager.discover(bundle)
    manager.install("devices", frozenset({"devices.read"}))
    manager.enable("devices")
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="vault"),
        utterance="Research the current state of light.desk for why it is off",
    )

    card = _deterministic_composition_action(intent, manager, Context())

    assert card is not None
    assert card.action.action_id == "devices.states.research"
    assert card.action.arguments == {
        "entity_id": "light.desk",
        "query": "why it is off",
    }


def test_device_research_context_sanitizes_public_query_and_reads_authorized_state(monkeypatch):
    from aegis import cli

    monkeypatch.setenv("AEGIS_AUTHORIZED_DEVICE_ENTITIES", "light.desk")
    monkeypatch.setattr(
        cli.DeviceStatesExecutor,
        "execute",
        lambda *_args: type(
            "Observation",
            (),
            {"evidence": {"states": [{"entity_id": "light.desk", "state": "off"}]}},
        )(),
    )

    result = cli._device_research_context("Research why light.desk is currently off")

    assert result == (
        "Research why the smart-home device is currently off",
        {
            "entity_id": "light.desk",
            "state": "off",
            "observed_at": None,
            "provenance": "authorized_observed_device_state",
        },
    )


def test_deterministic_homelab_page_action_uses_explicit_workspace_path():
    manager = PackManager()
    bundle = next(
        bundle for bundle in reference_bundles() if bundle.manifest.pack_id == "homelab-reports"
    )
    manager.discover(bundle)
    manager.install("homelab-reports", frozenset({"homelab.read", "workspace.write"}))
    manager.enable("homelab-reports")
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="vault"),
        utterance="Make me a little webpage showing my homelab as homelab.html",
    )
    card = _deterministic_composition_action(intent, manager, Context())
    assert card is not None
    assert card.action.action_id == "homelab-reports.inventory.to_workspace"
    assert card.action.arguments == {"target_path": "homelab.html"}


def test_deterministic_homelab_health_report_uses_bounded_workspace_action():
    manager = PackManager()
    bundle = next(
        bundle for bundle in reference_bundles() if bundle.manifest.pack_id == "homelab-reports"
    )
    manager.discover(bundle)
    manager.install("homelab-reports", frozenset({"homelab.read", "workspace.write"}))
    manager.enable("homelab-reports")
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="vault"),
        utterance="Create a homelab health report as health.html",
    )
    card = _deterministic_composition_action(intent, manager, Context())
    assert card is not None
    assert card.action.action_id == "homelab-reports.health.to_workspace"
    assert card.action.arguments == {"target_path": "health.html"}


def test_deterministic_message_send_requires_explicit_destination_and_channel():
    manager = PackManager()
    bundle = next(
        bundle for bundle in reference_bundles() if bundle.manifest.pack_id == "communications"
    )
    manager.discover(bundle)
    manager.install("communications", frozenset({"communications.read", "communications.send"}))
    manager.enable("communications")
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="vault"),
        utterance="Send a message to alice via sms account personal saying hello",
    )
    card = _deterministic_composition_action(intent, manager, Context())
    assert card is not None
    assert card.action.action_id == "communications.messages.send"
    assert card.action.arguments == {
        "target": "alice",
        "channel": "sms",
        "account": "personal",
        "body": "hello",
    }


def test_deterministic_researched_message_draft_preserves_bounded_source_marker():
    manager = PackManager()
    bundle = next(
        bundle
        for bundle in reference_bundles()
        if bundle.manifest.pack_id == "communication-drafts"
    )
    manager.discover(bundle)
    manager.install(bundle.manifest.pack_id, frozenset(bundle.manifest.permissions))
    manager.enable(bundle.manifest.pack_id)
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="vault"),
        utterance=(
            "Draft researched message to scotty with subject Server tips about casual settings, "
            "save it as research.md"
        ),
    )
    card = _deterministic_composition_action(intent, manager, Context())
    assert card is not None
    assert card.action.action_id == "communication-drafts.messages.draft"
    assert card.action.arguments["body_source"] == "bounded.research"
    assert card.action.arguments["query"] == "casual settings"


def test_deterministic_document_export_accepts_named_document_and_target():
    manager = PackManager()
    bundle = next(
        bundle for bundle in reference_bundles() if bundle.manifest.pack_id == "documents"
    )
    manager.discover(bundle)
    manager.install(bundle.manifest.pack_id, frozenset(bundle.manifest.permissions))
    manager.enable(bundle.manifest.pack_id)
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="vault"),
        utterance="Export Alpha Handbook to handbook-copy.md",
    )
    card = _deterministic_composition_action(intent, manager, Context())
    assert card is not None
    assert card.action.action_id == "documents.export_to_workspace"
    assert card.action.arguments == {
        "document_id": "Alpha Handbook",
        "target_path": "handbook-copy.md",
    }


def test_deterministic_document_summary_accepts_named_document_and_target():
    manager = PackManager()
    bundle = next(
        bundle for bundle in reference_bundles() if bundle.manifest.pack_id == "documents"
    )
    manager.discover(bundle)
    manager.install(bundle.manifest.pack_id, frozenset(bundle.manifest.permissions))
    manager.enable(bundle.manifest.pack_id)
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="vault"),
        utterance="Summarize Alpha Handbook to handbook-summary.md",
    )
    card = _deterministic_composition_action(intent, manager, Context())
    assert card is not None
    assert card.action.action_id == "documents.summarize_to_workspace"


def test_deterministic_document_search_accepts_explicit_query():
    manager = PackManager()
    bundle = next(
        bundle for bundle in reference_bundles() if bundle.manifest.pack_id == "documents"
    )
    manager.discover(bundle)
    manager.install(bundle.manifest.pack_id, frozenset(bundle.manifest.permissions))
    manager.enable(bundle.manifest.pack_id)
    card = _deterministic_composition_action(
        IntentFrame(
            principal=Principal(id="alice", vault_id="vault"),
            utterance="Search my documents for guidance",
        ),
        manager,
        Context(),
    )
    assert card is not None
    assert card.action.action_id == "documents.search"
    assert card.action.arguments == {"query": "guidance"}


def test_deterministic_document_search_workspace_accepts_query_and_target():
    manager = PackManager()
    bundle = next(
        bundle for bundle in reference_bundles() if bundle.manifest.pack_id == "documents"
    )
    manager.discover(bundle)
    manager.install(bundle.manifest.pack_id, frozenset(bundle.manifest.permissions))
    manager.enable(bundle.manifest.pack_id)
    card = _deterministic_composition_action(
        IntentFrame(
            principal=Principal(id="alice", vault_id="vault"),
            utterance="Find my documents for guidance and save results as search.md",
        ),
        manager,
        Context(),
    )
    assert card is not None
    assert card.action.action_id == "documents.search_to_workspace"
    assert card.action.arguments == {"query": "guidance", "target_path": "search.md"}


def test_workspace_multi_file_provenance_accepts_bounded_component_spans():
    manager = PackManager()
    bundle = next(
        bundle for bundle in reference_bundles() if bundle.manifest.pack_id == "workspace"
    )
    manager.discover(bundle)
    manager.install("workspace", frozenset({"workspace.write"}))
    manager.enable("workspace")
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="vault"),
        utterance=(
            "Create a workspace artifact with files index.html containing home and "
            "style.css containing body"
        ),
    )
    card = _deterministic_composition_action(intent, manager, Context())
    assert card is not None
    from aegis.reference_interaction import _ground_argument_provenance

    grounded = _ground_argument_provenance(intent, card, Context())
    assert not isinstance(grounded, Result)
    assert (
        _argument_provenance_error(
            grounded.action, intent.utterance, card=grounded, context=Context()
        )
        is None
    )


def test_deterministic_device_control_action_requires_explicit_entity_and_postcondition():
    manager = PackManager()
    bundle = next(
        bundle for bundle in reference_bundles() if bundle.manifest.pack_id == "device-controls"
    )
    manager.discover(bundle)
    manager.install("device-controls", frozenset(bundle.manifest.permissions))
    manager.enable("device-controls")
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="vault"),
        utterance="Turn on light.desk and verify it",
    )
    card = _deterministic_composition_action(intent, manager, Context())
    assert card is not None
    assert card.action.action_id == "device-controls.devices.command.execute"
    assert card.action.arguments == {
        "entity_id": "light.desk",
        "service": "turn_on",
        "expected_state": "on",
    }


def test_deterministic_device_workspace_report_action_requires_explicit_path():
    manager = PackManager()
    bundle = next(
        bundle for bundle in reference_bundles() if bundle.manifest.pack_id == "device-reports"
    )
    manager.discover(bundle)
    manager.install("device-reports", frozenset(bundle.manifest.permissions))
    manager.enable("device-reports")
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="vault"),
        utterance="Save authorized device states to workspace as devices.md",
    )
    card = _deterministic_composition_action(intent, manager, Context())
    assert card is not None
    assert card.action.action_id == "device-reports.devices.snapshot_to_workspace"
    assert card.action.arguments == {"target_path": "devices.md"}


def test_deterministic_communication_draft_action_preserves_explicit_arguments():
    manager = PackManager()
    bundle = next(
        bundle
        for bundle in reference_bundles()
        if bundle.manifest.pack_id == "communication-drafts"
    )
    manager.discover(bundle)
    permissions = frozenset(bundle.manifest.permissions)
    manager.install("communication-drafts", permissions)
    manager.enable("communication-drafts")
    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="vault"),
        utterance=(
            "Draft a message to Maya with subject Weekend plan saying "
            "We should meet Saturday, save it as drafts/weekend-plan.md"
        ),
    )
    card = _deterministic_composition_action(intent, manager, Context())
    assert card is not None
    assert card.action.action_id == "communication-drafts.messages.draft"
    assert card.action.arguments == {
        "recipient": "Maya",
        "subject": "Weekend plan",
        "body": "We should meet Saturday",
        "target_path": "drafts/weekend-plan.md",
    }


def test_bounded_model_fallback_accepts_non_authoritative_answer():
    from aegis.interaction import InteractionBoundary

    class Provider:
        def decide(self, request):
            assert request.classification_only is False
            assert request.action_cards == ()
            assert request.working_set.context == Context()
            return type("Response", (), {"raw": {"kind": "ANSWER", "answer": "A fish story"}})()

    boundary = InteractionBoundary(
        InteractionDependencies(
            connect=lambda _url: None,
            required=lambda _name: "unused",
            apply_migrations=lambda _connection: None,
            ensure_local_identity=lambda _connection, _principal: None,
            select_action=lambda _utterance, _manager: ("unused", None),
            openclaw_channel=lambda: None,
            local_identity=lambda: False,
            model_provider=lambda: Provider(),
        )
    )
    decision = boundary._fallback_decision(
        IntentFrame(
            principal=Principal(id="alice", vault_id="vault"), utterance="Write a fish story"
        ),
        (),
        Context(),
    )

    assert isinstance(decision, Decision)
    assert decision.kind is DecisionKind.ANSWER
    assert decision.answer == "A fish story"


def test_bounded_model_fallback_exposes_safe_invalid_decision_diagnostic():
    class Provider:
        def decide(self, _request):
            return type("Response", (), {"raw": {"kind": "ACTION"}})()

    boundary = InteractionBoundary(
        InteractionDependencies(
            connect=lambda _url: None,
            required=lambda _name: "unused",
            apply_migrations=lambda _connection: None,
            ensure_local_identity=lambda _connection, _principal: None,
            select_action=lambda _utterance, _manager: ("unused", None),
            openclaw_channel=lambda: None,
            local_identity=lambda: False,
            model_provider=lambda: Provider(),
        )
    )

    result = boundary._fallback_decision(
        IntentFrame(
            principal=Principal(id="alice", vault_id="vault"),
            utterance="what needs attention?",
        ),
        (),
        Context(),
    )

    assert isinstance(result, Result)
    assert result.state is ObjectiveState.BLOCKED
    assert result.evidence == {
        "provenance": "model_boundary",
        "authoritative": False,
        "failure_class": "invalid_model_decision",
        "failure_reason": "ACTION requires an action reference",
    }


def test_bounded_model_fallback_recovers_safe_answer_for_non_mutation_question():
    class Provider:
        def __init__(self):
            self.calls = 0

        def decide(self, _request):
            self.calls += 1
            if self.calls == 1:
                return type("Response", (), {"raw": {"kind": "ACTION"}})()
            return type(
                "Response",
                (),
                {"raw": {"kind": "ANSWER", "answer": "The restore drill is open."}},
            )()

    provider = Provider()
    boundary = InteractionBoundary(
        InteractionDependencies(
            connect=lambda _url: None,
            required=lambda _name: "unused",
            apply_migrations=lambda _connection: None,
            ensure_local_identity=lambda _connection, _principal: None,
            select_action=lambda _utterance, _manager: ("unused", None),
            openclaw_channel=lambda: None,
            local_identity=lambda: False,
            model_provider=lambda: provider,
        )
    )

    result = boundary._fallback_decision(
        IntentFrame(
            principal=Principal(id="alice", vault_id="vault"),
            utterance="what remains open?",
        ),
        (),
        Context(),
    )

    assert isinstance(result, Decision)
    assert result.kind is DecisionKind.ANSWER
    assert result.answer == "The restore drill is open."
    assert provider.calls == 2


def test_bounded_model_fallback_recovers_one_focused_authorized_read():
    class Provider:
        def decide(self, _request):
            return type(
                "Response",
                (),
                {
                    "raw": {
                        "kind": "ANSWER",
                        "semantic_mode": "READ",
                        "context_focus": "canonical_items",
                    }
                },
            )()

    boundary = InteractionBoundary(
        InteractionDependencies(
            connect=lambda _url: None,
            required=lambda _name: "unused",
            apply_migrations=lambda _connection: None,
            ensure_local_identity=lambda _connection, _principal: None,
            select_action=lambda _utterance, _manager: ("unused", None),
            openclaw_channel=lambda: None,
            local_identity=lambda: False,
            model_provider=lambda: Provider(),
        )
    )
    context = Context(
        values={"canonical_facts": {"canonical_items": ["rice", "milk"]}},
        sources=("authorized_canonical_context",),
    )

    decision = boundary._fallback_decision(
        IntentFrame(
            principal=Principal(id="alice", vault_id="vault"),
            utterance="what should I pick up?",
        ),
        (),
        context,
    )

    assert isinstance(decision, Decision)
    assert decision.answer == "Authorized groceries: rice, milk"
    assert decision.context_focus == "canonical_items"


def test_write_capable_fallback_routes_without_canonical_read_context():
    from aegis.interaction import InteractionBoundary

    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "tasks.complete"
    )
    principal = Principal(id="alice", vault_id="alice-vault")
    context = Context(
        values={"canonical_facts": {"canonical_items": ["rice"]}},
        sources=("authorized_canonical_result",),
    )

    class Provider:
        def __init__(self):
            self.requests = []

        def decide(self, request):
            self.requests.append(request)
            return type(
                "Response",
                (),
                {
                    "raw": {
                        "kind": "ACTION",
                        "action_ref": card.action.action_id,
                        "action_arguments": {"title": "frontier routing task"},
                    }
                },
            )()

    provider = Provider()
    boundary = InteractionBoundary(
        InteractionDependencies(
            connect=lambda _url: None,
            required=lambda _name: "unused",
            apply_migrations=lambda _connection: None,
            ensure_local_identity=lambda _connection, _principal: None,
            select_action=lambda _utterance, _manager: ("tasks", card),
            openclaw_channel=lambda: None,
            local_identity=lambda: False,
            model_provider=lambda: provider,
            structural_parser=lambda _utterance: StructuralCoverageSignal(
                anchors=(StructuralAnchor(source_span=(0, 1), kind="predicate"),)
            ),
        )
    )

    decision = boundary._fallback_decision(
        IntentFrame(principal=principal, utterance="I finished frontier routing task"),
        (card,),
        context,
    )

    assert isinstance(decision, Decision)
    assert decision.kind is DecisionKind.ACTION
    assert len(provider.requests) == 2
    assert provider.requests[0].classification_only is True
    assert provider.requests[0].action_cards == (card,)
    assert provider.requests[1].routing_only is True
    assert "canonical_facts" not in provider.requests[1].working_set.context.values


def test_semantic_action_reference_prevents_second_pass_read_drift():
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "tasks.create"
    )

    class Provider:
        def __init__(self):
            self.calls = 0

        def decide(self, _request):
            self.calls += 1
            return type(
                "Response",
                (),
                {
                    "raw": {
                        "kind": "ACTION",
                        "semantic_mode": "ACTION",
                        "action_ref": "tasks.create",
                        "action_arguments": {"title": "inspect the backup"},
                    }
                },
            )()

    provider = Provider()
    boundary = InteractionBoundary(
        InteractionDependencies(
            connect=lambda _url: None,
            required=lambda _name: "unused",
            apply_migrations=lambda _connection: None,
            ensure_local_identity=lambda _connection, _principal: None,
            select_action=lambda _utterance, _manager: ("tasks", card),
            openclaw_channel=lambda: None,
            local_identity=lambda: False,
            model_provider=lambda: provider,
            structural_parser=lambda _utterance: StructuralCoverageSignal(
                anchors=(StructuralAnchor(source_span=(0, 1), kind="predicate"),)
            ),
        )
    )

    decision = boundary._fallback_decision(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="put a reminder on my list to inspect the backup",
        ),
        (card,),
        Context(),
    )

    assert isinstance(decision, Decision)
    assert decision.kind is DecisionKind.ACTION
    assert decision.action is not None
    assert decision.action.action_id == "tasks.create"
    assert decision.action.arguments == {"title": "inspect the backup"}
    assert provider.calls == 1


def test_task_read_fast_path_requires_high_confidence_read_shape():
    assert TaskReadFastPath.matches("Show my tasks")
    assert TaskReadFastPath.matches("Give me a quick list of my open tasks")
    assert not TaskReadFastPath.matches("set task status get gud scrub complete")
    assert not TaskReadFastPath.matches("I finished the task get gud scrub")


def test_model_working_context_preserves_only_canonical_task_deadlines():
    principal = Principal(id="alice", vault_id="alice-vault")
    dated = Task(
        uuid4(),
        "apartment",
        "review the restore drill",
        "alice",
        due_at=datetime(2026, 9, 4, 12, 0),
    )
    undated = Task(uuid4(), "apartment", "check the pantry", "alice")

    class Tasks:
        def list(self, _principal):
            return [dated, undated]

    class Household:
        def list_groceries(self, _principal):
            return []

    context = build_reference_fallback_context(
        Context(), Tasks(), Household(), principal, "what deserves my attention?"
    )

    assert context.values["canonical_facts"]["canonical_tasks"] == [
        {
            "task_id": str(dated.task_id),
            "title": "review the restore drill",
            "status": "open",
            "due_at": "2026-09-04T12:00:00",
        },
        {"task_id": str(undated.task_id), "title": "check the pantry", "status": "open"},
    ]
    assert context.values["as_of_date"] == datetime.now(timezone.utc).date().isoformat()


def test_model_working_context_excludes_canonical_state_for_general_subjects():
    principal = Principal(id="alice", vault_id="alice-vault")

    class Tasks:
        def list(self, _principal):
            raise AssertionError("general knowledge must not load task context")

    class Household:
        def list_groceries(self, _principal):
            raise AssertionError("general knowledge must not load household context")

    context = build_reference_fallback_context(
        Context(), Tasks(), Household(), principal, "What is the current version of Rust?"
    )

    assert "canonical_facts" not in context.values


def test_household_read_does_not_match_rent_inside_current():
    from aegis.household import HouseholdReadFastPath

    assert HouseholdReadFastPath.matches("What is the current version of Rust?") is False


def test_household_read_keeps_explicit_obligation_questions():
    from aegis.household import HouseholdReadFastPath

    assert HouseholdReadFastPath.matches("What are my outstanding obligations?") is True
    assert HouseholdReadFastPath.matches("No, show me what is on my calendar.") is True
    assert HouseholdReadFastPath.matches("Can you tell me about my calendar?") is True
    assert HouseholdReadFastPath.matches("Can you tell me what chores are open?") is True
    assert (
        HouseholdReadFastPath.matches("Could you tell me what obligations are outstanding?") is True
    )


def test_grocery_read_accepts_polite_read_prefix():
    from aegis.household import GroceryReadFastPath

    assert GroceryReadFastPath.matches("Can you tell me what is on my grocery list?") is True
    assert GroceryReadFastPath.matches("Can you tell me what groceries are on my list?") is True


def test_household_read_does_not_invent_chore_ordinal_order():
    from aegis.household import Chore, HouseholdReadFastPath

    result = HouseholdReadFastPath(
        {
            "space_id": "home",
            "obligations": (),
            "chores": (Chore("chore-1", "Dishes", "alice", False),),
            "events": (),
        }
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which chore is first?",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "no canonical priority order" in result.message


def test_household_read_selects_latest_event_without_prior_context():
    from aegis.household import HouseholdEvent, HouseholdReadFastPath

    events = (
        HouseholdEvent("early", "early event", datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc)),
        HouseholdEvent("late", "latest event", datetime(2026, 9, 10, 10, 0, tzinfo=timezone.utc)),
    )
    result = HouseholdReadFastPath(
        {"space_id": "home", "obligations": (), "chores": (), "events": events}
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which event is latest?",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["event"]["title"] == "latest event"


def test_household_read_selects_latest_event_from_natural_wording():
    from aegis.household import HouseholdEvent, HouseholdReadFastPath

    events = (
        HouseholdEvent("early", "early event", datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc)),
        HouseholdEvent("late", "latest event", datetime(2026, 9, 10, 10, 0, tzinfo=timezone.utc)),
    )
    result = HouseholdReadFastPath(
        {"space_id": "home", "obligations": (), "chores": (), "events": events}
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="And what is the latest event?",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["event"]["title"] == "latest event"


def test_household_read_selects_latest_event_from_inverted_natural_wording():
    from aegis.household import HouseholdEvent, HouseholdReadFastPath

    events = (
        HouseholdEvent("early", "early event", datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc)),
        HouseholdEvent("late", "latest event", datetime(2026, 9, 10, 10, 0, tzinfo=timezone.utc)),
    )
    result = HouseholdReadFastPath(
        {"space_id": "home", "obligations": (), "chores": (), "events": events}
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="And which is the latest event?",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["event"]["title"] == "latest event"


def test_household_read_selects_next_event_from_natural_wording():
    from aegis.household import HouseholdEvent, HouseholdReadFastPath

    upcoming = datetime.now(timezone.utc) + timedelta(days=1)
    events = (HouseholdEvent("next", "next event", upcoming),)
    result = HouseholdReadFastPath(
        {"space_id": "home", "obligations": (), "chores": (), "events": events}
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="And when is the next event?",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["authorized_next_referent"]["title"] == "next event"


def test_household_read_selects_next_event_from_calendar_wording():
    from aegis.household import HouseholdEvent, HouseholdReadFastPath

    upcoming = datetime.now(timezone.utc) + timedelta(days=1)
    events = (HouseholdEvent("next", "next event", upcoming),)
    result = HouseholdReadFastPath(
        {"space_id": "home", "obligations": (), "chores": (), "events": events}
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What is next on my calendar?",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["authorized_next_referent"]["title"] == "next event"


def test_household_obligation_read_returns_obligations_not_events():
    from aegis.household import HouseholdReadFastPath

    obligation = type(
        "Obligation",
        (),
        {"title": "Utilities", "settled": False, "responsible_id": "alice", "amount": 100},
    )()
    result = HouseholdReadFastPath(
        {"space_id": "home", "obligations": (obligation,), "chores": (), "events": ()}
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What are my outstanding obligations?",
        )
    )

    assert result is not None
    assert "obligations" in result.evidence
    assert "events" not in result.evidence


def test_household_event_read_filters_this_weekend():
    from aegis.household import HouseholdEvent, HouseholdReadFastPath

    now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    saturday = now.date() + timedelta(
        days=5 - now.weekday() if now.weekday() < 5 else -(now.weekday() - 5)
    )
    result = HouseholdReadFastPath(
        {
            "space_id": "home",
            "obligations": (),
            "chores": (),
            "events": (
                HouseholdEvent(
                    "weekend", "Weekend inspection", datetime.combine(saturday, datetime.min.time())
                ),
                HouseholdEvent(
                    "later",
                    "Next weekend inspection",
                    datetime.combine(saturday + timedelta(days=7), datetime.min.time()),
                ),
            ),
        }
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What events are happening this weekend?",
        )
    )

    assert result is not None
    assert result.evidence["date_filter"] == "this_weekend"
    assert [event["title"] for event in result.evidence["events"]] == ["Weekend inspection"]


def test_household_event_read_filters_this_week():
    from aegis.household import HouseholdEvent, HouseholdReadFastPath

    now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    week_end = now.date() + timedelta(days=7 - now.weekday())
    result = HouseholdReadFastPath(
        {
            "space_id": "home",
            "obligations": (),
            "chores": (),
            "events": (
                HouseholdEvent("current", "This week appointment", now + timedelta(hours=1)),
                HouseholdEvent(
                    "later",
                    "Next week appointment",
                    datetime.combine(week_end + timedelta(days=1), datetime.min.time()),
                ),
            ),
        }
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What appointments do I have this week?",
        )
    )

    assert result is not None
    assert result.evidence["date_filter"] == "this_week"
    assert [event["title"] for event in result.evidence["events"]] == ["This week appointment"]


def test_household_event_read_filters_rest_of_week():
    from aegis.household import HouseholdEvent, HouseholdReadFastPath

    now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    week_end = now.date() + timedelta(days=7 - now.weekday())
    result = HouseholdReadFastPath(
        {
            "space_id": "home",
            "obligations": (),
            "chores": (),
            "events": (
                HouseholdEvent("current", "Later this week", now + timedelta(hours=1)),
                HouseholdEvent(
                    "later",
                    "Next week appointment",
                    datetime.combine(week_end + timedelta(days=1), datetime.min.time()),
                ),
            ),
        }
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What is on my calendar for the rest of the week?",
        )
    )

    assert result is not None
    assert result.evidence["date_filter"] == "rest_of_week"
    assert [event["title"] for event in result.evidence["events"]] == ["Later this week"]


def test_household_event_read_filters_current_and_next_month():
    from aegis.household import HouseholdEvent, HouseholdReadFastPath

    now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    month_start = now.date().replace(day=1)
    next_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
    following_month = (next_month.replace(day=28) + timedelta(days=4)).replace(day=1)
    events = (
        HouseholdEvent(
            "current",
            "Current month appointment",
            datetime.combine(month_start + timedelta(days=2), datetime.min.time()),
        ),
        HouseholdEvent(
            "next",
            "Next month appointment",
            datetime.combine(next_month + timedelta(days=1), datetime.min.time()),
        ),
    )

    for utterance, date_filter, expected in (
        ("What events do I have this month?", "this_month", "Current month appointment"),
        ("What appointments do I have next month?", "next_month", "Next month appointment"),
    ):
        result = HouseholdReadFastPath(
            {"space_id": "home", "obligations": (), "chores": (), "events": events}
        ).resolve(
            IntentFrame(
                principal=Principal(id="alice", vault_id="alice-vault"),
                utterance=utterance,
            )
        )
        assert result is not None
        assert result.evidence["date_filter"] == date_filter
        assert [event["title"] for event in result.evidence["events"]] == [expected]
    assert following_month > next_month


def test_household_implicit_planned_read_filters_this_week():
    from aegis.household import HouseholdEvent, HouseholdReadFastPath

    now = datetime.now(timezone.utc)
    result = HouseholdReadFastPath(
        {
            "space_id": "home",
            "obligations": (),
            "chores": (),
            "events": (
                HouseholdEvent("current", "Planned appointment", now + timedelta(hours=1)),
                HouseholdEvent("later", "Later appointment", now + timedelta(days=10)),
            ),
        }
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Do I have anything planned this week?",
        )
    )

    assert result is not None
    assert result.evidence["date_filter"] == "this_week"
    assert [event["title"] for event in result.evidence["events"]] == ["Planned appointment"]
    assert HouseholdReadFastPath.matches("What is on my schedule next week?")
    assert not HouseholdReadFastPath.matches("Schedule a meeting next week")


def test_household_implicit_happening_read_filters_this_weekend():
    from aegis.household import HouseholdEvent, HouseholdReadFastPath

    now = datetime.now(timezone.utc)
    saturday = now.date() + timedelta(
        days=5 - now.weekday() if now.weekday() < 5 else -(now.weekday() - 5)
    )
    result = HouseholdReadFastPath(
        {
            "space_id": "home",
            "obligations": (),
            "chores": (),
            "events": (
                HouseholdEvent(
                    "weekend", "Weekend inspection", datetime.combine(saturday, datetime.min.time())
                ),
                HouseholdEvent(
                    "later",
                    "Later inspection",
                    datetime.combine(saturday + timedelta(days=7), datetime.min.time()),
                ),
            ),
        }
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What's happening this weekend?",
        )
    )

    assert result is not None
    assert result.evidence["date_filter"] == "this_weekend"
    assert [event["title"] for event in result.evidence["events"]] == ["Weekend inspection"]


def test_household_implicit_going_on_read_filters_today():
    from aegis.household import HouseholdEvent, HouseholdReadFastPath

    now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    result = HouseholdReadFastPath(
        {
            "space_id": "home",
            "obligations": (),
            "chores": (),
            "events": (
                HouseholdEvent("today", "Today appointment", now + timedelta(hours=1)),
                HouseholdEvent("tomorrow", "Tomorrow appointment", now + timedelta(days=1)),
            ),
        }
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What do I have going on today?",
        )
    )

    assert result is not None
    assert result.evidence["date_filter"] == "today"
    assert [event["title"] for event in result.evidence["events"]] == ["Today appointment"]


def test_household_implicit_plans_read_filters_today():
    from aegis.household import HouseholdEvent, HouseholdReadFastPath

    now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    result = HouseholdReadFastPath(
        {
            "space_id": "home",
            "obligations": (),
            "chores": (),
            "events": (
                HouseholdEvent("today", "Today appointment", now + timedelta(hours=1)),
                HouseholdEvent("tomorrow", "Tomorrow appointment", now + timedelta(days=1)),
            ),
        }
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What are my plans for today?",
        )
    )

    assert result is not None
    assert result.evidence["date_filter"] == "today"
    assert [event["title"] for event in result.evidence["events"]] == ["Today appointment"]


def test_household_implicit_coming_up_read_filters_today():
    from aegis.household import HouseholdEvent, HouseholdReadFastPath

    now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    result = HouseholdReadFastPath(
        {
            "space_id": "home",
            "obligations": (),
            "chores": (),
            "events": (
                HouseholdEvent("today", "Today appointment", now + timedelta(hours=1)),
                HouseholdEvent("tomorrow", "Tomorrow appointment", now + timedelta(days=1)),
            ),
        }
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What is coming up today?",
        )
    )

    assert result is not None
    assert result.evidence["date_filter"] == "today"
    assert [event["title"] for event in result.evidence["events"]] == ["Today appointment"]


def test_household_coming_up_read_excludes_past_events():
    from aegis.household import HouseholdEvent, HouseholdReadFastPath

    now = datetime.now(timezone.utc)
    result = HouseholdReadFastPath(
        {
            "space_id": "home",
            "obligations": (),
            "chores": (),
            "events": (
                HouseholdEvent("past", "Past appointment", now - timedelta(hours=1)),
                HouseholdEvent("future", "Future appointment", now + timedelta(hours=1)),
            ),
        }
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What appointments do I have coming up?",
        )
    )

    assert result is not None
    assert result.evidence["date_filter"] == "upcoming"
    assert [event["title"] for event in result.evidence["events"]] == ["Future appointment"]


def test_household_implicit_meeting_read_filters_today():
    from aegis.household import HouseholdEvent, HouseholdReadFastPath

    now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    result = HouseholdReadFastPath(
        {
            "space_id": "home",
            "obligations": (),
            "chores": (),
            "events": (
                HouseholdEvent("today", "Today meeting", now + timedelta(hours=1)),
                HouseholdEvent("tomorrow", "Tomorrow meeting", now + timedelta(days=1)),
            ),
        }
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Do I have any meetings today?",
        )
    )

    assert result is not None
    assert result.evidence["date_filter"] == "today"
    assert [event["title"] for event in result.evidence["events"]] == ["Today meeting"]
    assert not HouseholdReadFastPath.matches("Schedule a meeting today")
    assert not HouseholdReadFastPath.matches("Could you book a meeting today?")


def test_domainless_today_priority_is_grounded_in_open_task_deadlines():
    from aegis.tasks import TaskPriorityFastPath

    due = Task(
        uuid4(),
        "home",
        "review the backup",
        "alice",
        due_at=datetime.now() + timedelta(hours=1),
    )
    later = Task(
        uuid4(),
        "home",
        "clean the garage",
        "alice",
        due_at=datetime.now() + timedelta(days=2),
    )

    result = TaskPriorityFastPath(
        type("Tasks", (), {"list": lambda _self, _p: [later, due]})()
    ).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What should I take care of today?",
        )
    )

    assert result is not None
    assert result.evidence["task"]["title"] == "review the backup"


def test_domainless_weekend_priority_uses_only_this_weekends_deadlines():
    from aegis.tasks import TaskPriorityFastPath

    now = datetime(2026, 9, 4, 12, tzinfo=timezone.utc)  # Friday
    weekend = Task(
        uuid4(),
        "home",
        "inspect the porch",
        "alice",
        due_at=datetime(2026, 9, 5, 12, tzinfo=timezone.utc),
    )
    later = Task(
        uuid4(),
        "home",
        "inspect the roof",
        "alice",
        due_at=datetime(2026, 9, 12, 12, tzinfo=timezone.utc),
    )

    class Store:
        def list(self, _principal):
            return (later, weekend)

    result = TaskPriorityFastPath(Store()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What should I take care of this weekend?",
        ),
        now,
    )

    assert result is not None
    assert result.evidence["priority_basis"] == "earliest_due_at_on_weekend"
    assert result.evidence["task"]["title"] == "inspect the porch"


def test_domainless_next_week_priority_uses_only_next_calendar_week():
    from aegis.tasks import TaskPriorityFastPath

    now = datetime(2026, 9, 4, 12, tzinfo=timezone.utc)  # Friday
    next_week = Task(
        uuid4(),
        "home",
        "inspect the roof",
        "alice",
        due_at=datetime(2026, 9, 7, 12, tzinfo=timezone.utc),
    )
    this_week = Task(
        uuid4(),
        "home",
        "inspect the porch",
        "alice",
        due_at=datetime(2026, 9, 5, 12, tzinfo=timezone.utc),
    )

    class Store:
        def list(self, _principal):
            return (this_week, next_week)

    result = TaskPriorityFastPath(Store()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What tasks should I focus on next week?",
        ),
        now,
    )

    assert result is not None
    assert result.evidence["priority_basis"] == "earliest_due_at_on_next_week"
    assert result.evidence["task"]["title"] == "inspect the roof"


def test_priority_followup_reuses_one_scalar_task_focus():
    from aegis.tasks import ContextualTaskPriorityFastPath

    context = Context(
        values={"canonical_facts": {"task": {"title": "review the backup", "status": "open"}}},
        sources=("authorized_canonical_result",),
    )
    result = ContextualTaskPriorityFastPath().resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which one is most urgent?",
        ),
        context,
    )

    assert result is not None
    assert result.message == "The most urgent referenced task is: review the backup"


def test_polite_task_collection_read_is_recognized():
    from aegis.tasks import TaskReadFastPath

    assert TaskReadFastPath.matches("Could you show me what tasks are still open?") is True


def test_model_working_context_preserves_plan_progress_source_marker():
    context = Context(
        values={
            "canonical_facts": {
                "plan_steps": [
                    {"index": 0, "state": "completed"},
                    {"index": 1, "state": "completed"},
                ]
            }
        },
        sources=("authorized_canonical_result",),
    )

    class UnreachableTasks:
        def list(self, _principal):
            raise AssertionError("plan progress should not load task fallback context")

    class UnreachableHousehold:
        def list_groceries(self, _principal):
            raise AssertionError("plan progress should not load household fallback context")

    result = build_reference_fallback_context(
        context,
        UnreachableTasks(),
        UnreachableHousehold(),
        Principal(id="alice", vault_id="alice-vault"),
        "What's left?",
    )

    assert result.sources == ("authorized_canonical_result",)
    assert (
        result.values["canonical_facts"]["plan_steps"]
        == context.values["canonical_facts"]["plan_steps"]
    )


def test_unique_prior_task_reference_grounds_scalar_priority_focus():
    from aegis.interaction_context import resolve_unique_prior_task_reference

    context = Context(
        sources=("authorized_canonical_result",),
        values={
            "canonical_facts": {
                "task": {
                    "task_id": "11111111-1111-4111-8111-111111111111",
                    "title": "renew insurance",
                    "status": "open",
                }
            }
        },
    )

    assert resolve_unique_prior_task_reference("Complete that one", context) == {
        "task_id": "11111111-1111-4111-8111-111111111111",
        "title": "renew insurance",
        "status": "open",
    }


def test_unique_prior_task_reference_grounds_recent_task_result():
    from aegis.interaction_context import resolve_unique_prior_task_reference

    context = Context(
        sources=("authorized_canonical_result",),
        values={
            "canonical_facts": {
                "collection": "tasks",
                "title": "check the loading dock lock",
                "status": "open",
            }
        },
    )

    assert resolve_unique_prior_task_reference("Complete that one", context) == {
        "title": "check the loading dock lock",
        "status": "open",
    }


def test_model_working_context_includes_bounded_authorized_household_attention():
    principal = Principal(id="alice", vault_id="alice-vault")
    chore = type("Chore", (), {"title": "clean the utility closet", "completed": False})()
    obligation = type(
        "Obligation",
        (),
        {"title": "Utilities", "settled": False, "responsible_id": "alice"},
    )()

    class Tasks:
        def list(self, _principal):
            return []

    class Household:
        def list_groceries(self, _principal):
            return []

        def read_snapshot(self, _principal):
            return {"chores": (chore,), "obligations": (obligation,)}

    context = build_reference_fallback_context(
        Context(), Tasks(), Household(), principal, "what needs attention?"
    )

    assert context.values["canonical_facts"]["canonical_chores"] == [
        {"title": "clean the utility closet", "completed": False}
    ]
    assert context.values["canonical_facts"]["canonical_obligations"] == [
        {"title": "Utilities", "settled": False, "responsible_id": "alice"}
    ]


def test_model_working_context_prioritization_uses_bounded_open_tasks():
    principal = Principal(id="alice", vault_id="alice-vault")
    completed = Task(
        uuid4(), "apartment", "old completed task", "alice", status=TaskStatus.COMPLETED
    )
    open_task = Task(uuid4(), "apartment", "open task", "alice")

    class Tasks:
        def list(self, _principal):
            return [completed, open_task]

    class Household:
        def list_groceries(self, _principal):
            return []

    context = build_reference_fallback_context(
        Context(), Tasks(), Household(), principal, "which task should i do first"
    )

    assert context.values["canonical_facts"]["canonical_tasks"] == [
        {"task_id": str(open_task.task_id), "title": "open task", "status": "open"}
    ]


def test_contextual_task_priority_does_not_capture_mutation_follow_up():
    from aegis.tasks import ContextualTaskPriorityFastPath

    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [{"title": "buy milk", "status": "open"}],
                }
            }
        }
    )
    result = ContextualTaskPriorityFastPath().resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="complete the first one",
        ),
        context,
    )
    assert result is None


def test_contextual_task_priority_does_not_cross_into_chore_follow_up():
    from aegis.tasks import ContextualTaskPriorityFastPath

    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [
                        {"title": "check the gate", "status": "open", "due_at": "2026-09-05"}
                    ],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )
    result = ContextualTaskPriorityFastPath().resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which chore is due first?",
        ),
        context,
    )

    assert result is None


def test_chore_priority_followup_does_not_substitute_task_projection():
    from aegis.household import ContextualChorePriorityFastPath

    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [{"title": "check the gate", "status": "open"}],
                }
            }
        },
        sources=("authorized_canonical_result",),
    )
    result = ContextualChorePriorityFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which chore is due first?",
        ),
        context,
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "preceding result was a task list" in result.message


def test_contextual_task_priority_accepts_start_with_follow_up():
    from aegis.tasks import ContextualTaskPriorityFastPath

    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [
                        {"title": "later task", "status": "open", "due_at": "2026-09-05"},
                        {"title": "first task", "status": "open", "due_at": "2026-09-02"},
                    ],
                }
            }
        }
    )
    result = ContextualTaskPriorityFastPath().resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which one should I start with?",
        ),
        context,
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.message.endswith("first task")
    assert result.evidence["canonical_tasks"] == context.values["referents"]["those"]["candidates"]


def test_contextual_task_priority_accepts_begin_with_follow_up():
    from aegis.tasks import ContextualTaskPriorityFastPath

    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [
                        {"title": "later task", "status": "open", "due_at": "2026-09-05"},
                        {"title": "first task", "status": "open", "due_at": "2026-09-02"},
                    ],
                }
            }
        }
    )
    result = ContextualTaskPriorityFastPath().resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which one should I begin with?",
        ),
        context,
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.message.endswith("first task")


def test_contextual_task_priority_accepts_handle_first_follow_up():
    from aegis.tasks import ContextualTaskPriorityFastPath

    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [
                        {"title": "later task", "status": "open", "due_at": "2026-09-05"},
                        {"title": "first task", "status": "open", "due_at": "2026-09-02"},
                    ],
                }
            }
        }
    )
    result = ContextualTaskPriorityFastPath().resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which one should I handle first?",
        ),
        context,
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.message.endswith("first task")


def test_contextual_task_priority_accepts_earliest_from_planning_referents():
    from aegis.tasks import ContextualTaskPriorityFastPath

    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [
                        {"title": "later task", "status": "open", "due_at": "2026-09-05"},
                        {"title": "earliest task", "status": "open", "due_at": "2026-09-02"},
                    ],
                }
            }
        }
    )
    result = ContextualTaskPriorityFastPath().resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which is the earliest one?",
        ),
        context,
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.message.endswith("earliest task")


def test_contextual_task_priority_wins_over_ordinal_wording_for_due_first():
    from aegis.tasks import ContextualTaskPriorityFastPath

    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [
                        {"title": "undated task", "status": "open"},
                        {"title": "earliest task", "status": "open", "due_at": "2026-09-02"},
                    ],
                }
            }
        }
    )
    result = ContextualTaskPriorityFastPath().resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which one is due first?",
        ),
        context,
    )

    assert result is not None
    assert result.message.endswith("earliest task")


def test_contextual_task_priority_does_not_capture_plain_first_item_read():
    from aegis.tasks import ContextualTaskPriorityFastPath

    context = Context(
        values={
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [{"title": "first task", "status": "open"}],
                }
            }
        }
    )
    result = ContextualTaskPriorityFastPath().resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What about the first one?",
        ),
        context,
    )

    assert result is None


def test_authorized_prior_context_contains_one_bounded_non_authoritative_turn():
    principal = Principal(id="alice", vault_id="alice-vault")
    correlation_id = uuid4()
    objective = Objective(
        intent=IntentFrame(
            principal=principal,
            utterance="What's on my grocery list?",
            correlation_id=correlation_id,
        ),
        correlation_id=correlation_id,
    )
    result = Result(
        objective_id=objective.id,
        state=ObjectiveState.COMPLETED,
        message="Your list contains rice.",
        evidence={"canonical_items": ["rice"]},
        correlation_id=correlation_id,
    )

    class Store:
        def get_objective_by_correlation(self, _correlation, _principal):
            return objective

        def get_result_for_correlation(self, _correlation, _principal):
            return result

    context = _context_from_prior_result(Store(), correlation_id, principal)

    assert context.values["recent_turns"] == [
        {
            "role": "user",
            "utterance": "What's on my grocery list?",
            "correlation_id": str(correlation_id),
        },
        {
            "role": "assistant",
            "message": "Your list contains rice.",
            "correlation_id": str(correlation_id),
        },
    ]
    assert context.values["referents"] == {
        "those": {
            "source": "canonical_facts",
            "fact_key": "canonical_items",
            "candidates": ["rice"],
        }
    }
    assert context.sources == ("authorized_canonical_result",)


def test_authorized_prior_context_preserves_empty_canonical_collection():
    principal = Principal(id="alice", vault_id="alice-vault")
    correlation_id = uuid4()
    objective = Objective(
        intent=IntentFrame(
            principal=principal,
            utterance="What tasks are due Friday?",
            correlation_id=correlation_id,
        ),
        correlation_id=correlation_id,
    )
    result = Result(
        objective_id=objective.id,
        state=ObjectiveState.COMPLETED,
        message="Tasks: (empty)",
        evidence={"canonical_tasks": []},
        correlation_id=correlation_id,
    )

    class Store:
        def get_objective_by_correlation(self, _correlation, _principal):
            return objective

        def get_result_for_correlation(self, _correlation, _principal):
            return result

    context = _context_from_prior_result(Store(), correlation_id, principal)

    assert context.values["referents"] == {
        "those": {
            "source": "canonical_facts",
            "fact_key": "canonical_tasks",
            "candidates": [],
        }
    }


def test_blocked_prior_context_preserves_clarification_without_referents():
    principal = Principal(id="alice", vault_id="alice-vault")
    correlation_id = uuid4()
    objective = Objective(
        intent=IntentFrame(
            principal=principal,
            utterance="What do I have on Friday?",
            correlation_id=correlation_id,
        ),
        correlation_id=correlation_id,
    )
    result = Result(
        objective_id=objective.id,
        state=ObjectiveState.BLOCKED,
        message="Could you clarify whether you mean tasks or appointments?",
        correlation_id=correlation_id,
    )

    class Store:
        def get_objective_by_correlation(self, _correlation, _principal):
            return objective

        def get_result_for_correlation(self, _correlation, _principal):
            return result

    context = _context_from_prior_result(Store(), correlation_id, principal)

    assert context.sources == ("authorized_prior_result",)
    assert context.values["referents"] == {}
    assert context.values["recent_turns"][1] == {
        "role": "assistant",
        "message": result.message,
        "correlation_id": str(correlation_id),
    }


def test_authorized_prior_plan_progress_preserves_step_state_for_restart_followup():
    principal = Principal(id="alice", vault_id="alice-vault")
    correlation_id = uuid4()
    objective = Objective(
        intent=IntentFrame(
            principal=principal,
            utterance="What remains?",
            correlation_id=correlation_id,
        ),
        correlation_id=correlation_id,
    )
    result = Result(
        objective_id=objective.id,
        state=ObjectiveState.COMPLETED,
        message="All 2 plan steps are complete.",
        evidence={
            "plan_progress": {"completed": 2, "total": 2},
            "plan_steps": [
                {"index": 0, "state": "completed"},
                {"index": 1, "state": "completed"},
            ],
        },
        correlation_id=correlation_id,
    )

    class Store:
        def get_objective_by_correlation(self, _correlation, _principal):
            return objective

        def get_result_for_correlation(self, _correlation, _principal):
            return result

    context = _context_from_prior_result(Store(), correlation_id, principal)

    assert context.values["canonical_facts"]["plan_steps"] == [
        {"index": 0, "state": "completed"},
        {"index": 1, "state": "completed"},
    ]


def test_authorized_priority_context_preserves_task_focus_for_temporal_followup():
    principal = Principal(id="alice", vault_id="alice-vault")
    correlation_id = uuid4()
    objective = Objective(
        intent=IntentFrame(
            principal=principal,
            utterance="Which one should I start with?",
            correlation_id=correlation_id,
        ),
        correlation_id=correlation_id,
    )
    result = Result(
        objective_id=objective.id,
        state=ObjectiveState.COMPLETED,
        message="Start with the first task.",
        evidence={"task": {"task_id": str(uuid4()), "title": "first task"}},
        correlation_id=correlation_id,
    )

    class Store:
        def get_objective_by_correlation(self, _correlation, _principal):
            return objective

        def get_result_for_correlation(self, _correlation, _principal):
            return result

    context = _context_from_prior_result(Store(), correlation_id, principal)

    assert context.values["canonical_facts"]["task"] == result.evidence["task"]


def test_authorized_task_context_preserves_due_candidates_beyond_ordinal_window():
    principal = Principal(id="alice", vault_id="alice-vault")
    correlation_id = uuid4()
    objective = Objective(
        intent=IntentFrame(
            principal=principal,
            utterance="show my open tasks",
            correlation_id=correlation_id,
        ),
        correlation_id=correlation_id,
    )
    tasks = [
        {"task_id": str(uuid4()), "title": f"undated {index}", "status": "open"}
        for index in range(11)
    ]
    tasks[-1]["due_at"] = "2026-09-03T00:00:00+00:00"
    result = Result(
        objective_id=objective.id,
        state=ObjectiveState.COMPLETED,
        message="Canonical task list read",
        evidence={"canonical_tasks": tasks},
        correlation_id=correlation_id,
    )

    class Store:
        def get_objective_by_correlation(self, _correlation, _principal):
            return objective

        def get_result_for_correlation(self, _correlation, _principal):
            return result

    context = _context_from_prior_result(Store(), correlation_id, principal)
    candidates = context.values["referents"]["those"]["candidates"]
    assert candidates[:10] == tasks[:10]
    assert candidates[-1] == tasks[-1]


def test_follow_up_result_can_trace_authorized_prior_objective_without_reusing_identity():
    prior_id = str(uuid4())
    context = Context(
        values={"prior_objective_id": prior_id},
        sources=("authorized_canonical_result",),
    )
    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="follow-up",
        correlation_id=uuid4(),
    )

    enriched = _with_continuation_context(result, context)
    assert enriched.objective_id != UUID(prior_id)
    assert enriched.evidence["continuation_of_objective_id"] == prior_id


def test_blocked_followup_preserves_bounded_authorized_referents():
    prior_id = str(uuid4())
    context = Context(
        values={
            "prior_objective_id": prior_id,
            "referents": {
                "those": {
                    "fact_key": "canonical_tasks",
                    "candidates": [{"title": "check the gate", "status": "open"}],
                }
            },
        },
        sources=("authorized_canonical_result",),
    )
    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.BLOCKED,
        message="Please choose an available ordinal.",
        correlation_id=uuid4(),
    )

    enriched = _with_continuation_context(result, context)

    assert enriched.evidence["canonical_tasks"] == [{"title": "check the gate", "status": "open"}]


def test_completed_followup_does_not_overwrite_fresh_canonical_evidence():
    context = Context(
        values={
            "prior_objective_id": str(uuid4()),
            "referents": {
                "those": {
                    "fact_key": "events",
                    "candidates": [{"title": "Friday event"}],
                }
            },
        },
        sources=("authorized_canonical_result",),
    )
    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="Events: (none)",
        evidence={"events": [], "date_filter": "tomorrow"},
        correlation_id=uuid4(),
    )

    enriched = _with_continuation_context(result, context)

    assert enriched.evidence["events"] == []
    assert enriched.evidence["date_filter"] == "tomorrow"


def test_context_reconstruction_preserves_original_objective_through_follow_up_result():
    principal = Principal(id="alice", vault_id="alice-vault")
    correlation_id = uuid4()
    original_id = uuid4()
    progress_id = uuid4()
    objective = Objective(
        id=progress_id,
        intent=IntentFrame(
            principal=principal,
            utterance="what is left?",
            correlation_id=correlation_id,
        ),
        correlation_id=correlation_id,
    )
    result = Result(
        objective_id=progress_id,
        state=ObjectiveState.COMPLETED,
        message="All requested changes are complete.",
        evidence={"continuation_of_objective_id": str(original_id), "plan_steps": []},
        correlation_id=correlation_id,
    )

    class Store:
        def get_objective_by_correlation(self, _correlation, _principal):
            return objective

        def get_result_for_correlation(self, _correlation, _principal):
            return result

    context = _context_from_prior_result(Store(), correlation_id, principal)
    assert context.values["prior_objective_id"] == str(original_id)


def test_prior_context_turn_and_referents_are_bounded():
    principal = Principal(id="alice", vault_id="alice-vault")
    correlation_id = uuid4()
    objective = Objective(
        intent=IntentFrame(
            principal=principal,
            utterance="x" * 2_000,
            correlation_id=correlation_id,
        ),
        correlation_id=correlation_id,
    )
    result = Result(
        objective_id=objective.id,
        state=ObjectiveState.COMPLETED,
        message="grounded list",
        evidence={"canonical_items": [f"item-{i}" for i in range(30)]},
        correlation_id=correlation_id,
    )

    class Store:
        def get_objective_by_correlation(self, _correlation, _principal):
            return objective

        def get_result_for_correlation(self, _correlation, _principal):
            return result

    context = _context_from_prior_result(Store(), correlation_id, principal)

    assert len(context.values["recent_turns"][0]["utterance"]) == 500
    assert len(context.values["referents"]["those"]["candidates"]) == 10


def test_prior_task_context_retains_deadline_candidates_within_bound():
    principal = Principal(id="alice", vault_id="alice-vault")
    correlation_id = uuid4()
    objective = Objective(
        intent=IntentFrame(
            principal=principal,
            utterance="Which responsibilities are still open?",
            correlation_id=correlation_id,
        ),
        correlation_id=correlation_id,
    )
    result = Result(
        objective_id=objective.id,
        state=ObjectiveState.COMPLETED,
        message="grounded task list",
        evidence={
            "canonical_tasks": [{"title": f"undated-{i}", "status": "open"} for i in range(10)]
            + [{"title": "deadline task", "status": "open", "due_at": "2026-09-03"}],
        },
        correlation_id=correlation_id,
    )

    class Store:
        def get_objective_by_correlation(self, _correlation, _principal):
            return objective

        def get_result_for_correlation(self, _correlation, _principal):
            return result

    context = _context_from_prior_result(Store(), correlation_id, principal)
    candidates = context.values["referents"]["those"]["candidates"]

    assert len(candidates) == 11
    assert candidates[0] == {"title": "undated-0", "status": "open"}
    assert candidates[-1] == {
        "title": "deadline task",
        "status": "open",
        "due_at": "2026-09-03",
    }
    assert context.values["canonical_facts"]["canonical_tasks"][0] == {
        "title": "deadline task",
        "status": "open",
        "due_at": "2026-09-03",
    }


def test_model_answer_can_carry_authorized_working_facts_without_becoming_truth():
    context = Context(
        values={"canonical_facts": {"canonical_items": ["rice", "beans"]}},
        sources=("authorized_task_candidates",),
    )

    assert _authorized_context_evidence(context) == {"canonical_items": ["rice", "beans"]}


def test_fresh_working_context_does_not_suppress_unrelated_mutation_cards():
    from aegis.interaction import InteractionDependencies

    card = reference_bundles()[0].cards[0]
    manager = PackManager()
    manager.discover(reference_bundles()[0])
    manager.install("tasks", frozenset({"tasks.write", "tasks.read"}))
    manager.enable("tasks")
    boundary = InteractionBoundary(
        InteractionDependencies(
            connect=lambda _url: None,
            required=lambda _name: "unused",
            apply_migrations=lambda _connection: None,
            ensure_local_identity=lambda _connection, _principal: None,
            select_action=lambda _utterance, _manager: ("tasks", card),
            openclaw_channel=lambda: None,
            local_identity=lambda: False,
            fallback_card_selector=reference_fallback_cards,
        )
    )
    fresh = Context(
        values={"canonical_facts": {"canonical_items": ["rice"]}},
        sources=("authorized_task_candidates",),
    )
    prior_grocery = fresh.model_copy(update={"sources": ("authorized_canonical_context",)})

    assert boundary._fallback_cards(manager, "please add a task", fresh)
    assert boundary._fallback_cards(manager, "add those", prior_grocery) == ()


def test_semantic_write_candidates_are_scoped_to_their_top_write_pack():
    from aegis.interaction import InteractionDependencies

    manager = PackManager()
    for bundle in reference_bundles():
        manager.discover(bundle)
        manager.install(bundle.manifest.pack_id, frozenset(bundle.manifest.permissions))
        manager.enable(bundle.manifest.pack_id)
    cards = tuple(manager.enabled_cards())
    task_complete = next(card for card in cards if card.action.action_id == "tasks.complete")
    grocery_list = next(card for card in cards if card.action.action_id == "kitchen.groceries.list")
    network_probe = next(card for card in cards if card.action.action_id == "network.probe")
    boundary = InteractionBoundary(
        InteractionDependencies(
            connect=lambda _url: None,
            required=lambda _name: "unused",
            apply_migrations=lambda _connection: None,
            ensure_local_identity=lambda _connection, _principal: None,
            select_action=lambda _utterance, _manager: ("tasks", task_complete),
            openclaw_channel=lambda: None,
            local_identity=lambda: False,
            capability_retriever=lambda _query, _manager: (
                network_probe,
                task_complete,
                grocery_list,
            ),
        )
    )

    selected = boundary._fallback_cards(manager, "informal completion", Context())

    assert [card.action.action_id for card in selected] == ["tasks.complete"]


def test_interaction_boundary_reuses_completed_plan_before_fast_paths(monkeypatch):
    from aegis import interaction

    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    correlation_id = uuid4()
    action = ActionSpec(
        action_id="tasks.create",
        capability="tasks.write",
        verification=VerificationContract(kind="readback"),
    )
    objective = Objective(
        intent=IntentFrame(
            principal=principal,
            utterance="Create a task and a chore.",
            correlation_id=correlation_id,
        ),
        correlation_id=correlation_id,
        steps=(action,),
    )
    prior = Result(
        objective_id=objective.id,
        state=ObjectiveState.COMPLETED,
        message="Completed all 1 plan steps",
        correlation_id=correlation_id,
    )

    class Connection:
        def close(self):
            pass

    class Store:
        def get_objective_by_correlation(self, _correlation, _principal):
            return objective

        def get_result(self, key):
            assert key == f"plan:{correlation_id}"
            return prior

    monkeypatch.setattr(interaction, "PostgresObjectiveStore", lambda _connection: Store())
    boundary = InteractionBoundary(
        InteractionDependencies(
            connect=lambda _url: Connection(),
            required=lambda _name: "postgresql://example",
            apply_migrations=lambda _connection: None,
            ensure_local_identity=lambda _connection, _principal: None,
            select_action=lambda _utterance, _manager: ("tasks", None),
            openclaw_channel=lambda: None,
            local_identity=lambda: False,
        )
    )

    replay = boundary.run("I need help with the apartment inspection.", principal, correlation_id)

    assert replay == prior


def test_interaction_boundary_does_not_short_circuit_blocked_plan_retry(monkeypatch):
    from aegis import interaction

    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    correlation_id = uuid4()
    action = ActionSpec(
        action_id="tasks.create",
        capability="tasks.write",
        verification=VerificationContract(kind="readback"),
    )
    objective = Objective(
        intent=IntentFrame(
            principal=principal,
            utterance="Create a task and a chore.",
            correlation_id=correlation_id,
        ),
        correlation_id=correlation_id,
        steps=(action,),
    )
    prior = Result(
        objective_id=objective.id,
        state=ObjectiveState.BLOCKED,
        message="step 1 denied",
        correlation_id=correlation_id,
    )
    fallback = Result(
        objective_id=uuid4(),
        state=ObjectiveState.BLOCKED,
        message="retry reached the interaction path",
        correlation_id=correlation_id,
    )

    class Connection:
        def close(self):
            pass

    class Store:
        def __init__(self):
            self.saved_result = None

        def get_objective_by_correlation(self, _correlation, _principal):
            return objective

        def get_result(self, key):
            assert key == f"plan:{correlation_id}"
            return prior

        def save_objective(self, _value):
            pass

        def save_result(self, _key, value):
            self.saved_result = value

    store = Store()

    class Clarification:
        @staticmethod
        def resolve(_intent):
            return fallback

    monkeypatch.setattr(interaction, "PostgresObjectiveStore", lambda _connection: store)
    boundary = InteractionBoundary(
        InteractionDependencies(
            connect=lambda _url: Connection(),
            required=lambda _name: "postgresql://example",
            apply_migrations=lambda _connection: None,
            ensure_local_identity=lambda _connection, _principal: None,
            select_action=lambda _utterance, _manager: ("tasks", None),
            openclaw_channel=lambda: None,
            local_identity=lambda: False,
            safety_fast_path_resolver=lambda _intent, _recovered, _model_enabled: fallback,
        )
    )

    result = boundary.run("resume the denied plan", principal, correlation_id)

    assert result == fallback
    assert store.saved_result == fallback


def test_interaction_boundary_persists_fast_path_result_for_status_recovery(monkeypatch):
    from aegis import interaction

    principal = Principal(id="alice", vault_id="alice-vault")
    correlation_id = uuid4()

    class Connection:
        def close(self):
            pass

    class Store:
        def __init__(self):
            self.objective = None
            self.result = None

        def get_objective_by_correlation(self, _correlation, _principal):
            return self.objective

        def correlation_bound(self, _correlation):
            return self.objective is not None

        def get_result(self, key):
            assert key == f"interaction:{correlation_id}"
            return self.result

        def save_objective(self, objective):
            self.objective = objective

        def save_result(self, _key, result):
            self.result = result

    store = Store()
    monkeypatch.setattr(interaction, "PostgresObjectiveStore", lambda _connection: store)
    boundary = InteractionBoundary(
        InteractionDependencies(
            connect=lambda _url: Connection(),
            required=lambda _name: "postgresql://example",
            apply_migrations=lambda _connection: None,
            ensure_local_identity=lambda _connection, _principal: None,
            select_action=lambda _utterance, _manager: ("tasks", None),
            openclaw_channel=lambda: None,
            local_identity=lambda: False,
        )
    )

    first = boundary.run("hello there", principal, correlation_id)
    replay = boundary.run("a different wording", principal, correlation_id)

    assert first.state is ObjectiveState.BLOCKED
    assert store.objective is not None
    assert store.result == first
    assert replay == first


def manager_with_reference_cards() -> PackManager:
    manager = PackManager()
    for pack in reference_packs():
        manager.discover(pack)
        manager.install(
            pack.manifest.pack_id,
            frozenset(pack.manifest.permissions),
        )
        manager.enable(pack.manifest.pack_id)
    return manager


def test_cli_routes_task_before_food_keyword() -> None:
    domain, card = _domain_and_action(
        "Create a task to buy cat food.", manager_with_reference_cards()
    )

    assert domain == "tasks"
    assert card.action.action_id == "tasks.create"
    assert card.action.arguments == {"title": "buy cat food."}


def test_cli_routes_put_task_on_list_as_mutation() -> None:
    domain, card = _domain_and_action(
        "Could you put a task on my list to keep an eye on the backup before the restore drill?",
        manager_with_reference_cards(),
    )

    assert domain == "tasks"
    assert card.action.action_id == "tasks.create"
    assert card.action.arguments == {"title": "keep an eye on the backup before the restore drill?"}


def test_cli_routes_want_to_put_task_on_list_as_mutation() -> None:
    domain, card = _domain_and_action(
        "I'd like to put a task on my list to verify the restore drill.",
        manager_with_reference_cards(),
    )

    assert domain == "tasks"
    assert card.action.action_id == "tasks.create"
    assert card.action.arguments == {"title": "verify the restore drill."}


def test_cli_carries_unambiguous_tomorrow_task_due_date():
    from datetime import datetime, timedelta

    _domain, card = _domain_and_action(
        "Could you put a task on my list to review the restore drill tomorrow?",
        manager_with_reference_cards(),
    )

    assert card.action.arguments["title"] == "review the restore drill"
    due_at = datetime.fromisoformat(card.action.arguments["due_at"])
    assert timedelta(hours=23) < due_at - datetime.now(timezone.utc) < timedelta(hours=25)


def test_cli_carries_unambiguous_next_week_task_due_date():
    from datetime import datetime, timedelta, timezone

    _domain, card = _domain_and_action(
        "Please create a task to review the restore drill next week.",
        manager_with_reference_cards(),
    )

    assert card.action.arguments["title"] == "review the restore drill"
    due_at = datetime.fromisoformat(card.action.arguments["due_at"])
    assert (
        timedelta(days=6, hours=23)
        < due_at - datetime.now(timezone.utc)
        < timedelta(days=7, hours=1)
    )


def test_model_task_deadline_must_be_grounded_in_request():
    now = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)

    assert ground_task_due_at(
        "please add send the rent receipt to my todos", "2026-09-02T00:00:00", now
    ) == (
        False,
        None,
    )
    assert ground_task_due_at(
        "please add send the rent receipt to my todos tomorrow",
        "2099-01-01T00:00:00+00:00",
        now,
    ) == (True, "2026-09-02T12:00:00+00:00")
    assert ground_task_due_at(
        "please add check the side gate tomorrow to my task list",
        None,
        now,
    ) == (True, "2026-09-02T12:00:00+00:00")
    assert ground_task_due_at(
        "add a task to inspect the side gate Friday",
        None,
        now,
    ) == (True, "2026-09-04T12:00:00+00:00")


def test_cli_routes_task_completion_to_complete_action() -> None:
    domain, card = _domain_and_action(
        "Complete the task Verify backup retention.", manager_with_reference_cards()
    )

    assert domain == "tasks"
    assert card.action.action_id == "tasks.complete"
    assert card.action.arguments == {"title": "verify backup retention."}


def test_cli_routes_chore_completion_to_complete_action() -> None:
    domain, card = _domain_and_action(
        "Complete the chore Clean the utility closet.", manager_with_reference_cards()
    )

    assert domain == "tasks"
    assert card.action.action_id == "tasks.chores.complete"
    assert card.action.arguments == {"title": "clean the utility closet."}


def test_cli_routes_mark_task_done_to_complete_action() -> None:
    domain, card = _domain_and_action(
        "Mark the task Verify backup retention as done.", manager_with_reference_cards()
    )

    assert domain == "tasks"
    assert card.action.action_id == "tasks.complete"
    assert card.action.arguments == {"title": "verify backup retention"}


def test_cli_routes_mark_chore_done_to_complete_action() -> None:
    domain, card = _domain_and_action(
        "Mark the chore Clean the utility closet as done.", manager_with_reference_cards()
    )

    assert domain == "tasks"
    assert card.action.action_id == "tasks.chores.complete"
    assert card.action.arguments == {"title": "clean the utility closet"}


def test_cli_prepares_grocery_action_card_arguments() -> None:
    domain, card = _domain_and_action("Add rice to groceries.", manager_with_reference_cards())

    assert domain == "kitchen"
    assert card.action.action_id == "kitchen.groceries.add"
    assert card.action.arguments == {"item": "rice"}


def test_cli_reports_incomplete_grocery_input_as_actionable_input_error() -> None:
    with pytest.raises(InteractionInputError, match="tell AEGIS what to add"):
        _domain_and_action("Please add to groceries.", manager_with_reference_cards())


def test_cli_retrieves_read_cards() -> None:
    manager = manager_with_reference_cards()

    _, groceries = _domain_and_action("What's on my grocery list?", manager)
    _, tasks = _domain_and_action("Show my tasks.", manager)

    assert groceries.action.action_id == "kitchen.groceries.list"
    assert tasks.action.action_id == "tasks.list"


def test_cli_routes_shopping_list_read_to_canonical_grocery_action() -> None:
    _, groceries = _domain_and_action(
        "What is on my shopping list?", manager_with_reference_cards()
    )

    assert groceries.action.action_id == "kitchen.groceries.list"


def test_local_identity_bootstrap_does_not_reactivate_revoked_membership():
    class Connection:
        def __init__(self) -> None:
            self.queries: list[tuple[str, tuple[object, ...]]] = []

        def execute(self, query: str, params: tuple[object, ...] = ()) -> None:
            self.queries.append((query, params))

        def commit(self) -> None:
            pass

    connection = Connection()
    _ensure_local_identity(
        connection,
        Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    membership_query = next(
        query for query, _ in connection.queries if "INSERT INTO space_memberships" in query
    )
    assert "DO NOTHING" in membership_query
    assert "active = TRUE" not in membership_query.split("DO NOTHING")[1]


def test_cli_help_is_available_without_runtime_configuration(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--help"])

    with pytest.raises(SystemExit) as raised:
        cli.main()

    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "--once REQUEST" in output
    assert "--no-banner" in output
    assert "--init" in output


def test_cli_version_is_available_without_runtime_configuration(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--version"])
    with pytest.raises(SystemExit) as raised:
        cli.main()

    assert raised.value.code == 0
    assert capsys.readouterr().out.startswith("aegis ")


def test_cli_migration_source_fallback_executes_sql(monkeypatch):
    from aegis import cli

    class Connection:
        def __init__(self):
            self.statements = []
            self.committed = False

        def execute(self, statement):
            self.statements.append(statement)

        def commit(self):
            self.committed = True

    connection = Connection()
    monkeypatch.delenv("AEGIS_AUTO_MIGRATE", raising=False)

    cli._apply_migrations(connection)

    assert len(connection.statements) == 16
    assert connection.statements[0].startswith("-- PostgreSQL canonical schema")
    assert connection.committed is True


def test_cli_init_creates_private_non_overwriting_template(monkeypatch, capsys, tmp_path):
    from aegis import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["aegis", "--init"])

    assert cli.main() == 0
    target = tmp_path / ".env"
    assert target.is_file()
    assert target.stat().st_mode & 0o777 == 0o600
    assert "AEGIS_DATABASE_URL=" in target.read_text(encoding="utf-8")
    assert "Created private configuration template: .env" in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", ["aegis", "--init"])
    assert cli.main() == 1
    assert "configuration file already exists" in capsys.readouterr().out


def test_run_interaction_threads_pack_runtime_registry_to_shared_boundary(monkeypatch):
    from aegis import cli
    from aegis.pack_runtime import PackRuntimeRegistry

    registry = PackRuntimeRegistry()
    captured = {}

    class Boundary:
        def __init__(self, dependencies):
            captured["dependencies"] = dependencies

        def run(self, *_args, **_kwargs):
            return "shared-result"

    monkeypatch.setattr(cli, "InteractionBoundary", Boundary)

    result = cli.run_interaction(
        "read the weather note",
        Principal(id="alice", vault_id="alice-vault"),
        runtime_registry=registry,
    )

    assert result == "shared-result"
    assert captured["dependencies"].runtime_registry is registry
    assert captured["dependencies"].action_grounder is cli.ground_reference_action_runtime
    assert captured["dependencies"].plan_runner is cli.run_reference_plan
    assert captured["dependencies"].decision_rewriter is cli.rewrite_reference_decision
    assert captured["dependencies"].fast_path_resolver is cli.resolve_reference_fast_paths


def test_run_interaction_unresolved_investigation_reports_installed_capabilities(monkeypatch):
    from aegis import cli
    from aegis.contracts import RequestedEffect
    from aegis.pack_runtime import PackRuntimeRegistry

    registry = PackRuntimeRegistry()
    registry.register("tasks.read", lambda *_args: object())
    captured = {}

    class Boundary:
        def __init__(self, dependencies):
            captured["dependencies"] = dependencies

        def run(self, *_args, **_kwargs):
            return "shared-result"

    monkeypatch.delenv("AEGIS_SEARCH_ENDPOINT", raising=False)
    monkeypatch.setattr(cli, "InteractionBoundary", Boundary)
    cli.run_interaction(
        "set up a cluster",
        Principal(id="alice", vault_id="alice-vault"),
        runtime_registry=registry,
    )

    intent = IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance="set up a cluster",
    )
    effect = RequestedEffect(source_spans=((0, 16),), normalized_effect="set up a cluster")
    result = captured["dependencies"].unresolved_requirement_investigator(
        intent, Context(), (effect,)
    )

    assert result.state is ObjectiveState.BLOCKED
    assert result.evidence["investigation"] == "authorized_capability_inventory"
    assert result.evidence["available_action_ids"] == ["tasks.read"]
    assert result.evidence["authoritative"] is False
    assert result.evidence["objective_open"] is True
    assert result.evidence["unsatisfied_requirements"][0]["effect_id"] == str(effect.effect_id)


def test_default_runtime_registry_covers_kitchen_mutation(monkeypatch):
    from aegis import cli
    from aegis.reference_packs import reference_bundles

    registry = cli._default_runtime_registry(lambda: object())
    assert "kitchen.groceries.add" in registry.action_ids()
    expected = {card.action.action_id for bundle in reference_bundles() for card in bundle.cards}
    assert set(registry.action_ids()) == expected


def test_browser_interaction_threads_pack_runtime_registry_to_shared_boundary(monkeypatch):
    from aegis import cli
    from aegis.pack_runtime import PackRuntimeRegistry

    registry = PackRuntimeRegistry()
    captured = {}
    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="verified third-party Pack result",
        correlation_id=uuid4(),
    )

    def interaction(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return result

    monkeypatch.setattr(cli, "run_interaction", interaction)

    response = cli._browser_interaction(
        "read the weather note",
        Principal(id="alice", vault_id="alice-vault"),
        runtime_registry=registry,
    )

    assert response["state"] == "completed"
    assert captured["kwargs"]["runtime_registry"] is registry


def test_browser_interaction_factory_binds_pack_registry(monkeypatch):
    from aegis import cli
    from aegis.pack_runtime import PackRuntimeRegistry

    registry = PackRuntimeRegistry()
    captured = {}
    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="verified Pack result",
        correlation_id=uuid4(),
    )

    def interaction(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return result

    monkeypatch.setattr(cli, "_browser_interaction", interaction)
    handler = cli.browser_interaction(registry)
    response = handler(
        "read the weather note",
        Principal(id="alice", vault_id="alice-vault"),
        uuid4(),
    )

    assert response is result
    assert captured["args"][0] == "read the weather note"
    assert captured["args"][4] is registry


def test_cli_init_refuses_symlink_target(monkeypatch, capsys, tmp_path):
    from aegis import cli

    monkeypatch.chdir(tmp_path)
    target = tmp_path / ".env"
    existing = tmp_path / "existing.env"
    existing.write_text("AEGIS_DATABASE_URL=untouched\n", encoding="utf-8")
    target.symlink_to(existing)
    monkeypatch.setattr("sys.argv", ["aegis", "--init"])

    assert cli.main() == 1
    assert "configuration file already exists" in capsys.readouterr().out
    assert existing.read_text(encoding="utf-8") == "AEGIS_DATABASE_URL=untouched\n"


def test_cli_init_reads_packaged_template(monkeypatch, capsys, tmp_path):
    from importlib.resources import files

    from aegis import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["aegis", "--init"])

    assert cli.main() == 0
    assert (tmp_path / ".env").read_text(encoding="utf-8") == files("aegis").joinpath(
        "aegis.env.example"
    ).read_text(encoding="utf-8")
    capsys.readouterr()


def test_cli_env_file_loads_aegis_settings_without_overriding_shell(monkeypatch, tmp_path):
    from aegis import cli

    env_file = tmp_path / "aegis.env"
    env_file.write_text(
        "AEGIS_DATABASE_URL=postgresql://file\nAEGIS_PRINCIPAL_ID=file-user\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AEGIS_DATABASE_URL", "postgresql://shell")
    monkeypatch.delenv("AEGIS_PRINCIPAL_ID", raising=False)

    cli._load_env_file(str(env_file))

    assert cli.os.environ["AEGIS_DATABASE_URL"] == "postgresql://shell"
    assert cli.os.environ["AEGIS_PRINCIPAL_ID"] == "file-user"


def test_cli_env_file_rejects_shell_syntax_and_non_aegis_keys(tmp_path):
    from aegis import cli

    env_file = tmp_path / "aegis.env"
    env_file.write_text("export AEGIS_DATABASE_URL=secret\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid env file key"):
        cli._load_env_file(str(env_file))


def test_cli_env_file_rejects_duplicate_keys(tmp_path):
    from aegis import cli

    env_file = tmp_path / "aegis.env"
    env_file.write_text(
        "AEGIS_DATABASE_URL=postgresql://first\nAEGIS_DATABASE_URL=postgresql://second\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate env file key"):
        cli._load_env_file(str(env_file))


def test_cli_auto_discovers_local_env_file(monkeypatch, capsys, tmp_path):
    from aegis import cli
    from aegis.health import HealthReport

    (tmp_path / ".env").write_text("AEGIS_PRINCIPAL_ID=discovered-user\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AEGIS_PRINCIPAL_ID", raising=False)
    monkeypatch.setattr("sys.argv", ["aegis", "--check", "--json"])

    def report():
        assert cli.os.environ["AEGIS_PRINCIPAL_ID"] == "discovered-user"
        return HealthReport(healthy=True, ready=True, components=())

    monkeypatch.setattr(cli, "_runtime_report", report)

    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["ready"] is True


def test_cli_env_file_json_failure_is_machine_safe(monkeypatch, capsys, tmp_path):
    from aegis import cli

    env_file = tmp_path / "aegis.env"
    env_file.write_text("not-a-setting\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["aegis", "--check", "--json", "--env-file", str(env_file)])

    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "code": "configuration_invalid",
        "error": "configuration file is invalid",
        "state": "failed",
    }


def test_cli_once_routes_through_handle(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--once", "Show my tasks."])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(cli, "handle", lambda utterance, principal: f"handled: {utterance}")

    cli.main()

    assert capsys.readouterr().out == "handled: Show my tasks.\n"


def test_cli_once_returns_failure_status_for_handled_error(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--once", "Show my tasks."])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )

    def fail(_utterance, _principal):
        raise ValueError("request is ambiguous")

    monkeypatch.setattr(cli, "handle", fail)

    assert cli.main() == 1
    assert capsys.readouterr().out == "Not completed — request is ambiguous\n"


def test_cli_once_json_serializes_canonical_result(monkeypatch, capsys):
    from aegis import cli

    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="task list read",
        evidence={"canonical_tasks": []},
        correlation_id=uuid4(),
    )
    monkeypatch.setattr("sys.argv", ["aegis", "--once", "Show my tasks", "--json"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(cli, "run_interaction", lambda *_: result)

    assert cli.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "completed"
    assert payload["objective_id"] == str(result.objective_id)
    assert payload["evidence"] == {"canonical_tasks": []}


def test_cli_once_json_returns_failure_for_non_completed_result(monkeypatch, capsys):
    from aegis import cli

    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.BLOCKED,
        message="authorization denied",
        correlation_id=uuid4(),
    )
    monkeypatch.setattr("sys.argv", ["aegis", "--once", "do it", "--json"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(cli, "run_interaction", lambda *_: result)

    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out)["state"] == "blocked"


def test_cli_once_hides_database_failure_details(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--once", "Show my tasks."])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(
        cli,
        "run_interaction",
        lambda *_: (_ for _ in ()).throw(cli.psycopg.OperationalError("password=private-secret")),
    )

    assert cli.main() == 1
    output = capsys.readouterr().out
    assert output == (
        "Not completed — request unavailable; run "
        "'./scripts/aegis --check' and verify AEGIS_DATABASE_URL\n"
    )
    assert "private-secret" not in output


def test_cli_once_hides_unexpected_request_failure_details(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--once", "Show my tasks."])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(
        cli,
        "handle",
        lambda *_: (_ for _ in ()).throw(Exception("provider private detail")),
    )

    assert cli.main() == 1
    output = capsys.readouterr().out
    assert output == (
        "Not completed — request unavailable; run "
        "'./scripts/aegis --check' and verify configured services\n"
    )
    assert "provider private detail" not in output


def test_interactive_cli_contains_database_failure(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--no-banner"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(
        cli,
        "handle",
        lambda *_: (_ for _ in ()).throw(cli.psycopg.OperationalError("password=private-secret")),
    )
    prompts = iter(["Show my tasks.", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(prompts))

    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "request unavailable; run './scripts/aegis --check'" in output
    assert "private-secret" not in output


def test_interactive_cli_hides_runtime_failure_details(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--no-banner"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(
        cli,
        "handle",
        lambda *_: (_ for _ in ()).throw(RuntimeError("gateway-password=private-secret")),
    )
    prompts = iter(["Show my tasks.", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(prompts))

    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "request unavailable; run './scripts/aegis --check'" in output
    assert "private-secret" not in output


def test_interactive_cli_hides_unexpected_request_failure_details(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--no-banner"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(
        cli,
        "handle",
        lambda *_: (_ for _ in ()).throw(Exception("provider private detail")),
    )
    prompts = iter(["Show my tasks.", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(prompts))

    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "request unavailable; run './scripts/aegis --check'" in output
    assert "provider private detail" not in output


def test_cli_once_json_returns_stable_error_for_request_failure(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--once", "do it", "--json"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(
        cli,
        "run_interaction",
        lambda *_: (_ for _ in ()).throw(RuntimeError("secret implementation detail")),
    )

    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "code": "request_unavailable",
        "error": "request unavailable",
        "state": "failed",
    }


def test_cli_once_json_contains_value_error_details(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--once", "do it", "--json"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(
        cli,
        "run_interaction",
        lambda *_: (_ for _ in ()).throw(ValueError("private parser detail")),
    )

    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "code": "request_unavailable",
        "error": "request unavailable",
        "state": "failed",
    }


def test_cli_once_json_returns_stable_error_for_denied_request(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--once", "private read", "--json"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(
        cli,
        "run_interaction",
        lambda *_: (_ for _ in ()).throw(PermissionError("private Vault")),
    )

    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "code": "request_denied",
        "error": "request denied",
        "state": "failed",
    }


def test_cli_once_json_returns_stable_error_for_identity_failure(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--once", "do it", "--json"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: (_ for _ in ()).throw(RuntimeError("secret identity detail")),
    )

    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "code": "identity_unavailable",
        "error": "identity unavailable",
        "state": "failed",
    }


def test_cli_human_identity_errors_are_generic(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--once", "Show my tasks."])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: (_ for _ in ()).throw(
            cli.psycopg.OperationalError("password=private-secret host=internal-db")
        ),
    )

    assert cli.main() == 1
    output = capsys.readouterr().out
    assert output == (
        "Not completed — identity unavailable; run './scripts/aegis --check' and "
        "verify identity configuration\n"
    )
    assert "private-secret" not in output
    assert "internal-db" not in output


def test_cli_human_unexpected_identity_errors_are_generic(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--once", "Show my tasks."])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: (_ for _ in ()).throw(Exception("provider private detail")),
    )

    assert cli.main() == 1
    output = capsys.readouterr().out
    assert output == (
        "Not completed — identity unavailable; run './scripts/aegis --check' and "
        "verify identity configuration\n"
    )
    assert "provider private detail" not in output


def test_cli_json_requires_check_or_once(monkeypatch):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--json"])
    with pytest.raises(SystemExit) as raised:
        cli.main()

    assert raised.value.code == 2


def test_cli_rejects_invalid_browser_port(monkeypatch):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--web", "--port", "70000"])
    with pytest.raises(SystemExit) as raised:
        cli.main()

    assert raised.value.code == 2


def test_cli_web_reports_bootstrap_failure(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--web", "--port", "8081"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(
        cli,
        "_prepare_local_web_runtime",
        lambda _principal: (_ for _ in ()).throw(RuntimeError("missing database")),
    )
    monkeypatch.setattr(cli, "serve", lambda *_args, **_kwargs: None)

    assert cli.main() == 0
    assert capsys.readouterr().out == (
        "AEGIS runtime is not ready; the browser will show diagnostics. "
        "Run './scripts/aegis --check' to see remediation.\n"
        "Starting AEGIS Constellation at http://127.0.0.1:8081\n"
    )


def test_cli_web_reports_port_conflict_with_remediation(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--web", "--port", "8081"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(cli, "_prepare_local_web_runtime", lambda _principal: None)
    monkeypatch.setattr(
        cli,
        "serve",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError(cli.errno.EADDRINUSE, "address occupied")
        ),
    )

    assert cli.main() == 1
    assert capsys.readouterr().out == (
        "Starting AEGIS Constellation at http://127.0.0.1:8081\n"
        "Not completed — browser port 8081 is already in use; choose another with --port\n"
    )


def test_cli_web_hides_database_failure_details(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--web", "--port", "8082"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(
        cli,
        "_prepare_local_web_runtime",
        lambda _principal: (_ for _ in ()).throw(
            cli.psycopg.OperationalError("password=private-secret")
        ),
    )
    monkeypatch.setattr(cli, "serve", lambda *_args, **_kwargs: None)

    assert cli.main() == 0
    output = capsys.readouterr().out
    assert output == (
        "AEGIS runtime is not ready; the browser will show diagnostics. "
        "Run './scripts/aegis --check' to see remediation.\n"
        "Starting AEGIS Constellation at http://127.0.0.1:8082\n"
    )
    assert "private-secret" not in output


def test_cli_check_reports_missing_required_configuration(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--check"])
    for name in ("AEGIS_DATABASE_URL", "AEGIS_OLLAMA_URL"):
        monkeypatch.delenv(name, raising=False)

    assert cli.main() == 1
    output = capsys.readouterr().out
    assert "AEGIS runtime: NOT READY" in output
    assert "postgres: FAIL (required) — set AEGIS_DATABASE_URL" in output
    assert "ollama: FAIL (required)" in output
    assert "openclaw: OK (optional)" in output


def test_identity_health_rejects_incomplete_bearer_configuration(monkeypatch):
    from aegis import cli

    monkeypatch.setenv("AEGIS_KEYCLOAK_ACCESS_TOKEN", "token")
    monkeypatch.delenv("AEGIS_KEYCLOAK_ISSUER", raising=False)

    assert cli._identity_health() == (
        False,
        "incomplete bearer identity configuration; set both AEGIS_KEYCLOAK_ISSUER and "
        "AEGIS_KEYCLOAK_ACCESS_TOKEN",
    )


@pytest.mark.parametrize("configured", ["token", "issuer"])
def test_principal_does_not_fall_back_to_local_identity_for_incomplete_bearer_config(
    monkeypatch, configured
):
    from aegis import cli

    monkeypatch.delenv("AEGIS_KEYCLOAK_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("AEGIS_KEYCLOAK_ISSUER", raising=False)
    monkeypatch.setenv(
        "AEGIS_KEYCLOAK_ACCESS_TOKEN" if configured == "token" else "AEGIS_KEYCLOAK_ISSUER",
        "token" if configured == "token" else "https://keycloak.example/realms/aegis",
    )

    with pytest.raises(RuntimeError, match="incomplete bearer identity configuration"):
        cli._principal()


def test_identity_health_hides_bearer_mapping_failure(monkeypatch):
    from aegis import cli

    monkeypatch.setenv("AEGIS_KEYCLOAK_ACCESS_TOKEN", "token")
    monkeypatch.setenv("AEGIS_KEYCLOAK_ISSUER", "https://keycloak.example/realms/aegis")
    monkeypatch.setenv("AEGIS_DATABASE_URL", "postgresql://db")
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: (_ for _ in ()).throw(RuntimeError("password=private-secret")),
    )

    healthy, detail = cli._identity_health()

    assert healthy is False
    assert detail == (
        "bearer identity is unavailable; verify the token, Keycloak issuer, and "
        "canonical subject mapping"
    )
    assert "private-secret" not in detail


def test_postgres_health_reports_partial_canonical_schema(monkeypatch):
    from aegis import cli

    class Cursor:
        def fetchone(self):
            return ("objectives", None, "space_memberships")

    class Connection:
        def execute(self, *_args, **_kwargs):
            return Cursor()

        def close(self):
            pass

    monkeypatch.setattr(cli.psycopg, "connect", lambda *_args, **_kwargs: Connection())

    component = cli._postgres_health("postgresql://operator:secret@example.test/aegis")

    assert component.healthy is False
    assert component.detail == (
        "connection succeeded but the canonical schema is incomplete; apply migrations before "
        "starting AEGIS"
    )


def test_postgres_health_connection_failure_has_safe_remediation(monkeypatch):
    from aegis import cli

    monkeypatch.setattr(
        cli.psycopg,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            cli.psycopg.OperationalError("password=private-secret")
        ),
    )

    component = cli._postgres_health("postgresql://operator:secret@example.test/aegis")

    assert component.healthy is False
    assert component.detail == (
        "connection failed: OperationalError; verify AEGIS_DATABASE_URL and ensure PostgreSQL is "
        "running"
    )


def test_postgres_health_contains_unexpected_driver_failure(monkeypatch):
    from aegis import cli

    monkeypatch.setattr(
        cli.psycopg,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(Exception("password=private-secret")),
    )

    component = cli._postgres_health("postgresql://operator:secret@example.test/aegis")

    assert component.healthy is False
    assert component.detail == (
        "database health check failed; verify AEGIS_DATABASE_URL and ensure PostgreSQL is running"
    )
    assert "private-secret" not in component.detail


def test_ollama_health_contains_unexpected_client_failure(monkeypatch):
    from aegis import cli

    monkeypatch.setattr(
        cli.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(Exception("token=private-secret")),
    )

    component = cli._ollama_health("http://ollama.example:11434", "qwen3:8b")

    assert component.healthy is False
    assert component.detail == (
        "Ollama health check failed at http://ollama.example:11434; check "
        "`curl http://ollama.example:11434/api/tags`, start Ollama, or set "
        "AEGIS_OLLAMA_URL to its reachable address"
    )
    assert "private-secret" not in component.detail
    assert "private-secret" not in component.detail


def test_ollama_model_digest_is_metadata_only_and_missing_is_safe(monkeypatch):
    from aegis import cli

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"models":[{"name":"qwen3:8b","digest":"sha256:abc"}]}'

    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())

    assert cli._ollama_model_digest("http://ollama.example:11434", "qwen3:8b") == "sha256:abc"
    assert cli._ollama_model_digest("http://ollama.example:11434", "missing") is None


def test_postgres_health_rejects_malformed_connection_configuration():
    from aegis import cli

    component = cli._postgres_health("not-a-postgresql-url")

    assert component.healthy is False
    assert component.detail == "invalid database configuration; verify AEGIS_DATABASE_URL"


def test_postgres_health_detects_generated_template_placeholder():
    from aegis import cli

    component = cli._postgres_health("postgresql://USER:PASSWORD@127.0.0.1:5432/DBNAME")

    assert component.healthy is False
    assert component.detail == (
        "configuration contains template placeholders; replace AEGIS_DATABASE_URL"
    )


def test_cli_check_json_is_machine_readable(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--check", "--json"])
    monkeypatch.delenv("AEGIS_DATABASE_URL", raising=False)
    monkeypatch.delenv("AEGIS_OLLAMA_URL", raising=False)

    assert cli.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ready"] is False
    assert {item["name"] for item in report["components"]} == {
        "postgres",
        "ollama",
        "openclaw",
        "structural_parser",
        "identity",
    }
    assert report["runtime"]["execution_mode"] == "source-checkout"


def test_cli_check_rejects_missing_ollama_model(monkeypatch, capsys):
    from aegis import cli

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"models": [{"name": "another-model"}]}'

    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr("sys.argv", ["aegis", "--check"])
    monkeypatch.delenv("AEGIS_DATABASE_URL", raising=False)
    monkeypatch.setenv("AEGIS_OLLAMA_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("AEGIS_OLLAMA_MODEL", "qwen3:8b")

    assert cli.main() == 1
    output = capsys.readouterr().out
    assert "ollama: FAIL (required)" in output
    assert "ollama pull qwen3:8b" in output


def test_cli_check_explains_unreachable_ollama_endpoint(monkeypatch, capsys):
    from aegis import cli

    def unavailable(*_args, **_kwargs):
        raise cli.urllib.error.URLError("connection refused")

    monkeypatch.setattr(cli.urllib.request, "urlopen", unavailable)
    monkeypatch.setattr("sys.argv", ["aegis", "--check"])
    monkeypatch.delenv("AEGIS_DATABASE_URL", raising=False)
    monkeypatch.setenv("AEGIS_OLLAMA_URL", "http://127.0.0.1:11434")

    assert cli.main() == 1
    output = capsys.readouterr().out
    assert (
        "API unavailable at http://127.0.0.1:11434: URLError; "
        "check `curl http://127.0.0.1:11434/api/tags`" in output
    )


def test_cli_check_rejects_invalid_ollama_endpoint(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--check"])
    monkeypatch.delenv("AEGIS_DATABASE_URL", raising=False)
    monkeypatch.setenv("AEGIS_OLLAMA_URL", "ftp://ollama.example.invalid")

    assert cli.main() == 1
    output = capsys.readouterr().out
    assert "ollama: FAIL (required) — invalid URL" in output
    assert "set AEGIS_OLLAMA_URL to an http:// or https:// endpoint" in output


def test_safe_endpoint_does_not_render_url_credentials():
    from aegis.cli import _safe_endpoint

    assert _safe_endpoint("http://secret:password@example.test:11434/api") == (
        "http://example.test:11434/api"
    )


def test_browser_app_uses_core_callbacks_for_state_and_messages():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    seen: list[tuple[str, str]] = []
    app = BrowserApp(
        principal,
        lambda utterance, current, _correlation_id: (
            seen.append((utterance, current.id)) or "canonical answer"
        ),
        lambda current: {"nodes": [{"id": "tasks", "label": "Tasks", "detail": current.id}]},
    )

    status, content_type, payload = app.dispatch("GET", "/api/constellation")
    assert status == 200
    assert content_type == "application/json"
    assert json.loads(payload)["nodes"][0]["detail"] == "alice"

    status, _, payload = app.dispatch(
        "POST",
        "/api/message",
        b'{"utterance":"Show my tasks.","correlation_id":"00000000-0000-4000-8000-000000000001"}',
    )
    assert status == 200
    response = json.loads(payload)
    assert response["message"] == "canonical answer"
    assert response["correlation_id"] == "00000000-0000-4000-8000-000000000001"
    assert UUID(response["session_id"])
    assert seen == [("Show my tasks.", "alice")]


def test_browser_app_exposes_principal_scoped_workspace_inventory():
    principal = Principal(id="alice", vault_id="vault")
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        workspace_state=lambda current: {
            "workspaces": [{"workspace_id": current.id, "files": ["index.html"]}]
        },
        session_token="session-secret",
    )
    status, _, payload = app.dispatch(
        "GET", "/api/workspace", headers={"X-Aegis-Session": "session-secret"}
    )
    assert status == 200
    assert json.loads(payload)["workspaces"] == [{"workspace_id": "alice", "files": ["index.html"]}]


def test_browser_app_workspace_file_exposes_observed_sha256():
    principal = Principal(id="alice", vault_id="vault")
    workspace_id = str(uuid4())
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        workspace_file=lambda current, requested_workspace, path: {
            "workspace_id": requested_workspace,
            "path": path,
            "content": "artifact",
            "sha256": hashlib.sha256(b"artifact").hexdigest(),
        },
        session_token="session-secret",
    )
    status, _, payload = app.dispatch(
        "GET",
        f"/api/workspace/file?workspace_id={workspace_id}&path=index.html",
        headers={"X-Aegis-Session": "session-secret"},
    )
    assert status == 200
    assert json.loads(payload)["sha256"] == hashlib.sha256(b"artifact").hexdigest()


def test_browser_app_keeps_bounded_workspace_inventory_readable_over_twenty_entries():
    principal = Principal(id="alice", vault_id="vault")
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        workspace_state=lambda _current: {
            "workspaces": [
                {"workspace_id": str(index), "files": ["artifact.md"]} for index in range(21)
            ]
        },
        session_token="session-secret",
    )
    status, _, payload = app.dispatch(
        "GET", "/api/workspace", headers={"X-Aegis-Session": "session-secret"}
    )
    assert status == 200
    assert len(json.loads(payload)["workspaces"]) == 21


def test_browser_app_exposes_composition_metadata_without_execution_authority():
    principal = Principal(id="alice", vault_id="vault")
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        composition_state=lambda current: {
            "compositions": [{"id": "docs-workspace", "owner": current.id}]
        },
        session_token="session-secret",
    )
    status, _, payload = app.dispatch(
        "GET", "/api/compositions", headers={"X-Aegis-Session": "session-secret"}
    )
    assert status == 200
    assert json.loads(payload)["compositions"] == [{"id": "docs-workspace", "owner": "alice"}]


def test_browser_app_compositions_surface_exposes_readable_workflow_cards():
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_: "unused",
        lambda _: {"nodes": []},
        composition_state=lambda _current: {
            "compositions": [
                {
                    "id": "research-draft",
                    "label": "Research → Draft",
                    "description": "Create an unsent sourced draft",
                    "surfaces": ["Research", "Communications"],
                    "authority": "Core authorization required",
                }
            ]
        },
        session_token="session-secret",
    )
    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    html = payload.decode()
    assert "trusted cross-capability workflow(s)" in html
    assert "Surfaces: ${composition.surfaces.join(' · ')}" in html
    assert "Authority: ${composition.authority" in html


def test_browser_app_exposes_pack_lifecycle_without_granting_permissions():
    principal = Principal(id="alice", vault_id="vault")
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        pack_state=lambda current: {
            "packs": [
                {
                    "pack_id": "communications",
                    "status": "discovered",
                    "granted_permissions": [],
                    "owner_next_step": "explicit owner approval is required",
                    "owner": current.id,
                }
            ]
        },
        session_token="session-secret",
    )
    status, _, payload = app.dispatch(
        "GET", "/api/packs", headers={"X-Aegis-Session": "session-secret"}
    )
    assert status == 200
    pack = json.loads(payload)["packs"][0]
    assert pack["status"] == "discovered"
    assert pack["granted_permissions"] == []
    assert pack["owner"] == "alice"


def test_browser_app_exposes_bounded_calendar_projection():
    principal = Principal(id="alice", vault_id="vault")
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        calendar_state=lambda current: {
            "source": "configured_calendar",
            "events": [{"event_id": "event-1", "title": current.id}],
        },
        session_token="session-secret",
    )
    status, _, payload = app.dispatch(
        "GET", "/api/calendar", headers={"X-Aegis-Session": "session-secret"}
    )
    assert status == 200
    assert json.loads(payload)["events"] == [{"event_id": "event-1", "title": "alice"}]


def test_browser_app_exposes_authorized_documents_and_scoped_read():
    principal = Principal(id="alice", vault_id="vault")
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        documents_state=lambda current: {
            "source": "authorized_document_fixture",
            "documents": [{"document_id": "doc-1", "title": current.id}],
        },
        document_file=lambda current, document_id: {
            "document_id": document_id,
            "title": current.id,
            "text": "scoped content",
        },
        session_token="session-secret",
    )
    status, _, payload = app.dispatch(
        "GET", "/api/documents", headers={"X-Aegis-Session": "session-secret"}
    )
    assert status == 200
    assert json.loads(payload)["documents"][0]["title"] == "alice"

    status, _, payload = app.dispatch(
        "GET",
        "/api/documents/file?document_id=doc-1",
        headers={"X-Aegis-Session": "session-secret"},
    )
    assert status == 200
    assert json.loads(payload)["text"] == "scoped content"


def test_browser_app_rejects_malformed_document_read_query():
    principal = Principal(id="alice", vault_id="vault")
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        document_file=lambda *_: {"text": "must not be called"},
        session_token="session-secret",
    )
    status, _, _ = app.dispatch(
        "GET",
        "/api/documents/file?document_id=doc-1&extra=x",
        headers={"X-Aegis-Session": "session-secret"},
    )
    assert status == 400


def test_browser_app_exposes_daily_driver_scoreboard_without_authority():
    principal = Principal(id="alice", vault_id="vault")
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        daily_driver_state=lambda current: {
            "source_basis_sha": "abc123",
            "statuses": {"documents": "INSTALLED"},
            "metrics": {"owner_visible_capability_surfaces": 17},
            "provider_gates": ["Google Calendar remains pending"],
            "boundary": "descriptive only",
            "owner": current.id,
        },
        session_token="session-secret",
    )
    status, _, payload = app.dispatch(
        "GET", "/api/daily-driver", headers={"X-Aegis-Session": "session-secret"}
    )
    assert status == 200
    result = json.loads(payload)
    assert result["statuses"]["documents"] == "INSTALLED"
    assert result["metrics"]["owner_visible_capability_surfaces"] == 17
    assert result["owner"] == "alice"


def test_browser_app_exposes_bounded_device_projection():
    principal = Principal(id="alice", vault_id="vault")
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        device_state=lambda current: {
            "source": "authorized_device_fixture",
            "devices": [{"entity_id": "light.desk", "state": "off", "owner": current.id}],
        },
        session_token="session-secret",
    )
    status, _, payload = app.dispatch(
        "GET", "/api/devices", headers={"X-Aegis-Session": "session-secret"}
    )
    assert status == 200
    assert json.loads(payload)["devices"][0]["state"] == "off"


def test_browser_app_exposes_communications_outcome_projection():
    principal = Principal(id="alice", vault_id="vault")
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        communications_state=lambda current: {
            "messages": [
                {
                    "target": current.id,
                    "provider_status": "PROVIDER_ACCEPTED",
                    "provider_readback_proven": True,
                    "delivery_proven": False,
                }
            ],
            "provider_boundary": "acceptance is not delivery",
        },
        session_token="session-secret",
    )
    status, _, payload = app.dispatch(
        "GET", "/api/communications", headers={"X-Aegis-Session": "session-secret"}
    )
    assert status == 200
    outcome = json.loads(payload)["messages"][0]
    assert outcome["target"] == "alice"
    assert outcome["provider_status"] == "PROVIDER_ACCEPTED"
    assert outcome["provider_readback_proven"] is True
    assert outcome["delivery_proven"] is False


def test_browser_app_exposes_truthful_today_projection():
    principal = Principal(id="alice", vault_id="vault")
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        today_state=lambda _current: {
            "canonical": {
                "open_tasks": [{"title": "Review backup"}],
                "completed_tasks": [{"title": "Buy milk", "status": "completed"}],
            },
            "external_calendar": {"source": "external_calendar_fixture", "events": []},
            "truth_boundary": "canonical state is distinct from external evidence",
        },
        session_token="session-secret",
    )
    status, _, payload = app.dispatch(
        "GET", "/api/today", headers={"X-Aegis-Session": "session-secret"}
    )
    assert status == 200
    assert json.loads(payload)["canonical"]["open_tasks"] == [{"title": "Review backup"}]
    assert json.loads(payload)["canonical"]["completed_tasks"][0]["status"] == "completed"


def test_browser_app_exposes_objective_capability_needs():
    principal = Principal(id="alice", vault_id="vault")
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        objectives_state=lambda current: {
            "objectives": [
                {
                    "objective_id": "objective-1",
                    "state": "blocked",
                    "updated_at": "2026-09-05T00:00:00+00:00",
                    "utterance": "Set up an unsupported service",
                    "capability_needs": [{"status": "open", "requested_effect": current.id}],
                }
            ]
        },
        session_token="session-secret",
    )
    status, _, payload = app.dispatch(
        "GET", "/api/objectives", headers={"X-Aegis-Session": "session-secret"}
    )
    assert status == 200
    assert json.loads(payload)["objectives"][0]["capability_needs"][0]["status"] == "open"


def test_browser_app_routes_pack_enablement_through_explicit_owner_callback():
    principal = Principal(id="alice", vault_id="vault")
    seen: list[tuple[str, dict[str, object]]] = []

    def enable(current, request):
        seen.append((current.id, request))
        return {"pack_id": request["pack_id"], "status": "enabled"}

    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        pack_enable=enable,
        session_token="session-secret",
    )
    status, _, payload = app.dispatch(
        "POST",
        "/api/packs/enable",
        b'{"pack_id":"communications","permissions":["communications.read"],"confirm":true}',
        headers={"X-Aegis-Session": "session-secret"},
    )
    assert status == 200
    assert json.loads(payload) == {"pack_id": "communications", "status": "enabled"}
    assert seen == [
        (
            "alice",
            {
                "pack_id": "communications",
                "permissions": ["communications.read"],
                "confirm": True,
            },
        )
    ]


def test_browser_app_routes_workspace_creation_through_authorized_callback():
    principal = Principal(id="alice", vault_id="vault")
    seen: list[tuple[str, dict[str, object]]] = []

    def create(current, request):
        seen.append((current.id, request))
        return {"status": "verified", "artifact": "index.html"}

    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        workspace_create=create,
        session_token="session-secret",
    )
    status, _, payload = app.dispatch(
        "POST",
        "/api/workspace",
        b'{"operation":"create_artifact"}',
        headers={"X-Aegis-Session": "session-secret"},
    )
    assert status == 200
    assert json.loads(payload) == {"status": "verified", "artifact": "index.html"}
    assert seen == [("alice", {"operation": "create_artifact"})]


def test_browser_app_session_gate_rejects_unauthenticated_api_requests():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_args: "answer",
        lambda _current: {"nodes": []},
        session_token="session-secret",
    )

    status, _, payload = app.dispatch("GET", "/api/constellation")
    assert status == 401
    assert json.loads(payload) == {
        "code": "identity_unavailable",
        "error": "identity unavailable",
    }

    status, _, payload = app.dispatch(
        "GET", "/api/constellation", headers={"X-Aegis-Session": "session-secret"}
    )
    assert status == 200
    assert json.loads(payload) == {"nodes": [], "edges": [], "details": {}}

    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    assert "session-secret" in payload.decode()


def test_browser_app_communications_surface_offers_core_resolved_source_choice():
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_args: "unused",
        lambda _current: {"nodes": []},
        session_token="session-secret",
    )
    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    html = payload.decode()
    assert "Use authorized grocery list" in html
    assert "Canonical sources are resolved by Core" in html
    assert "Text my grocery list to" in html


def test_browser_app_task_and_household_views_expose_core_completion_affordances():
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_args: "unused",
        lambda _current: {"nodes": []},
        session_token="session-secret",
    )
    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    html = payload.decode()
    assert "Complete the task" in html
    assert "Mark the chore" in html


def test_browser_app_documents_surface_exposes_bounded_search():
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_args: "unused",
        lambda _current: {"nodes": []},
        session_token="session-secret",
    )
    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    html = payload.decode()
    assert "Search authorized documents" in html
    assert "Find my documents for" in html
    assert "bounded to documents authorized" in html


def test_browser_app_workspace_surface_exposes_sandboxed_static_preview():
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_args: "unused",
        lambda _current: {"nodes": []},
        session_token="session-secret",
    )
    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    html = payload.decode()
    assert "Preview ${path}" in html
    assert "setAttribute('sandbox', '')" in html
    assert "Static preview of ${path}" in html


def test_browser_app_documents_surface_exposes_search_to_workspace_composition():
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_args: "unused",
        lambda _current: {"nodes": []},
        session_token="session-secret",
    )
    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    html = payload.decode()
    assert "Save results to Workspace" in html
    assert "Find my documents for ${query} and save results as ${path}" in html
    assert "independent verification path" in html


def test_browser_app_workspace_surface_exposes_scoped_download():
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_args: "unused",
        lambda _current: {"nodes": []},
        session_token="session-secret",
    )
    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    html = payload.decode()
    assert "Download ${path}" in html
    assert "URL.createObjectURL" in html
    assert "workspace/file?workspace_id=" in html
    assert "Observed SHA-256: ${file.sha256" in html


def test_browser_app_constellation_exposes_conventional_navigation_and_bounded_focus():
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_args: "unused",
        lambda _current: {"nodes": []},
        session_token="session-secret",
    )
    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    html = payload.decode()
    assert "Open ${targetNav?.textContent || targetView} view" in html
    assert "Ask about ${node.label}" in html
    assert "Tell me about ${node.label}" in html


def test_browser_app_research_surface_exposes_research_to_workspace_composition():
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_args: "unused",
        lambda _current: {"nodes": []},
        session_token="session-secret",
    )
    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    html = payload.decode()
    assert "Save sourced research to Workspace" in html
    assert "Research and save notes" in html
    assert "Research ${question} and save notes as ${targetPath}" in html
    assert "Public evidence remains non-canonical" in html


def test_browser_app_research_surface_exposes_unsent_communication_draft():
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_args: "unused",
        lambda _current: {"nodes": []},
        session_token="session-secret",
    )
    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    html = payload.decode()
    assert "Draft a researched message" in html
    assert "Research and create unsent draft" in html
    assert "Draft researched message to ${values[0]} with subject" in html
    assert "no message is sent" in html


def test_browser_app_calendar_surface_exposes_read_only_task_attention():
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_args: "unused",
        lambda _current: {"nodes": []},
        session_token="session-secret",
    )
    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    html = payload.decode()
    assert "Tasks before shared events" in html
    assert "Calendar + Tasks attention" in html
    assert "Read-only Calendar + Tasks attention" in html


def test_browser_app_calendar_surface_exposes_provider_readback_cancellation():
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_args: "unused",
        lambda _current: {"nodes": []},
        session_token="session-secret",
    )
    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    html = payload.decode()
    assert "Cancel an authorized event" in html
    assert "Cancel calendar event ${eventRecord.event_id}" in html
    assert "absence readback" in html


def test_browser_app_calendar_surface_exposes_read_only_conflict_inspection():
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_args: "unused",
        lambda _current: {"nodes": []},
        session_token="session-secret",
    )
    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    html = payload.decode()
    assert "Scheduling conflicts" in html
    assert "No overlapping timed events detected." in html
    assert "Conflict inspection is read-only." in html


def test_browser_app_today_surface_exposes_calendar_conflicts():
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_args: "unused",
        lambda _current: {"nodes": []},
        session_token="session-secret",
    )
    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    html = payload.decode()
    assert "Scheduling conflicts" in html
    assert "No overlapping timed events detected." in html
    assert "payload.external_calendar?.conflicts" in html


def test_browser_app_calendar_surface_exposes_provider_readback_update():
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_args: "unused",
        lambda _current: {"nodes": []},
        session_token="session-secret",
    )
    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    html = payload.decode()
    assert "Update an authorized event" in html
    assert "changed fields are independently read back" in html
    assert "Update calendar event ${eventRecord.event_id} to" in html


def test_browser_app_calendar_surface_exposes_workspace_snapshot():
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_args: "unused",
        lambda _current: {"nodes": []},
        session_token="session-secret",
    )
    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    html = payload.decode()
    assert "Save calendar snapshot to Workspace" in html
    assert "Save my calendar snapshot to Workspace as agenda.md" in html


def test_browser_app_devices_surface_exposes_workspace_snapshot():
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_args: "unused",
        lambda _current: {"nodes": []},
        session_token="session-secret",
    )
    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    html = payload.decode()
    assert "Save device snapshot to Workspace" in html
    assert "Save authorized device states to Workspace as devices.md" in html


def test_browser_app_communications_surface_exposes_provider_outcome_distinction():
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_args: "unused",
        lambda _current: {"nodes": []},
        session_token="session-secret",
    )
    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    html = payload.decode()
    assert "Provider outcome status" in html
    assert "delivery not proven" in html
    assert "DRAFTED" in html
    assert "provider readback unavailable" in html


def test_browser_app_objectives_surface_exposes_capability_need_investigation_boundary():
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_args: "unused",
        lambda _current: {"nodes": []},
        session_token="session-secret",
    )
    status, _, payload = app.dispatch("GET", "/")
    assert status == 200
    html = payload.decode()
    assert "Candidate resolutions" in html
    assert "Investigation: ${investigation}" in html
    assert "discovery does not grant installation" in html


def test_browser_app_passes_optional_context_correlation_to_shared_boundary():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    seen: list[object] = []
    context_id = uuid4()

    def contextual(_utterance, _principal, _correlation, prior_correlation):
        seen.append(prior_correlation)
        return "contextual answer"

    app = BrowserApp(
        principal,
        lambda *_args: "unused",
        lambda _current: {"nodes": []},
        contextual_interaction=contextual,
    )
    status, _content_type, payload = app.dispatch(
        "POST",
        "/api/message",
        json.dumps(
            {
                "utterance": "Which of those should I buy first?",
                "correlation_id": str(uuid4()),
                "context_correlation_id": str(context_id),
            }
        ).encode(),
    )

    assert status == 200
    assert seen == [context_id]
    assert json.loads(payload)["message"] == "contextual answer"


def test_browser_app_preserves_supplied_session_identity():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    session_id = uuid4()
    app = BrowserApp(
        principal,
        lambda *_args: "answer",
        lambda _current: {"nodes": []},
    )

    status, _, payload = app.dispatch(
        "POST",
        "/api/message",
        json.dumps(
            {
                "utterance": "show tasks",
                "correlation_id": str(uuid4()),
                "session_id": str(session_id),
            }
        ).encode(),
    )

    assert status == 200
    assert json.loads(payload)["session_id"] == str(session_id)


def test_browser_app_rejects_malformed_session_id_before_core():
    called = False

    def interaction(*_args):
        nonlocal called
        called = True
        return "unreachable"

    app = BrowserApp(
        Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
        interaction,
        lambda _: {"nodes": []},
    )
    status, _, payload = app.dispatch(
        "POST",
        "/api/message",
        b'{"utterance":"show tasks","session_id":"bad"}',
    )

    assert status == 400
    assert json.loads(payload) == {"code": "invalid_request", "error": "invalid request"}
    assert called is False


def test_browser_app_records_bounded_feedback_for_a_response_correlation():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    seen: list[tuple[str, str | None]] = []
    app = BrowserApp(
        principal,
        lambda *_args: "answer",
        lambda _current: {"nodes": []},
        feedback=lambda _principal, correlation, outcome, reason: seen.append(
            (str(correlation), outcome if reason is None else f"{outcome}:{reason}")
        ),
    )
    correlation = uuid4()
    status, _, payload = app.dispatch(
        "POST",
        "/api/feedback",
        json.dumps({"correlation_id": str(correlation), "outcome": "not_helpful"}).encode(),
    )
    assert status == 200
    assert json.loads(payload) == {"recorded": True, "correlation_id": str(correlation)}
    assert seen == [(str(correlation), "not_helpful")]


def test_feedback_harvest_creates_non_replaying_defect_candidates():
    from aegis.feedback_triage import harvest_defect_candidates

    candidates = harvest_defect_candidates(
        [
            {
                "event_id": "event-1",
                "objective_id": "objective-1",
                "correlation_id": "correlation-1",
                "created_at": "now",
                "outcome": "not_helpful",
                "reason": "objective_failed",
                "result_state": "completed",
                "retryable": False,
            },
            {"outcome": "helpful", "reason": None},
        ]
    )

    assert candidates[0]["classification"] == "objective_failure"
    assert candidates[0]["correlation_id"] == "correlation-1"
    assert candidates[0]["reproduction_required"] is True
    assert candidates[0]["replay_consequential_action"] is False
    assert candidates[0]["privacy_classification"] == "owner_scoped_metadata_only"
    assert candidates[0]["regression_eligibility"] == "human_review_required"
    assert candidates[0]["duplicate_signature"]


def test_feedback_harvest_deduplicates_same_owner_correlation_and_failure():
    from aegis.feedback_triage import harvest_defect_candidates

    feedback = [
        {
            "event_id": "event-new",
            "correlation_id": "same-correlation",
            "outcome": "not_helpful",
            "reason": "incorrect",
            "result_state": "completed",
        },
        {
            "event_id": "event-old",
            "correlation_id": "same-correlation",
            "outcome": "not_helpful",
            "reason": "incorrect",
            "result_state": "completed",
        },
    ]

    candidates = harvest_defect_candidates(feedback)

    assert len(candidates) == 1
    assert candidates[0]["event_id"] == "event-new"


def test_reference_completion_grounding_blocks_unrelated_canonical_target():
    from aegis.contracts import ActionCard, ActionSpec, VerificationContract
    from aegis.personal import PersonalState
    from aegis.reference_interaction import ground_reference_action
    from aegis.tasks import Task

    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    intent = IntentFrame(principal=principal, utterance="Finish the backup")
    card = ActionCard(
        action=ActionSpec(
            action_id="tasks.complete",
            capability="tasks.complete",
            arguments={"title": "restore drill"},
            verification=VerificationContract(kind="readback"),
        ),
        summary="Complete a task",
        relevance=1,
        argument_keys=("title",),
    )

    class Tasks:
        def list(self, _principal):
            return (Task(uuid4(), "apartment", "restore drill", "alice"),)

    blocked = ground_reference_action(
        intent,
        card,
        Tasks(),
        object(),
        PersonalState(),
        None,
        None,
        None,
        None,
    )
    assert isinstance(blocked, Result)
    assert blocked.state is ObjectiveState.BLOCKED

    grounded = ground_reference_action(
        intent.model_copy(update={"utterance": "Finish the restore drill"}),
        card,
        Tasks(),
        object(),
        PersonalState(),
        None,
        None,
        None,
        None,
    )
    assert isinstance(grounded, ActionCard)


def test_cli_feedback_without_database_reports_actionable_error(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.delenv("AEGIS_DATABASE_URL", raising=False)
    monkeypatch.setattr("sys.argv", ["aegis", "--feedback", "--harvest", "--json"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )

    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "code": "feedback_unavailable",
        "error": "feedback unavailable",
        "state": "failed",
    }


def test_browser_app_rejects_undocumented_feedback_without_recording():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    called = False

    def record(*_args):
        nonlocal called
        called = True

    app = BrowserApp(
        principal,
        lambda *_args: "unused",
        lambda _: {"nodes": []},
        feedback=record,
    )
    status, _, payload = app.dispatch(
        "POST",
        "/api/feedback",
        json.dumps(
            {"correlation_id": str(uuid4()), "outcome": "helpful", "raw": "secret"}
        ).encode(),
    )
    assert status == 400
    assert json.loads(payload) == {"code": "invalid_request", "error": "invalid request"}
    assert called is False


def test_browser_app_rejects_empty_messages_and_unknown_routes():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(principal, lambda *_: "unused", lambda _: {"nodes": []})

    status, _, payload = app.dispatch("POST", "/api/message", b'{"utterance":" "}')
    assert status == 400
    assert "non-empty" in json.loads(payload)["error"]

    status, _, _ = app.dispatch("GET", "/missing")
    assert status == 404


def test_browser_app_rejects_invalid_utf8_without_decoder_details():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(principal, lambda *_: "unreachable", lambda _: {"nodes": []})

    status, _, payload = app.dispatch("POST", "/api/message", b"\xff")

    assert status == 400
    assert json.loads(payload) == {
        "code": "invalid_request",
        "error": "invalid request",
    }


def test_browser_app_fails_closed_when_state_is_not_authorized():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(principal, lambda *_: "unused", lambda _: _deny())

    status, _, payload = app.dispatch("GET", "/api/constellation")
    assert status == 403
    assert json.loads(payload) == {
        "code": "state_access_denied",
        "error": "state access denied",
    }


def test_browser_app_rejects_malformed_constellation_projection():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(principal, lambda *_: "unused", lambda _: {"nodes": [{"id": "missing label"}]})

    status, _, payload = app.dispatch("GET", "/api/constellation")

    assert status == 503
    assert json.loads(payload) == {"code": "state_unavailable", "error": "state unavailable"}


@pytest.mark.parametrize(
    "projection",
    [
        {
            "nodes": [
                {"id": "tasks", "label": "Tasks"},
                {"id": "tasks", "label": "Duplicate"},
            ]
        },
        {
            "nodes": [{"id": "tasks", "label": "Tasks"}],
            "edges": [{"source": "aegis", "target": "tasks"}],
        },
    ],
)
def test_browser_rejects_ambiguous_constellation_relationships(projection):
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(principal, lambda *_: "unused", lambda _: projection)

    status, _, payload = app.dispatch("GET", "/api/constellation")

    assert status == 503
    assert json.loads(payload) == {"code": "state_unavailable", "error": "state unavailable"}


def test_browser_api_resolves_identity_for_each_request():
    principals = iter(
        (
            Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
            Principal(id="bob", vault_id="bob-vault", space_ids=("apartment",)),
        )
    )
    seen: list[str] = []
    app = BrowserApp(
        lambda: next(principals),
        lambda _utterance, current, _correlation_id: seen.append(current.id) or "answer",
        lambda current: {"nodes": [{"id": current.id, "label": current.id}]},
    )

    app.dispatch("GET", "/api/constellation")
    app.dispatch("POST", "/api/message", b'{"utterance":"show tasks"}')

    assert seen == ["bob"]


def test_browser_api_rejects_unavailable_identity():
    app = BrowserApp(
        lambda: (_ for _ in ()).throw(ValueError("malformed claims")),
        lambda *_: "unreachable",
        lambda _: {"nodes": []},
    )

    status, _, payload = app.dispatch("GET", "/api/constellation")
    assert status == 401
    assert json.loads(payload) == {
        "code": "identity_unavailable",
        "error": "identity unavailable",
    }


def test_browser_reuses_correlation_id_for_retry_safe_delivery():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    seen: list[str] = []
    app = BrowserApp(
        principal,
        lambda _utterance, _principal, correlation_id: (
            seen.append(str(correlation_id)) or {"message": "completed", "state": "completed"}
        ),
        lambda _: {"nodes": []},
    )
    body = (
        b'{"utterance":"Add rice to groceries.",'
        b'"correlation_id":"00000000-0000-4000-8000-000000000007"}'
    )

    first_status, _, first_payload = app.dispatch("POST", "/api/message", body)
    second_status, _, second_payload = app.dispatch("POST", "/api/message", body)

    assert first_status == second_status == 200
    assert seen == [
        "00000000-0000-4000-8000-000000000007",
        "00000000-0000-4000-8000-000000000007",
    ]
    assert (
        json.loads(first_payload)["correlation_id"] == json.loads(second_payload)["correlation_id"]
    )


def test_browser_exposes_core_retryability_without_inventing_it():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: {
            "message": "Model unavailable; request can be retried",
            "state": "failed",
            "retryable": True,
        },
        lambda _: {"nodes": []},
    )

    status, _, payload = app.dispatch("POST", "/api/message", b'{"utterance":"show tasks"}')

    assert status == 200
    assert json.loads(payload)["retryable"] is True


def test_browser_request_status_is_a_bounded_canonical_projection():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    correlation_id = "00000000-0000-4000-8000-000000000008"
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        request_status=lambda _principal, _correlation: {
            "correlation_id": correlation_id,
            "objective_id": "00000000-0000-4000-8000-000000000009",
            "state": "completed",
            "message": "canonical readback verified",
        },
    )

    status, _, payload = app.dispatch("GET", f"/api/request-status?correlation_id={correlation_id}")

    assert status == 200
    assert json.loads(payload) == {
        "correlation_id": correlation_id,
        "objective_id": "00000000-0000-4000-8000-000000000009",
        "state": "completed",
        "message": "canonical readback verified",
    }


def test_browser_request_status_rejects_bad_query_and_contains_provider_failure():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        request_status=lambda *_: (_ for _ in ()).throw(Exception("private state")),
    )

    status, _, payload = app.dispatch("GET", "/api/request-status?correlation_id=bad")
    assert status == 400
    assert json.loads(payload) == {"code": "invalid_request", "error": "invalid request"}

    status, _, payload = app.dispatch(
        "GET",
        "/api/request-status?correlation_id=00000000-0000-4000-8000-000000000008",
    )
    assert status == 503
    assert json.loads(payload) == {"code": "state_unavailable", "error": "state unavailable"}


@pytest.mark.parametrize(
    "request_status",
    [
        lambda *_: (_ for _ in ()).throw(ValueError("provider implementation detail")),
        lambda *_: {"correlation_id": "00000000-0000-4000-8000-000000000008", "state": "bogus"},
    ],
)
def test_browser_request_status_contains_provider_validation_failures(request_status):
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        request_status=request_status,
    )

    status, _, payload = app.dispatch(
        "GET",
        "/api/request-status?correlation_id=00000000-0000-4000-8000-000000000008",
    )

    assert status == 503
    assert json.loads(payload) == {"code": "state_unavailable", "error": "state unavailable"}


def test_browser_rejects_malformed_correlation_id_before_core():
    called = False

    def interaction(*_args):
        nonlocal called
        called = True
        return "unreachable"

    app = BrowserApp(
        Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
        interaction,
        lambda _: {"nodes": []},
    )
    status, _, payload = app.dispatch(
        "POST", "/api/message", b'{"utterance":"show tasks","correlation_id":"bad"}'
    )

    assert status == 400
    assert json.loads(payload) == {"code": "invalid_request", "error": "invalid request"}
    assert called is False


def test_browser_rejects_malformed_json_with_generic_error_before_core():
    called = False

    def interaction(*_args):
        nonlocal called
        called = True
        return "unreachable"

    app = BrowserApp(
        Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
        interaction,
        lambda _: {"nodes": []},
    )
    status, _, payload = app.dispatch("POST", "/api/message", b'{"utterance":')

    assert status == 400
    assert json.loads(payload) == {"code": "invalid_request", "error": "invalid request"}
    assert called is False


def test_browser_rejects_undocumented_request_fields_before_core():
    called = False

    def interaction(*_args):
        nonlocal called
        called = True
        return "unreachable"

    app = BrowserApp(
        Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
        interaction,
        lambda _: {"nodes": []},
    )
    status, _, payload = app.dispatch(
        "POST", "/api/message", b'{"utterance":"show tasks","private_debug":true}'
    )

    assert status == 400
    assert json.loads(payload) == {
        "code": "invalid_request",
        "error": "request contains undocumented fields",
    }
    assert called is False


def test_browser_rejects_undocumented_interaction_fields():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: {"message": "answer", "private_debug": "secret"},
        lambda _: {"nodes": []},
    )

    status, _, payload = app.dispatch("POST", "/api/message", b'{"utterance":"show tasks"}')

    assert status == 503
    assert json.loads(payload) == {
        "code": "request_unavailable",
        "error": "request unavailable",
    }


def test_browser_rejects_unknown_interaction_lifecycle_state():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: {"message": "answer", "state": "invented_state"},
        lambda _: {"nodes": []},
    )

    status, _, payload = app.dispatch("POST", "/api/message", b'{"utterance":"show tasks"}')

    assert status == 503
    assert json.loads(payload) == {
        "code": "request_unavailable",
        "error": "request unavailable",
    }


def test_browser_health_uses_structured_readiness_without_identity():
    from aegis.health import ComponentHealth, HealthReport

    app = BrowserApp(
        lambda: (_ for _ in ()).throw(ValueError("identity unavailable")),
        lambda *_: "unreachable",
        lambda _: {"nodes": []},
        lambda: HealthReport(
            healthy=False,
            ready=False,
            components=(
                ComponentHealth(
                    name="postgres", healthy=False, required=True, detail="unavailable"
                ),
            ),
        ),
    )

    status, _, payload = app.dispatch("GET", "/api/health")
    assert status == 200
    assert json.loads(payload)["ready"] is False

    status, _, payload = app.dispatch("GET", "/api/ready")
    assert status == 503
    assert json.loads(payload)["ready"] is False


def test_browser_contains_unexpected_identity_provider_failure():
    app = BrowserApp(
        lambda: (_ for _ in ()).throw(Exception("identity database password")),
        lambda *_: "unreachable",
        lambda _: {"nodes": []},
    )

    status, _, payload = app.dispatch("GET", "/api/constellation")

    assert status == 401
    assert json.loads(payload) == {
        "code": "identity_unavailable",
        "error": "identity unavailable",
    }


def test_browser_ready_is_healthy_for_ready_report_without_identity():
    from aegis.health import HealthReport

    app = BrowserApp(
        lambda: (_ for _ in ()).throw(ValueError("identity unavailable")),
        lambda *_: "unreachable",
        lambda _: {"nodes": []},
        lambda: HealthReport(healthy=True, ready=True, components=()),
    )

    status, _, payload = app.dispatch("GET", "/api/ready")
    assert status == 200
    assert json.loads(payload)["ready"] is True


def test_browser_health_provider_failure_is_generic_and_recoverable():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: "unreachable",
        lambda _: {"nodes": []},
        lambda: (_ for _ in ()).throw(RuntimeError("database password")),
    )

    status, _, payload = app.dispatch("GET", "/api/health")

    assert status == 503
    assert json.loads(payload) == {
        "code": "health_unavailable",
        "error": "runtime status unavailable",
    }


def test_browser_rejects_malformed_health_payload_without_server_exception():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: "unreachable",
        lambda _: {"nodes": []},
        lambda: ["not a health report"],
    )

    status, _, payload = app.dispatch("GET", "/api/health")

    assert status == 503
    assert json.loads(payload) == {
        "code": "health_unavailable",
        "error": "runtime status unavailable",
    }


def test_browser_contains_unexpected_service_exceptions_at_the_boundary():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))

    app = BrowserApp(
        principal,
        lambda *_: (_ for _ in ()).throw(Exception("database password")),
        lambda _: (_ for _ in ()).throw(Exception("private row")),
        lambda: (_ for _ in ()).throw(Exception("provider secret")),
    )

    status, _, payload = app.dispatch("GET", "/api/health")
    assert status == 503
    assert json.loads(payload) == {
        "code": "health_unavailable",
        "error": "runtime status unavailable",
    }

    status, _, payload = app.dispatch("GET", "/api/constellation")
    assert status == 503
    assert json.loads(payload) == {"code": "state_unavailable", "error": "state unavailable"}

    status, _, payload = app.dispatch("POST", "/api/message", b'{"utterance":"show tasks"}')
    assert status == 503
    assert json.loads(payload) == {
        "code": "request_unavailable",
        "error": "request unavailable",
    }


def test_browser_does_not_treat_core_value_error_as_client_validation():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: (_ for _ in ()).throw(ValueError("database password=private-secret")),
        lambda _: {"nodes": []},
    )

    status, _, payload = app.dispatch("POST", "/api/message", b'{"utterance":"show tasks"}')

    assert status == 503
    assert json.loads(payload) == {
        "code": "request_unavailable",
        "error": "request unavailable",
    }
    assert b"private-secret" not in payload


def test_browser_contains_non_json_provider_payloads():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": [], "details": {"bad": object()}},
    )

    status, _, payload = app.dispatch("GET", "/api/constellation")

    assert status == 503
    assert json.loads(payload) == {
        "code": "state_unavailable",
        "error": "state unavailable",
    }


def test_browser_bounds_serialized_response_size():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": [], "details": {"oversized": "x" * 1_000_001}},
    )

    status, _, payload = app.dispatch("GET", "/api/constellation")

    assert status == 503
    assert json.loads(payload) == {
        "code": "response_unavailable",
        "error": "response unavailable",
    }


def test_browser_rejects_oversized_request_body():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(principal, lambda *_: "unreachable", lambda _: {"nodes": []})

    status, _, payload = app.dispatch("POST", "/api/message", b"x" * 20_001)
    assert status == 413
    assert json.loads(payload) == {
        "code": "request_too_large",
        "error": "request too large",
    }


def test_browser_surface_has_transcript_and_duplicate_submission_guard():
    from aegis.web import _INDEX_HTML

    assert 'id="theme-toggle"' in _INDEX_HTML
    assert "themeStorageKey = 'aegis.theme'" in _INDEX_HTML
    assert "color-scheme:dark" in _INDEX_HTML
    assert "let initialTheme = 'dark';" in _INDEX_HTML
    assert 'class="conversation-panel"' in _INDEX_HTML
    assert '<nav class="product-nav" aria-label="AEGIS views">' in _INDEX_HTML
    for view in (
        "Today",
        "Tasks",
        "Calendar",
        "Household",
        "Systems",
        "Research",
        "Packs",
        "Documents",
        "Daily driver",
    ):
        assert f">{view}</button>" in _INDEX_HTML
    assert "Research is available through conversation" in _INDEX_HTML
    assert "async function loadDocuments()" in _INDEX_HTML
    assert "Read document" in _INDEX_HTML
    assert "async function loadDailyDriver()" in _INDEX_HTML
    assert "/api/daily-driver" in _INDEX_HTML
    assert 'class="intro"' in _INDEX_HTML
    assert 'id="status-badge"' in _INDEX_HTML
    assert "setOutcomeStatus(result.state)" in _INDEX_HTML
    assert '<details class="secondary" aria-label="Canonical state">' in _INDEX_HTML
    assert "<summary>Canonical state</summary>" in _INDEX_HTML
    assert 'placeholder="Talk to AEGIS…"' in _INDEX_HTML
    assert '<textarea id="utterance"' in _INDEX_HTML
    assert "Enter to send · Shift+Enter for a new line" in _INDEX_HTML
    assert 'id="conversation"' in _INDEX_HTML
    assert 'id="activity"' in _INDEX_HTML
    assert 'id="health-details"' in _INDEX_HTML
    assert 'id="node-filter"' in _INDEX_HTML
    assert 'id="node-filter-status"' in _INDEX_HTML
    assert 'aria-describedby="node-filter-status"' in _INDEX_HTML
    assert (
        'id="node-filter-status" class="muted" aria-live="polite" aria-atomic="true"' in _INDEX_HTML
    )
    assert "component.detail" in _INDEX_HTML
    assert "send.disabled = true" in _INDEX_HTML
    assert "input.disabled = true" in _INDEX_HTML
    assert "form.setAttribute('aria-busy', 'true')" in _INDEX_HTML
    assert "form.setAttribute('aria-busy', 'false')" in _INDEX_HTML
    assert "nodes.setAttribute('aria-busy', 'true')" in _INDEX_HTML
    assert "nodes.setAttribute('aria-busy', 'false')" in _INDEX_HTML
    assert "appendConversationMessage('aegis-message'" in _INDEX_HTML
    assert "appendConversationMessage('owner-message'" in _INDEX_HTML
    assert "requestSubmit()" in _INDEX_HTML
    assert "scrollIntoView({block: 'nearest', behavior: 'smooth'})" in _INDEX_HTML
    assert 'role="log"' in _INDEX_HTML
    assert "renderDetailValue(details[node.id])" in _INDEX_HTML
    assert "aria-pressed" in _INDEX_HTML
    assert 'role="region"' in _INDEX_HTML
    assert 'aria-label="Selected node details"' in _INDEX_HTML
    assert "card.setAttribute('aria-label'" in _INDEX_HTML
    assert "selectedNode" in _INDEX_HTML
    assert "const nodeCards = new Map()" in _INDEX_HTML
    assert "const selectNode = (node, card)" in _INDEX_HTML
    assert "navigableViews.has(node.detail_view)" in _INDEX_HTML
    assert "Open relationship to" in _INDEX_HTML
    assert "target.focus(); target.click()" in _INDEX_HTML
    assert "function applyNodeFilter()" in _INDEX_HTML
    assert "authorizedProjectionLoaded" in _INDEX_HTML
    assert "Authorized nodes unavailable." in _INDEX_HTML
    assert "No authorized nodes match" in _INDEX_HTML
    assert "renderedEdgeRows" in _INDEX_HTML
    assert (
        "selectedNode = null;\n  document.getElementById('detail').replaceChildren();"
        in _INDEX_HTML
    )
    assert "No canonical records available" in _INDEX_HTML
    assert "Show ${value.length} canonical records" in _INDEX_HTML
    assert "key.replaceAll('_', ' ')" in _INDEX_HTML
    assert "retryableCodes.has(result.code)" in _INDEX_HTML
    assert "const lifecycleLabels = Object.freeze" in _INDEX_HTML
    assert "const errorLabels = Object.freeze" in _INDEX_HTML
    assert "function errorLabel(code)" in _INDEX_HTML
    assert "function clearHealthDetails()" in _INDEX_HTML
    assert "clearHealthDetails();" in _INDEX_HTML
    assert "function lifecycleLabel(state)" in _INDEX_HTML
    assert "lifecycleLabel(result.state)" in _INDEX_HTML
    assert "lifecycleLabel(status.state)" in _INDEX_HTML
    assert "Request status recovered" in _INDEX_HTML
    assert "new AbortController()" in _INDEX_HTML
    assert "request_timeout" in _INDEX_HTML
    assert "refreshRequestTimeoutMs = 10000" in _INDEX_HTML
    assert "async function fetchWithTimeout(resource, options = {})" in _INDEX_HTML
    assert "setTimeout(() => controller.abort(), refreshRequestTimeoutMs)" in _INDEX_HTML
    assert "fetchWithTimeout('/api/health')" in _INDEX_HTML
    assert "fetchWithTimeout('/api/constellation')" in _INDEX_HTML
    assert "outcome is unknown" in _INDEX_HTML
    assert 'id="refresh"' in _INDEX_HTML
    assert "refreshState()" in _INDEX_HTML
    assert "state_access_denied" in _INDEX_HTML
    assert "State refresh failed" in _INDEX_HTML
    assert "${errorLabel(code)} (${code})." in _INDEX_HTML
    assert "await loadHealth();" in _INDEX_HTML
    assert "await loadState();" in _INDEX_HTML
    assert "response.ok" in _INDEX_HTML
    assert "apiFetch('/api/message'" in _INDEX_HTML
    assert "apiFetch('/api/feedback'" in _INDEX_HTML
    assert "if (response.status !== 401 || resource === '/') return response;" in _INDEX_HTML
    assert 'const refreshed = html.match(/<meta name="aegis-session-token"' in _INDEX_HTML
    assert "return fetch(resource, {...options, headers: retryHeaders});" in _INDEX_HTML
    assert "clearAuthorizedDisplays()" in _INDEX_HTML
    assert "document.getElementById('step-status').textContent = '';" in _INDEX_HTML
    assert "document.querySelector('#chat button').textContent = 'Send';" in _INDEX_HTML
    assert "Authorization lost; authorized state cleared." in _INDEX_HTML
    assert "conversation').replaceChildren()" in _INDEX_HTML
    assert "sessionStorage" in _INDEX_HTML
    assert "aegis.session-id" in _INDEX_HTML
    assert "session_id:conversationSessionId" in _INDEX_HTML
    assert "A previous request may still be in progress" in _INDEX_HTML
    assert "persistPendingRequest(utterance, correlationId)" in _INDEX_HTML
    assert "/api/request-status?correlation_id=" in _INDEX_HTML
    assert "recoverPendingRequest();" in _INDEX_HTML
    assert "maxRecoveryPolls = 60" in _INDEX_HTML
    assert "recoveryRequestTimeoutMs = 10000" in _INDEX_HTML
    assert "setTimeout(() => controller.abort(), recoveryRequestTimeoutMs)" in _INDEX_HTML
    assert "signal: controller.signal" in _INDEX_HTML
    assert "Status checks paused after five minutes." in _INDEX_HTML
    assert "Outcome unknown; checking canonical status. Retry remains explicit." in _INDEX_HTML
    assert "Status check unavailable; retry remains explicit." in _INDEX_HTML
    assert "inProgressStates.has(status.state)" in _INDEX_HTML
    assert "Retry remains explicit." in _INDEX_HTML
    assert "scheduleRecoveryPoll();" in _INDEX_HTML
    assert "recoveryPollMs = 5000" in _INDEX_HTML
    assert "if (result.state === 'completed') refreshState();" in _INDEX_HTML
    assert "if (status.state === 'completed') refreshState();" in _INDEX_HTML
    assert "loadState().catch(() => {})" not in _INDEX_HTML
    assert "Status: ${errorLabel(result.code)}" in _INDEX_HTML
    assert "result.retryable === true" in _INDEX_HTML
    assert "status.retryable === true" in _INDEX_HTML


def test_browser_transport_disables_caching_and_referrer_disclosure():
    from pathlib import Path

    source = Path(__file__).parents[1].joinpath("src/aegis/web.py").read_text()

    assert 'self.send_header("Cache-Control", "no-store")' in source
    assert 'self.send_header("Pragma", "no-cache")' in source
    assert 'self.send_header("Referrer-Policy", "no-referrer")' in source


def test_browser_transport_ignores_expected_client_disconnect():
    from aegis.web import _write_response_payload

    class DisconnectedWriter:
        def write(self, _payload):
            raise BrokenPipeError

    _write_response_payload(DisconnectedWriter(), b"response")


def test_browser_transport_restricts_embedding_and_resource_execution():
    from pathlib import Path

    source = Path(__file__).parents[1].joinpath("src/aegis/web.py").read_text()

    assert 'self.send_header("X-Frame-Options", "DENY")' in source
    assert "Content-Security-Policy" in source
    assert "default-src 'none'" in source
    assert "connect-src 'self'" in source
    assert "frame-ancestors 'none'" in source
    assert 'self.send_header("Retry-After", str(_RETRY_AFTER_SECONDS))' in source


def test_browser_server_closes_cleanly_on_keyboard_interrupt(monkeypatch):
    from aegis import web

    instances = []

    class FakeServer:
        def __init__(self, _address, _handler):
            self.closed = False
            instances.append(self)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            self.closed = True

    monkeypatch.setattr(web, "ThreadingHTTPServer", FakeServer)

    web.serve("127.0.0.1", 18099, lambda: None, lambda *_: "unused", lambda _: {})

    assert instances[0].closed is True


def test_task_read_fast_path_returns_membership_checked_canonical_tasks():
    task = Task(uuid4(), "apartment", "replace filter", "alice", status=TaskStatus.OPEN)

    class Store:
        def list(self, principal):
            assert principal.id == "alice"
            return (task,)

    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    intent = IntentFrame(
        principal=principal,
        utterance="Show my tasks",
        correlation_id=uuid4(),
    )

    result = TaskReadFastPath(Store()).resolve(intent)

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["canonical_tasks"] == [
        {"task_id": str(task.task_id), "title": "replace filter", "status": "open"}
    ]
    assert not TaskReadFastPath.matches("Create a task to replace filter")
    assert not TaskReadFastPath.matches(
        "Please create a task to compare the inspection checklist with the backup runbook"
    )
    assert not TaskReadFastPath.matches(
        "Could you put a task on my list to keep an eye on the backup?"
    )
    assert not TaskReadFastPath.matches("I'd like to put a task on my list to verify the drill")
    assert not TaskReadFastPath.matches("Mark the task Verify backup retention as done")
    assert not TaskReadFastPath.matches("Which task should I do first?")
    assert TaskReadFastPath.matches("Can you tell me what tasks are open")


def test_task_read_fast_path_resolves_explicit_ordinal_from_canonical_order():
    first = Task(uuid4(), "apartment", "replace filter", "alice", status=TaskStatus.OPEN)
    second = Task(uuid4(), "apartment", "inspect latch", "alice", status=TaskStatus.COMPLETED)

    class Store:
        def list(self, _principal):
            return (first, second)

    result = TaskReadFastPath(Store()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
            utterance="Show me the second task.",
        )
    )

    assert result is not None
    assert result.message == "Task: inspect latch (completed)"
    assert result.evidence["authorized_ordinal_referent"]["task_id"] == str(second.task_id)
    assert result.evidence["task"]["task_id"] == str(second.task_id)
    assert [item["title"] for item in result.evidence["canonical_tasks"]] == [
        "replace filter",
        "inspect latch",
    ]


def test_task_priority_fast_path_uses_earliest_open_deadline():
    soon = Task(
        uuid4(),
        "apartment",
        "renew insurance",
        "alice",
        due_at=datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc),
    )
    later = Task(
        uuid4(),
        "apartment",
        "schedule inspection",
        "alice",
        due_at=datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc),
    )

    class Store:
        def list(self, _principal):
            return (later, soon)

    result = TaskPriorityFastPath(Store()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="which task should i do first",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["priority_basis"] == "earliest_due_at"
    assert result.evidence["task"]["title"] == "renew insurance"
    assert {task["title"] for task in result.evidence["canonical_tasks"]} == {
        "renew insurance",
        "schedule inspection",
    }


def test_task_priority_fast_path_uses_latest_open_deadline():
    soon = Task(
        uuid4(),
        "apartment",
        "renew insurance",
        "alice",
        due_at=datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc),
    )
    later = Task(
        uuid4(),
        "apartment",
        "review the porch lights",
        "alice",
        due_at=datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc),
    )

    class Store:
        def list(self, _principal):
            return (soon, later)

    result = TaskPriorityFastPath(Store()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which task is due latest?",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["priority_basis"] == "latest_due_at"
    assert result.evidence["task"]["title"] == "review the porch lights"


def test_task_priority_fast_path_accepts_implicit_daily_priority_language():
    soon = Task(
        uuid4(),
        "apartment",
        "call the dentist",
        "alice",
        due_at=datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc),
    )

    class Store:
        def list(self, _principal):
            return (soon,)

    for utterance in (
        "what should I take care of first today?",
        "which task should I focus on first?",
        "what should I prioritize today?",
        "what should I work on next?",
        "what is next on my list?",
        "what is the next thing on my list?",
        "what needs my attention first?",
        "what is the most urgent task?",
        "which one is due first?",
    ):
        assert TaskPriorityFastPath.matches(utterance)
        result = TaskPriorityFastPath(Store()).resolve(
            IntentFrame(
                principal=Principal(id="alice", vault_id="alice-vault"),
                utterance=utterance,
            )
        )
        assert result is not None
        assert result.state is ObjectiveState.COMPLETED
        assert result.evidence["task"]["title"] == "call the dentist"

    assert not TaskPriorityFastPath.matches("which groceries should I buy first?")


def test_task_collection_request_keeps_an_ordered_referent_for_follow_up():
    utterance = "Show me the open tasks I should focus on today"
    assert TaskReadFastPath.matches(utterance)
    assert not TaskPriorityFastPath.matches(utterance)


def test_task_priority_fast_path_filters_temporal_should_do_request():
    now = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    tomorrow = Task(
        uuid4(),
        "apartment",
        "call the dentist",
        "alice",
        due_at=now + timedelta(days=1),
    )
    later = Task(
        uuid4(),
        "apartment",
        "renew insurance",
        "alice",
        due_at=now + timedelta(days=2),
    )

    class Store:
        def list(self, _principal):
            return (tomorrow, later)

    result = TaskPriorityFastPath(Store()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="what should I do tomorrow?",
        ),
        now=now,
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["priority_basis"] == "earliest_due_at_on_tomorrow"
    assert result.evidence["task"]["title"] == "call the dentist"


def test_task_priority_fast_path_uses_owner_local_date_at_utc_boundary():
    from zoneinfo import ZoneInfo

    owner_zone = ZoneInfo("America/Chicago")
    now = datetime(2026, 9, 5, 12, 0, tzinfo=owner_zone)
    tomorrow_evening = datetime(2026, 9, 6, 19, 11, tzinfo=owner_zone)
    later = Task(
        uuid4(),
        "apartment",
        "check the patio latch",
        "alice",
        due_at=tomorrow_evening.astimezone(timezone.utc),
    )

    class Store:
        def list(self, _principal):
            return (later,)

    result = TaskPriorityFastPath(Store()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Which task is due first tomorrow?",
        ),
        now=now,
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["priority_basis"] == "earliest_due_at_on_tomorrow"
    assert result.evidence["task"]["title"] == "check the patio latch"


def test_task_read_fast_path_exposes_canonical_due_at():
    from datetime import datetime, timezone

    due_at = datetime(2026, 9, 8, 22, 0, tzinfo=timezone.utc)
    task = Task(uuid4(), "apartment", "review restore drill", "alice", due_at=due_at)

    class Store:
        def list(self, _principal):
            return (task,)

    result = TaskReadFastPath(Store()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
            utterance="Show my tasks",
        )
    )

    assert result is not None
    assert result.evidence["canonical_tasks"][0]["due_at"] == due_at.isoformat()


def test_task_read_fast_path_filters_relative_due_window():
    from datetime import datetime, timedelta

    now = datetime.now().astimezone()
    tomorrow = Task(uuid4(), "apartment", "tomorrow task", "alice", due_at=now + timedelta(days=1))
    next_month = Task(uuid4(), "apartment", "later task", "alice", due_at=now + timedelta(days=30))

    class Store:
        def list(self, _principal):
            return (tomorrow, next_month)

    result = TaskReadFastPath(Store()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
            utterance="Show my tasks due tomorrow",
        )
    )

    assert result is not None
    assert result.evidence["due_filter"] == "tomorrow"
    assert [item["title"] for item in result.evidence["canonical_tasks"]] == ["tomorrow task"]


def test_task_read_fast_path_accepts_informal_temporal_work_terms():
    from datetime import datetime, timedelta

    now = datetime.now().astimezone()
    tomorrow = Task(uuid4(), "apartment", "tomorrow task", "alice", due_at=now + timedelta(days=1))
    later = Task(uuid4(), "apartment", "later task", "alice", due_at=now + timedelta(days=3))

    class Store:
        def list(self, _principal):
            return (tomorrow, later)

    for utterance in (
        "What do I need to do tomorrow?",
        "What do I need to knock out tomorrow?",
        "What's on my plate tomorrow?",
        "What am I supposed to do tomorrow?",
        "What needs doing tomorrow?",
        "What do I have to do tomorrow?",
    ):
        result = TaskReadFastPath(Store()).resolve(
            IntentFrame(
                principal=Principal(id="alice", vault_id="alice-vault"),
                utterance=utterance,
            )
        )
        assert result is not None
        assert result.evidence["due_filter"] == "tomorrow"
        assert [item["title"] for item in result.evidence["canonical_tasks"]] == ["tomorrow task"]


def test_task_read_fast_path_accepts_natural_open_task_remainder_wording():
    class Store:
        def list(self, _principal):
            return (
                Task(uuid4(), "apartment", "open task", "alice"),
                Task(uuid4(), "apartment", "finished task", "alice", status=TaskStatus.COMPLETED),
            )

    result = TaskReadFastPath(Store()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What else do I need to get done?",
        )
    )

    assert result is not None
    assert result.evidence["status_filter"] == "open"
    assert result.evidence["due_filter"] == "all"
    assert [item["title"] for item in result.evidence["canonical_tasks"]] == ["open task"]


def test_task_read_fast_path_orders_temporal_results_by_deadline():
    from datetime import datetime, timedelta

    local_now = datetime.now().astimezone()
    tomorrow = (local_now + timedelta(days=1)).date()

    later = Task(
        uuid4(),
        "apartment",
        "later tomorrow task",
        "alice",
        due_at=datetime.combine(tomorrow, datetime.min.time().replace(hour=18), local_now.tzinfo),
    )
    earlier = Task(
        uuid4(),
        "apartment",
        "earlier tomorrow task",
        "alice",
        due_at=datetime.combine(tomorrow, datetime.min.time().replace(hour=8), local_now.tzinfo),
    )

    class Store:
        def list(self, _principal):
            return (later, earlier)

    result = TaskReadFastPath(Store()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Show my tasks due tomorrow",
        )
    )

    assert result is not None
    assert [item["title"] for item in result.evidence["canonical_tasks"]] == [
        "earlier tomorrow task",
        "later tomorrow task",
    ]


def test_task_read_fast_path_uses_owner_local_date_for_relative_due_window():
    from datetime import datetime, timedelta, timezone

    local_now = datetime.now().astimezone()
    local_tomorrow = (local_now + timedelta(days=1)).date()
    local_midnight = datetime.combine(local_tomorrow, datetime.min.time(), local_now.tzinfo)
    tomorrow = Task(
        uuid4(),
        "apartment",
        "local tomorrow task",
        "alice",
        due_at=local_midnight.astimezone(timezone.utc),
    )

    class Store:
        def list(self, _principal):
            return (tomorrow,)

    result = TaskReadFastPath(Store()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What's due tomorrow?",
        )
    )

    assert result is not None
    assert result.evidence["due_filter"] == "tomorrow"
    assert [item["title"] for item in result.evidence["canonical_tasks"]] == ["local tomorrow task"]


def test_task_read_fast_path_accepts_implicit_temporal_task_language():
    from datetime import datetime, timedelta

    now = datetime.now().astimezone()
    tomorrow = Task(
        uuid4(),
        "apartment",
        "tomorrow task",
        "alice",
        due_at=now + timedelta(days=1),
    )
    completed_tomorrow = Task(
        uuid4(),
        "apartment",
        "completed tomorrow task",
        "alice",
        due_at=now + timedelta(days=1),
        status=TaskStatus.COMPLETED,
    )
    later = Task(
        uuid4(),
        "apartment",
        "later task",
        "alice",
        due_at=now + timedelta(days=30),
    )

    class Store:
        def list(self, _principal):
            return (tomorrow, completed_tomorrow, later)

    for utterance in ("what's due tomorrow?", "what do I need to get done tomorrow?"):
        assert TaskReadFastPath.matches(utterance)
        result = TaskReadFastPath(Store()).resolve(
            IntentFrame(
                principal=Principal(id="alice", vault_id="alice-vault"),
                utterance=utterance,
            )
        )
        assert result is not None
        assert result.evidence["due_filter"] == "tomorrow"
        assert result.evidence["status_filter"] == "open"
        assert [item["title"] for item in result.evidence["canonical_tasks"]] == ["tomorrow task"]

    result = TaskReadFastPath(Store()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="Show me the tasks I need to finish tomorrow",
        )
    )
    assert result is not None
    assert result.evidence["due_filter"] == "tomorrow"
    assert [item["title"] for item in result.evidence["canonical_tasks"]] == ["tomorrow task"]


def test_task_read_fast_path_matches_conversational_temporal_followup():
    assert TaskReadFastPath.matches("And what tasks are due tomorrow?")
    assert TaskReadFastPath.matches("But what do I need to get done Friday?")


def test_task_read_fast_path_filters_structural_get_done_today_request():
    from datetime import datetime, timedelta

    now = datetime.now().astimezone()
    today = Task(uuid4(), "apartment", "today task", "alice", due_at=now.replace(hour=12))
    tomorrow = Task(uuid4(), "apartment", "tomorrow task", "alice", due_at=now + timedelta(days=1))

    class Store:
        def list(self, _principal):
            return (today, tomorrow)

    utterance = "What else do I need to get done today?"
    assert TaskReadFastPath.matches(utterance)
    result = TaskReadFastPath(Store()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance=utterance,
        )
    )
    assert result is not None
    assert result.evidence["due_filter"] == "today"
    assert [item["title"] for item in result.evidence["canonical_tasks"]] == ["today task"]


def test_task_read_fast_path_filters_this_week_to_open_current_week():
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    this_week = Task(
        uuid4(), "apartment", "this week task", "alice", due_at=now + timedelta(hours=1)
    )
    next_week = Task(
        uuid4(), "apartment", "next week task", "alice", due_at=now + timedelta(days=8)
    )
    completed = Task(
        uuid4(),
        "apartment",
        "completed this week task",
        "alice",
        due_at=now + timedelta(hours=1),
        status=TaskStatus.COMPLETED,
    )

    class Store:
        def list(self, _principal):
            return (this_week, next_week, completed)

    utterance = "What do I need to get done this week?"
    assert TaskReadFastPath.matches(utterance)
    result = TaskReadFastPath(Store()).resolve(
        IntentFrame(principal=Principal(id="alice", vault_id="alice-vault"), utterance=utterance)
    )
    assert result is not None
    assert result.evidence["due_filter"] == "this_week"
    assert result.evidence["status_filter"] == "open"
    assert [item["title"] for item in result.evidence["canonical_tasks"]] == ["this week task"]


def test_task_read_fast_path_filters_weekday_without_due_verb():
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    target = (now + timedelta(days=(0 - now.weekday()) % 7)).date()
    monday_task = Task(
        uuid4(),
        "apartment",
        "monday task",
        "alice",
        due_at=datetime.combine(target, datetime.min.time(), now.astimezone().tzinfo),
    )
    later = Task(uuid4(), "apartment", "later task", "alice", due_at=now + timedelta(days=30))

    class Store:
        def list(self, _principal):
            return (monday_task, later)

    utterance = "Show the Monday tasks"
    assert TaskReadFastPath.matches(utterance)
    result = TaskReadFastPath(Store()).resolve(
        IntentFrame(principal=Principal(id="alice", vault_id="alice-vault"), utterance=utterance)
    )
    assert result is not None
    assert result.evidence["due_filter"] == "weekday:monday"
    assert [item["title"] for item in result.evidence["canonical_tasks"]] == ["monday task"]


def test_task_read_fast_path_next_same_weekday_means_next_occurrence():
    from datetime import datetime, timedelta

    now = datetime.now().astimezone()
    weekday = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")[
        now.weekday()
    ]
    this_week = Task(uuid4(), "apartment", "today's task", "alice", due_at=now)
    next_week = Task(
        uuid4(),
        "apartment",
        "next week's task",
        "alice",
        due_at=now + timedelta(days=7),
    )

    class Store:
        def list(self, _principal):
            return (this_week, next_week)

    utterance = f"Show tasks due next {weekday}"
    assert TaskReadFastPath.matches(utterance)
    result = TaskReadFastPath.resolve(
        TaskReadFastPath(Store()),
        IntentFrame(principal=Principal(id="alice", vault_id="alice-vault"), utterance=utterance),
    )
    assert result is not None
    assert result.evidence["due_filter"] == f"weekday:next:{weekday}"
    assert [item["title"] for item in result.evidence["canonical_tasks"]] == ["next week's task"]


def test_task_priority_fast_path_yields_to_due_weekday_reads():
    assert not TaskPriorityFastPath.matches("What tasks are due next Monday?")


def test_task_read_fast_path_filters_remaining_task_list_to_open_tasks():
    class Store:
        def list(self, _principal):
            return (
                Task(uuid4(), "apartment", "open task", "alice"),
                Task(uuid4(), "apartment", "finished task", "alice", status=TaskStatus.COMPLETED),
            )

    utterance = "What is left on my task list?"
    assert TaskReadFastPath.matches(utterance)
    result = TaskReadFastPath(Store()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance=utterance,
        )
    )
    assert result is not None
    assert result.evidence["status_filter"] == "open"
    assert [item["title"] for item in result.evidence["canonical_tasks"]] == ["open task"]


def test_context_reset_strips_dash_before_new_objective():
    from aegis.utterance import strip_context_reset

    assert strip_context_reset("Actually, never mind — what chores need attention?") == (
        "what chores need attention?"
    )


def test_context_reset_normalization_does_not_revoke_context_for_capitalization():
    from aegis.utterance import strip_context_reset

    utterance = "When is it?"
    assert strip_context_reset(utterance) == "when is it?"
    assert strip_context_reset(utterance) == " ".join(utterance.casefold().split())


def test_task_read_fast_path_filters_open_tasks_before_weekend():
    from datetime import datetime, timedelta

    now = datetime.now().astimezone()
    days_until_saturday = (5 - now.weekday()) % 7 or 7
    fixture_now = now.replace(hour=12, minute=0, second=0, microsecond=0)
    before_weekend = Task(
        # Keep the fixture on the current calendar date in every runner
        # timezone, including UTC just before midnight.
        uuid4(),
        "apartment",
        "before weekend task",
        "alice",
        due_at=fixture_now,
    )
    after_weekend = Task(
        uuid4(),
        "apartment",
        "after weekend task",
        "alice",
        due_at=fixture_now + timedelta(days=days_until_saturday + 1),
    )
    completed = Task(
        uuid4(),
        "apartment",
        "completed before weekend",
        "alice",
        due_at=fixture_now,
        status=TaskStatus.COMPLETED,
    )

    class Store:
        def list(self, _principal):
            return (before_weekend, after_weekend, completed)

    utterance = "What tasks should I worry about before the weekend?"
    assert TaskReadFastPath.matches(utterance)
    assert TaskReadFastPath.matches("What should I get done before the weekend?")
    result = TaskReadFastPath(Store()).resolve(
        IntentFrame(principal=Principal(id="alice", vault_id="alice-vault"), utterance=utterance)
    )
    assert result is not None
    assert result.evidence["status_filter"] == "open"
    assert result.evidence["due_filter"] == "before_weekend"
    assert [item["title"] for item in result.evidence["canonical_tasks"]] == ["before weekend task"]


def test_task_read_fast_path_filters_open_tasks_this_weekend():
    from datetime import timedelta

    alice = Principal(id="alice", vault_id="alice-vault")
    now = datetime.now().astimezone()
    saturday_offset = 5 - now.weekday() if now.weekday() < 5 else -(now.weekday() - 5)
    saturday = (now + timedelta(days=saturday_offset)).date()
    weekend_task = Task(
        uuid4(),
        "apartment",
        "weekend task",
        "alice",
        due_at=datetime.combine(saturday, datetime.min.time(), tzinfo=now.tzinfo),
    )
    later_task = Task(
        uuid4(),
        "apartment",
        "later task",
        "alice",
        due_at=datetime.combine(
            saturday + timedelta(days=2), datetime.min.time(), tzinfo=now.tzinfo
        ),
    )

    class Store:
        def list(self, _principal):
            return (later_task, weekend_task)

    result = TaskReadFastPath(Store()).resolve(
        IntentFrame(principal=alice, utterance="What tasks do I need to get done this weekend?")
    )

    assert result is not None
    assert result.evidence["due_filter"] == "this_weekend"
    assert [item["title"] for item in result.evidence["canonical_tasks"]] == ["weekend task"]


def test_task_read_fast_path_filters_this_weekday_due_window():
    from datetime import datetime, timedelta

    now = datetime.now().astimezone()
    target = (now + timedelta(days=(4 - now.weekday()) % 7)).date()
    friday_task = Task(
        uuid4(),
        "apartment",
        "friday task",
        "alice",
        due_at=datetime.combine(target, datetime.min.time(), now.astimezone().tzinfo),
    )
    later = Task(uuid4(), "apartment", "later task", "alice", due_at=now + timedelta(days=30))

    class Store:
        def list(self, _principal):
            return (friday_task, later)

    utterance = "what do I need to get done this Friday?"
    assert TaskReadFastPath.matches(utterance)
    result = TaskReadFastPath(Store()).resolve(
        IntentFrame(principal=Principal(id="alice", vault_id="alice-vault"), utterance=utterance)
    )
    assert result is not None
    assert result.evidence["due_filter"] == "weekday:friday"
    assert [item["title"] for item in result.evidence["canonical_tasks"]] == ["friday task"]


def test_task_read_fast_path_next_week_uses_next_calendar_week():
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    next_monday = now.date() + timedelta(days=7 - now.weekday())
    first = Task(
        uuid4(),
        "apartment",
        "next Monday task",
        "alice",
        due_at=datetime.combine(next_monday, datetime.min.time().replace(hour=12), timezone.utc),
    )
    last = Task(
        uuid4(),
        "apartment",
        "next Sunday task",
        "alice",
        due_at=datetime.combine(
            next_monday + timedelta(days=6), datetime.min.time().replace(hour=12), timezone.utc
        ),
    )
    following = Task(
        uuid4(),
        "apartment",
        "following week task",
        "alice",
        due_at=datetime.combine(
            next_monday + timedelta(days=7), datetime.min.time().replace(hour=12), timezone.utc
        ),
    )

    class Store:
        def list(self, _principal):
            return (first, last, following)

    result = TaskReadFastPath(Store()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault"),
            utterance="What tasks are due next week?",
        )
    )
    assert result is not None
    assert result.evidence["due_filter"] == "next_week"
    assert [item["title"] for item in result.evidence["canonical_tasks"]] == [
        "next Monday task",
        "next Sunday task",
    ]


def test_task_read_fast_path_filters_explicit_status_language():
    completed = Task(uuid4(), "apartment", "done task", "alice", status=TaskStatus.COMPLETED)
    open_task = Task(uuid4(), "apartment", "open task", "alice", status=TaskStatus.OPEN)

    class Store:
        def list(self, _principal):
            return (completed, open_task)

    result = TaskReadFastPath(Store()).resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
            utterance="Show my completed tasks",
        )
    )

    assert result is not None
    assert result.evidence["status_filter"] == "completed"
    assert result.evidence["canonical_tasks"] == [
        {"task_id": str(completed.task_id), "title": "done task", "status": "completed"}
    ]


def test_task_intent_clarification_blocks_unresolved_task_verbs():
    result = TaskIntentClarificationFastPath.resolve(
        IntentFrame(
            principal=Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
            utterance="Can you take care of the task Verify backup retention?",
        )
    )

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert "complete the task" in result.message


def test_reference_pack_ui_metadata_is_optional_and_non_authoritative():
    bundles = reference_bundles()

    assert {bundle.manifest.ui.label for bundle in bundles if bundle.manifest.ui} == {
        "Calendar",
        "Calendar Reports",
        "Calendar Communications",
        "Communications",
        "Documents",
        "Tasks",
        "Kitchen",
        "Homelab",
        "Homelab Reports",
        "Network",
        "Workspace",
        "Devices",
        "Communication Drafts",
        "Device Controls",
        "Device Reports",
    }
    assert all(bundle.manifest.permissions for bundle in bundles)


def test_browser_interaction_exposes_canonical_result_status(monkeypatch):
    from aegis import cli

    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.BLOCKED,
        message="authorization denied",
        correlation_id=uuid4(),
    )
    monkeypatch.setattr(cli, "run_interaction", lambda *_: result)

    payload = cli._browser_interaction(
        "Add rice to groceries.",
        Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )

    assert payload["state"] == "blocked"
    assert payload["code"] == "blocked"
    assert payload["message"] == "Not completed — authorization denied"
    assert payload["detail"] == "authorization denied"
    assert payload["objective_id"] == str(result.objective_id)


def test_cli_formats_safe_cross_domain_planning_summary():
    from aegis import cli

    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="Cross-domain planning context assembled from canonical state",
        correlation_id=uuid4(),
        evidence={
            "planning": {
                "affordability": {
                    "affordable": False,
                    "purchase_cents": 5000,
                    "shared_obligations_cents": 12000,
                    "shortfall_cents": 2000,
                    "balance_cents": 999999,
                },
                "open_tasks": [{"title": "Review backup runbook", "task_id": "private-id"}],
            }
        },
    )

    formatted = cli._format(result)

    assert formatted == (
        "Planning: affordable: no (purchase $50.00; shared obligations $120.00); "
        "open tasks: Review backup runbook"
    )
    assert "balance" not in formatted
    assert "private-id" not in formatted


def test_cli_does_not_replace_model_answer_with_authorized_working_set():
    from aegis import cli

    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="No due-today deadline is recorded.",
        correlation_id=uuid4(),
        evidence={
            "provenance": "model_generated",
            "authoritative": False,
            "answer_mode": "READ",
            "canonical_items": ["rice"],
            "canonical_tasks": [{"title": "review restore drill", "status": "open"}],
        },
    )

    assert cli._format(result) == "No due-today deadline is recorded."


def test_cli_formats_repeated_canonical_groceries_as_counts_without_changing_truth():
    from aegis import cli

    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="Canonical grocery list read",
        correlation_id=uuid4(),
        evidence={"canonical_items": ["rice", "beans", "rice", "rice"]},
    )

    assert cli._format(result) == "Groceries: rice (x3), beans"
    assert result.evidence["canonical_items"] == ["rice", "beans", "rice", "rice"]


def test_cli_formats_cross_domain_memory_and_obligation_summary():
    from aegis import cli

    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="Cross-domain planning context assembled from canonical state",
        correlation_id=uuid4(),
        evidence={
            "planning": {
                "open_obligations": [{"title": "Utilities", "responsible_id": "alice"}],
                "memories": [{"content": "The backup needs a restore drill"}],
            }
        },
    )

    assert cli._format(result) == (
        "Planning: open obligations: Utilities; relevant memories: The backup needs a restore drill"
    )


def test_cli_formats_cross_domain_open_chores():
    from aegis import cli

    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="Cross-domain planning context assembled from canonical state",
        correlation_id=uuid4(),
        evidence={"planning": {"open_chores": [{"title": "Clean the kitchen"}]}},
    )

    assert cli._format(result) == "Planning: open chores: Clean the kitchen"


def test_cli_formats_completed_chore_as_completion():
    from aegis import cli

    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="canonical chore readback verified",
        correlation_id=uuid4(),
        evidence={"collection": "chores", "title": "Dishes", "completed": True},
    )

    assert cli._format(result) == "Done — completed chore: Dishes"


def test_browser_interaction_projects_bounded_canonical_step_status(monkeypatch):
    from aegis import cli

    child_objective = uuid4()
    child_correlation = uuid4()
    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="Completed all 2 plan steps",
        correlation_id=uuid4(),
        evidence={
            "steps": [
                {
                    "index": 0,
                    "action_id": "tasks.create",
                    "state": "completed",
                    "objective_id": str(child_objective),
                    "correlation_id": str(child_correlation),
                    "message": "canonical task readback verified",
                    "evidence": {"title": "private detail is not projected"},
                }
            ]
        },
    )
    monkeypatch.setattr(cli, "run_interaction", lambda *_: result)

    payload = cli._browser_interaction(
        "Create a task and a chore.",
        Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )

    assert payload["steps"] == (
        {
            "index": 0,
            "action_id": "tasks.create",
            "state": "completed",
            "objective_id": str(child_objective),
            "correlation_id": str(child_correlation),
            "message": "canonical task readback verified",
        },
    )


def test_constellation_state_keeps_current_pack_ui_metadata(monkeypatch):
    from aegis import cli
    from aegis.network import HomelabInventory
    from aegis.personal import PersonalState

    class Connection:
        def close(self):
            pass

    monkeypatch.setenv("AEGIS_DATABASE_URL", "postgresql://example")
    monkeypatch.setattr(cli.psycopg, "connect", lambda *_args, **_kwargs: Connection())
    monkeypatch.setattr(cli, "_apply_migrations", lambda _connection: None)
    monkeypatch.setattr(
        cli.PostgresHouseholdStore,
        "read_snapshot",
        lambda _store, _principal: {"groceries": ()},
    )
    monkeypatch.setattr(cli.PostgresTaskStore, "list", lambda _store, _principal: ())
    monkeypatch.setattr(cli.PostgresPackStore, "load", lambda _store: ())
    monkeypatch.setattr(
        cli.PostgresPersonalStateStore,
        "load_for_principal",
        lambda _store, _principal: PersonalState(),
    )
    monkeypatch.setattr(cli.PostgresFinanceSnapshotStore, "load", lambda _store, _owner: None)
    monkeypatch.setattr(
        cli.PostgresNetworkStore, "load", lambda _store, _principal: HomelabInventory()
    )
    monkeypatch.setattr(
        cli.PostgresHomelabStore,
        "load",
        lambda _store, _principal, _runtime: type("Pack", (), {"hosts": {}, "services": {}})(),
    )

    state = cli._constellation_state(
        Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    )

    labels = {node["label"] for node in state["nodes"]}
    assert {"AEGIS", "Homelab", "Kitchen", "Network", "Tasks"} <= labels
    assert any(
        node["id"].startswith("pack-tasks-area-") and node["category"] == "capability"
        for node in state["nodes"]
    )
    assert any(
        edge["source"] == "pack-tasks" and edge["target"].startswith("pack-tasks-area-")
        for edge in state["edges"]
    )
    assert state["nodes"][0]["category"] == "core"
    assert state["nodes"][0]["detail_view"] == "overview"
    assert state["nodes"][1]["category"] == "domain"
    assert state["nodes"][1]["detail_view"] == "list"
    node_ids = [node["id"] for node in state["nodes"]]
    assert node_ids.count("pack-homelab") == 1
    assert node_ids.count("domain-homelab") == 0
    assert node_ids.count("pack-network") == 1
    assert node_ids.count("domain-network") == 0
    assert "composition-calendar-to-workspace" in node_ids
    assert any(
        edge["source"] == "pack-calendar" and edge["target"] == "composition-calendar-to-workspace"
        for edge in state["edges"]
    )
    assert state["details"]["composition-calendar-to-workspace"]["authority"]


def test_constellation_graph_keeps_record_rows_in_detail_views(monkeypatch):
    from aegis import cli
    from aegis.network import HomelabInventory
    from aegis.personal import PersonalState, Project

    class Connection:
        def close(self):
            pass

    project = Project(uuid4(), "Private project", datetime(2026, 1, 1))
    task = Task(uuid4(), "apartment", "Private task", "alice")
    personal = PersonalState(projects={project.project_id: project})
    monkeypatch.setenv("AEGIS_DATABASE_URL", "postgresql://example")
    monkeypatch.setattr(cli.psycopg, "connect", lambda *_args, **_kwargs: Connection())
    monkeypatch.setattr(cli, "_apply_migrations", lambda _connection: None)
    monkeypatch.setattr(
        cli.PostgresHouseholdStore,
        "read_snapshot",
        lambda _store, _principal: {"groceries": ()},
    )
    monkeypatch.setattr(cli.PostgresTaskStore, "list", lambda _store, _principal: (task,))
    monkeypatch.setattr(cli.PostgresPackStore, "load", lambda _store: ())
    monkeypatch.setattr(
        cli.PostgresPersonalStateStore,
        "load_for_principal",
        lambda _store, _principal: personal,
    )
    monkeypatch.setattr(cli.PostgresFinanceSnapshotStore, "load", lambda _store, _owner: None)
    monkeypatch.setattr(
        cli.PostgresNetworkStore, "load", lambda _store, _principal: HomelabInventory()
    )
    monkeypatch.setattr(
        cli.PostgresHomelabStore,
        "load",
        lambda _store, _principal, _runtime: type("Pack", (), {"hosts": {}, "services": {}})(),
    )

    state = cli._constellation_state(
        Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    )
    node_ids = {node["id"] for node in state["nodes"]}

    assert f"project-{project.project_id}" not in node_ids
    assert f"task-{task.task_id}" not in node_ids
    assert state["details"]["domain-personal"]["projects"][0]["name"] == "Private project"
    assert state["details"]["pack-tasks"]["tasks"][0]["title"] == "Private task"


def _deny() -> dict[str, object]:
    raise PermissionError("revoked")
