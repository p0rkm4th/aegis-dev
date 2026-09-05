from uuid import uuid4

from aegis.contracts import ExecutionRequest, Observation, Principal, VerificationContract
from aegis.devices import FixtureDeviceGateway
from aegis.documents import Document, FixtureDocumentProvider
from aegis.pack_lifecycle import PackBundle
from aegis.reference_packs import (
    DeviceControlExecutor,
    DeviceControlVerifier,
    DocumentWorkspaceExecutor,
    DocumentWorkspaceVerifier,
    prepare_reference_action,
    reference_bundles,
    reference_packs,
)
from aegis.reference_runtime import default_runtime_registry
from aegis.workspace import WorkspaceManager


def test_first_party_packs_use_the_generic_pack_bundle_contract() -> None:
    packs = reference_packs()

    assert packs
    assert all(isinstance(pack, PackBundle) for pack in packs)
    assert reference_bundles() == packs
    assert {pack.manifest.pack_id for pack in packs} == {
        "calendar",
        "communications",
        "documents",
        "tasks",
        "kitchen",
        "homelab",
        "network",
        "workspace",
        "devices",
        "communication-drafts",
        "device-controls",
        "device-reports",
    }


def test_workspace_pack_uses_generic_runtime_and_readback(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AEGIS_WORKSPACE_ROOT", str(tmp_path))
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "workspace.artifact.create"
    )
    principal = Principal(id="alice", vault_id="alice-vault")
    runtime = default_runtime_registry(lambda: None).resolve(card, None, principal)
    objective_id = uuid4()
    action = card.action.model_copy(
        update={"arguments": {"path": "index.html", "content": "<html>ok</html>"}}
    )
    assert runtime.prepare is not None
    action = runtime.prepare(action, principal, objective_id)
    observation = runtime.executor.execute(
        ExecutionRequest(
            objective_id=objective_id,
            action_id=uuid4(),
            action=action,
            idempotency_key="artifact-1",
        )
    )
    assert observation.command_succeeded is True
    assert action.verification is not None
    assert runtime.verifier.verify(observation, action.verification).verified is True


def test_research_workspace_composition_preserves_sources_and_non_authority(tmp_path, monkeypatch):
    monkeypatch.setenv("AEGIS_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "AEGIS_RESEARCH_FIXTURE_JSON",
        '[{"title":"Config guide","url":"https://fixture.test/config","text":"Use mode=casual."}]',
    )
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "workspace.research_notes.create"
    )
    action = card.action.model_copy(
        update={"arguments": {"query": "configuration mode", "target_path": "notes.md"}}
    )
    runtime = default_runtime_registry(lambda: None).resolve(
        card, None, Principal(id="alice", vault_id="alice-vault")
    )
    observation = runtime.executor.execute(
        ExecutionRequest(
            objective_id=uuid4(), action_id=uuid4(), action=action, idempotency_key="research-1"
        )
    )
    assert observation.command_succeeded is True
    assert observation.evidence["authoritative"] is False
    assert runtime.verifier.verify(observation, card.action.verification).verified is False


def test_communication_draft_is_scoped_and_unsent(tmp_path, monkeypatch):
    monkeypatch.setenv("AEGIS_WORKSPACE_ROOT", str(tmp_path))
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "communication-drafts.messages.draft"
    )
    action = card.action.model_copy(
        update={
            "arguments": {
                "recipient": "owner@example.test",
                "subject": "Staging update",
                "body": "The staging check is complete.",
                "target_path": "drafts/staging-update.md",
            }
        }
    )
    principal = Principal(id="alice", vault_id="alice-vault")
    runtime = default_runtime_registry(lambda: None).resolve(card, None, principal)
    objective_id = uuid4()
    assert runtime.prepare is not None
    action = runtime.prepare(action, principal, objective_id)
    observation = runtime.executor.execute(
        ExecutionRequest(
            objective_id=objective_id, action_id=uuid4(), action=action, idempotency_key="draft-1"
        )
    )
    assert observation.command_succeeded is True
    assert observation.evidence["sent"] is False
    assert action.verification is not None
    assert runtime.verifier.verify(observation, action.verification).verified is True


def test_devices_pack_reads_authorized_entity_state_through_generic_runtime() -> None:
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "devices.states.list"
    )
    principal = Principal(id="alice", vault_id="alice-vault")
    runtime = default_runtime_registry(lambda: None).resolve(card, None, principal)
    observation = runtime.executor.execute(
        ExecutionRequest(
            objective_id=uuid4(), action_id=uuid4(), action=card.action, idempotency_key="device-1"
        )
    )
    assert observation.command_succeeded is True
    assert observation.evidence["source"] == "home_assistant_fixture"
    assert runtime.verifier.verify(observation, card.action.verification).verified is True


def test_device_controls_pack_verifies_low_risk_fixture_readback(monkeypatch) -> None:
    monkeypatch.setenv("AEGIS_AUTHORIZED_DEVICE_ENTITIES", "light.desk")
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "device-controls.devices.command.execute"
    )
    action = card.action.model_copy(
        update={
            "arguments": {
                "entity_id": "light.desk",
                "service": "turn_on",
                "expected_state": "on",
            }
        }
    )
    runtime = default_runtime_registry(lambda: None).resolve(
        card, None, Principal(id="alice", vault_id="alice-vault")
    )
    objective_id = uuid4()
    assert runtime.prepare is not None
    principal = Principal(id="alice", vault_id="alice-vault")
    action = runtime.prepare(action, principal, objective_id)
    observation = runtime.executor.execute(
        ExecutionRequest(
            objective_id=objective_id,
            action_id=uuid4(),
            action=action,
            idempotency_key="device-control-1",
        )
    )
    assert observation.command_succeeded is True
    assert action.verification is not None
    assert runtime.verifier.verify(observation, action.verification).verified is True


def test_device_verifier_rejects_forged_execution_without_matching_provider_state() -> None:
    verifier = DeviceControlVerifier(FixtureDeviceGateway({"light.desk": {"state": "off"}}))
    observation = Observation(
        execution_id=uuid4(),
        evidence={"device_execution": {"verified": True, "entity_id": "light.desk"}},
        command_succeeded=True,
    )
    contract = VerificationContract(
        kind="custom",
        expected={"entity_id": "light.desk", "expected_state": "on"},
    )
    assert verifier.verify(observation, contract).verified is False


def test_device_service_success_with_wrong_provider_state_is_not_verified(monkeypatch) -> None:
    class NoMutationGateway(FixtureDeviceGateway):
        def call_service(self, command):
            del command

    monkeypatch.setenv("AEGIS_AUTHORIZED_DEVICE_ENTITIES", "light.desk")
    gateway = NoMutationGateway({"light.desk": {"state": "off"}})
    executor = DeviceControlExecutor(gateway)
    verifier = DeviceControlVerifier(gateway)
    principal = Principal(id="alice", vault_id="alice-vault")
    objective_id = uuid4()
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "device-controls.devices.command.execute"
    )
    action = card.action.model_copy(
        update={
            "arguments": {
                "entity_id": "light.desk",
                "service": "turn_on",
                "expected_state": "on",
            }
        }
    )
    action = prepare_reference_action(action, principal, objective_id)
    observation = executor.execute(
        ExecutionRequest(
            objective_id=objective_id,
            action_id=uuid4(),
            action=action,
            idempotency_key="no-mutation",
        )
    )
    assert action.verification is not None
    assert observation.command_succeeded is True
    assert verifier.verify(observation, action.verification).verified is False


def test_device_controls_pack_returns_structured_scope_denial(monkeypatch) -> None:
    monkeypatch.setenv("AEGIS_AUTHORIZED_DEVICE_ENTITIES", "light.desk")
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "device-controls.devices.command.execute"
    )
    action = card.action.model_copy(
        update={
            "arguments": {
                "entity_id": "light.garage",
                "service": "turn_on",
                "expected_state": "on",
            }
        }
    )
    runtime = default_runtime_registry(lambda: None).resolve(
        card, None, Principal(id="alice", vault_id="alice-vault")
    )
    observation = runtime.executor.execute(
        ExecutionRequest(
            objective_id=uuid4(),
            action_id=uuid4(),
            action=action,
            idempotency_key="device-control-denied",
        )
    )
    assert observation.command_succeeded is False
    assert runtime.verifier.verify(observation, card.action.verification).verified is False
    assert (
        "outside the authorized device scope" in observation.evidence["device_execution"]["reason"]
    )


def test_device_reports_pack_verifies_workspace_composition(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AEGIS_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("AEGIS_AUTHORIZED_DEVICE_ENTITIES", "homeassistant.status")
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "device-reports.devices.snapshot_to_workspace"
    )
    action = card.action.model_copy(update={"arguments": {"target_path": "devices.md"}})
    runtime = default_runtime_registry(lambda: None).resolve(
        card, None, Principal(id="alice", vault_id="alice-vault")
    )
    observation = runtime.executor.execute(
        ExecutionRequest(
            objective_id=uuid4(),
            action_id=uuid4(),
            action=action,
            idempotency_key="device-report-1",
        )
    )
    assert observation.command_succeeded is True
    assert runtime.verifier.verify(observation, card.action.verification).verified is False


def test_workspace_pack_verifies_a_multi_file_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AEGIS_WORKSPACE_ROOT", str(tmp_path))
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "workspace.artifact.create"
    )
    principal = Principal(id="alice", vault_id="alice-vault")
    runtime = default_runtime_registry(lambda: None).resolve(card, None, principal)
    files = {"index.html": "<link rel=stylesheet href=style.css>", "style.css": "body{}"}
    action = card.action.model_copy(update={"arguments": {"files": files}})
    objective_id = uuid4()
    assert runtime.prepare is not None
    action = runtime.prepare(action, principal, objective_id)
    observation = runtime.executor.execute(
        ExecutionRequest(
            objective_id=objective_id,
            action_id=uuid4(),
            action=action,
            idempotency_key="artifact-2",
        )
    )
    assert observation.command_succeeded is True
    assert observation.evidence["files"] == ["index.html", "style.css"]
    assert action.verification is not None
    assert runtime.verifier.verify(observation, action.verification).verified is True


def _prepared_workspace_action(runtime, card, principal, objective_id, files):
    action = card.action.model_copy(update={"arguments": {"files": files}})
    assert runtime.prepare is not None
    return runtime.prepare(action, principal, objective_id)


def test_workspace_verifier_reopens_actual_scope_after_execution(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AEGIS_WORKSPACE_ROOT", str(tmp_path))
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "workspace.artifact.create"
    )
    principal = Principal(id="alice", vault_id="alice-vault")
    runtime = default_runtime_registry(lambda: None).resolve(card, None, principal)
    objective_id = uuid4()
    action = _prepared_workspace_action(
        runtime, card, principal, objective_id, {"index.html": "ok"}
    )
    request = ExecutionRequest(
        objective_id=objective_id, action_id=uuid4(), action=action, idempotency_key="adversarial"
    )
    observation = runtime.executor.execute(request)
    assert action.verification is not None
    workspace = WorkspaceManager(tmp_path).for_objective(principal.id, objective_id)
    workspace._path("index.html").unlink()
    assert runtime.verifier.verify(observation, action.verification).verified is False
    workspace.write("index.html", "changed")
    assert runtime.verifier.verify(observation, action.verification).verified is False


def test_workspace_verifier_rejects_forgery_wrong_scope_and_redirect(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AEGIS_WORKSPACE_ROOT", str(tmp_path))
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "workspace.artifact.create"
    )
    alice = Principal(id="alice", vault_id="alice-vault")
    bob = Principal(id="bob", vault_id="bob-vault")
    runtime = default_runtime_registry(lambda: None).resolve(card, None, alice)
    objective_id = uuid4()
    action = _prepared_workspace_action(runtime, card, alice, objective_id, {"index.html": "ok"})
    assert action.verification is not None
    forged = Observation(
        execution_id=uuid4(),
        evidence={
            "workspace": "artifact",
            "validated": True,
            "workspace_id": "other",
            "path": "/host/escape/index.html",
        },
        command_succeeded=True,
    )
    assert runtime.verifier.verify(forged, action.verification).verified is False
    WorkspaceManager(tmp_path).for_objective(bob.id, objective_id).write("index.html", "ok")
    assert runtime.verifier.verify(forged, action.verification).verified is False
    WorkspaceManager(tmp_path).for_objective(alice.id, uuid4()).write("index.html", "ok")
    assert runtime.verifier.verify(forged, action.verification).verified is False


def test_workspace_verifier_requires_exact_files_and_rejects_symlink(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AEGIS_WORKSPACE_ROOT", str(tmp_path))
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "workspace.artifact.create"
    )
    principal = Principal(id="alice", vault_id="alice-vault")
    runtime = default_runtime_registry(lambda: None).resolve(card, None, principal)
    objective_id = uuid4()
    files = {"index.html": "ok", "style.css": "body{}"}
    action = _prepared_workspace_action(runtime, card, principal, objective_id, files)
    request = ExecutionRequest(
        objective_id=objective_id, action_id=uuid4(), action=action, idempotency_key="exact"
    )
    observation = runtime.executor.execute(request)
    assert action.verification is not None
    workspace = WorkspaceManager(tmp_path).for_objective(principal.id, objective_id)
    (workspace.root / "style.css").unlink()
    assert runtime.verifier.verify(observation, action.verification).verified is False
    workspace.write("style.css", "wrong")
    assert runtime.verifier.verify(observation, action.verification).verified is False
    (workspace.root / "style.css").unlink()
    (workspace.root / "style.css").symlink_to(workspace.root / "index.html")
    assert runtime.verifier.verify(observation, action.verification).verified is False


def test_document_export_pack_composes_authorized_read_with_workspace(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AEGIS_WORKSPACE_ROOT", str(tmp_path))
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "documents.export_to_workspace"
    )
    principal = Principal(id="alice", vault_id="alice-vault")
    executor = DocumentWorkspaceExecutor(
        principal,
        FixtureDocumentProvider((Document("doc-1", "Starter", "Keep this scoped."),)),
    )
    action = card.action.model_copy(
        update={"arguments": {"document_id": "doc-1", "target_path": "starter.md"}}
    )
    observation = executor.execute(
        ExecutionRequest(
            objective_id=uuid4(), action_id=uuid4(), action=action, idempotency_key="export-1"
        )
    )
    assert observation.command_succeeded is True
    assert observation.evidence["source"] == "authorized_document"
    assert DocumentWorkspaceVerifier(principal).verify(
        observation, card.action.verification
    ).verified is False
