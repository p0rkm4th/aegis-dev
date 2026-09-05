from uuid import uuid4

from aegis.contracts import ExecutionRequest, Principal
from aegis.documents import Document, FixtureDocumentProvider
from aegis.pack_lifecycle import PackBundle
from aegis.reference_packs import (
    DocumentWorkspaceExecutor,
    DocumentWorkspaceVerifier,
    reference_bundles,
    reference_packs,
)
from aegis.reference_runtime import default_runtime_registry


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
    action = card.action.model_copy(
        update={"arguments": {"path": "index.html", "content": "<html>ok</html>"}}
    )
    observation = runtime.executor.execute(
        ExecutionRequest(
            objective_id=uuid4(), action_id=uuid4(), action=action, idempotency_key="artifact-1"
        )
    )
    assert observation.command_succeeded is True
    assert runtime.verifier.verify(observation, card.action.verification).verified is True


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
    observation = runtime.executor.execute(
        ExecutionRequest(
            objective_id=uuid4(), action_id=uuid4(), action=action, idempotency_key="artifact-2"
        )
    )
    assert observation.command_succeeded is True
    assert observation.evidence["files"] == ["index.html", "style.css"]
    assert runtime.verifier.verify(observation, card.action.verification).verified is True


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
    assert (
        DocumentWorkspaceVerifier().verify(observation, card.action.verification).verified is True
    )
