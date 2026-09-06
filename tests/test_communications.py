import subprocess
from types import SimpleNamespace
from uuid import uuid4

import pytest

import aegis.reference_packs as reference_packs_module
from aegis.communications import (
    FixtureCommunicationProvider,
    FixtureCommunicationSendProvider,
    Message,
    OpenClawCliCommunicationSendProvider,
    OutboundMessage,
    SendStatus,
    communications_evidence,
    configured_communication_targets,
)
from aegis.contracts import ActionSpec, ExecutionRequest, Observation, Principal
from aegis.reference_packs import (
    CommunicationsSendExecutor,
    prepare_reference_action,
    reference_bundles,
)
from aegis.reference_runtime import default_runtime_registry


def test_approved_communication_targets_are_exact_and_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv(
        "AEGIS_APPROVED_COMMUNICATION_TARGETS",
        '[{"target":"sms:+15555550123","channel":"sms","account":"personal"}]',
    )
    assert configured_communication_targets() == frozenset(
        {("sms:+15555550123", "sms", "personal")}
    )
    monkeypatch.setenv("AEGIS_APPROVED_COMMUNICATION_TARGETS", "not-json")
    with pytest.raises(ValueError):
        configured_communication_targets()


def test_send_executor_rejects_target_outside_approved_boundary() -> None:
    provider = FixtureCommunicationSendProvider()
    executor = CommunicationsSendExecutor(
        provider,
        frozenset({("sms:+15555550123", "sms", "personal")}),
    )
    action = {
        "target": "sms:+15555550124",
        "body": "Milk",
        "channel": "sms",
        "account": "personal",
    }
    observation = executor.execute(
        ExecutionRequest(
            objective_id=uuid4(),
            action_id=uuid4(),
            action=ActionSpec(
                action_id="communications.messages.send",
                capability="communications.messages.send",
                arguments=action,
            ),
            idempotency_key="blocked-target",
        )
    )
    assert observation.command_succeeded is False
    assert observation.evidence["communication_send"] == "target_not_approved"
    assert provider.sent == []


def test_canonical_grocery_source_is_fixed_before_generic_send(monkeypatch) -> None:
    class FakeHouseholdStore:
        def __init__(self, _connection):
            pass

        def list_groceries(self, _principal):
            return ("milk", "rice")

    monkeypatch.setattr(reference_packs_module, "PostgresHouseholdStore", FakeHouseholdStore)
    from aegis.contracts import ActionSpec

    prepared = prepare_reference_action(
        ActionSpec(
            action_id="communications.messages.send",
            capability="communications.messages.send",
            arguments={
                "target": "scotty",
                "channel": "sms",
                "account": "personal",
                "body_source": "canonical.groceries",
            },
        ),
        Principal(id="alice", vault_id="vault"),
        uuid4(),
        object(),
    )
    assert prepared.arguments["body"] == "Grocery list:\n- milk\n- rice"


def test_bounded_research_source_is_fixed_before_unsent_draft(monkeypatch) -> None:
    monkeypatch.setenv(
        "AEGIS_RESEARCH_FIXTURE_JSON",
        '[{"title":"Guide","url":"https://fixture.test/guide",'
        '"text":"Use the casual setting.","snippet":"casual"}]',
    )
    from aegis.contracts import ActionSpec

    prepared = prepare_reference_action(
        ActionSpec(
            action_id="communication-drafts.messages.draft",
            capability="communication-drafts.messages.draft",
            arguments={
                "recipient": "scotty",
                "subject": "Settings",
                "query": "casual settings",
                "target_path": "research.md",
                "body_source": "bounded.research",
            },
        ),
        Principal(id="alice", vault_id="vault"),
        uuid4(),
    )
    assert "Use the casual setting." in prepared.arguments["body"]


def test_fixture_communications_are_bounded_and_read_only() -> None:
    evidence = communications_evidence(
        FixtureCommunicationProvider(
            (Message("m-1", "Ada", "Hello", "Read this."),)
        ).list_messages()
    )
    assert evidence["source"] == "authorized_communications_fixture"
    assert evidence["messages"][0]["sender"] == "Ada"


def test_communications_pack_uses_generic_verified_read_runtime() -> None:
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "communications.messages.list"
    )
    runtime = default_runtime_registry(lambda: None).resolve(
        card, None, Principal(id="alice", vault_id="vault")
    )
    observation = runtime.executor.execute(
        ExecutionRequest(
            objective_id=uuid4(),
            action_id=uuid4(),
            action=card.action,
            idempotency_key="messages-1",
        )
    )
    assert observation.command_succeeded is True
    assert runtime.verifier.verify(observation, card.action.verification).verified is True


def test_fixture_send_is_idempotent_and_reports_acceptance_not_delivery() -> None:
    provider = FixtureCommunicationSendProvider()
    message = OutboundMessage(target="scotty", body="Milk", channel="sms")
    first = provider.send(message, "send-1")
    second = provider.send(message, "send-1")
    assert first.status is SendStatus.PROVIDER_ACCEPTED
    assert second.provider_message_id == first.provider_message_id
    assert len(provider.sent) == 1


def test_fixture_send_verifier_reads_back_exact_provider_message() -> None:
    provider = FixtureCommunicationSendProvider()
    principal = Principal(id="alice", vault_id="vault")
    action = ActionSpec(
        action_id="communications.messages.send",
        capability="communications.messages.send",
        arguments={"target": "scotty", "body": "Milk", "channel": "sms", "account": "household"},
    )
    prepared = prepare_reference_action(action, principal, uuid4())
    executor = CommunicationsSendExecutor(provider)
    observation = executor.execute(
        ExecutionRequest(
            objective_id=uuid4(), action_id=uuid4(), action=prepared, idempotency_key="readback-1"
        )
    )
    result = reference_packs_module.CommunicationsSendVerifier(provider).verify(
        observation, prepared.verification
    )
    assert result.verified is True
    assert result.evidence["independent_provider_readback"] is True
    assert result.evidence["independent_delivery"] is False


def test_calendar_snapshot_message_prepares_body_before_generic_send() -> None:
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "communications.messages.send"
    )
    action = card.action.model_copy(
        update={
            "arguments": {
                "target": "scotty",
                "channel": "sms",
                "account": "household",
                "body_source": "canonical.calendar",
            }
        }
    )
    principal = Principal(id="alice", vault_id="vault")
    prepared = prepare_reference_action(action, principal, uuid4())
    expected_body = prepared.verification.expected["body"]
    assert isinstance(expected_body, str)
    assert expected_body.startswith("# Authorized calendar snapshot")

    provider = FixtureCommunicationSendProvider()
    observation = CommunicationsSendExecutor(provider).execute(
        ExecutionRequest(
            objective_id=uuid4(),
            action_id=uuid4(),
            action=prepared,
            idempotency_key="calendar-send-1",
        )
    )
    result = reference_packs_module.CommunicationsSendVerifier(provider).verify(
        observation, prepared.verification
    )
    assert result.verified is True
    assert result.evidence["independent_provider_readback"] is True


def test_research_message_fixes_evidence_before_generic_send(monkeypatch) -> None:
    monkeypatch.setenv(
        "AEGIS_RESEARCH_FIXTURE_JSON",
        '[{"title":"Guide","url":"https://fixture.test/guide",'
        '"text":"Use the casual setting.","snippet":"casual"}]',
    )
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "communications.messages.send"
    )
    action = card.action.model_copy(
        update={
            "arguments": {
                "target": "scotty",
                "channel": "sms",
                "account": "household",
                "body_source": "bounded.research",
                "query": "casual server settings",
            }
        }
    )
    principal = Principal(id="alice", vault_id="vault")
    prepared = prepare_reference_action(action, principal, uuid4())
    expected_body = prepared.verification.expected["body"]
    assert isinstance(expected_body, str)
    assert "Use the casual setting." in expected_body

    provider = FixtureCommunicationSendProvider()
    observation = CommunicationsSendExecutor(provider).execute(
        ExecutionRequest(
            objective_id=uuid4(),
            action_id=uuid4(),
            action=prepared,
            idempotency_key="research-send-1",
        )
    )
    result = reference_packs_module.CommunicationsSendVerifier(provider).verify(
        observation, prepared.verification
    )
    assert result.verified is True
    assert result.evidence["independent_provider_readback"] is True


def test_weather_message_fixes_public_forecast_before_generic_send(monkeypatch) -> None:
    monkeypatch.setenv("AEGIS_WEATHER_LATITUDE", "41.881832")
    monkeypatch.setenv("AEGIS_WEATHER_LONGITUDE", "-87.623177")
    monkeypatch.setenv(
        "AEGIS_WEATHER_FORECAST_FIXTURE_JSON",
        '[{"date":"2026-09-07","temperature_max_c":22,"temperature_min_c":12,'
        '"precipitation_probability_max":20,"sunrise":"06:20","sunset":"19:10"}]',
    )
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "communications.messages.send"
    )
    action = card.action.model_copy(
        update={
            "arguments": {
                "target": "scotty",
                "channel": "sms",
                "account": "household",
                "body_source": "public.weather",
            }
        }
    )
    prepared = prepare_reference_action(action, Principal(id="alice", vault_id="vault"), uuid4())
    expected_body = prepared.verification.expected["body"]
    assert isinstance(expected_body, str)
    assert "2026-09-07" in expected_body

    provider = FixtureCommunicationSendProvider()
    observation = CommunicationsSendExecutor(provider).execute(
        ExecutionRequest(
            objective_id=uuid4(),
            action_id=uuid4(),
            action=prepared,
            idempotency_key="weather-send-1",
        )
    )
    result = reference_packs_module.CommunicationsSendVerifier(provider).verify(
        observation, prepared.verification
    )
    assert result.verified is True
    assert result.evidence["independent_provider_readback"] is True


def test_document_message_fixes_authorized_content_before_generic_send(monkeypatch) -> None:
    monkeypatch.setenv(
        "AEGIS_DOCUMENT_FIXTURE_JSON",
        '[{"document_id":"alpha-handbook","title":"Alpha Handbook",'
        '"text":"Authorized guidance.","source":"owner"}]',
    )
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "communications.messages.send"
    )
    action = card.action.model_copy(
        update={
            "arguments": {
                "target": "scotty",
                "channel": "sms",
                "account": "household",
                "body_source": "canonical.document",
                "document_id": "alpha-handbook",
            }
        }
    )
    prepared = prepare_reference_action(action, Principal(id="alice", vault_id="vault"), uuid4())
    expected_body = prepared.verification.expected["body"]
    assert expected_body == "Alpha Handbook\n\nAuthorized guidance."

    provider = FixtureCommunicationSendProvider()
    observation = CommunicationsSendExecutor(provider).execute(
        ExecutionRequest(
            objective_id=uuid4(),
            action_id=uuid4(),
            action=prepared,
            idempotency_key="document-send-1",
        )
    )
    result = reference_packs_module.CommunicationsSendVerifier(provider).verify(
        observation, prepared.verification
    )
    assert result.verified is True
    assert result.evidence["independent_provider_readback"] is True


def test_homelab_health_message_fixes_observation_before_generic_send(monkeypatch) -> None:
    monkeypatch.setattr(
        reference_packs_module,
        "_canonical_homelab_service",
        lambda connection, principal, service_id: SimpleNamespace(
            name="Acceptance Plex", service_id=service_id, health_endpoint="http://plex.test/health"
        ),
    )
    monkeypatch.setattr(
        reference_packs_module, "_health_read", lambda endpoint: (False, "http_503")
    )
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "communications.messages.send"
    )
    action = card.action.model_copy(
        update={
            "arguments": {
                "target": "scotty",
                "channel": "sms",
                "account": "household",
                "body_source": "canonical.homelab_health",
                "service": "acceptance-plex",
            }
        }
    )
    prepared = prepare_reference_action(
        action, Principal(id="alice", vault_id="vault"), uuid4(), connection=object()
    )
    assert prepared.verification.expected["body"] == (
        "Homelab health for Acceptance Plex (acceptance-plex): http_503."
    )

    provider = FixtureCommunicationSendProvider()
    observation = CommunicationsSendExecutor(provider).execute(
        ExecutionRequest(
            objective_id=uuid4(),
            action_id=uuid4(),
            action=prepared,
            idempotency_key="homelab-health-send-1",
        )
    )
    result = reference_packs_module.CommunicationsSendVerifier(provider).verify(
        observation, prepared.verification
    )
    assert result.verified is True
    assert result.evidence["independent_provider_readback"] is True


def test_workspace_artifact_message_reads_scoped_content_before_generic_send(
    tmp_path, monkeypatch
) -> None:
    workspace_id = str(uuid4())
    monkeypatch.setenv("AEGIS_WORKSPACE_ROOT", str(tmp_path))
    root = tmp_path / "alice" / workspace_id
    root.mkdir(parents=True)
    (root / "report.md").write_text("Authorized workspace report.", encoding="utf-8")
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "workspace-communications.artifact.send"
    )
    action = card.action.model_copy(
        update={
            "arguments": {
                "target": "scotty",
                "channel": "sms",
                "account": "household",
                "body_source": "canonical.workspace_artifact",
                "workspace_id": workspace_id,
                "path": "report.md",
            }
        }
    )
    prepared = prepare_reference_action(action, Principal(id="alice", vault_id="vault"), uuid4())
    assert prepared.verification.expected["body"] == "Authorized workspace report."

    provider = FixtureCommunicationSendProvider()
    observation = CommunicationsSendExecutor(provider).execute(
        ExecutionRequest(
            objective_id=uuid4(),
            action_id=uuid4(),
            action=prepared,
            idempotency_key="workspace-artifact-send-1",
        )
    )
    result = reference_packs_module.CommunicationsSendVerifier(provider).verify(
        observation, prepared.verification
    )
    assert result.verified is True
    assert result.evidence["independent_provider_readback"] is True


def test_fixture_send_verifier_rejects_forged_acceptance_without_readback() -> None:
    provider = FixtureCommunicationSendProvider()
    principal = Principal(id="alice", vault_id="vault")
    action = ActionSpec(
        action_id="communications.messages.send",
        capability="communications.messages.send",
        arguments={"target": "scotty", "body": "Milk", "channel": "sms", "account": "household"},
    )
    prepared = prepare_reference_action(action, principal, uuid4())
    observation = Observation(
        execution_id=uuid4(),
        command_succeeded=True,
        evidence={
            "communication_send": {
                "status": "PROVIDER_ACCEPTED",
                "provider_message_id": "fixture:missing",
                "target": "scotty",
                "channel": "sms",
            }
        },
    )
    # A missing provider readback cannot be upgraded by forged acceptance evidence.
    assert (
        reference_packs_module.CommunicationsSendVerifier(provider)
        .verify(observation, prepared.verification)
        .verified
        is False
    )


def test_openclaw_cli_provider_adapts_explicit_message_send_without_claiming_delivery() -> None:
    calls: list[list[str]] = []

    def run(args: list[str]):
        import subprocess

        calls.append(args)
        return subprocess.CompletedProcess(args, 0, '{"messageId":"openclaw-42"}\n', "")

    provider = OpenClawCliCommunicationSendProvider(executable="/usr/bin/openclaw", runner=run)
    message = OutboundMessage(
        target="sms:+15555550123", body="Milk", channel="sms", account="personal"
    )
    result = provider.send(message, "send-42")
    replay = provider.send(message, "send-42")

    assert result.status is SendStatus.PROVIDER_ACCEPTED
    assert result.provider_message_id == "openclaw-42"
    assert "delivery is not independently proven" in result.detail
    assert replay == result
    assert calls == [
        [
            "/usr/bin/openclaw",
            "message",
            "send",
            "--channel",
            "sms",
            "--target",
            "sms:+15555550123",
            "--message",
            "Milk",
            "--json",
            "--account",
            "personal",
        ]
    ]


def test_openclaw_cli_provider_accepts_explicit_delivery_evidence() -> None:
    def run(_args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            _args, 0, '{"messageId":"openclaw-43","delivered":true}\n', ""
        )

    provider = OpenClawCliCommunicationSendProvider(runner=run)
    result = provider.send(
        OutboundMessage(target="scotty", body="hello", channel="sms", account="household"),
        "delivery-key",
    )

    assert result.status is SendStatus.DELIVERED
    assert result.provider_message_id == "openclaw-43"


def test_openclaw_cli_provider_downgrades_missing_acceptance() -> None:
    def run(args: list[str]):
        import subprocess

        return subprocess.CompletedProcess(args, 0, "{}", "")

    provider = OpenClawCliCommunicationSendProvider(runner=run)
    result = provider.send(OutboundMessage(target="scotty", body="Milk"), "send-43")

    assert result.status is SendStatus.SEND_ATTEMPTED
    assert result.provider_message_id is None


def test_communications_send_pack_uses_explicit_provider_contract() -> None:
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "communications.messages.send"
    )
    runtime = default_runtime_registry(lambda: None).resolve(
        card, None, Principal(id="alice", vault_id="vault")
    )
    action = card.action.model_copy(
        update={
            "arguments": {
                "target": "scotty",
                "body": "Milk",
                "channel": "sms",
                "account": "household",
            }
        }
    )
    observation = runtime.executor.execute(
        ExecutionRequest(
            objective_id=uuid4(), action_id=uuid4(), action=action, idempotency_key="send-1"
        )
    )
    result = runtime.verifier.verify(observation, card.action.verification)
    assert result.verified is True
    assert result.evidence["communication_send_status"] == "PROVIDER_ACCEPTED"
    assert result.evidence["independent_delivery"] is False
