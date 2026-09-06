import json
from datetime import datetime, timezone
from uuid import uuid4

from aegis.calendar import (
    CalendarEvent,
    FixtureCalendarProvider,
    FixtureCalendarWriteProvider,
    GoogleCalendarRestProvider,
    GoogleCalendarWriteProvider,
    calendar_conflicts,
    calendar_events_evidence,
    configured_calendar_provider,
)
from aegis.contracts import ExecutionRequest, Principal
from aegis.documents import Document, FixtureDocumentProvider, documents_evidence
from aegis.reference_packs import (
    CalendarCancelExecutor,
    CalendarCancelVerifier,
    prepare_reference_action,
    reference_bundles,
)
from aegis.reference_runtime import default_runtime_registry


def test_fixture_calendar_provider_returns_bounded_provider_neutral_events() -> None:
    event = CalendarEvent("event-1", "Game night", datetime(2026, 9, 6, tzinfo=timezone.utc))
    provider = FixtureCalendarProvider((event,))
    evidence = calendar_events_evidence(provider.list_events())
    assert evidence["source"] == "external_calendar_fixture"
    assert evidence["events"][0]["title"] == "Game night"


def test_calendar_conflicts_returns_only_overlapping_timed_pairs() -> None:
    events = (
        CalendarEvent(
            "b",
            "Second",
            datetime(2026, 9, 6, 10, tzinfo=timezone.utc),
            datetime(2026, 9, 6, 11, tzinfo=timezone.utc),
        ),
        CalendarEvent(
            "a",
            "First",
            datetime(2026, 9, 6, 10, 30, tzinfo=timezone.utc),
            datetime(2026, 9, 6, 11, 30, tzinfo=timezone.utc),
        ),
        CalendarEvent("open", "Open ended", datetime(2026, 9, 6, 12, tzinfo=timezone.utc)),
    )
    assert calendar_conflicts(events) == (
        {
            "event_id": "a",
            "event_title": "First",
            "conflicts_with": "b",
            "conflicting_title": "Second",
        },
    )


def test_configured_calendar_provider_loads_bounded_snapshot(monkeypatch) -> None:
    monkeypatch.setenv(
        "AEGIS_CALENDAR_FIXTURE_JSON",
        json.dumps(
            [
                {
                    "event_id": "event-2",
                    "title": "Doctor",
                    "starts_at": "2026-09-07T14:00:00+00:00",
                }
            ]
        ),
    )
    provider = configured_calendar_provider()
    events = provider.list_events()
    assert events[0].event_id == "event-2"
    assert events[0].source == "configured_calendar"


def test_configured_calendar_provider_rejects_unbounded_snapshot(monkeypatch) -> None:
    monkeypatch.setenv("AEGIS_CALENDAR_FIXTURE_JSON", json.dumps([{}] * 51))
    try:
        configured_calendar_provider()
    except ValueError as exc:
        assert "at most 50" in str(exc)
    else:
        raise AssertionError("unbounded calendar snapshot was accepted")


def test_google_calendar_provider_reads_bounded_events(monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return (
                b'{"items":[{"id":"g-1","summary":"Dentist","start":'
                b'{"dateTime":"2026-09-08T15:00:00+00:00"}}]}'
            )

    seen = []
    monkeypatch.setattr(
        "aegis.calendar.urllib.request.urlopen",
        lambda request, timeout: seen.append((request, timeout)) or Response(),
    )
    events = GoogleCalendarRestProvider("token").list_events()
    assert events[0].title == "Dentist"
    assert events[0].source == "google_calendar"
    assert "singleEvents=true" in seen[0][0].full_url
    assert seen[0][0].get_header("Authorization") == "Bearer token"


def test_google_calendar_write_creates_then_separately_reads_back(monkeypatch) -> None:
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return json.dumps(self.payload).encode("utf-8")

    item = {
        "id": "g-created",
        "summary": "Dinner",
        "start": {"dateTime": "2026-09-08T19:00:00+00:00"},
        "end": {"dateTime": "2026-09-08T20:00:00+00:00"},
    }
    seen = []
    monkeypatch.setattr(
        "aegis.calendar.urllib.request.urlopen",
        lambda request, timeout: seen.append((request, timeout)) or Response(item),
    )
    provider = GoogleCalendarWriteProvider("token")
    event = CalendarEvent(
        "pending",
        "Dinner",
        datetime(2026, 9, 8, 19, tzinfo=timezone.utc),
        datetime(2026, 9, 8, 20, tzinfo=timezone.utc),
    )
    created = provider.create_event(event, "calendar-1")
    read_back = provider.get_event(created.event_id)
    assert created.event_id == "g-created"
    assert read_back is not None and read_back.title == "Dinner"
    assert seen[0][0].method == "POST"
    assert seen[1][0].get_method() == "GET"
    assert b"calendar-1" in seen[0][0].data


def test_fixture_documents_preserve_authorized_read_provenance() -> None:
    evidence = documents_evidence(
        FixtureDocumentProvider((Document("doc-1", "Notes", "Keep this scoped."),)).list_documents()
    )
    assert evidence["source"] == "authorized_document_fixture"
    assert evidence["documents"][0]["text"] == "Keep this scoped."


def test_calendar_pack_uses_generic_verified_read_runtime() -> None:
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "calendar.events.list"
    )
    runtime = default_runtime_registry(lambda: None).resolve(
        card, None, Principal(id="alice", vault_id="vault")
    )
    observation = runtime.executor.execute(
        ExecutionRequest(
            objective_id=uuid4(),
            action_id=uuid4(),
            action=card.action,
            idempotency_key="calendar-1",
        )
    )
    verification = runtime.verifier.verify(observation, card.action.verification)
    assert observation.command_succeeded is True
    assert verification.verified is True


def test_calendar_create_reads_back_the_provider_event_and_is_idempotent() -> None:
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "calendar.events.create"
    )
    runtime = default_runtime_registry(lambda: None).resolve(
        card, None, Principal(id="alice", vault_id="vault")
    )
    action = card.action.model_copy(
        update={
            "arguments": {
                "title": "Dinner",
                "starts_at": "2026-09-07T19:00:00+00:00",
                "ends_at": "2026-09-07T20:00:00+00:00",
            }
        }
    )
    request = ExecutionRequest(
        objective_id=uuid4(), action_id=uuid4(), action=action, idempotency_key="calendar-1"
    )
    prepared = runtime.prepare(
        action, Principal(id="alice", vault_id="vault"), request.objective_id
    )
    observation = runtime.executor.execute(request.model_copy(update={"action": prepared}))
    verification = runtime.verifier.verify(observation, prepared.verification)
    assert verification.verified is True
    assert verification.evidence["provider_readback"] is True
    assert isinstance(runtime.executor.provider, FixtureCalendarWriteProvider)


def test_calendar_cancel_reads_back_provider_absence_and_is_idempotent() -> None:
    provider = FixtureCalendarWriteProvider()
    created = provider.create_event(
        CalendarEvent("pending", "Dinner", datetime(2026, 9, 7, tzinfo=timezone.utc)),
        "cancel-calendar-1",
    )
    card = next(
        card
        for bundle in reference_bundles()
        for card in bundle.cards
        if card.action.action_id == "calendar.events.cancel"
    )
    principal = Principal(id="alice", vault_id="vault")
    objective_id = uuid4()
    action = card.action.model_copy(update={"arguments": {"event_id": created.event_id}})
    prepared = prepare_reference_action(action, principal, objective_id)
    executor = CalendarCancelExecutor(provider)
    verifier = CalendarCancelVerifier(provider)
    request = ExecutionRequest(
        objective_id=objective_id,
        action_id=uuid4(),
        action=prepared,
        idempotency_key="cancel-calendar-1",
    )
    observation = executor.execute(request)
    result = verifier.verify(observation, prepared.verification)
    replay = executor.execute(request)
    assert result.verified is True
    assert result.evidence["provider_readback_absent"] is True
    assert replay.command_succeeded is True
    assert provider.get_event(created.event_id) is None
