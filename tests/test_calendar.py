from datetime import datetime, timezone
from uuid import uuid4

from aegis.calendar import CalendarEvent, FixtureCalendarProvider, calendar_events_evidence
from aegis.contracts import ExecutionRequest, Principal
from aegis.reference_packs import reference_bundles
from aegis.reference_runtime import default_runtime_registry


def test_fixture_calendar_provider_returns_bounded_provider_neutral_events() -> None:
    event = CalendarEvent("event-1", "Game night", datetime(2026, 9, 6, tzinfo=timezone.utc))
    provider = FixtureCalendarProvider((event,))
    evidence = calendar_events_evidence(provider.list_events())
    assert evidence["source"] == "external_calendar_fixture"
    assert evidence["events"][0]["title"] == "Game night"


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
