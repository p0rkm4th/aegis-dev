"""Provider-neutral, read-only external calendar contract.

The first slice deliberately uses deterministic fixtures.  A live provider can
implement the same port later without changing Core authorization or UI code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class CalendarEvent:
    event_id: str
    title: str
    starts_at: datetime
    ends_at: datetime | None = None
    source: str = "external"


class CalendarProvider(Protocol):
    def list_events(self) -> tuple[CalendarEvent, ...]: ...


@dataclass(frozen=True)
class FixtureCalendarProvider:
    """Deterministic provider for local acceptance and contract tests."""

    events: tuple[CalendarEvent, ...] = ()

    def list_events(self) -> tuple[CalendarEvent, ...]:
        return self.events


def configured_calendar_provider() -> CalendarProvider:
    """Load a bounded provider-neutral calendar feed for owner acceptance.

    The environment contract is intentionally data-only; it is a fixture or
    adapter-fed snapshot, not an authorization mechanism or calendar server.
    A live provider can implement ``CalendarProvider`` without changing Core.
    """

    raw = os.environ.get("AEGIS_CALENDAR_FIXTURE_JSON")
    if not raw:
        return FixtureCalendarProvider()
    payload = json.loads(raw)
    if not isinstance(payload, list) or len(payload) > 50:
        raise ValueError("calendar fixture must be a list of at most 50 events")
    events: list[CalendarEvent] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("calendar fixture event must be an object")
        event_id = item.get("event_id")
        title = item.get("title")
        starts_at = item.get("starts_at")
        ends_at = item.get("ends_at")
        if (
            not isinstance(event_id, str)
            or not event_id
            or len(event_id) > 200
            or not isinstance(title, str)
            or not title
            or len(title) > 2_000
            or not isinstance(starts_at, str)
            or len(starts_at) > 100
            or (ends_at is not None and not isinstance(ends_at, str))
        ):
            raise ValueError("calendar fixture event fields are invalid")
        try:
            start = datetime.fromisoformat(starts_at)
            end = datetime.fromisoformat(ends_at) if ends_at is not None else None
        except ValueError as exc:
            raise ValueError("calendar fixture timestamps must be ISO-8601") from exc
        events.append(CalendarEvent(event_id, title, start, end, source="configured_calendar"))
    return FixtureCalendarProvider(tuple(events))


def calendar_events_evidence(events: tuple[CalendarEvent, ...]) -> dict[str, object]:
    """Serialize bounded provider output without exposing provider internals."""

    return {
        "source": "external_calendar_fixture",
        "events": [
            {
                "event_id": event.event_id,
                "title": event.title,
                "starts_at": event.starts_at.isoformat(),
                "ends_at": event.ends_at.isoformat() if event.ends_at else None,
                "source": event.source,
            }
            for event in events[:50]
        ],
    }
