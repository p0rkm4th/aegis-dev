"""Provider-neutral, read-only external calendar contract.

The first slice deliberately uses deterministic fixtures.  A live provider can
implement the same port later without changing Core authorization or UI code.
"""

from __future__ import annotations

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
