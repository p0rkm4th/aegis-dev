"""Provider-neutral, read-only external calendar contract.

The first slice deliberately uses deterministic fixtures.  A live provider can
implement the same port later without changing Core authorization or UI code.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
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
class GoogleCalendarRestProvider:
    """Bounded Google Calendar ``events.list`` read adapter."""

    access_token: str
    calendar_id: str = "primary"
    endpoint: str = "https://www.googleapis.com/calendar/v3"
    timeout_seconds: float = 5.0

    def list_events(self) -> tuple[CalendarEvent, ...]:
        parsed = urllib.parse.urlsplit(self.endpoint)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Google Calendar endpoint must use HTTPS")
        if not self.access_token or len(self.access_token) > 8_000:
            raise ValueError("Google Calendar access token is invalid")
        path = "/calendars/{}/events".format(urllib.parse.quote(self.calendar_id, safe=""))
        query = urllib.parse.urlencode(
            {"maxResults": 50, "orderBy": "startTime", "singleEvents": "true"}
        )
        request = urllib.request.Request(
            urllib.parse.urlunsplit(
                (parsed.scheme, parsed.netloc, parsed.path.rstrip("/") + path, query, "")
            ),
            headers={"Authorization": f"Bearer {self.access_token}", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read(1_000_001))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError("Google Calendar read failed") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise RuntimeError("Google Calendar response was invalid")
        events: list[CalendarEvent] = []
        for item in payload["items"][:50]:
            if not isinstance(item, dict):
                continue
            event_id = item.get("id")
            title = item.get("summary") or "(untitled event)"
            start = item.get("start")
            end = item.get("end")
            if (
                not isinstance(event_id, str)
                or not isinstance(title, str)
                or not isinstance(start, dict)
            ):
                continue
            starts_at = _google_timestamp(start)
            ends_at = _google_timestamp(end) if isinstance(end, dict) else None
            if starts_at is None:
                continue
            events.append(
                CalendarEvent(event_id[:200], title[:2_000], starts_at, ends_at, "google_calendar")
            )
        return tuple(events)


def _google_timestamp(value: dict[str, object]) -> datetime | None:
    timestamp = value.get("dateTime") or value.get("date")
    if not isinstance(timestamp, str):
        return None
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    return parsed


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

    token = os.environ.get("AEGIS_GOOGLE_CALENDAR_TOKEN")
    if token:
        return GoogleCalendarRestProvider(
            token,
            calendar_id=os.environ.get("AEGIS_GOOGLE_CALENDAR_ID", "primary"),
            endpoint=os.environ.get(
                "AEGIS_GOOGLE_CALENDAR_ENDPOINT",
                "https://www.googleapis.com/calendar/v3",
            ),
        )
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
