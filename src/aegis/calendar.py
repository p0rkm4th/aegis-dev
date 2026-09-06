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


class CalendarWriteProvider(Protocol):
    def create_event(self, event: CalendarEvent, idempotency_key: str) -> CalendarEvent: ...

    def update_event(self, event_id: str, event: CalendarEvent) -> CalendarEvent: ...

    def get_event(self, event_id: str) -> CalendarEvent | None: ...

    def delete_event(self, event_id: str) -> None: ...

    def find_event_by_idempotency_key(self, idempotency_key: str) -> CalendarEvent | None: ...


@dataclass(frozen=True)
class GoogleCalendarWriteProvider:
    """Bounded Google Calendar create/readback adapter."""

    access_token: str
    calendar_id: str = "primary"
    endpoint: str = "https://www.googleapis.com/calendar/v3"
    timeout_seconds: float = 5.0

    def _url(self, event_id: str | None = None) -> str:
        parsed = urllib.parse.urlsplit(self.endpoint)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Google Calendar endpoint must use HTTPS")
        if not self.access_token or len(self.access_token) > 8_000:
            raise ValueError("Google Calendar access token is invalid")
        path = "/calendars/{}/events".format(urllib.parse.quote(self.calendar_id, safe=""))
        if event_id is not None:
            path += "/" + urllib.parse.quote(event_id, safe="")
        return urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path.rstrip("/") + path, "", "")
        )

    def create_event(self, event: CalendarEvent, idempotency_key: str) -> CalendarEvent:
        payload = {
            "summary": event.title,
            "start": {"dateTime": event.starts_at.isoformat()},
            "end": {"dateTime": (event.ends_at or event.starts_at).isoformat()},
            "extendedProperties": {"private": {"aegis_idempotency_key": idempotency_key}},
        }
        request = urllib.request.Request(
            self._url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                item = json.loads(response.read(1_000_001))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError("Google Calendar create failed") from exc
        parsed = _google_event(item)
        if parsed is None:
            raise RuntimeError("Google Calendar create response was invalid")
        return parsed

    def update_event(self, event_id: str, event: CalendarEvent) -> CalendarEvent:
        payload = {
            "summary": event.title,
            "start": {"dateTime": event.starts_at.isoformat()},
            "end": {"dateTime": (event.ends_at or event.starts_at).isoformat()},
        }
        request = urllib.request.Request(
            self._url(event_id),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="PATCH",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                item = json.loads(response.read(1_000_001))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError("Google Calendar update failed") from exc
        parsed = _google_event(item)
        if parsed is None:
            raise RuntimeError("Google Calendar update response was invalid")
        return parsed

    def get_event(self, event_id: str) -> CalendarEvent | None:
        request = urllib.request.Request(
            self._url(event_id),
            headers={"Authorization": f"Bearer {self.access_token}", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                item = json.loads(response.read(1_000_001))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise RuntimeError("Google Calendar readback failed") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError("Google Calendar readback failed") from exc
        return _google_event(item)

    def find_event_by_idempotency_key(self, idempotency_key: str) -> CalendarEvent | None:
        if not idempotency_key or len(idempotency_key) > 200:
            raise ValueError("calendar idempotency key is invalid")
        parsed = urllib.parse.urlsplit(self._url())
        query = urllib.parse.urlencode(
            [("privateExtendedProperty", f"aegis_idempotency_key={idempotency_key}")]
        )
        request = urllib.request.Request(
            urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, "")),
            headers={"Authorization": f"Bearer {self.access_token}", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read(1_000_001))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError("Google Calendar idempotency lookup failed") from exc
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list) or len(items) > 50:
            raise RuntimeError("Google Calendar idempotency response was invalid")
        return next(
            (
                event
                for item in items
                if isinstance(item, dict)
                for event in [_google_event(item)]
                if event
            ),
            None,
        )

    def delete_event(self, event_id: str) -> None:
        request = urllib.request.Request(
            self._url(event_id),
            headers={"Authorization": f"Bearer {self.access_token}", "Accept": "application/json"},
            method="DELETE",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds):
                pass
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return
            raise RuntimeError("Google Calendar delete failed") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError("Google Calendar delete failed") from exc


@dataclass
class FixtureCalendarWriteProvider:
    """Deterministic create/readback provider; live credentials remain separate."""

    events: dict[str, CalendarEvent]

    def __init__(self) -> None:
        self.events = {}

    def create_event(self, event: CalendarEvent, idempotency_key: str) -> CalendarEvent:
        event_id = f"fixture:{idempotency_key}"
        existing = self.events.get(event_id)
        if existing is not None:
            return existing
        created = CalendarEvent(
            event_id=event_id,
            title=event.title,
            starts_at=event.starts_at,
            ends_at=event.ends_at,
            source="fixture_calendar",
        )
        self.events[event_id] = created
        return created

    def get_event(self, event_id: str) -> CalendarEvent | None:
        return self.events.get(event_id)

    def find_event_by_idempotency_key(self, idempotency_key: str) -> CalendarEvent | None:
        return self.events.get(f"fixture:{idempotency_key}")

    def update_event(self, event_id: str, event: CalendarEvent) -> CalendarEvent:
        if event_id not in self.events:
            raise RuntimeError("calendar event does not exist")
        updated = CalendarEvent(
            event_id=event_id,
            title=event.title,
            starts_at=event.starts_at,
            ends_at=event.ends_at,
            source="fixture_calendar",
        )
        self.events[event_id] = updated
        return updated

    def delete_event(self, event_id: str) -> None:
        self.events.pop(event_id, None)


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


def configured_calendar_write_provider() -> CalendarWriteProvider:
    token = os.environ.get("AEGIS_GOOGLE_CALENDAR_TOKEN")
    if token:
        return GoogleCalendarWriteProvider(
            token,
            calendar_id=os.environ.get("AEGIS_GOOGLE_CALENDAR_ID", "primary"),
            endpoint=os.environ.get(
                "AEGIS_GOOGLE_CALENDAR_ENDPOINT", "https://www.googleapis.com/calendar/v3"
            ),
        )
    return FixtureCalendarWriteProvider()


def _google_event(item: object) -> CalendarEvent | None:
    if not isinstance(item, dict):
        return None
    event_id = item.get("id")
    title = item.get("summary") or "(untitled event)"
    start, end = item.get("start"), item.get("end")
    if not isinstance(event_id, str) or not isinstance(title, str) or not isinstance(start, dict):
        return None
    starts_at = _google_timestamp(start)
    ends_at = _google_timestamp(end) if isinstance(end, dict) else None
    if starts_at is None:
        return None
    return CalendarEvent(event_id[:200], title[:2_000], starts_at, ends_at, "google_calendar")


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


def calendar_conflicts(events: tuple[CalendarEvent, ...]) -> tuple[dict[str, str], ...]:
    """Return bounded pairwise overlaps for events with explicit end times."""

    conflicts: list[dict[str, str]] = []
    bounded = tuple((event, event.ends_at) for event in events[:50] if event.ends_at is not None)
    for index, (event, event_end) in enumerate(bounded):
        for other, other_end in bounded[index + 1 :]:
            if max(event.starts_at, other.starts_at) >= min(event_end, other_end):
                continue
            first, second = sorted((event, other), key=lambda item: item.event_id)
            conflicts.append(
                {
                    "event_id": first.event_id,
                    "event_title": first.title,
                    "conflicts_with": second.event_id,
                    "conflicting_title": second.title,
                }
            )
    return tuple(conflicts[:50])


def calendar_snapshot_content(events: tuple[CalendarEvent, ...]) -> str:
    """Create stable, bounded report bytes for a Calendar → Workspace snapshot."""

    return (
        "# Authorized calendar snapshot\n\n"
        + json.dumps(calendar_events_evidence(events)["events"], indent=2)
        + "\n"
    )
