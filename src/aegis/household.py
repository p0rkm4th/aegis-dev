"""Shared household Space state with membership checks below the model layer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Any, Protocol, cast
from uuid import uuid4

from .contracts import (
    Context,
    ExecutionRequest,
    IntentFrame,
    ObjectiveState,
    Observation,
    Principal,
    Result,
    VerificationContract,
    VerificationResult,
)
from .read_applicability import ReadApplicability, assess_read_applicability
from .utterance import (
    is_correction_request,
    is_mutation_request,
    is_question_request,
    strip_correction_prefix,
)

_WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


@dataclass(frozen=True)
class Chore:
    chore_id: str
    title: str
    assignee_id: str
    completed: bool = False


@dataclass(frozen=True)
class HouseholdEvent:
    event_id: str
    title: str
    starts_at: datetime


@dataclass(frozen=True)
class HouseholdObligation:
    obligation_id: str
    title: str
    amount: int
    responsible_id: str
    settled: bool = False


@dataclass(frozen=True)
class GroceryItem:
    """Stable-ID shopping item; unknown quantities remain unknown."""

    grocery_id: str
    display_name: str
    normalized_key: str
    desired_quantity: float | None = None
    unit: str | None = None
    state: str = "needed"
    pantry_item_id: str | None = None
    updated_at: datetime | None = None
    version: int = 0


@dataclass(frozen=True)
class PantryItem:
    """Stable-ID pantry state with deliberately nullable facts."""

    item_id: str
    display_name: str
    normalized_key: str
    quantity: float | None = None
    unit: str | None = None
    storage_location: str | None = None
    best_by: str | None = None
    minimum_quantity: float | None = None
    updated_at: datetime | None = None
    version: int = 0


def normalize_food_key(display_name: str) -> str:
    """Create a deterministic matching key without inventing food facts."""

    return " ".join(display_name.casefold().split())


def migrate_grocery_strings(items: list[str]) -> dict[str, GroceryItem]:
    """Idempotently migrate legacy string groceries to stable-ID records."""

    migrated: dict[str, GroceryItem] = {}
    for index, raw in enumerate(items):
        display_name = str(raw).strip()
        if not display_name:
            continue
        normalized = normalize_food_key(display_name)
        digest = sha256(f"{normalized}\0{index}".encode()).hexdigest()[:16]
        grocery_id = f"grocery-{digest}"
        migrated[grocery_id] = GroceryItem(grocery_id, display_name, normalized)
    return migrated


class HouseholdStateConnection(Protocol):
    def execute(self, query: str, params: tuple[object, ...] = ()) -> Any: ...

    def commit(self) -> None: ...


class PostgresHouseholdStore:
    """Persist shared Space state while leaving membership to policy storage."""

    def __init__(self, connection: HouseholdStateConnection) -> None:
        self.connection = connection

    def save(self, space: HouseholdSpace) -> None:
        if not space.space_id:
            raise ValueError("household state requires a Space")
        payload = {
            "groceries": list(space.groceries),
            "grocery_items": [
                {
                    "grocery_id": item.grocery_id,
                    "display_name": item.display_name,
                    "normalized_key": item.normalized_key,
                    "desired_quantity": item.desired_quantity,
                    "unit": item.unit,
                    "state": item.state,
                    "pantry_item_id": item.pantry_item_id,
                    "updated_at": item.updated_at.isoformat() if item.updated_at else None,
                    "version": item.version,
                }
                for item in space.grocery_items.values()
            ],
            "pantry_items": [
                {
                    "item_id": item.item_id,
                    "display_name": item.display_name,
                    "normalized_key": item.normalized_key,
                    "quantity": item.quantity,
                    "unit": item.unit,
                    "storage_location": item.storage_location,
                    "best_by": item.best_by,
                    "minimum_quantity": item.minimum_quantity,
                    "updated_at": item.updated_at.isoformat() if item.updated_at else None,
                    "version": item.version,
                }
                for item in space.pantry_items.values()
            ],
            "grocery_mutations": space.grocery_mutations,
            "chores": [
                {
                    "chore_id": chore.chore_id,
                    "title": chore.title,
                    "assignee_id": chore.assignee_id,
                    "completed": chore.completed,
                }
                for chore in space.chores.values()
            ],
            "events": [
                {
                    "event_id": event.event_id,
                    "title": event.title,
                    "starts_at": event.starts_at.isoformat(),
                }
                for event in space.events.values()
            ],
            "obligations": [
                {
                    "obligation_id": obligation.obligation_id,
                    "title": obligation.title,
                    "amount": obligation.amount,
                    "responsible_id": obligation.responsible_id,
                    "settled": obligation.settled,
                }
                for obligation in space.obligations.values()
            ],
            "chore_mutations": space.chore_mutations,
            "event_mutations": space.event_mutations,
        }
        self.connection.execute(
            "INSERT INTO household_spaces (space_id, payload) VALUES (%s, %s) "
            "ON CONFLICT (space_id) DO UPDATE SET payload = EXCLUDED.payload, "
            "updated_at = now()",
            (space.space_id, json.dumps(payload, sort_keys=True)),
        )
        self.connection.commit()

    def load(self, space_id: str, members: set[str]) -> HouseholdSpace:
        if not space_id:
            raise ValueError("household state requires a Space")
        row = self.connection.execute(
            "SELECT payload FROM household_spaces WHERE space_id = %s", (space_id,)
        ).fetchone()
        if row is None:
            return HouseholdSpace(space_id, set(members))
        payload = row[0] if isinstance(row[0], dict) else json.loads(str(row[0]))
        chores = {
            str(item["chore_id"]): Chore(
                str(item["chore_id"]),
                str(item["title"]),
                str(item["assignee_id"]),
                bool(item.get("completed", False)),
            )
            for item in payload.get("chores", [])
        }
        events = {
            str(item["event_id"]): HouseholdEvent(
                str(item["event_id"]),
                str(item["title"]),
                datetime.fromisoformat(str(item["starts_at"])),
            )
            for item in payload.get("events", [])
        }
        obligations = {
            str(item["obligation_id"]): HouseholdObligation(
                str(item["obligation_id"]),
                str(item["title"]),
                int(item["amount"]),
                str(item["responsible_id"]),
                bool(item.get("settled", False)),
            )
            for item in payload.get("obligations", [])
        }
        groceries = [str(item) for item in payload.get("groceries", [])]
        grocery_items = {
            str(item["grocery_id"]): GroceryItem(
                grocery_id=str(item["grocery_id"]),
                display_name=str(item["display_name"]),
                normalized_key=str(item["normalized_key"]),
                desired_quantity=(
                    float(item["desired_quantity"])
                    if item.get("desired_quantity") is not None
                    else None
                ),
                unit=str(item["unit"]) if item.get("unit") is not None else None,
                state=str(item.get("state", "needed")),
                pantry_item_id=(
                    str(item["pantry_item_id"]) if item.get("pantry_item_id") is not None else None
                ),
                updated_at=(
                    datetime.fromisoformat(str(item["updated_at"]))
                    if item.get("updated_at")
                    else None
                ),
                version=int(item.get("version", 0)),
            )
            for item in payload.get("grocery_items", [])
        }
        if not grocery_items:
            grocery_items = migrate_grocery_strings(groceries)
        pantry_items = {
            str(item["item_id"]): PantryItem(
                item_id=str(item["item_id"]),
                display_name=str(item["display_name"]),
                normalized_key=str(item["normalized_key"]),
                quantity=float(item["quantity"]) if item.get("quantity") is not None else None,
                unit=str(item["unit"]) if item.get("unit") is not None else None,
                storage_location=(
                    str(item["storage_location"])
                    if item.get("storage_location") is not None
                    else None
                ),
                best_by=str(item["best_by"]) if item.get("best_by") is not None else None,
                minimum_quantity=(
                    float(item["minimum_quantity"])
                    if item.get("minimum_quantity") is not None
                    else None
                ),
                updated_at=(
                    datetime.fromisoformat(str(item["updated_at"]))
                    if item.get("updated_at")
                    else None
                ),
                version=int(item.get("version", 0)),
            )
            for item in payload.get("pantry_items", [])
        }
        grocery_mutations = {
            str(key): str(item) for key, item in payload.get("grocery_mutations", {}).items()
        }
        chore_mutations = {
            str(key): str(item) for key, item in payload.get("chore_mutations", {}).items()
        }
        event_mutations = {
            str(key): str(item) for key, item in payload.get("event_mutations", {}).items()
        }
        return HouseholdSpace(
            space_id,
            set(members),
            groceries,
            chores,
            events,
            obligations,
            grocery_mutations,
            chore_mutations,
            event_mutations,
            grocery_items,
            pantry_items,
        )

    def add_grocery(self, principal: Principal, item: str, idempotency_key: str) -> None:
        """Apply one authorized, replay-safe grocery mutation to canonical state."""
        if not idempotency_key:
            raise ValueError("grocery mutation requires an idempotency key")
        space_id = self._space_for(principal)
        members = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT principal_id FROM space_memberships WHERE space_id = %s AND active = TRUE",
                (space_id,),
            ).fetchall()
        }
        space = self.load(space_id, members)
        space.add_grocery(principal, item, idempotency_key)
        self.save(space)

    def grocery_recorded(self, principal: Principal, item: str, idempotency_key: str) -> bool:
        """Read canonical grocery mutation evidence after membership validation."""
        space_id = self._space_for(principal)
        members = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT principal_id FROM space_memberships WHERE space_id = %s AND active = TRUE",
                (space_id,),
            ).fetchall()
        }
        space = self.load(space_id, members)
        return space.grocery_mutations.get(idempotency_key) == item

    def list_groceries(self, principal: Principal) -> tuple[str, ...]:
        """Read canonical groceries only for an active Space member."""
        space_id = self._space_for(principal)
        members = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT principal_id FROM space_memberships WHERE space_id = %s AND active = TRUE",
                (space_id,),
            ).fetchall()
        }
        return tuple(self.load(space_id, members).groceries)

    def list_grocery_items(self, principal: Principal) -> tuple[GroceryItem, ...]:
        """Read stable-ID grocery records for an active Space member."""
        space_id = self._space_for(principal)
        members = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT principal_id FROM space_memberships WHERE space_id = %s AND active = TRUE",
                (space_id,),
            ).fetchall()
        }
        return tuple(self.load(space_id, members).grocery_items.values())

    def list_pantry_items(self, principal: Principal) -> tuple[PantryItem, ...]:
        """Read canonical Pantry records for an active Space member."""
        space_id = self._space_for(principal)
        members = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT principal_id FROM space_memberships WHERE space_id = %s AND active = TRUE",
                (space_id,),
            ).fetchall()
        }
        return tuple(self.load(space_id, members).pantry_items.values())

    def add_pantry_item(self, principal: Principal, item: PantryItem) -> PantryItem:
        """Persist one stable-ID Pantry record after membership recheck."""
        space_id = self._space_for(principal)
        members = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT principal_id FROM space_memberships WHERE space_id = %s AND active = TRUE",
                (space_id,),
            ).fetchall()
        }
        space = self.load(space_id, members)
        result = space.add_pantry(principal, item)
        self.save(space)
        return result

    def read_snapshot(self, principal: Principal) -> dict[str, object]:
        """Read shared household state after rechecking current membership."""
        space_id = self._space_for(principal)
        members = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT principal_id FROM space_memberships WHERE space_id = %s AND active = TRUE",
                (space_id,),
            ).fetchall()
        }
        return self.load(space_id, members).snapshot(principal)

    def add_chore(
        self,
        principal: Principal,
        title: str,
        assignee_id: str,
        idempotency_key: str,
    ) -> Chore:
        space_id = self._space_for(principal)
        members = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT principal_id FROM space_memberships WHERE space_id = %s AND active = TRUE",
                (space_id,),
            ).fetchall()
        }
        space = self.load(space_id, members)
        chore = space.add_chore(
            principal, Chore(f"chore-{uuid4().hex}", title, assignee_id), idempotency_key
        )
        self.save(space)
        return chore

    def complete_chore(self, principal: Principal, chore_id: str) -> Chore:
        space_id = self._space_for(principal)
        members = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT principal_id FROM space_memberships WHERE space_id = %s AND active = TRUE",
                (space_id,),
            ).fetchall()
        }
        space = self.load(space_id, members)
        chore = space.complete_chore(principal, chore_id)
        self.save(space)
        return chore

    def add_event(
        self,
        principal: Principal,
        title: str,
        starts_at: datetime,
        idempotency_key: str,
    ) -> HouseholdEvent:
        space_id = self._space_for(principal)
        members = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT principal_id FROM space_memberships WHERE space_id = %s AND active = TRUE",
                (space_id,),
            ).fetchall()
        }
        space = self.load(space_id, members)
        event = space.add_event(
            principal,
            HouseholdEvent(f"event-{uuid4().hex}", title, starts_at),
            idempotency_key,
        )
        self.save(space)
        return event

    def _space_for(self, principal: Principal) -> str:
        if not principal.space_ids:
            raise PermissionError("household operation requires an explicit Space")
        space_id = principal.space_ids[0]
        row = self.connection.execute(
            "SELECT 1 FROM space_memberships "
            "WHERE principal_id = %s AND space_id = %s AND active = TRUE",
            (principal.id, space_id),
        ).fetchone()
        if row is None:
            raise PermissionError("principal is not an active Space member")
        return space_id


@dataclass
class HouseholdSpace:
    space_id: str
    members: set[str]
    groceries: list[str] = field(default_factory=list)
    chores: dict[str, Chore] = field(default_factory=dict)
    events: dict[str, HouseholdEvent] = field(default_factory=dict)
    obligations: dict[str, HouseholdObligation] = field(default_factory=dict)
    grocery_mutations: dict[str, str] = field(default_factory=dict)
    chore_mutations: dict[str, str] = field(default_factory=dict)
    event_mutations: dict[str, str] = field(default_factory=dict)
    grocery_items: dict[str, GroceryItem] = field(default_factory=dict)
    pantry_items: dict[str, PantryItem] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.grocery_items and self.groceries:
            self.grocery_items = migrate_grocery_strings(self.groceries)

    def _require_member(self, principal: Principal) -> None:
        if principal.id not in self.members or self.space_id not in principal.space_ids:
            raise PermissionError("principal is not an active member of this Space")

    def add_grocery(
        self, principal: Principal, item: str, idempotency_key: str | None = None
    ) -> None:
        self._require_member(principal)
        if not item.strip():
            raise ValueError("grocery item is required")
        if idempotency_key is not None:
            if self.grocery_mutations.get(idempotency_key) == item:
                return
            if idempotency_key in self.grocery_mutations:
                raise ValueError("grocery idempotency key is bound to another item")
        self.groceries.append(item)
        migrated = migrate_grocery_strings(self.groceries)
        self.grocery_items.update(migrated)
        if idempotency_key is not None:
            self.grocery_mutations[idempotency_key] = item

    def mark_grocery_purchased(self, principal: Principal, grocery_id: str) -> GroceryItem:
        self._require_member(principal)
        current = self.grocery_items.get(grocery_id)
        if current is None:
            raise KeyError("grocery item is unavailable")
        updated = GroceryItem(
            **{**current.__dict__, "state": "purchased", "version": current.version + 1}
        )
        self.grocery_items[grocery_id] = updated
        return updated

    def remove_grocery(self, principal: Principal, grocery_id: str) -> GroceryItem:
        self._require_member(principal)
        current = self.grocery_items.get(grocery_id)
        if current is None:
            raise KeyError("grocery item is unavailable")
        updated = GroceryItem(
            **{**current.__dict__, "state": "removed", "version": current.version + 1}
        )
        self.grocery_items[grocery_id] = updated
        return updated

    def add_pantry(self, principal: Principal, item: PantryItem) -> PantryItem:
        self._require_member(principal)
        if not item.display_name.strip():
            raise ValueError("pantry item is required")
        current = self.pantry_items.get(item.item_id)
        if current is not None and current != item:
            raise ValueError("pantry item id already exists")
        self.pantry_items[item.item_id] = item
        return item

    def update_pantry(
        self, principal: Principal, item: PantryItem, expected_version: int
    ) -> PantryItem:
        self._require_member(principal)
        current = self.pantry_items.get(item.item_id)
        if current is None:
            raise KeyError("pantry item is unavailable")
        if current.version != expected_version:
            raise ValueError("stale pantry item version")
        updated = PantryItem(**{**item.__dict__, "version": current.version + 1})
        self.pantry_items[item.item_id] = updated
        return updated

    def consume_pantry(
        self, principal: Principal, item_id: str, quantity: float, expected_version: int
    ) -> PantryItem:
        self._require_member(principal)
        if quantity <= 0:
            raise ValueError("consumption quantity must be positive")
        current = self.pantry_items.get(item_id)
        if current is None:
            raise KeyError("pantry item is unavailable")
        if current.version != expected_version:
            raise ValueError("stale pantry item version")
        if current.quantity is None:
            raise ValueError("pantry quantity is unknown")
        updated = PantryItem(
            **{
                **current.__dict__,
                "quantity": max(0, current.quantity - quantity),
                "version": current.version + 1,
            }
        )
        self.pantry_items[item_id] = updated
        return updated

    def add_chore(
        self, principal: Principal, chore: Chore, idempotency_key: str | None = None
    ) -> Chore:
        self._require_member(principal)
        if chore.assignee_id not in self.members:
            raise ValueError("chore assignee is not a Space member")
        if idempotency_key is not None:
            existing_id = self.chore_mutations.get(idempotency_key)
            if existing_id is not None:
                existing = self.chores[existing_id]
                if existing.title != chore.title or existing.assignee_id != chore.assignee_id:
                    raise ValueError("chore idempotency key is bound to different arguments")
                return existing
        self.chores[chore.chore_id] = chore
        if idempotency_key is not None:
            self.chore_mutations[idempotency_key] = chore.chore_id
        return chore

    def complete_chore(self, principal: Principal, chore_id: str) -> Chore:
        self._require_member(principal)
        chore = self.chores.get(chore_id)
        if chore is None:
            raise KeyError("chore is unavailable")
        completed = Chore(chore.chore_id, chore.title, chore.assignee_id, True)
        self.chores[chore_id] = completed
        return completed

    def add_event(
        self, principal: Principal, event: HouseholdEvent, idempotency_key: str | None = None
    ) -> HouseholdEvent:
        self._require_member(principal)
        if idempotency_key is not None:
            existing_id = self.event_mutations.get(idempotency_key)
            if existing_id is not None:
                existing = self.events[existing_id]
                if existing.title != event.title or existing.starts_at != event.starts_at:
                    raise ValueError("event idempotency key is bound to different arguments")
                return existing
        self.events[event.event_id] = event
        if idempotency_key is not None:
            self.event_mutations[idempotency_key] = event.event_id
        return event

    def add_obligation(self, principal: Principal, obligation: HouseholdObligation) -> None:
        self._require_member(principal)
        if obligation.responsible_id not in self.members:
            raise ValueError("obligation owner is not a Space member")
        self.obligations[obligation.obligation_id] = obligation

    def snapshot(self, principal: Principal) -> dict[str, object]:
        self._require_member(principal)
        return {
            "space_id": self.space_id,
            "groceries": tuple(self.groceries),
            "grocery_items": tuple(self.grocery_items.values()),
            "pantry_items": tuple(self.pantry_items.values()),
            "chores": tuple(self.chores.values()),
            "events": tuple(self.events.values()),
            "obligations": tuple(self.obligations.values()),
        }


class HouseholdReadFastPath:
    """Deterministic, allowlisted reads over shared household state."""

    _TRIGGERS = (
        "household",
        "chore",
        "chores",
        "inspection",
        "utility",
        "utilities",
        "rent",
        "obligation",
        "obligations",
        "event",
        "events",
        "calendar",
        "scheduled",
        "appointment",
        "schedule",
        "appointments",
    )
    _READ_PREFIXES = (
        "what",
        "show",
        "list",
        "which",
        "see",
        "display",
        "give me",
        "is",
        "are",
        "when",
        "where",
        "who",
    )

    def __init__(self, snapshot: dict[str, object]) -> None:
        self.snapshot = snapshot

    @classmethod
    def matches(cls, utterance: str) -> bool:
        text = strip_correction_prefix(utterance).casefold()
        # Preserve an explicit event/household topic when it is introduced as
        # a conversational continuation after another authorized domain.
        text = re.sub(r"^(?:and|but)\s+", "", text)
        text = re.sub(r"^(?:can you|could you|please) tell me about\s+", "what is ", text)
        text = re.sub(r"^(?:can you|could you|please) tell me what\s+", "what ", text)
        if is_mutation_request(text):
            return False
        collection_correction = is_correction_request(utterance) and text.strip(".!?") in {
            "just chores",
            "just the chores",
            "only chores",
            "only the chores",
            "chores only",
        }
        if any(term in text for term in ("task", "tasks", "todo", "to-do", "grocery")):
            return False
        task_objective = any(
            term in text for term in ("need to", "take care of", "prepare", "before")
        )
        explicit_event = any(
            term in text
            for term in (
                "event",
                "events",
                "calendar",
                "scheduled",
                "schedule",
                "appointment",
            )
        )
        implicit_event = "happening" in text and any(
            term in text
            for term in (
                "today",
                "tomorrow",
                "this weekend",
                "next weekend",
                "this week",
                "next week",
                "this month",
                "next month",
            )
        )
        implicit_schedule = (
            any(
                term in text
                for term in ("going on", "plans", "planned", "coming up", "meeting", "meetings")
            )
            and any(
                term in text
                for term in (
                    "today",
                    "tomorrow",
                    "this weekend",
                    "next weekend",
                    "this week",
                    "next week",
                    "this month",
                    "next month",
                )
            )
            and is_question_request(text)
        )
        if task_objective and not explicit_event:
            return False
        return (
            collection_correction
            or implicit_event
            or implicit_schedule
            or any(re.search(rf"\b{re.escape(trigger)}\b", text) for trigger in cls._TRIGGERS)
            and (text.startswith(cls._READ_PREFIXES) or text in cls._TRIGGERS)
        )

    def resolve(self, intent: IntentFrame) -> Result | None:
        if not self.matches(intent.utterance):
            return None
        text = intent.utterance.casefold()
        normalized = " ".join(strip_correction_prefix(intent.utterance).casefold().split()).strip(
            ".!?"
        )
        normalized = re.sub(r"^(?:and|but)\s+", "", normalized)
        normalized = re.sub(
            r"^(?:can you|could you|please) tell me about\s+", "what is ", normalized
        )
        normalized = re.sub(r"^(?:can you|could you|please) tell me what\s+", "what ", normalized)
        if (
            re.fullmatch(
                r"(?:what is|what's|show me|which is) the "
                r"(first|second|third|fourth|last) chore|"
                r"which chore is (first|second|third|fourth|last)",
                normalized,
            )
            is not None
        ):
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message=(
                    "I cannot determine which chore comes first because chores have "
                    "no canonical priority order."
                ),
                correlation_id=intent.correlation_id,
            )
        latest = normalized in {
            "which event is latest",
            "what event is latest",
            "what is the latest event",
            "which is the latest event",
        }
        earliest = normalized in {
            "which event is earliest",
            "what event is earliest",
            "what is the earliest event",
            "which is the earliest event",
        }
        next_event = normalized in {
            "when is the next event",
            "what is the next event",
            "what is next on my calendar",
            "what's next on my calendar",
        }
        if next_event:
            events = cast(tuple[HouseholdEvent, ...], self.snapshot["events"])
            now = datetime.now(timezone.utc)
            upcoming = []
            for event in events:
                starts_at = event.starts_at
                if starts_at.tzinfo is None:
                    starts_at = starts_at.replace(tzinfo=timezone.utc)
                starts_at = starts_at.astimezone(timezone.utc)
                if starts_at >= now:
                    upcoming.append((starts_at, event))
            if not upcoming:
                return Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.BLOCKED,
                    message="I cannot find an upcoming event in the calendar.",
                    correlation_id=intent.correlation_id,
                )
            starts_at, selected = min(upcoming, key=lambda item: item[0])
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.COMPLETED,
                message=(
                    f"Event: {selected.title}; "
                    f"starts {starts_at.astimezone().strftime('%Y-%m-%d %H:%M %Z').strip()}"
                ),
                evidence={
                    "collection": "events",
                    "priority_basis": "canonical_next_event_starts_at",
                    "authorized_next_referent": {
                        "event_id": selected.event_id,
                        "title": selected.title,
                        "starts_at": selected.starts_at.isoformat(),
                    },
                    "event": {
                        "event_id": selected.event_id,
                        "title": selected.title,
                        "starts_at": selected.starts_at.isoformat(),
                    },
                },
                correlation_id=intent.correlation_id,
            )
        if latest or earliest:
            events = cast(tuple[HouseholdEvent, ...], self.snapshot["events"])
            if not events:
                return Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.BLOCKED,
                    message="I cannot order the calendar because it has no events.",
                    correlation_id=intent.correlation_id,
                )

            def event_time(event: HouseholdEvent) -> datetime:
                value = event.starts_at
                return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

            selected = (max if latest else min)(events, key=event_time)
            starts_at = event_time(selected).astimezone()
            label = "latest" if latest else "earliest"
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.COMPLETED,
                message=(
                    f"Based on the {label} recorded time: {selected.title}; "
                    f"starts {starts_at.strftime('%Y-%m-%d %H:%M %Z').strip()}"
                ),
                evidence={
                    "collection": "events",
                    "priority_basis": f"canonical_{label}_event_starts_at",
                    "event": {
                        "event_id": selected.event_id,
                        "title": selected.title,
                        "starts_at": selected.starts_at.isoformat(),
                    },
                },
                correlation_id=intent.correlation_id,
            )
        if re.search(r"\b(?:due|deadline|priority|prioritize)\b", text) and (
            "first" in text or "earliest" in text or "soonest" in text
        ):
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message=(
                    "Chores in the canonical household list have no deadlines or priority order."
                ),
                correlation_id=intent.correlation_id,
            )
        evidence: dict[str, object] = {"space_id": self.snapshot["space_id"]}
        if any(
            word in text for word in ("utility", "utilities", "rent", "obligation", "obligations")
        ):
            obligations = cast(tuple[HouseholdObligation, ...], self.snapshot["obligations"])
            evidence["obligations"] = [
                {
                    "title": obligation.title,
                    "responsible_id": obligation.responsible_id,
                    "amount": obligation.amount,
                    "settled": obligation.settled,
                }
                for obligation in obligations
            ]
            evidence["canonical_obligations"] = [
                {
                    "obligation_id": str(obligation.obligation_id),
                    "title": obligation.title,
                    "responsible_id": obligation.responsible_id,
                    "amount": obligation.amount,
                    "settled": obligation.settled,
                }
                for obligation in obligations
                if hasattr(obligation, "obligation_id")
            ]
        elif any(word in text for word in ("chore", "chores")):
            chores = cast(tuple[Chore, ...], self.snapshot["chores"])
            if any(term in text for term in ("completed", "finished")):
                chores = tuple(chore for chore in chores if chore.completed)
                status_filter = "completed"
            else:
                chores = tuple(chore for chore in chores if not chore.completed)
                status_filter = "open"
            evidence["chores"] = [
                {
                    "chore_id": chore.chore_id,
                    "title": chore.title,
                    "assignee_id": chore.assignee_id,
                    "completed": chore.completed,
                }
                for chore in chores
            ]
            evidence["status_filter"] = status_filter
        elif (
            any(
                word in text
                for word in (
                    "event",
                    "events",
                    "calendar",
                    "scheduled",
                    "schedule",
                    "appointment",
                )
            )
            or (
                "happening" in text
                and any(
                    term in text
                    for term in (
                        "today",
                        "tomorrow",
                        "this weekend",
                        "next weekend",
                        "this week",
                        "next week",
                        "this month",
                        "next month",
                    )
                )
            )
            or (
                any(
                    term in text
                    for term in (
                        "going on",
                        "plans",
                        "planned",
                        "coming up",
                        "meeting",
                        "meetings",
                    )
                )
                and any(
                    term in text
                    for term in (
                        "today",
                        "tomorrow",
                        "this weekend",
                        "next weekend",
                        "this week",
                        "next week",
                        "this month",
                        "next month",
                    )
                )
            )
        ):
            events = cast(tuple[HouseholdEvent, ...], self.snapshot["events"])
            date_filter = "all"
            target_date = None
            now = datetime.now(timezone.utc)
            if "tomorrow" in text:
                target_date = (now + timedelta(days=1)).date()
                date_filter = "tomorrow"
            elif "today" in text:
                target_date = now.date()
                date_filter = "today"
            elif "coming up" in text:
                date_filter = "upcoming"
                events = tuple(
                    event
                    for event in events
                    if (
                        event.starts_at.replace(tzinfo=timezone.utc)
                        if event.starts_at.tzinfo is None
                        else event.starts_at.astimezone(timezone.utc)
                    )
                    >= now
                )
            elif "this month" in text or "next month" in text:
                month_start = now.date().replace(day=1)
                following_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
                month_end = (
                    (following_month.replace(day=28) + timedelta(days=4)).replace(day=1)
                    if "next month" in text
                    else following_month
                )
                range_start = following_month if "next month" in text else month_start
                date_filter = "next_month" if "next month" in text else "this_month"
                events = tuple(
                    event
                    for event in events
                    if range_start
                    <= (
                        event.starts_at.replace(tzinfo=timezone.utc)
                        if event.starts_at.tzinfo is None
                        else event.starts_at.astimezone(timezone.utc)
                    ).date()
                    < month_end
                )
            elif "rest of the week" in text or "rest of week" in text:
                week_end = now.date() + timedelta(days=7 - now.weekday())
                date_filter = "rest_of_week"
                events = tuple(
                    event
                    for event in events
                    if now.date()
                    <= (
                        event.starts_at.replace(tzinfo=timezone.utc)
                        if event.starts_at.tzinfo is None
                        else event.starts_at.astimezone(timezone.utc)
                    ).date()
                    < week_end
                )
            elif "this week" in text and "this weekend" not in text:
                week_end = now.date() + timedelta(days=7 - now.weekday())
                date_filter = "this_week"
                events = tuple(
                    event
                    for event in events
                    if now.date()
                    <= (
                        event.starts_at.replace(tzinfo=timezone.utc)
                        if event.starts_at.tzinfo is None
                        else event.starts_at.astimezone(timezone.utc)
                    ).date()
                    < week_end
                )
            if target_date is not None:
                events = tuple(
                    event
                    for event in events
                    if (
                        event.starts_at.replace(tzinfo=timezone.utc)
                        if event.starts_at.tzinfo is None
                        else event.starts_at.astimezone(timezone.utc)
                    ).date()
                    == target_date
                )
            elif "this weekend" in text or "next weekend" in text:
                saturday_offset = 5 - now.weekday() if now.weekday() < 5 else -(now.weekday() - 5)
                if "next weekend" in text:
                    saturday_offset += 7
                weekend_start = (now + timedelta(days=saturday_offset)).date()
                weekend_end = weekend_start + timedelta(days=2)
                date_filter = "next_weekend" if "next weekend" in text else "this_weekend"
                events = tuple(
                    event
                    for event in events
                    if weekend_start
                    <= (
                        event.starts_at.replace(tzinfo=timezone.utc)
                        if event.starts_at.tzinfo is None
                        else event.starts_at.astimezone(timezone.utc)
                    ).date()
                    < weekend_end
                )
            elif "next week" in text and "next weekend" not in text:
                week_start = (now + timedelta(days=7)).date()
                week_end = week_start + timedelta(days=7)
                date_filter = "next_week"
                events = tuple(
                    event
                    for event in events
                    if week_start
                    <= (
                        event.starts_at.replace(tzinfo=timezone.utc)
                        if event.starts_at.tzinfo is None
                        else event.starts_at.astimezone(timezone.utc)
                    ).date()
                    < week_end
                )
            else:
                weekday_match = re.search(
                    r"\b(?:this\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
                    text,
                )
                if weekday_match is not None:
                    target = _WEEKDAYS.index(weekday_match.group(1))
                    days_ahead = (target - now.weekday()) % 7
                    target_date = (now + timedelta(days=days_ahead)).date()
                    date_filter = f"weekday:{weekday_match.group(1)}"
                    events = tuple(
                        event
                        for event in events
                        if (
                            event.starts_at.replace(tzinfo=timezone.utc)
                            if event.starts_at.tzinfo is None
                            else event.starts_at.astimezone(timezone.utc)
                        ).date()
                        == target_date
                    )
            evidence["events"] = [
                {"title": event.title, "starts_at": event.starts_at.isoformat()} for event in events
            ]
            evidence["canonical_events"] = [
                {
                    "event_id": str(event.event_id),
                    "title": event.title,
                    "starts_at": event.starts_at.isoformat(),
                }
                for event in events
            ]
            evidence["date_filter"] = date_filter
        else:
            events = cast(tuple[HouseholdEvent, ...], self.snapshot["events"])
            evidence["events"] = [
                {"title": event.title, "starts_at": event.starts_at.isoformat()} for event in events
            ]
            evidence["canonical_events"] = [
                {
                    "event_id": str(event.event_id),
                    "title": event.title,
                    "starts_at": event.starts_at.isoformat(),
                }
                for event in events
            ]
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.COMPLETED,
            message="Shared household state read",
            evidence=evidence,
            correlation_id=intent.correlation_id,
        )


class ContextualChorePriorityFastPath:
    """Refuse to invent chore priority when canonical chores have no deadlines."""

    _TERMS = ("first", "next", "priority", "prioritize", "focus", "start", "begin")

    @classmethod
    def resolve(cls, intent: IntentFrame, context: Context) -> Result | None:
        text = " ".join(intent.utterance.casefold().split())
        if (
            context.sources != ("authorized_canonical_result",)
            or is_mutation_request(text)
            or not any(term in text for term in cls._TERMS)
        ):
            return None
        ordinal_only = (
            re.search(r"\b(?:the\s+)?(?:first|second|third|fourth|last)\s+(?:one|chore)\b", text)
            is not None
        )
        explicit_priority = any(
            term in text
            for term in (
                "needs attention",
                "priority",
                "prioritize",
                "focus",
                "start with",
                "begin with",
            )
        )
        if ordinal_only and not explicit_priority:
            return None
        referents = context.values.get("referents")
        those = referents.get("those") if isinstance(referents, dict) else None
        if (
            isinstance(those, dict)
            and those.get("fact_key") == "canonical_tasks"
            and re.search(r"\b(?:chore|chores)\b", text)
        ):
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message=(
                    "That follow-up asks about chores, but the preceding result was a task "
                    "list. I can show the chores separately, but I cannot infer a chore "
                    "deadline from the task list."
                ),
                correlation_id=intent.correlation_id,
            )
        if not isinstance(those, dict) or those.get("fact_key") != "canonical_chores":
            return None
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=(
                "I cannot determine which chore comes first because the referenced chores "
                "have no canonical deadlines."
            ),
            correlation_id=intent.correlation_id,
        )


class GroceryReadFastPath:
    """Handle an unambiguous grocery-list read from canonical household state."""

    _READ_PREFIXES = (
        "what",
        "show",
        "list",
        "which",
        "see",
        "display",
        "is",
        "are",
        "how",
    )

    def __init__(self, store: PostgresHouseholdStore) -> None:
        self.store = store

    @classmethod
    def matches(cls, utterance: str) -> bool:
        text = strip_correction_prefix(utterance).casefold()
        # A short conversational continuation can introduce an explicit
        # household-domain read with a conjunction.  Normalize only the
        # leading connective; the domain noun and read prefix remain required.
        text = re.sub(r"^(?:and|but)\s+", "", text)
        text = re.sub(r"^only\s+", "", text)
        text = re.sub(
            r"^(?:can you tell me|could you tell me|can you|could you|please)\s+", "", text
        )
        if is_mutation_request(text):
            return False
        grocery_noun = re.search(r"\b(?:grocery|groceries)\b", text) is not None
        shopping_list = "shopping list" in text
        natural_shopping = "left to buy" in text
        store_pickup = re.search(r"\bpick\s+up\b.*\bstore\b", text) is not None
        unsupported_inventory = (
            assess_read_applicability(text, "kitchen.shopping_list") is ReadApplicability.NO_MATCH
        )
        return (
            grocery_noun
            or shopping_list
            or natural_shopping
            or store_pickup
            or unsupported_inventory
        ) and text.startswith(cls._READ_PREFIXES)

    def resolve(self, intent: IntentFrame) -> Result | None:
        if not self.matches(intent.utterance):
            return None
        if re.search(r"\b(?:due|urgent|priority|prioritize)\b", intent.utterance.casefold()) and (
            "first" in intent.utterance.casefold()
            or "earliest" in intent.utterance.casefold()
            or "soonest" in intent.utterance.casefold()
        ):
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message="The grocery list has no canonical deadline or priority order.",
                correlation_id=intent.correlation_id,
            )
        applicability = assess_read_applicability(intent.utterance, "kitchen.shopping_list")
        if applicability is ReadApplicability.NO_MATCH:
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message=("I track the grocery shopping list, but not pantry or on-hand inventory."),
                evidence={"semantic_scope": "kitchen.shopping_list", "applicability": "NO_MATCH"},
                correlation_id=intent.correlation_id,
            )
        if applicability is ReadApplicability.CLARIFY:
            if any(
                term in intent.utterance.casefold()
                for term in ("today", "tomorrow", "yesterday", "this week", "next week")
            ):
                return Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.BLOCKED,
                    message=(
                        "The canonical grocery list is not date-specific. "
                        "Please ask for the shopping list, or provide a dated source."
                    ),
                    evidence={
                        "semantic_scope": "kitchen.shopping_list",
                        "applicability": "CLARIFY",
                        "reason": "temporal_scope_unavailable",
                    },
                    correlation_id=intent.correlation_id,
                )
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message="Do you mean items to buy, or inventory you already have?",
                evidence={"semantic_scope": "kitchen.shopping_list", "applicability": "CLARIFY"},
                correlation_id=intent.correlation_id,
            )
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.COMPLETED,
            message="Canonical grocery list read",
            evidence={
                "collection": "groceries",
                "semantic_scope": "kitchen.shopping_list",
                "canonical_items": list(self.store.list_groceries(intent.principal)),
            },
            correlation_id=intent.correlation_id,
        )


class PostgresChoreExecutor:
    """Adapt replay-safe shared chore creation to the Core Executor port."""

    def __init__(self, store: PostgresHouseholdStore, principal: Principal) -> None:
        self.store = store
        self.principal = principal

    def execute(self, request: ExecutionRequest) -> Observation:
        if request.action.action_id not in {"tasks.chores.create", "tasks.chores.complete"}:
            return Observation(
                execution_id=request.action_id,
                evidence={"unknown_action": request.action.action_id},
                command_succeeded=False,
            )
        title = request.action.arguments.get("title")
        assignee_id = request.action.arguments.get("assignee_id", self.principal.id)
        if not isinstance(title, str) or not title.strip() or not isinstance(assignee_id, str):
            return Observation(
                execution_id=request.action_id,
                evidence={"invalid_chore": True},
                command_succeeded=False,
            )
        try:
            if request.action.action_id == "tasks.chores.create":
                chore = self.store.add_chore(
                    self.principal, title.strip(), assignee_id, request.idempotency_key
                )
            else:
                snapshot = self.store.read_snapshot(self.principal)
                chore_id = request.action.arguments.get("chore_id")
                normalized = title.casefold().strip().rstrip(".!?")
                if isinstance(chore_id, str) and chore_id.strip():
                    matches = tuple(
                        chore
                        for chore in cast(tuple[Chore, ...], snapshot["chores"])
                        if chore.chore_id == chore_id
                    )
                else:
                    matches = tuple(
                        chore
                        for chore in cast(tuple[Chore, ...], snapshot["chores"])
                        if chore.title.casefold().strip().rstrip(".!?") == normalized
                    )
                if len(matches) != 1:
                    return Observation(
                        execution_id=request.action_id,
                        evidence={
                            "collection": "chores",
                            "title": title,
                            "chore_unavailable": len(matches) == 0,
                            "ambiguous_chore_title": len(matches) > 1,
                        },
                        command_succeeded=False,
                    )
                chore = (
                    matches[0]
                    if matches[0].completed
                    else self.store.complete_chore(self.principal, matches[0].chore_id)
                )
        except (PermissionError, ValueError) as exc:
            return Observation(
                execution_id=request.action_id,
                evidence={"persistence_error": str(exc)},
                command_succeeded=False,
            )
        return Observation(
            execution_id=request.action_id,
            evidence={
                "collection": "chores",
                "chore_id": chore.chore_id,
                "title": chore.title,
                "assignee_id": chore.assignee_id,
                "completed": chore.completed,
                "idempotency_key": request.idempotency_key,
            },
            command_succeeded=True,
        )


class ChoreCompletionFastPath:
    """Resolve shared chore titles before model/executor dispatch can guess."""

    @staticmethod
    def resolve(intent: IntentFrame, title: str, chores: tuple[Chore, ...]) -> Result | None:
        normalized = title.casefold().strip().rstrip(".!?")
        matches = tuple(
            chore for chore in chores if chore.title.casefold().strip().rstrip(".!?") == normalized
        )
        if len(matches) == 1:
            return None
        if len(matches) == 0:
            message = (
                f"I couldn't find one chore named '{title}'. "
                "Ask to complete a chore that appears in household chores."
            )
        else:
            message = (
                f"I found multiple chores named '{title}'. "
                "Please include more detail so I complete only the intended chore."
            )
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=message,
            correlation_id=intent.correlation_id,
        )


class PostgresChoreVerifier:
    """Verify chore creation by independently reading canonical shared state."""

    def __init__(self, store: PostgresHouseholdStore, principal: Principal) -> None:
        self.store = store
        self.principal = principal

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        if contract.kind != "readback" or not observation.command_succeeded:
            return VerificationResult(
                verified=False, evidence=observation.evidence, reason="chore execution failed"
            )
        snapshot = self.store.read_snapshot(self.principal)
        chores = cast(tuple[Chore, ...], snapshot["chores"])
        chore_id = observation.evidence.get("chore_id")
        verified = any(
            chore.chore_id == chore_id
            and chore.title == observation.evidence.get("title")
            and chore.assignee_id == observation.evidence.get("assignee_id")
            and chore.completed == observation.evidence.get("completed", False)
            for chore in chores
        )
        return VerificationResult(
            verified=verified,
            evidence={**observation.evidence, "canonical_chore_count": len(chores)},
            reason=(
                "canonical chore readback verified"
                if verified
                else "canonical chore readback failed"
            ),
        )


class PostgresEventExecutor:
    """Adapt replay-safe shared event creation to the Core Executor port."""

    def __init__(self, store: PostgresHouseholdStore, principal: Principal) -> None:
        self.store = store
        self.principal = principal

    def execute(self, request: ExecutionRequest) -> Observation:
        if request.action.action_id != "tasks.events.create":
            return Observation(
                execution_id=request.action_id,
                evidence={"unknown_action": request.action.action_id},
                command_succeeded=False,
            )
        title = request.action.arguments.get("title")
        starts_at = request.action.arguments.get("starts_at")
        if not isinstance(title, str) or not title.strip() or not isinstance(starts_at, str):
            return Observation(
                execution_id=request.action_id,
                evidence={"invalid_event": True},
                command_succeeded=False,
            )
        try:
            event = self.store.add_event(
                self.principal,
                title.strip(),
                datetime.fromisoformat(starts_at),
                request.idempotency_key,
            )
        except (PermissionError, ValueError) as exc:
            return Observation(
                execution_id=request.action_id,
                evidence={"persistence_error": str(exc)},
                command_succeeded=False,
            )
        return Observation(
            execution_id=request.action_id,
            evidence={
                "collection": "events",
                "event_id": event.event_id,
                "title": event.title,
                "starts_at": event.starts_at.isoformat(),
                "idempotency_key": request.idempotency_key,
            },
            command_succeeded=True,
        )


class PostgresEventVerifier:
    """Verify event creation by independently reading canonical shared state."""

    def __init__(self, store: PostgresHouseholdStore, principal: Principal) -> None:
        self.store = store
        self.principal = principal

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        if contract.kind != "readback" or not observation.command_succeeded:
            return VerificationResult(
                verified=False, evidence=observation.evidence, reason="event execution failed"
            )
        snapshot = self.store.read_snapshot(self.principal)
        events = cast(tuple[HouseholdEvent, ...], snapshot["events"])
        event_id = observation.evidence.get("event_id")
        verified = any(
            event.event_id == event_id
            and event.title == observation.evidence.get("title")
            and event.starts_at.isoformat() == observation.evidence.get("starts_at")
            for event in events
        )
        return VerificationResult(
            verified=verified,
            evidence={**observation.evidence, "canonical_event_count": len(events)},
            reason=(
                "canonical event readback verified"
                if verified
                else "canonical event readback failed"
            ),
        )
