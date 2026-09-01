"""Shared household Space state with membership checks below the model layer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, cast
from uuid import uuid4

from .contracts import IntentFrame, ObjectiveState, Principal, Result


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
        grocery_mutations = {
            str(key): str(item) for key, item in payload.get("grocery_mutations", {}).items()
        }
        return HouseholdSpace(
            space_id,
            set(members),
            groceries,
            chores,
            events,
            obligations,
            grocery_mutations,
        )

    def add_grocery(self, principal: Principal, item: str, idempotency_key: str) -> None:
        """Apply one authorized, replay-safe grocery mutation to canonical state."""
        if not idempotency_key:
            raise ValueError("grocery mutation requires an idempotency key")
        space_id = self._space_for(principal)
        members = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT principal_id FROM space_memberships "
                "WHERE space_id = %s AND active = TRUE",
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
                "SELECT principal_id FROM space_memberships "
                "WHERE space_id = %s AND active = TRUE",
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
                "SELECT principal_id FROM space_memberships "
                "WHERE space_id = %s AND active = TRUE",
                (space_id,),
            ).fetchall()
        }
        return tuple(self.load(space_id, members).groceries)

    def read_snapshot(self, principal: Principal) -> dict[str, object]:
        """Read shared household state after rechecking current membership."""
        space_id = self._space_for(principal)
        members = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT principal_id FROM space_memberships "
                "WHERE space_id = %s AND active = TRUE",
                (space_id,),
            ).fetchall()
        }
        return self.load(space_id, members).snapshot(principal)

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
        if idempotency_key is not None:
            self.grocery_mutations[idempotency_key] = item

    def add_chore(self, principal: Principal, chore: Chore) -> None:
        self._require_member(principal)
        if chore.assignee_id not in self.members:
            raise ValueError("chore assignee is not a Space member")
        self.chores[chore.chore_id] = chore

    def add_event(self, principal: Principal, event: HouseholdEvent) -> None:
        self._require_member(principal)
        self.events[event.event_id] = event

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
            "chores": tuple(self.chores.values()),
            "events": tuple(self.events.values()),
            "obligations": tuple(self.obligations.values()),
        }


class HouseholdReadFastPath:
    """Deterministic, allowlisted reads over shared household state."""

    _TRIGGERS = ("household", "chore", "chores", "inspection", "utility", "utilities", "rent")

    def __init__(self, snapshot: dict[str, object]) -> None:
        self.snapshot = snapshot

    @classmethod
    def matches(cls, utterance: str) -> bool:
        text = utterance.casefold()
        return any(trigger in text for trigger in cls._TRIGGERS)

    def resolve(self, intent: IntentFrame) -> Result | None:
        if not self.matches(intent.utterance):
            return None
        text = intent.utterance.casefold()
        evidence: dict[str, object] = {"space_id": self.snapshot["space_id"]}
        if any(word in text for word in ("utility", "utilities", "rent")):
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
        elif any(word in text for word in ("chore", "chores")):
            chores = cast(tuple[Chore, ...], self.snapshot["chores"])
            evidence["chores"] = [
                {
                    "title": chore.title,
                    "assignee_id": chore.assignee_id,
                    "completed": chore.completed,
                }
                for chore in chores
            ]
        else:
            events = cast(tuple[HouseholdEvent, ...], self.snapshot["events"])
            evidence["events"] = [
                {"title": event.title, "starts_at": event.starts_at.isoformat()}
                for event in events
            ]
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.COMPLETED,
            message="Shared household state read",
            evidence=evidence,
            correlation_id=intent.correlation_id,
        )
