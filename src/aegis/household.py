"""Shared household Space state with membership checks below the model layer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, cast
from uuid import uuid4

from .contracts import (
    ExecutionRequest,
    IntentFrame,
    ObjectiveState,
    Observation,
    Principal,
    Result,
    VerificationContract,
    VerificationResult,
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
        if text.startswith(("add ", "create ", "update ", "complete ", "remove ")):
            return False
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
                {"title": event.title, "starts_at": event.starts_at.isoformat()} for event in events
            ]
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.COMPLETED,
            message="Shared household state read",
            evidence=evidence,
            correlation_id=intent.correlation_id,
        )


class PostgresChoreExecutor:
    """Adapt replay-safe shared chore creation to the Core Executor port."""

    def __init__(self, store: PostgresHouseholdStore, principal: Principal) -> None:
        self.store = store
        self.principal = principal

    def execute(self, request: ExecutionRequest) -> Observation:
        if request.action.action_id != "tasks.chores.create":
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
            chore = self.store.add_chore(
                self.principal, title.strip(), assignee_id, request.idempotency_key
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
                "idempotency_key": request.idempotency_key,
            },
            command_succeeded=True,
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
