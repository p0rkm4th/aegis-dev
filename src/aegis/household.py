"""Shared household Space state with membership checks below the model layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .contracts import Principal


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


@dataclass
class HouseholdSpace:
    space_id: str
    members: set[str]
    groceries: list[str] = field(default_factory=list)
    chores: dict[str, Chore] = field(default_factory=dict)
    events: dict[str, HouseholdEvent] = field(default_factory=dict)
    obligations: dict[str, HouseholdObligation] = field(default_factory=dict)

    def _require_member(self, principal: Principal) -> None:
        if principal.id not in self.members or self.space_id not in principal.space_ids:
            raise PermissionError("principal is not an active member of this Space")

    def add_grocery(self, principal: Principal, item: str) -> None:
        self._require_member(principal)
        if not item.strip():
            raise ValueError("grocery item is required")
        self.groceries.append(item)

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
