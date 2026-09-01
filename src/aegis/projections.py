"""Deterministic privacy projections; raw private inputs never enter the result."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import Principal


@dataclass(frozen=True)
class PrivateContribution:
    owner_id: str
    amount: int
    source_resource_id: str


@dataclass(frozen=True)
class SharedObligation:
    name: str
    amount: int


@dataclass(frozen=True)
class HouseholdProjection:
    space_id: str
    obligation_total: int
    equal_share: int
    contributions: dict[str, int]
    settlements: dict[str, int]


class ProjectionAuthorization(Protocol):
    def may_derive(self, requester: Principal, owner_id: str, space_id: str) -> bool: ...


class ProjectionStore(Protocol):
    def save(self, projection: HouseholdProjection) -> None: ...

    def load(
        self, space_id: str, principal: Principal, members: set[str]
    ) -> HouseholdProjection | None: ...


class PostgresProjectionStore:
    """Persist only the allowlisted derived household projection."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def save(self, projection: HouseholdProjection) -> None:
        payload = {
            "obligation_total": projection.obligation_total,
            "equal_share": projection.equal_share,
            "contributions": projection.contributions,
            "settlements": projection.settlements,
        }
        self.connection.execute(
            "INSERT INTO household_projections (space_id, payload) VALUES (%s, %s) "
            "ON CONFLICT (space_id) DO UPDATE SET payload = EXCLUDED.payload, "
            "updated_at = now()",
            (projection.space_id, json.dumps(payload, sort_keys=True)),
        )
        self.connection.commit()

    def load(
        self, space_id: str, principal: Principal, members: set[str]
    ) -> HouseholdProjection | None:
        if principal.id not in members or space_id not in principal.space_ids:
            raise PermissionError("principal is not an active member of this Space")
        row = self.connection.execute(
            "SELECT payload FROM household_projections WHERE space_id = %s", (space_id,)
        ).fetchone()
        if row is None:
            return None
        payload = row[0] if isinstance(row[0], dict) else json.loads(str(row[0]))
        contributions = {
            str(owner): int(amount) for owner, amount in payload.get("contributions", {}).items()
        }
        settlements = {
            str(owner): int(amount) for owner, amount in payload.get("settlements", {}).items()
        }
        return HouseholdProjection(
            space_id=space_id,
            obligation_total=int(payload["obligation_total"]),
            equal_share=int(payload["equal_share"]),
            contributions=contributions,
            settlements=settlements,
        )


class PrivacyProjectionService:
    """Build an allowlisted shared projection below the model layer."""

    def __init__(
        self, authorization: ProjectionAuthorization, store: ProjectionStore | None = None
    ) -> None:
        self.authorization = authorization
        self.store = store

    def build(
        self,
        requester: Principal,
        space_id: str,
        members: tuple[str, ...],
        obligations: tuple[SharedObligation, ...],
        contributions: tuple[PrivateContribution, ...],
    ) -> HouseholdProjection:
        if not members or any(not member for member in members):
            raise ValueError("a projection requires non-empty member ids")
        allowed: dict[str, int] = {}
        for contribution in contributions:
            if not self.authorization.may_derive(requester, contribution.owner_id, space_id):
                raise PermissionError(f"private contribution denied for {contribution.owner_id}")
            if contribution.owner_id not in members:
                raise PermissionError("contribution owner is not a member of this Space")
            allowed[contribution.owner_id] = (
                allowed.get(contribution.owner_id, 0) + contribution.amount
            )
        total = sum(obligation.amount for obligation in obligations)
        share, remainder = divmod(total, len(members))
        settlements = {
            member: allowed.get(member, 0) - share - (1 if index < remainder else 0)
            for index, member in enumerate(members)
        }
        projection = HouseholdProjection(
            space_id=space_id,
            obligation_total=total,
            equal_share=share,
            contributions=dict(allowed),
            settlements=settlements,
        )
        if self.store is not None:
            self.store.save(projection)
        return projection
