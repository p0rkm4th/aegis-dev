"""Deterministic privacy projections; raw private inputs never enter the result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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


class PrivacyProjectionService:
    """Build an allowlisted shared projection below the model layer."""

    def __init__(self, authorization: ProjectionAuthorization) -> None:
        self.authorization = authorization

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
        return HouseholdProjection(
            space_id=space_id,
            obligation_total=total,
            equal_share=share,
            contributions=dict(allowed),
            settlements=settlements,
        )
