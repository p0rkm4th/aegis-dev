"""Identity and relationship authorization ports with a structural fake."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from .contracts import Principal


class Role(StrEnum):
    OWNER = "owner"
    MEMBER = "member"


@dataclass(frozen=True)
class Vault:
    vault_id: str
    owner_id: str


@dataclass(frozen=True)
class Space:
    space_id: str
    name: str


@dataclass(frozen=True)
class Membership:
    principal_id: str
    space_id: str
    role: Role
    active: bool = True


@dataclass(frozen=True)
class Resource:
    resource_id: str
    owner_vault_id: str
    shared_space_id: str | None = None


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    reason: str


class IdentityProvider(Protocol):
    def principal_from_claims(self, claims: dict[str, Any]) -> Principal: ...


class RelationshipAuthorizer(Protocol):
    def can_read(self, principal: Principal, resource_id: str) -> AccessDecision: ...


class KeycloakIdentityProvider:
    """Map already-validated OIDC claims to Aegis identity; never validates tokens itself."""

    def principal_from_claims(self, claims: dict[str, Any]) -> Principal:
        subject = claims.get("sub")
        vault_id = claims.get("aegis_vault_id")
        if not isinstance(subject, str) or not subject:
            raise ValueError("validated identity is missing sub")
        if not isinstance(vault_id, str) or not vault_id:
            raise ValueError("validated identity is missing aegis_vault_id")
        spaces = claims.get("aegis_space_ids", ())
        if not isinstance(spaces, (list, tuple)) or not all(isinstance(s, str) for s in spaces):
            raise ValueError("aegis_space_ids must be a sequence of strings")
        return Principal(id=subject, vault_id=vault_id, space_ids=tuple(spaces))


class OpenFGAClient(Protocol):
    def check(self, user: str, relation: str, object_id: str) -> bool: ...


class OpenFGAAuthorization:
    """Relationship adapter; OpenFGA decides relations, Aegis still owns semantics."""

    def __init__(self, client: OpenFGAClient) -> None:
        self.client = client

    def can_read(self, principal: Principal, resource_id: str) -> AccessDecision:
        allowed = self.client.check(f"user:{principal.id}", "can_read", f"resource:{resource_id}")
        return AccessDecision(
            allowed, "OpenFGA relationship allows read" if allowed else "OpenFGA denied read"
        )


class InMemoryAuthorization:
    """Fail-closed relationship checks; no prompt or model input participates."""

    def __init__(self) -> None:
        self.vaults: dict[str, Vault] = {}
        self.spaces: dict[str, Space] = {}
        self.memberships: dict[tuple[str, str], Membership] = {}
        self.resources: dict[str, Resource] = {}

    def add_vault(self, vault: Vault) -> None:
        self.vaults[vault.vault_id] = vault

    def add_space(self, space: Space) -> None:
        self.spaces[space.space_id] = space

    def add_membership(self, membership: Membership) -> None:
        self.memberships[(membership.principal_id, membership.space_id)] = membership

    def add_resource(self, resource: Resource) -> None:
        self.resources[resource.resource_id] = resource

    def can_read(self, principal: Principal, resource_id: str) -> AccessDecision:
        resource = self.resources.get(resource_id)
        if resource is None:
            return AccessDecision(False, "resource does not exist")
        if resource.owner_vault_id == principal.vault_id:
            return AccessDecision(True, "principal owns resource Vault")
        if resource.shared_space_id is None:
            return AccessDecision(False, "resource is private to another Vault")
        membership = self.memberships.get((principal.id, resource.shared_space_id))
        if membership is None or not membership.active:
            return AccessDecision(False, "principal is not an active Space member")
        return AccessDecision(True, "active Space membership permits read")

    def revoke(self, principal_id: str, space_id: str) -> None:
        current = self.memberships.get((principal_id, space_id))
        if current is not None:
            self.memberships[(principal_id, space_id)] = Membership(
                principal_id=current.principal_id,
                space_id=current.space_id,
                role=current.role,
                active=False,
            )
