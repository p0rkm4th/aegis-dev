"""Identity and relationship authorization ports with a structural fake."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .contracts import AuthorizationRequest, PolicyDecision, Principal


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


class ExternalPrincipalResolver(Protocol):
    """Resolve a validated provider subject to AEGIS's canonical principal ID."""

    def __call__(self, external_subject: str) -> str: ...


class PostgresExternalPrincipalResolver:
    """Look up the canonical principal mapped to an immutable external subject."""

    def __init__(self, connect: Callable[[str], Any], database_url: str) -> None:
        self.connect = connect
        self.database_url = database_url

    def __call__(self, external_subject: str) -> str:
        try:
            connection = self.connect(self.database_url)
            try:
                row = connection.execute(
                    "SELECT id FROM aegis_principals WHERE external_subject = %s",
                    (external_subject,),
                ).fetchone()
            finally:
                connection.close()
        except Exception as exc:
            raise RuntimeError("canonical identity mapping is unavailable") from exc
        if row is None or not row[0]:
            raise PermissionError("external identity is not provisioned in AEGIS")
        return str(row[0])


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


class KeycloakOIDCClient:
    """Resolve a bearer token through Keycloak userinfo, then map its claims."""

    def __init__(
        self,
        issuer: str,
        timeout: float = 10.0,
        principal_resolver: ExternalPrincipalResolver | None = None,
    ) -> None:
        if not issuer.startswith(("http://", "https://")):
            raise ValueError("Keycloak issuer must use HTTP or HTTPS")
        self.userinfo_endpoint = f"{issuer.rstrip('/')}/protocol/openid-connect/userinfo"
        self.timeout = timeout
        self.identity = KeycloakIdentityProvider()
        self.principal_resolver = principal_resolver

    def principal_from_access_token(self, access_token: str) -> Principal:
        if not access_token:
            raise ValueError("access token is required")
        request = Request(
            self.userinfo_endpoint,
            headers={"Authorization": f"Bearer {access_token}"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                claims = json.load(response)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            raise RuntimeError("Keycloak userinfo request failed") from exc
        if not isinstance(claims, dict):
            raise RuntimeError("Keycloak userinfo response was invalid")
        principal = self.identity.principal_from_claims(claims)
        if self.principal_resolver is None:
            return principal
        canonical_id = self.principal_resolver(principal.id)
        if not canonical_id:
            raise RuntimeError("external identity resolved to an empty principal")
        return principal.model_copy(update={"id": canonical_id})


class OpenFGAClient(Protocol):
    def check(self, user: str, relation: str, object_id: str) -> bool: ...


class OpenFGAHttpClient:
    """Small standard-library client for the OpenFGA check endpoint."""

    def __init__(
        self,
        api_url: str,
        store_id: str,
        model_id: str | None = None,
        token: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        if not api_url.startswith(("http://", "https://")):
            raise ValueError("OpenFGA API URL must use HTTP or HTTPS")
        if not store_id:
            raise ValueError("OpenFGA store ID is required")
        self.endpoint = f"{api_url.rstrip('/')}/stores/{store_id}/check"
        self.model_id = model_id
        self.token = token
        self.timeout = timeout

    def check(self, user: str, relation: str, object_id: str) -> bool:
        body: dict[str, Any] = {
            "tuple_key": {"user": user, "relation": relation, "object": object_id}
        }
        if self.model_id:
            body["authorization_model_id"] = self.model_id
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(
            self.endpoint,
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            raise RuntimeError("OpenFGA check request failed") from exc
        if not isinstance(result, dict) or not isinstance(result.get("allowed"), bool):
            raise RuntimeError("OpenFGA check response was invalid")
        return bool(result["allowed"])


class OpenFGAAuthorization:
    """Relationship adapter; OpenFGA decides relations, Aegis still owns semantics."""

    def __init__(self, client: OpenFGAClient) -> None:
        self.client = client

    def can_read(self, principal: Principal, resource_id: str) -> AccessDecision:
        allowed = self.client.check(f"user:{principal.id}", "can_read", f"resource:{resource_id}")
        return AccessDecision(
            allowed, "OpenFGA relationship allows read" if allowed else "OpenFGA denied read"
        )


class SqlQueryConnection(Protocol):
    def execute(self, query: str, params: tuple[object, ...] = ()) -> Any: ...


class PostgresSpacePolicy:
    """Fail-closed capability policy backed by canonical Space membership."""

    def __init__(
        self,
        connection: SqlQueryConnection,
        permission_roles: dict[str, frozenset[Role]],
    ) -> None:
        self.connection = connection
        self.permission_roles = permission_roles

    def authorize(self, request: AuthorizationRequest) -> PolicyDecision:
        space_id = request.principal.space_ids[0] if request.principal.space_ids else None
        if space_id is None:
            return PolicyDecision(allowed=False, reason="action requires an explicit Space")
        row = self.connection.execute(
            "SELECT role FROM space_memberships "
            "WHERE principal_id = %s AND space_id = %s AND active = TRUE",
            (request.principal.id, space_id),
        ).fetchone()
        if row is None:
            return PolicyDecision(allowed=False, reason="principal is not an active Space member")
        try:
            role = Role(str(row[0]))
        except ValueError:
            return PolicyDecision(allowed=False, reason="membership role is invalid")
        for permission in request.action.required_permissions:
            allowed_roles = self.permission_roles.get(permission)
            if allowed_roles is None or role not in allowed_roles:
                return PolicyDecision(
                    allowed=False,
                    reason=f"Space policy denies permission: {permission}",
                )
        return PolicyDecision(
            allowed=True, reason="active PostgreSQL Space membership permits action"
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
