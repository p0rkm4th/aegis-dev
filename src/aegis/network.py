"""Authorized network scope and homelab inventory primitives."""

from __future__ import annotations

import ipaddress
import json
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from .contracts import AuthorizationRequest, PolicyDecision, Principal


class ScopeDenied(PermissionError):
    """A target is reachable or discovered but not explicitly authorized."""


@dataclass(frozen=True)
class AuthorizedNetworkScope:
    scope_id: str
    cidrs: tuple[str, ...]
    purpose: str
    active: bool = True

    def __post_init__(self) -> None:
        if not self.scope_id or not self.cidrs or not self.purpose:
            raise ValueError("scope id, CIDR, and purpose are required")
        for cidr in self.cidrs:
            ipaddress.ip_network(cidr, strict=False)

    def authorizes(self, address: str) -> bool:
        if not self.active:
            return False
        target = ipaddress.ip_address(address)
        return any(target in ipaddress.ip_network(cidr, strict=False) for cidr in self.cidrs)

    def require(self, address: str) -> None:
        if not self.authorizes(address):
            raise ScopeDenied(f"target {address} is outside authorized scope {self.scope_id}")


@dataclass(frozen=True)
class DiscoveredDevice:
    address: str
    hostname: str | None = None
    services: tuple[str, ...] = ()


@dataclass
class HomelabInventory:
    devices: dict[str, DiscoveredDevice] = field(default_factory=dict)
    scopes: dict[str, AuthorizedNetworkScope] = field(default_factory=dict)

    def record_discovery(self, device: DiscoveredDevice) -> None:
        ipaddress.ip_address(device.address)
        self.devices[device.address] = device

    def add_scope(self, scope: AuthorizedNetworkScope) -> None:
        self.scopes[scope.scope_id] = scope

    def require_action_scope(self, scope_id: str, address: str) -> None:
        try:
            scope = self.scopes[scope_id]
        except KeyError as exc:
            raise ScopeDenied(f"unknown authorization scope {scope_id}") from exc
        scope.require(address)

    def authorized_devices(self, scope_id: str) -> tuple[DiscoveredDevice, ...]:
        try:
            scope = self.scopes[scope_id]
        except KeyError as exc:
            raise ScopeDenied(f"unknown authorization scope {scope_id}") from exc
        return tuple(device for device in self.devices.values() if scope.authorizes(device.address))


class NetworkStateConnection(Protocol):
    def execute(self, query: str, params: tuple[object, ...] = ()) -> Any: ...

    def commit(self) -> None: ...


class PostgresNetworkStore:
    """Persist authorized network state partitioned by the member's Space."""

    def __init__(self, connection: NetworkStateConnection) -> None:
        self.connection = connection

    def save_scope(self, principal: Principal, scope: AuthorizedNetworkScope) -> None:
        space_id = self._space_for(principal)
        self.connection.execute(
            "INSERT INTO network_scopes (space_id, scope_id, cidrs, purpose, active) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (space_id, scope_id) DO UPDATE SET "
            "cidrs = EXCLUDED.cidrs, purpose = EXCLUDED.purpose, active = EXCLUDED.active, "
            "updated_at = now()",
            (space_id, scope.scope_id, json.dumps(list(scope.cidrs)), scope.purpose, scope.active),
        )
        self.connection.commit()

    def save_device(self, principal: Principal, device: DiscoveredDevice) -> None:
        space_id = self._space_for(principal)
        ipaddress.ip_address(device.address)
        self.connection.execute(
            "INSERT INTO network_devices (space_id, address, hostname, services) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (space_id, address) DO UPDATE SET "
            "hostname = EXCLUDED.hostname, services = EXCLUDED.services, updated_at = now()",
            (space_id, device.address, device.hostname, json.dumps(list(device.services))),
        )
        self.connection.commit()

    def load(self, principal: Principal) -> HomelabInventory:
        space_id = self._space_for(principal)
        scopes = {}
        for scope_id, cidrs, purpose, active in self.connection.execute(
            "SELECT scope_id, cidrs, purpose, active FROM network_scopes "
            "WHERE space_id = %s ORDER BY scope_id",
            (space_id,),
        ).fetchall():
            values = cidrs if isinstance(cidrs, list) else json.loads(str(cidrs))
            scopes[str(scope_id)] = AuthorizedNetworkScope(
                str(scope_id), tuple(str(value) for value in values), str(purpose), bool(active)
            )
        devices = {}
        for address, hostname, services in self.connection.execute(
            "SELECT address, hostname, services FROM network_devices "
            "WHERE space_id = %s ORDER BY address",
            (space_id,),
        ).fetchall():
            values = services if isinstance(services, list) else json.loads(str(services))
            device = DiscoveredDevice(
                str(address),
                str(hostname) if hostname is not None else None,
                tuple(str(value) for value in values),
            )
            devices[device.address] = device
        return HomelabInventory(devices=devices, scopes=scopes)

    def _space_for(self, principal: Principal) -> str:
        if not principal.space_ids:
            raise PermissionError("network state requires an explicit Space")
        space_id = principal.space_ids[0]
        row = self.connection.execute(
            "SELECT 1 FROM space_memberships WHERE principal_id = %s AND space_id = %s "
            "AND active = TRUE",
            (principal.id, space_id),
        ).fetchone()
        if row is None:
            raise PermissionError("principal is not an active Space member")
        return space_id


class NetworkScopePolicy:
    """Compose Space permission with explicit persisted network-scope authority."""

    def __init__(self, space_policy: Any, network_store: PostgresNetworkStore) -> None:
        self.space_policy = space_policy
        self.network_store = network_store

    def authorize(self, request: AuthorizationRequest) -> PolicyDecision:
        base = self.space_policy.authorize(request)
        if not base.allowed:
            return cast(PolicyDecision, base)
        if request.action.action_id != "network.probe":
            return PolicyDecision(
                allowed=False, reason="network policy does not permit this action"
            )
        address = request.action.arguments.get("address")
        scope_id = request.action.arguments.get("scope_id")
        if not isinstance(address, str) or not isinstance(scope_id, str):
            return PolicyDecision(
                allowed=False, reason="network probe requires address and scope_id"
            )
        try:
            self.network_store.load(request.principal).scopes[scope_id].require(address)
        except (KeyError, ValueError, PermissionError) as exc:
            return PolicyDecision(allowed=False, reason=f"network scope denies target: {exc}")
        return PolicyDecision(
            allowed=True, reason="active Space membership and network scope permit probe"
        )
