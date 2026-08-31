"""Authorized network scope and homelab inventory primitives."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field


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
