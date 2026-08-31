"""Scoped Homelab inventory and verified service actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .network import DiscoveredDevice, HomelabInventory


@dataclass(frozen=True)
class Host:
    host_id: str
    address: str
    hostname: str
    resources: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Service:
    service_id: str
    host_id: str
    name: str
    health_endpoint: str


@dataclass(frozen=True)
class ServiceActionResult:
    service_id: str
    attempted: bool
    verified: bool
    message: str


class HomelabRuntime(Protocol):
    def restart(self, service: Service) -> bool: ...
    def health(self, service: Service) -> bool: ...


class HomelabPack:
    def __init__(self, inventory: HomelabInventory, runtime: HomelabRuntime) -> None:
        self.inventory = inventory
        self.runtime = runtime
        self.hosts: dict[str, Host] = {}
        self.services: dict[str, Service] = {}

    def add_host(self, host: Host) -> None:
        self.inventory.record_discovery(DiscoveredDevice(host.address, host.hostname))
        self.hosts[host.host_id] = host

    def add_service(self, service: Service) -> None:
        if service.host_id not in self.hosts:
            raise ValueError("service references unknown host")
        self.services[service.service_id] = service

    def restart_service(self, scope_id: str, service_id: str) -> ServiceActionResult:
        try:
            service = self.services[service_id]
            host = self.hosts[service.host_id]
        except KeyError as exc:
            raise KeyError("unknown Homelab service") from exc
        self.inventory.require_action_scope(scope_id, host.address)
        attempted = self.runtime.restart(service)
        if not attempted:
            return ServiceActionResult(service_id, False, False, "restart was rejected")
        verified = self.runtime.health(service)
        return ServiceActionResult(
            service_id,
            True,
            verified,
            "service health verified" if verified else "restart attempted; health failed",
        )
