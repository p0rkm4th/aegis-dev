"""Scoped Homelab inventory and verified service actions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from .contracts import Principal
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


class PostgresHomelabStore:
    """Persist Space-scoped Homelab host and service inventory."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def save(self, principal: "Principal", pack: "HomelabPack") -> None:
        space_id = _space_for(self.connection, principal)
        payload = {
            "hosts": [
                {
                    "host_id": host.host_id,
                    "address": host.address,
                    "hostname": host.hostname,
                    "resources": host.resources,
                }
                for host in pack.hosts.values()
            ],
            "services": [
                {
                    "service_id": service.service_id,
                    "host_id": service.host_id,
                    "name": service.name,
                    "health_endpoint": service.health_endpoint,
                }
                for service in pack.services.values()
            ],
        }
        self.connection.execute(
            "INSERT INTO homelab_inventories (space_id, payload) VALUES (%s, %s) "
            "ON CONFLICT (space_id) DO UPDATE SET payload = EXCLUDED.payload, "
            "updated_at = now()",
            (space_id, json.dumps(payload, sort_keys=True)),
        )
        self.connection.commit()

    def load(self, principal: "Principal", runtime: HomelabRuntime) -> "HomelabPack":
        space_id = _space_for(self.connection, principal)
        row = self.connection.execute(
            "SELECT payload FROM homelab_inventories WHERE space_id = %s", (space_id,)
        ).fetchone()
        pack = HomelabPack(HomelabInventory(), runtime)
        if row is None:
            return pack
        payload = row[0] if isinstance(row[0], dict) else json.loads(str(row[0]))
        for item in payload.get("hosts", []):
            pack.add_host(
                Host(
                    str(item["host_id"]),
                    str(item["address"]),
                    str(item["hostname"]),
                    {str(key): int(value) for key, value in item.get("resources", {}).items()},
                )
            )
        for item in payload.get("services", []):
            pack.add_service(
                Service(
                    str(item["service_id"]),
                    str(item["host_id"]),
                    str(item["name"]),
                    str(item["health_endpoint"]),
                )
            )
        return pack


def _space_for(connection: Any, principal: "Principal") -> str:
    if not principal.space_ids:
        raise PermissionError("Homelab state requires an explicit Space")
    space_id = principal.space_ids[0]
    row = connection.execute(
        "SELECT 1 FROM space_memberships WHERE principal_id = %s AND space_id = %s "
        "AND active = TRUE",
        (principal.id, space_id),
    ).fetchone()
    if row is None:
        raise PermissionError("principal is not an active Space member")
    return space_id


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
