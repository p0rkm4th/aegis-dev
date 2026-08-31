"""Read-oriented device contracts and a narrow Home Assistant adapter."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from pydantic import Field

from .contracts import StrictModel


class DeviceState(StrictModel):
    entity_id: str = Field(min_length=1)
    state: str = Field(min_length=1)
    attributes: dict[str, Any] = {}
    observed_at: datetime


class DeviceCommand(StrictModel):
    entity_id: str = Field(min_length=1)
    service: str = Field(min_length=1)
    data: dict[str, Any] = {}


class DeviceGateway(Protocol):
    def get_state(self, entity_id: str) -> dict[str, Any]: ...

    def call_service(self, command: dict[str, Any]) -> None: ...


class DevicePolicy(Protocol):
    def allow_command(self, command: DeviceCommand) -> bool: ...


class HomeAssistantAdapter:
    """Home Assistant remains device truth; AEGIS supplies policy at the seam."""

    def __init__(self, gateway: DeviceGateway, policy: DevicePolicy) -> None:
        self.gateway = gateway
        self.policy = policy

    def read_state(self, entity_id: str, observed_at: datetime) -> DeviceState:
        raw = self.gateway.get_state(entity_id)
        return DeviceState(
            entity_id=entity_id,
            state=str(raw["state"]),
            attributes=dict(raw.get("attributes", {})),
            observed_at=observed_at,
        )

    def execute(self, command: DeviceCommand) -> None:
        if not self.policy.allow_command(command):
            raise PermissionError("Home Assistant command denied by policy")
        self.gateway.call_service(command.model_dump(mode="json"))
