"""Read-oriented device contracts and a narrow Home Assistant adapter."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from pydantic import Field

from .contracts import StrictModel
from .personal import Provenance


class DeviceState(StrictModel):
    entity_id: str = Field(min_length=1)
    state: str = Field(min_length=1)
    attributes: dict[str, Any] = {}
    observed_at: datetime
    provenance: Provenance = Provenance.OBSERVED


class DeviceCommand(StrictModel):
    entity_id: str = Field(min_length=1)
    service: str = Field(min_length=1)
    data: dict[str, Any] = {}
    expected_state: str | None = None


class DeviceExecution(StrictModel):
    command: DeviceCommand
    accepted: bool
    observed_state: DeviceState | None = None
    verified: bool
    reason: str


class DeviceGateway(Protocol):
    def get_state(self, entity_id: str) -> dict[str, Any]: ...

    def call_service(self, command: dict[str, Any]) -> None: ...


class FixtureDeviceGateway:
    """Deterministic Home Assistant-shaped gateway for contract acceptance."""

    def __init__(self, states: dict[str, dict[str, Any]] | None = None) -> None:
        self.states = states or {}

    def get_state(self, entity_id: str) -> dict[str, Any]:
        return dict(self.states.get(entity_id, {"state": "unknown", "attributes": {}}))

    def call_service(self, command: dict[str, Any]) -> None:
        del command


def device_states_evidence(states: tuple[DeviceState, ...]) -> dict[str, Any]:
    return {
        "source": "home_assistant_fixture",
        "states": [state.model_dump(mode="json") for state in states[:50]],
    }


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

    def execute(self, command: DeviceCommand, observed_at: datetime) -> DeviceExecution:
        if not self.policy.allow_command(command):
            raise PermissionError("Home Assistant command denied by policy")
        self.gateway.call_service(command.model_dump(mode="json"))
        state = self.read_state(command.entity_id, observed_at)
        verified = command.expected_state is not None and state.state == command.expected_state
        return DeviceExecution(
            command=command,
            accepted=True,
            observed_state=state,
            verified=verified,
            reason=(
                "device readback verified"
                if verified
                else "command accepted; expected postcondition unavailable or failed"
            ),
        )
