"""Read-oriented device contracts and a narrow Home Assistant adapter."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Protocol
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

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

    def list_states(self) -> tuple[dict[str, Any], ...]: ...

    def call_service(self, command: dict[str, Any]) -> None: ...


class FixtureDeviceGateway:
    """Deterministic Home Assistant-shaped gateway for contract acceptance."""

    def __init__(self, states: dict[str, dict[str, Any]] | None = None) -> None:
        self.states = states or {}

    def get_state(self, entity_id: str) -> dict[str, Any]:
        return dict(self.states.get(entity_id, {"state": "unknown", "attributes": {}}))

    def list_states(self) -> tuple[dict[str, Any], ...]:
        return tuple({"entity_id": entity_id, **value} for entity_id, value in self.states.items())

    def call_service(self, command: dict[str, Any]) -> None:
        entity_id = command.get("entity_id")
        service = command.get("service")
        if isinstance(entity_id, str) and isinstance(service, str):
            if service == "turn_on":
                self.states.setdefault(entity_id, {"attributes": {}})["state"] = "on"
            elif service == "turn_off":
                self.states.setdefault(entity_id, {"attributes": {}})["state"] = "off"


class HomeAssistantRestGateway:
    """Bounded read-only Home Assistant REST client.

    The token is accepted only through construction by the configured runtime;
    it is never included in errors or evidence.  Mutation endpoints are not
    implemented by this gateway.
    """

    def __init__(self, base_url: str, token: str, timeout: float = 5.0) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Home Assistant URL must be an absolute HTTP(S) URL")
        self.base_url = base_url.rstrip("/") + "/"
        self.token = token
        self.timeout = timeout

    def _get_json(self, path: str) -> Any:
        request = Request(
            urljoin(self.base_url, path.lstrip("/")),
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            return response.read(1_000_001)

    def list_states(self) -> tuple[dict[str, Any], ...]:
        import json

        payload = json.loads(self._get_json("api/states"))
        if not isinstance(payload, list):
            raise ValueError("Home Assistant states response is not a list")
        return tuple(item for item in payload[:50] if isinstance(item, dict))

    def get_state(self, entity_id: str) -> dict[str, Any]:
        import json

        payload = json.loads(self._get_json(f"api/states/{entity_id}"))
        if not isinstance(payload, dict):
            raise ValueError("Home Assistant state response is not an object")
        return payload

    def call_service(self, command: dict[str, Any]) -> None:
        del command
        raise PermissionError("Home Assistant REST gateway is read-only")


class HomeAssistantRestControlGateway(HomeAssistantRestGateway):
    """Separate, explicitly selected low-risk service-call adapter."""

    def call_service(self, command: dict[str, Any]) -> None:
        entity_id = command.get("entity_id")
        service = command.get("service")
        if not isinstance(entity_id, str) or not isinstance(service, str):
            raise ValueError("Home Assistant service command is invalid")
        if "." in service:
            domain, service_name = service.split(".", 1)
        else:
            domain, service_name = entity_id.split(".", 1)[0], service
        if domain not in {"light", "switch", "input_boolean"} or service_name not in {
            "turn_on",
            "turn_off",
        }:
            raise PermissionError("Home Assistant service is outside the bounded control policy")
        request = Request(
            urljoin(self.base_url, f"api/services/{domain}/{service_name}"),
            data=json.dumps({"entity_id": entity_id}).encode(),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            response.read(1_000_001)


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

    def read_states(self, observed_at: datetime) -> tuple[DeviceState, ...]:
        states: list[DeviceState] = []
        for raw in self.gateway.list_states()[:50]:
            entity_id = raw.get("entity_id")
            if isinstance(entity_id, str) and entity_id:
                states.append(
                    DeviceState(
                        entity_id=entity_id,
                        state=str(raw.get("state", "unknown")),
                        attributes=dict(raw.get("attributes", {})),
                        observed_at=observed_at,
                    )
                )
        return tuple(states)

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
