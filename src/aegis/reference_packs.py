"""Small reference Packs used to prove generic Core semantics."""

from __future__ import annotations

import shlex
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .contracts import (
    ActionCard,
    ActionSpec,
    ExecutionRequest,
    Observation,
    VerificationContract,
    VerificationResult,
)
from .gateway_rpc import (
    CorrelatedRpcClient,
    OpenClawGatewayRpc,
    OpenClawWebSocketChannel,
    RpcProtocolError,
)
from .household import PostgresHouseholdStore
from .pack_lifecycle import PackBundle, PackManifest


@dataclass(frozen=True)
class Pack:
    pack_id: str
    version: str
    cards: tuple[ActionCard, ...]


def reference_packs() -> tuple[Pack, ...]:
    return (
        Pack(
            "tasks",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="tasks.create",
                        capability="tasks.create",
                        required_permissions=("tasks.write",),
                        verification=VerificationContract(kind="readback"),
                    ),
                    summary="Create a task",
                    relevance=1,
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="tasks.list",
                        capability="tasks.read",
                        required_permissions=("tasks.read",),
                        verification=VerificationContract(kind="readback"),
                    ),
                    summary="Show tasks",
                    relevance=1,
                ),
            ),
        ),
        Pack(
            "kitchen",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="kitchen.groceries.add",
                        capability="kitchen.groceries.write",
                        required_permissions=("kitchen.write",),
                        verification=VerificationContract(kind="readback"),
                    ),
                    summary="Add an item to groceries",
                    relevance=1,
                ),
                ActionCard(
                    action=ActionSpec(
                        action_id="kitchen.groceries.list",
                        capability="kitchen.groceries.read",
                        required_permissions=("kitchen.read",),
                        verification=VerificationContract(kind="readback"),
                    ),
                    summary="Show grocery list",
                    relevance=1,
                ),
            ),
        ),
        Pack(
            "homelab",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="homelab.service.restart",
                        capability="homelab.service.restart",
                        required_permissions=("homelab.service.restart",),
                        verification=VerificationContract(kind="health"),
                    ),
                    summary="Restart a service and verify health",
                    relevance=1,
                ),
            ),
        ),
        Pack(
            "network",
            "0.1.0",
            (
                ActionCard(
                    action=ActionSpec(
                        action_id="network.probe",
                        capability="network.probe",
                        required_permissions=("network.read",),
                        verification=VerificationContract(kind="health"),
                    ),
                    summary="Probe an authorized network target",
                    relevance=1,
                ),
            ),
        ),
    )


def reference_bundles() -> tuple[PackBundle, ...]:
    """Manifest-backed versions of the three reference capabilities."""
    permissions = {
        "tasks": ("tasks.write", "tasks.read"),
        "kitchen": ("kitchen.write", "kitchen.read"),
        "homelab": ("homelab.service.restart",),
        "network": ("network.read",),
    }
    return tuple(
        PackBundle(
            manifest=PackManifest(
                pack_id=pack.pack_id,
                version=pack.version,
                permissions=permissions[pack.pack_id],
            ),
            cards=pack.cards,
        )
        for pack in reference_packs()
    )


@dataclass
class ReferenceWorld:
    tasks: list[dict[str, Any]] = field(default_factory=list)
    groceries: list[str] = field(default_factory=list)
    services: dict[str, str] = field(default_factory=lambda: {"test-service": "healthy"})


class ReferenceExecutor:
    """Deterministic fake adapter with readback evidence, not fake success."""

    def __init__(self, world: ReferenceWorld) -> None:
        self.world = world
        self._completed_keys: set[str] = set()

    def execute(self, request: ExecutionRequest) -> Observation:
        if request.idempotency_key in self._completed_keys:
            return Observation(
                execution_id=uuid4(),
                evidence={"replayed": True, "idempotency_key": request.idempotency_key},
                command_succeeded=True,
            )
        self._completed_keys.add(request.idempotency_key)
        args = request.action.arguments
        if request.action.action_id == "tasks.create":
            title = str(args["title"])
            self.world.tasks.append({"title": title, "status": "open"})
            evidence = {"collection": "tasks", "title": title}
        elif request.action.action_id == "kitchen.groceries.add":
            item = str(args["item"])
            self.world.groceries.append(item)
            evidence = {"collection": "groceries", "item": item}
        elif request.action.action_id == "homelab.service.restart":
            service = str(args["service"])
            self.world.services[service] = "healthy"
            evidence = {"service": service, "health": self.world.services[service]}
        else:
            return Observation(
                execution_id=uuid4(), evidence={"unknown_action": True}, command_succeeded=False
            )
        return Observation(execution_id=uuid4(), evidence=evidence, command_succeeded=True)


class ReferenceVerifier:
    def __init__(self, world: ReferenceWorld) -> None:
        self.world = world

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        evidence = observation.evidence
        if not observation.command_succeeded:
            return VerificationResult(verified=False, evidence=evidence, reason="adapter failed")
        if contract.kind == "readback" and evidence.get("collection") == "tasks":
            ok = any(t["title"] == evidence["title"] for t in self.world.tasks)
        elif contract.kind == "readback" and evidence.get("collection") == "groceries":
            ok = evidence["item"] in self.world.groceries
        elif contract.kind == "health":
            ok = self.world.services.get(str(evidence.get("service"))) == "healthy"
        else:
            ok = False
        return VerificationResult(
            verified=ok,
            evidence=evidence,
            reason="canonical readback verified" if ok else "canonical readback failed",
        )


class OpenClawGroceryExecutor:
    """Execute the existing grocery action through a paired OpenClaw Gateway.

    The acceptance store is an external newline-delimited record. The shell
    command is idempotency-guarded, and the command marker is only transport
    evidence; the verifier independently reads the record afterward.
    """

    def __init__(
        self,
        channel: OpenClawWebSocketChannel,
        state_path: str,
        canonical_store: PostgresHouseholdStore | None = None,
        principal: Any | None = None,
    ) -> None:
        if not channel.persistent:
            raise ValueError("OpenClaw grocery execution requires a persistent channel")
        self.channel = channel
        self.gateway = OpenClawGatewayRpc(CorrelatedRpcClient(channel))
        self.state_path = str(Path(state_path))
        self.canonical_store = canonical_store
        self.principal = principal

    def execute(self, request: ExecutionRequest) -> Observation:
        if request.action.action_id != "kitchen.groceries.add":
            return Observation(
                execution_id=request.action_id,
                evidence={"unknown_action": request.action.action_id},
                command_succeeded=False,
            )
        item = request.action.arguments.get("item")
        if not isinstance(item, str) or not item.strip():
            return Observation(
                execution_id=request.action_id,
                evidence={"invalid_item": True},
                command_succeeded=False,
            )
        marker = f"AEGIS_GROCERY_DONE_{uuid4().hex}"
        # Avoid control characters in PTY input: fish interprets a literal tab
        # as completion input rather than passing it to the command.
        record = f"{request.idempotency_key}|{item.strip()}"
        script = (
            f"touch {shlex.quote(self.state_path)}; "
            f"grep -Fqx -- {shlex.quote(record)} {shlex.quote(self.state_path)} "
            f"|| printf '%s\\n' {shlex.quote(record)} >> {shlex.quote(self.state_path)}; "
            f"printf '%s\\n' {shlex.quote(marker)}"
        )
        try:
            opened = self.gateway.terminal_open({"rows": 40, "cols": 120})
            session_id = opened["sessionId"]
            self.gateway.terminal_input(
                {"sessionId": session_id, "data": f"sh -c {shlex.quote(script)}\r"}
            )
            saw_marker = False
            terminal_output = ""
            for _ in range(32):
                event = self.channel.receive_event("terminal.data")
                terminal_output += str(event.get("data", ""))
                if any(line.strip() == marker for line in terminal_output.splitlines()):
                    saw_marker = True
                    break
            self.gateway.terminal_close({"sessionId": session_id})
        except (KeyError, RpcProtocolError) as exc:
            raise RpcProtocolError(
                "OpenClaw grocery execution did not produce a terminal outcome"
            ) from exc
        if saw_marker and self.canonical_store is not None and self.principal is not None:
            try:
                self.canonical_store.add_grocery(
                    self.principal, item.strip(), request.idempotency_key
                )
            except (PermissionError, ValueError) as exc:
                return Observation(
                    execution_id=request.action_id,
                    evidence={
                        "gateway": "openclaw",
                        "external_state_path": self.state_path,
                        "idempotency_key": request.idempotency_key,
                        "canonical_persistence_error": str(exc),
                    },
                    command_succeeded=False,
                )
        return Observation(
            execution_id=request.action_id,
            evidence={
                "gateway": "openclaw",
                "external_state_path": self.state_path,
                "idempotency_key": request.idempotency_key,
                "terminal_marker": marker,
                "terminal_marker_observed": saw_marker,
                "terminal_output_bytes": len(terminal_output),
            },
            command_succeeded=saw_marker,
        )


class OpenClawGroceryVerifier:
    """Independently read external grocery state after Gateway execution."""

    def __init__(
        self,
        canonical_store: PostgresHouseholdStore | None = None,
        principal: Any | None = None,
    ) -> None:
        self.canonical_store = canonical_store
        self.principal = principal

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        if contract.kind != "readback" or not observation.command_succeeded:
            return VerificationResult(
                verified=False,
                evidence=observation.evidence,
                reason="Gateway execution or verification contract failed",
            )
        path = observation.evidence.get("external_state_path")
        key = observation.evidence.get("idempotency_key")
        if not isinstance(path, str) or not isinstance(key, str):
            return VerificationResult(
                verified=False,
                evidence=observation.evidence,
                reason="missing external readback identity",
            )
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            return VerificationResult(
                verified=False,
                evidence={**observation.evidence, "readback_error": str(exc)},
                reason="external readback failed",
            )
        matches = [line for line in lines if line.startswith(f"{key}|")]
        verified = len(matches) == 1
        canonical_verified = True
        if verified and self.canonical_store is not None and self.principal is not None:
            try:
                canonical_verified = self.canonical_store.grocery_recorded(
                    self.principal, matches[0].split("|", 1)[1], key
                )
            except (PermissionError, ValueError):
                canonical_verified = False
        verified = verified and canonical_verified
        return VerificationResult(
            verified=verified,
            evidence={
                **observation.evidence,
                "external_records_for_key": len(matches),
                "canonical_grocery_verified": canonical_verified,
            },
            reason="external and canonical grocery readback verified"
            if verified
            else "external grocery postcondition failed",
        )


class OpenClawHomelabExecutor:
    """Run a preconfigured service restart through OpenClaw terminal execution.

    Commands are supplied by the trusted adapter configuration, never by model
    arguments. The model can select a known service identifier only.
    """

    def __init__(
        self,
        channel: OpenClawWebSocketChannel,
        services: dict[str, str],
    ) -> None:
        if not channel.persistent:
            raise ValueError("Homelab execution requires a persistent channel")
        self.channel = channel
        self.gateway = OpenClawGatewayRpc(CorrelatedRpcClient(channel))
        self.services = services

    def execute(self, request: ExecutionRequest) -> Observation:
        if request.action.action_id != "homelab.service.restart":
            return Observation(
                execution_id=request.action_id,
                evidence={"unknown_action": request.action.action_id},
                command_succeeded=False,
            )
        service = request.action.arguments.get("service")
        command = self.services.get(str(service))
        if command is None:
            return Observation(
                execution_id=request.action_id,
                evidence={"unknown_service": service},
                command_succeeded=False,
            )
        marker = f"AEGIS_HOMELAB_DONE_{uuid4().hex}"
        script = f"{command}; printf '%s\\n' {shlex.quote(marker)}"
        terminal_output = ""
        try:
            opened = self.gateway.terminal_open({"rows": 40, "cols": 120})
            session_id = opened["sessionId"]
            self.gateway.terminal_input(
                {"sessionId": session_id, "data": f"sh -c {shlex.quote(script)}\r"}
            )
            saw_marker = False
            for _ in range(32):
                event = self.channel.receive_event("terminal.data")
                terminal_output += str(event.get("data", ""))
                if any(line.strip() == marker for line in terminal_output.splitlines()):
                    saw_marker = True
                    break
            self.gateway.terminal_close({"sessionId": session_id})
        except (KeyError, RpcProtocolError) as exc:
            raise RpcProtocolError("Homelab restart did not produce a terminal outcome") from exc
        return Observation(
            execution_id=request.action_id,
            evidence={
                "gateway": "openclaw",
                "service": str(service),
                "terminal_marker": marker,
                "terminal_marker_observed": saw_marker,
                "terminal_output_bytes": len(terminal_output),
            },
            command_succeeded=saw_marker,
        )


class OpenClawHomelabVerifier:
    """Verify service health with an independent HTTP read, not Gateway output."""

    def __init__(self, health_endpoints: dict[str, str], timeout: float = 5.0) -> None:
        self.health_endpoints = health_endpoints
        self.timeout = timeout

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        service = observation.evidence.get("service")
        endpoint = self.health_endpoints.get(str(service))
        if contract.kind != "health" or not observation.command_succeeded or endpoint is None:
            return VerificationResult(
                verified=False,
                evidence=observation.evidence,
                reason="Homelab restart or health contract failed",
            )
        try:
            with urllib.request.urlopen(endpoint, timeout=self.timeout) as response:
                status = response.status
                body_bytes = len(response.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return VerificationResult(
                verified=False,
                evidence={**observation.evidence, "health_error": str(exc)},
                reason="independent service health read failed",
            )
        verified = status == 200
        return VerificationResult(
            verified=verified,
            evidence={
                **observation.evidence,
                "health_endpoint": endpoint,
                "health_status": status,
                "health_body_bytes": body_bytes,
            },
            reason="independent service health verified" if verified else "service health failed",
        )


class OpenClawNetworkProbeExecutor:
    """Run a bounded ping through OpenClaw after Core scope authorization."""

    def __init__(self, channel: OpenClawWebSocketChannel) -> None:
        if not channel.persistent:
            raise ValueError("Network probing requires a persistent channel")
        self.channel = channel
        self.gateway = OpenClawGatewayRpc(CorrelatedRpcClient(channel))

    def execute(self, request: ExecutionRequest) -> Observation:
        if request.action.action_id != "network.probe":
            return Observation(
                execution_id=request.action_id,
                evidence={"unknown_action": request.action.action_id},
                command_succeeded=False,
            )
        address = request.action.arguments.get("address")
        port = request.action.arguments.get("port")
        if not isinstance(address, str) or not isinstance(port, int) or not 1 <= port <= 65535:
            return Observation(
                execution_id=request.action_id,
                evidence={"invalid_endpoint": True},
                command_succeeded=False,
            )
        marker = f"AEGIS_NETWORK_DONE_{uuid4().hex}"
        script = (
            f"curl --fail --silent --show-error --max-time 3 "
            f"-o /dev/null -w HTTP_%{{http_code}}\\n "
            f"{shlex.quote(f'http://{address}:{port}/')} 2>&1; code=$?; "
            f"printf \"%s %s\\n\" {shlex.quote(marker)} $code"
        )
        terminal_output = ""
        try:
            opened = self.gateway.terminal_open({"rows": 40, "cols": 120})
            session_id = opened["sessionId"]
            self.gateway.terminal_input(
                {"sessionId": session_id, "data": f"sh -c {shlex.quote(script)}\r"}
            )
            for _ in range(32):
                event = self.channel.receive_event("terminal.data")
                terminal_output += str(event.get("data", ""))
                if marker in terminal_output:
                    break
            self.gateway.terminal_close({"sessionId": session_id})
        except (KeyError, RpcProtocolError) as exc:
            raise RpcProtocolError("Network probe did not produce a terminal outcome") from exc
        success = any(line.strip().endswith(" 0") for line in terminal_output.splitlines())
        return Observation(
            execution_id=request.action_id,
            evidence={
                "gateway": "openclaw",
                "address": address,
                "port": request.action.arguments.get("port"),
                "terminal_marker": marker,
                "terminal_output_bytes": len(terminal_output),
                "terminal_output_tail": terminal_output[-500:],
            },
            command_succeeded=success,
        )


class OpenClawNetworkProbeVerifier:
    """Independently verify target reachability with a TCP connection."""

    def __init__(self, timeout: float = 3.0) -> None:
        self.timeout = timeout

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        address = observation.evidence.get("address")
        port = observation.evidence.get("port")
        if contract.kind != "health" or not observation.command_succeeded:
            return VerificationResult(
                verified=False,
                evidence=observation.evidence,
                reason="network probe failed",
            )
        if not isinstance(address, str) or not isinstance(port, int):
            return VerificationResult(
                verified=False,
                evidence=observation.evidence,
                reason="network endpoint is invalid",
            )
        try:
            with socket.create_connection((address, port), timeout=self.timeout):
                pass
        except OSError as exc:
            return VerificationResult(
                False,
                {**observation.evidence, "readback_error": str(exc)},
                "independent network readback failed",
            )
        return VerificationResult(
            True,
            {**observation.evidence, "tcp_readback": "connected"},
            "independent network reachability verified",
        )


class PostgresGroceryListExecutor:
    """Adapt the canonical Kitchen grocery read to the generic Executor port."""

    def __init__(self, store: PostgresHouseholdStore, principal: Any) -> None:
        self.store = store
        self.principal = principal

    def execute(self, request: ExecutionRequest) -> Observation:
        if request.action.action_id != "kitchen.groceries.list":
            return Observation(
                execution_id=request.action_id,
                evidence={"unknown_action": request.action.action_id},
                command_succeeded=False,
            )
        return Observation(
            execution_id=request.action_id,
            evidence={
                "collection": "groceries",
                "items": list(self.store.list_groceries(self.principal)),
            },
            command_succeeded=True,
        )


class PostgresGroceryListVerifier:
    """Independently compare a grocery list observation with canonical state."""

    def __init__(self, store: PostgresHouseholdStore, principal: Any) -> None:
        self.store = store
        self.principal = principal

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        if contract.kind != "readback" or not observation.command_succeeded:
            return VerificationResult(
                verified=False, evidence=observation.evidence, reason="grocery list read failed"
            )
        expected = observation.evidence.get("items")
        actual = list(self.store.list_groceries(self.principal))
        verified = expected == actual
        return VerificationResult(
            verified=verified,
            evidence={**observation.evidence, "canonical_items": actual},
            reason=(
                "canonical grocery list verified"
                if verified
                else "canonical grocery list changed"
            ),
        )
