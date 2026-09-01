"""Small reference Packs used to prove generic Core semantics."""

from __future__ import annotations

import shlex
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
    )


def reference_bundles() -> tuple[PackBundle, ...]:
    """Manifest-backed versions of the three reference capabilities."""
    permissions = {
        "tasks": ("tasks.write",),
        "kitchen": ("kitchen.write",),
        "homelab": ("homelab.service.restart",),
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

    def __init__(self, channel: OpenClawWebSocketChannel, state_path: str) -> None:
        if not channel.persistent:
            raise ValueError("OpenClaw grocery execution requires a persistent channel")
        self.channel = channel
        self.gateway = OpenClawGatewayRpc(CorrelatedRpcClient(channel))
        self.state_path = str(Path(state_path))

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
        return VerificationResult(
            verified=verified,
            evidence={**observation.evidence, "external_records_for_key": len(matches)},
            reason="independent external grocery readback verified"
            if verified
            else "external grocery postcondition failed",
        )
