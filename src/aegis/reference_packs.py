"""Small reference Packs used to prove generic Core semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
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
