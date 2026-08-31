"""Minimal, local-first trading-card collection Pack implementation."""

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


@dataclass
class CardCollection:
    items: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, *, set_name: str | None = None, number: str | None = None) -> None:
        if not name.strip():
            raise ValueError("card name is required")
        self.items.append({"name": name, "set": set_name, "number": number})


def card_collection_card(pack_id: str) -> ActionCard:
    return ActionCard(
        action=ActionSpec(
            action_id=f"{pack_id}.items.add",
            capability=f"{pack_id}.items.add",
            required_permissions=("collection.write",),
            verification=VerificationContract(kind="readback"),
        ),
        summary="Add a trading card to the collection",
        relevance=1.0,
    )


class CardCollectionExecutor:
    def __init__(self, collection: CardCollection, action_id: str) -> None:
        self.collection = collection
        self.action_id = action_id

    def execute(self, request: ExecutionRequest) -> Observation:
        if request.action.action_id != self.action_id:
            return Observation(
                execution_id=uuid4(), evidence={"unknown_action": True}, command_succeeded=False
            )
        args = request.action.arguments
        self.collection.add(str(args["name"]), set_name=args.get("set"), number=args.get("number"))
        return Observation(
            execution_id=uuid4(),
            evidence={"collection": "trading_cards", "name": str(args["name"])},
            command_succeeded=True,
        )


class CardCollectionVerifier:
    def __init__(self, collection: CardCollection) -> None:
        self.collection = collection

    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult:
        name = observation.evidence.get("name")
        verified = (
            observation.command_succeeded
            and contract.kind == "readback"
            and any(item["name"] == name for item in self.collection.items)
        )
        return VerificationResult(
            verified=verified,
            evidence=observation.evidence,
            reason="collection readback verified" if verified else "collection readback failed",
        )
