"""Fail-closed, evidence-first state for authorized security-lab work."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4

from .network import HomelabInventory, ScopeDenied


@dataclass(frozen=True)
class LabFinding:
    finding_id: UUID
    scope_id: str
    target: str
    statement: str
    evidence: tuple[str, ...]
    observed_at: datetime


@dataclass
class SecurityLab:
    """Record bounded lab observations only after explicit scope authorization.

    This boundary deliberately does not perform active testing. Any future
    tool adapter must call ``authorize_target`` before it can operate.
    """

    inventory: HomelabInventory
    findings: dict[UUID, LabFinding] = field(default_factory=dict)

    def authorize_target(self, scope_id: str, target: str) -> None:
        self.inventory.require_action_scope(scope_id, target)

    def record_finding(
        self,
        scope_id: str,
        target: str,
        statement: str,
        evidence: tuple[str, ...],
    ) -> LabFinding:
        self.authorize_target(scope_id, target)
        if not statement.strip() or not evidence or any(not item.strip() for item in evidence):
            raise ValueError("security finding requires a statement and evidence")
        finding = LabFinding(
            uuid4(), scope_id, target, statement, evidence, datetime.now(timezone.utc)
        )
        self.findings[finding.finding_id] = finding
        return finding


__all__ = ["LabFinding", "ScopeDenied", "SecurityLab"]
