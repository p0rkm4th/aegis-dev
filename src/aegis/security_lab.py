"""Fail-closed, evidence-first state for authorized security-lab work."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from .contracts import Principal
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


class PostgresSecurityLabStore:
    """Persist authorized-lab findings under the requester's Space and Vault."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def save(self, principal: Principal, finding: LabFinding) -> None:
        if not principal.space_ids:
            raise PermissionError("security findings require an explicit Space")
        space_id = principal.space_ids[0]
        active = self.connection.execute(
            "SELECT 1 FROM space_memberships WHERE principal_id = %s AND space_id = %s "
            "AND active = TRUE",
            (principal.id, space_id),
        ).fetchone()
        if active is None:
            raise PermissionError("principal is not an active Space member")
        payload = {
            "scope_id": finding.scope_id,
            "target": finding.target,
            "statement": finding.statement,
            "evidence": list(finding.evidence),
            "observed_at": finding.observed_at.isoformat(),
        }
        self.connection.execute(
            "INSERT INTO security_lab_findings (finding_id, owner_id, space_id, payload) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (finding_id) DO UPDATE SET "
            "payload = EXCLUDED.payload, updated_at = now()",
            (str(finding.finding_id), principal.id, space_id, json.dumps(payload, sort_keys=True)),
        )
        self.connection.commit()

    def load(self, principal: Principal, finding_id: UUID) -> LabFinding:
        row = self.connection.execute(
            "SELECT owner_id, payload FROM security_lab_findings WHERE finding_id = %s",
            (str(finding_id),),
        ).fetchone()
        if row is None:
            raise KeyError("security finding is unavailable")
        if str(row[0]) != principal.id:
            raise PermissionError("security finding belongs to another Vault")
        payload = row[1] if isinstance(row[1], dict) else json.loads(str(row[1]))
        return LabFinding(
            finding_id,
            str(payload["scope_id"]),
            str(payload["target"]),
            str(payload["statement"]),
            tuple(str(item) for item in payload["evidence"]),
            datetime.fromisoformat(str(payload["observed_at"])),
        )


__all__ = ["LabFinding", "ScopeDenied", "SecurityLab"]
