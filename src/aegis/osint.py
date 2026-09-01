"""Source-grounded OSINT investigation state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from .contracts import ActionCard, Principal
from .pack_lifecycle import PackBundle, PackManifest


@dataclass(frozen=True)
class Source:
    source_id: UUID
    locator: str
    title: str
    collected_at: datetime


@dataclass(frozen=True)
class Finding:
    finding_id: UUID
    statement: str
    source_ids: tuple[UUID, ...]
    confidence: float


@dataclass
class Investigation:
    investigation_id: UUID = field(default_factory=uuid4)
    sources: dict[UUID, Source] = field(default_factory=dict)
    findings: dict[UUID, Finding] = field(default_factory=dict)

    def add_source(self, locator: str, title: str, collected_at: datetime) -> Source:
        if not locator or not title:
            raise ValueError("source locator and title are required")
        source = Source(uuid4(), locator, title, collected_at)
        self.sources[source.source_id] = source
        return source

    def add_finding(
        self, statement: str, source_ids: tuple[UUID, ...], confidence: float
    ) -> Finding:
        if not statement.strip() or not source_ids or not 0 <= confidence <= 1:
            raise ValueError("finding requires statement, sources, and bounded confidence")
        if any(source_id not in self.sources for source_id in source_ids):
            raise ValueError("finding references an unknown source")
        finding = Finding(uuid4(), statement, source_ids, confidence)
        self.findings[finding.finding_id] = finding
        return finding


class PostgresInvestigationStore:
    """Persist source-grounded investigations partitioned by Vault owner."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def save(self, principal: Principal, investigation: Investigation) -> None:
        payload = {
            "sources": [
                {
                    "source_id": str(source.source_id),
                    "locator": source.locator,
                    "title": source.title,
                    "collected_at": source.collected_at.isoformat(),
                }
                for source in investigation.sources.values()
            ],
            "findings": [
                {
                    "finding_id": str(finding.finding_id),
                    "statement": finding.statement,
                    "source_ids": [str(source_id) for source_id in finding.source_ids],
                    "confidence": finding.confidence,
                }
                for finding in investigation.findings.values()
            ],
        }
        self.connection.execute(
            "INSERT INTO osint_investigations (investigation_id, owner_id, payload) "
            "VALUES (%s, %s, %s) ON CONFLICT (investigation_id) DO UPDATE SET "
            "payload = EXCLUDED.payload, updated_at = now()",
            (
                str(investigation.investigation_id),
                principal.id,
                json.dumps(payload, sort_keys=True),
            ),
        )
        self.connection.commit()

    def load(self, principal: Principal, investigation_id: UUID) -> Investigation:
        row = self.connection.execute(
            "SELECT owner_id, payload FROM osint_investigations WHERE investigation_id = %s",
            (str(investigation_id),),
        ).fetchone()
        if row is None:
            raise KeyError("OSINT investigation is unavailable")
        if str(row[0]) != principal.id:
            raise PermissionError("OSINT investigation belongs to another Vault")
        payload = row[1] if isinstance(row[1], dict) else json.loads(str(row[1]))
        investigation = Investigation(investigation_id)
        for item in payload.get("sources", []):
            source = Source(
                UUID(str(item["source_id"])),
                str(item["locator"]),
                str(item["title"]),
                datetime.fromisoformat(str(item["collected_at"])),
            )
            investigation.sources[source.source_id] = source
        for item in payload.get("findings", []):
            finding = Finding(
                UUID(str(item["finding_id"])),
                str(item["statement"]),
                tuple(UUID(str(source_id)) for source_id in item["source_ids"]),
                float(item["confidence"]),
            )
            if any(source_id not in investigation.sources for source_id in finding.source_ids):
                raise ValueError("persisted finding references an unknown source")
            investigation.findings[finding.finding_id] = finding
        return investigation


@dataclass(frozen=True)
class CapabilityGap:
    description: str
    requested_by: str
    requested_permissions: tuple[str, ...] = ()


@dataclass(frozen=True)
class PackProposal:
    pack_id: str
    purpose: str
    permissions: tuple[str, ...]
    dependencies: tuple[str, ...]
    requires_approval: bool = True


class ForgeStatus(StrEnum):
    DRAFT = "draft"
    VALIDATED = "validated"
    APPROVED = "approved"
    INSTALLED = "installed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ForgeRecord:
    proposal: PackProposal
    status: ForgeStatus
    license_class: str | None = None


class Forge:
    """Draft-only Forge stage; install is a separate explicit owner operation."""

    def propose(self, gap: CapabilityGap) -> PackProposal:
        if not gap.description.strip() or not gap.requested_by.strip():
            raise ValueError("capability gap requires requester and description")
        normalized = "-".join(gap.description.casefold().split())[:48].strip("-")
        return PackProposal(
            pack_id=f"generated-{normalized}",
            purpose=gap.description,
            permissions=gap.requested_permissions,
            dependencies=(),
        )

    def install(self, proposal: PackProposal, approved: bool) -> None:
        if not approved:
            raise PermissionError("Forge installation requires explicit approval")
        raise NotImplementedError(
            "installation requires Pack validation and owner approval integration"
        )


class ForgeLifecycle:
    """Fail-closed validation and approval state for generated Pack proposals."""

    def __init__(self) -> None:
        self.records: dict[str, ForgeRecord] = {}

    def validate(
        self,
        proposal: PackProposal,
        *,
        license_class: str,
        sandbox_passed: bool,
        tests_passed: bool,
    ) -> ForgeRecord:
        if license_class not in {"COPY_SAFE", "INTERFACE_ONLY", "CLEAN_ROOM_ONLY"}:
            raise PermissionError("Forge proposal has unapproved license provenance")
        if not sandbox_passed or not tests_passed:
            raise ValueError("Forge proposal requires passing sandbox and tests")
        record = ForgeRecord(proposal, ForgeStatus.VALIDATED, license_class)
        self.records[proposal.pack_id] = record
        return record

    def approve(self, pack_id: str, owner_approved: bool) -> ForgeRecord:
        record = self._require(pack_id)
        if record.status is not ForgeStatus.VALIDATED:
            raise ValueError("only validated proposals can be approved")
        if not owner_approved:
            self.records[pack_id] = ForgeRecord(
                record.proposal, ForgeStatus.REJECTED, record.license_class
            )
            raise PermissionError("Forge installation requires owner approval")
        record = ForgeRecord(record.proposal, ForgeStatus.APPROVED, record.license_class)
        self.records[pack_id] = record
        return record

    def install(self, pack_id: str) -> ForgeRecord:
        record = self._require(pack_id)
        if record.status is not ForgeStatus.APPROVED:
            raise PermissionError("Forge installation is not approved")
        record = ForgeRecord(record.proposal, ForgeStatus.INSTALLED, record.license_class)
        self.records[pack_id] = record
        return record

    def to_bundle(
        self, pack_id: str, cards: tuple[ActionCard, ...], *, version: str = "0.1.0"
    ) -> PackBundle:
        """Materialize a validated, installed proposal for Pack lifecycle review.

        Conversion does not install or enable the Pack.  PackManager remains the
        authority for namespace, permission, and lifecycle validation.
        """
        record = self._require(pack_id)
        if record.status is not ForgeStatus.INSTALLED:
            raise PermissionError("only installed Forge proposals can become Pack bundles")
        return PackBundle(
            manifest=PackManifest(
                pack_id=record.proposal.pack_id,
                version=version,
                permissions=record.proposal.permissions,
                dependencies=record.proposal.dependencies,
            ),
            cards=cards,
        )

    def _require(self, pack_id: str) -> ForgeRecord:
        try:
            return self.records[pack_id]
        except KeyError as exc:
            raise KeyError("unknown Forge proposal") from exc
