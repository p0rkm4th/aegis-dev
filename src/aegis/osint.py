"""Source-grounded OSINT investigation state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4


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


@dataclass(frozen=True)
class CapabilityGap:
    description: str
    requested_by: str


@dataclass(frozen=True)
class PackProposal:
    pack_id: str
    purpose: str
    permissions: tuple[str, ...]
    dependencies: tuple[str, ...]
    requires_approval: bool = True


class Forge:
    """Draft-only Forge stage; install is a separate explicit owner operation."""

    def propose(self, gap: CapabilityGap) -> PackProposal:
        if not gap.description.strip() or not gap.requested_by.strip():
            raise ValueError("capability gap requires requester and description")
        normalized = "-".join(gap.description.casefold().split())[:48].strip("-")
        return PackProposal(
            pack_id=f"generated-{normalized}",
            purpose=gap.description,
            permissions=(),
            dependencies=(),
        )

    def install(self, proposal: PackProposal, approved: bool) -> None:
        if not approved:
            raise PermissionError("Forge installation requires explicit approval")
        raise NotImplementedError(
            "installation requires Pack validation and owner approval integration"
        )
