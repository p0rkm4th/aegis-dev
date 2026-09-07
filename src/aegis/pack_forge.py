"""Bounded structural Pack scaffolding from the executable Pack contract.

Forge compiles declarations only.  It never installs, enables, grants authority,
executes code, or asks a model to invent Pack semantics.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field, field_validator

from .contracts import ActionCard, ActionSpec, StrictModel, VerificationContract
from .pack_lifecycle import PackBundle, PackManifest, PackUI, validate_pack_bundle

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")


class PackProposalV0(StrictModel):
    """Strict input containing only fields represented by current Pack contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    pack_id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9-]*$")
    version: str = Field(min_length=1)
    permissions: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    ui: PackUI | None = None
    cards: tuple[ActionCard, ...] = Field(max_length=100)

    @field_validator("permissions", "dependencies")
    @classmethod
    def validate_names(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("Pack declarations must be unique")
        if any(not _IDENTIFIER.fullmatch(value) for value in values):
            raise ValueError("Pack declarations must use stable lowercase identifiers")
        return values

    @field_validator("cards")
    @classmethod
    def validate_card_identifiers(cls, values: tuple[ActionCard, ...]) -> tuple[ActionCard, ...]:
        for card in values:
            if not _IDENTIFIER.fullmatch(card.action.action_id):
                raise ValueError("ActionCard action IDs must use stable identifiers")
            if not _IDENTIFIER.fullmatch(card.action.capability):
                raise ValueError("ActionCard capability IDs must use stable identifiers")
        return values


def build_workspace_candidate_proposal(
    requested_effect: str,
    *,
    candidate: dict[str, Any],
) -> PackProposalV0:
    """Build the one safe candidate shape currently supported by owner Forge review.

    CapabilityNeed research is untrusted input.  The candidate can identify the
    requested direction, but it cannot choose permissions, dependencies, runtime
    code, or lifecycle state.  The resulting proposal is therefore a structural
    Workspace stub with a deliberately unimplemented runtime.
    """

    if not requested_effect.strip():
        raise ValueError("Forge candidate requires a requested effect")
    if (
        candidate.get("kind") != "workspace_solution"
        or candidate.get("capability") != "workspace.artifact.create"
        or candidate.get("requires_owner_input") is not True
    ):
        raise ValueError("candidate is not an owner-selectable Workspace solution")
    # Natural-language effects are untrusted labels, not Pack identifiers.
    # Keep the generated namespace deterministic while removing punctuation
    # (and other non-ASCII identifier characters) before strict validation.
    normalized = re.sub(r"[^a-z0-9]+", "-", requested_effect.casefold())[:48].strip("-")
    if not normalized:
        raise ValueError("Forge candidate requires a stable requested effect")
    pack_id = f"generated-{normalized}"
    action_id = f"{pack_id}.artifact.create"
    return PackProposalV0(
        pack_id=pack_id,
        version="0.1.0",
        permissions=("workspace.write",),
        dependencies=(),
        ui=PackUI(
            label=f"Candidate · {requested_effect.strip()[:72]}",
            category="candidate",
            detail_view="workspace",
        ),
        cards=(
            ActionCard(
                action=ActionSpec(
                    action_id=action_id,
                    capability=f"{pack_id}.artifact.create",
                    required_permissions=("workspace.write",),
                    verification=VerificationContract(kind="readback"),
                ),
                summary="Prepare a bounded Workspace artifact after implementation review",
                relevance=1,
            ),
        ),
    )


def compile_pack_proposal(proposal: PackProposalV0) -> PackBundle:
    """Purely compile and validate a proposal into the production Pack bundle type."""

    # Revalidate at the boundary so callers cannot bypass strict proposal
    # validation through Pydantic's low-level model_copy(update=...) helper.
    proposal = PackProposalV0.model_validate(proposal.model_dump())
    bundle = PackBundle(
        manifest=PackManifest(
            pack_id=proposal.pack_id,
            version=proposal.version,
            permissions=proposal.permissions,
            dependencies=proposal.dependencies,
            ui=proposal.ui,
        ),
        cards=proposal.cards,
    )
    validate_pack_bundle(bundle)
    return bundle


def materialize_pack_skeleton(
    proposal: PackProposalV0,
    destination: Path,
    *,
    preview: bool = False,
) -> tuple[str, ...]:
    """Render a validated proposal through the pinned, local, task-free template."""

    bundle = compile_pack_proposal(proposal)
    try:
        from copier import run_copy
    except ImportError as exc:  # pragma: no cover - exercised by deployment packaging
        raise RuntimeError("Pack Forge requires the pinned Copier development extra") from exc

    template = Path(__file__).resolve().parents[2] / "templates" / "pack_forge"
    if not template.is_dir():
        raise RuntimeError("trusted local Pack Forge template is unavailable")
    data: dict[str, Any] = {
        "pack_id": bundle.manifest.pack_id,
        "version": bundle.manifest.version,
        "permissions": list(bundle.manifest.permissions),
        "dependencies": list(bundle.manifest.dependencies),
        "cards": [card.model_dump(mode="json") for card in bundle.cards],
    }
    run_copy(
        str(template),
        destination,
        data=data,
        defaults=True,
        skip_tasks=True,
        pretend=preview,
        quiet=True,
    )
    if preview:
        return tuple(
            str(path.relative_to(template)).removesuffix(".jinja")
            for path in sorted(template.rglob("*"))
            if path.is_file() and path.name not in {"copier.yml", "_copier.yml"}
        )
    return tuple(
        str(path.relative_to(destination))
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    )
