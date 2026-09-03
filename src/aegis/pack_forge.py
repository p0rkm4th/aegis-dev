"""Bounded structural Pack scaffolding from the executable Pack contract.

Forge compiles declarations only.  It never installs, enables, grants authority,
executes code, or asks a model to invent Pack semantics.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field, field_validator

from .contracts import ActionCard, StrictModel
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
