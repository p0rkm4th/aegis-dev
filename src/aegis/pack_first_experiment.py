"""Isolated Pack-first cognition experiment.

This module deliberately stops at bounded candidate selection.  It does not
execute actions, install Packs, grant permissions, or participate in the
production interaction boundary.  Its output is evidence for comparing the
incumbent with two Pack-first variants.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .contracts import ActionCard
from .pack_lifecycle import PackManager, PackStatus


class PackRouterStatus(StrEnum):
    SELECTED = "SELECTED"
    NO_VALID_PACK = "NO_VALID_PACK"
    NEED_CONTEXT = "NEED_CONTEXT"
    UNSUPPORTED = "UNSUPPORTED"


class PackRouterResponse(BaseModel):
    """Strict, non-authoritative output from an experimental Pack router."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    selected_pack_ids: tuple[str, ...] = Field(default=(), max_length=4)
    status: PackRouterStatus


class PackRouter(Protocol):
    def __call__(self, prompt: str) -> Any: ...


class PackPrefilter(Protocol):
    def __call__(self, utterance: str, manager: PackManager) -> Sequence[str]: ...


@dataclass(frozen=True)
class PackCatalogEntry:
    pack_id: str
    label: str
    category: str
    status: PackStatus
    capabilities: tuple[str, ...]
    action_summaries: tuple[str, ...]


@dataclass(frozen=True)
class PackRouteMeasurement:
    variant: str
    utterance: str
    status: PackRouterStatus
    selected_pack_ids: tuple[str, ...]
    valid: bool
    correct: bool | None
    model_calls: int
    prompt_bytes: int
    context_bytes: int
    latency_ms: float
    failure_category: str | None = None


def compact_pack_catalog(
    manager: PackManager, pack_ids: Iterable[str] | None = None
) -> tuple[PackCatalogEntry, ...]:
    """Expose only Pack metadata; never canonical owner data."""

    allowed = set(pack_ids) if pack_ids is not None else None
    entries: list[PackCatalogEntry] = []
    for bundle, status, _grants in manager.lifecycle_snapshot():
        pack_id = bundle.manifest.pack_id
        if allowed is not None and pack_id not in allowed:
            continue
        ui = bundle.manifest.ui
        entries.append(
            PackCatalogEntry(
                pack_id=pack_id,
                label=ui.label if ui is not None else pack_id,
                category=ui.category if ui is not None else "capability",
                status=status,
                capabilities=tuple(card.action.capability for card in bundle.cards[:8]),
                action_summaries=tuple(card.summary[:180] for card in bundle.cards[:8]),
            )
        )
    return tuple(entries)


def router_prompt(utterance: str, catalog: Sequence[PackCatalogEntry]) -> str:
    """Build the intentionally compact Pack-only router context."""

    lines = ["Select Packs for this request.", f"Request: {utterance[:1_000]}", "Packs:"]
    for entry in catalog:
        lines.append(
            f"- {entry.pack_id} | {entry.label} | {entry.category} | {entry.status.value} | "
            f"capabilities: {', '.join(entry.capabilities[:6])}"
        )
    lines.append(
        'Return JSON only: {"status":"SELECTED|NO_VALID_PACK|NEED_CONTEXT|UNSUPPORTED",'
        '"selected_pack_ids":["pack-id"]}'
    )
    return "\n".join(lines)


def parse_router_response(raw: Any) -> PackRouterResponse:
    """Parse malformed experimental output fail-closed."""

    response = PackRouterResponse.model_validate(raw)
    if len(set(response.selected_pack_ids)) != len(response.selected_pack_ids):
        raise ValueError("Pack router selected a duplicate Pack")
    if response.status is not PackRouterStatus.SELECTED and response.selected_pack_ids:
        raise ValueError("non-selected Pack router status cannot contain Pack IDs")
    if response.status is PackRouterStatus.SELECTED and not response.selected_pack_ids:
        raise ValueError("selected Pack router status requires a Pack")
    return response


def validate_selected_packs(response: PackRouterResponse, manager: PackManager) -> tuple[str, ...]:
    """Validate lifecycle without changing lifecycle or authority."""

    selected: list[str] = []
    for pack_id in response.selected_pack_ids:
        try:
            status = manager.status(pack_id)
        except KeyError as exc:
            raise ValueError(f"unknown Pack selected: {pack_id}") from exc
        if status is not PackStatus.ENABLED:
            raise ValueError(f"Pack is not enabled: {pack_id}")
        selected.append(pack_id)
    return tuple(selected)


def selected_cards(manager: PackManager, pack_ids: Iterable[str]) -> tuple[ActionCard, ...]:
    """Retrieve cards only from already-enabled selected Packs."""

    allowed = set(pack_ids)
    return tuple(
        card
        for card in manager.enabled_cards()
        if card.action.action_id.split(".", 1)[0] in allowed
    )


def _measure_router(
    variant: str,
    utterance: str,
    manager: PackManager,
    router: PackRouter,
    catalog: Sequence[PackCatalogEntry],
    expected_pack_ids: frozenset[str] | None,
) -> PackRouteMeasurement:
    prompt = router_prompt(utterance, catalog)
    started = time.perf_counter()
    try:
        response = parse_router_response(router(prompt))
        selected = validate_selected_packs(response, manager)
    except (TypeError, ValueError) as exc:
        return PackRouteMeasurement(
            variant=variant,
            utterance=utterance,
            status=PackRouterStatus.NO_VALID_PACK,
            selected_pack_ids=(),
            valid=False,
            correct=False if expected_pack_ids is not None else None,
            model_calls=1,
            prompt_bytes=len(prompt.encode()),
            context_bytes=0,
            latency_ms=(time.perf_counter() - started) * 1_000,
            failure_category=type(exc).__name__,
        )
    return PackRouteMeasurement(
        variant=variant,
        utterance=utterance,
        status=response.status,
        selected_pack_ids=selected,
        valid=True,
        correct=(set(selected) == set(expected_pack_ids))
        if expected_pack_ids is not None
        else None,
        model_calls=1,
        prompt_bytes=len(prompt.encode()),
        context_bytes=0,
        latency_ms=(time.perf_counter() - started) * 1_000,
    )


def measure_incumbent(
    utterance: str,
    incumbent: Callable[[str], Sequence[ActionCard]],
    expected_pack_ids: frozenset[str] | None = None,
) -> PackRouteMeasurement:
    """Measure the unchanged production candidate selector as control."""

    started = time.perf_counter()
    cards = tuple(incumbent(utterance))
    selected = tuple(sorted({card.action.action_id.split(".", 1)[0] for card in cards}))
    return PackRouteMeasurement(
        variant="incumbent",
        utterance=utterance,
        status=PackRouterStatus.SELECTED if selected else PackRouterStatus.NO_VALID_PACK,
        selected_pack_ids=selected,
        valid=True,
        correct=(set(selected) == set(expected_pack_ids))
        if expected_pack_ids is not None
        else None,
        model_calls=0,
        prompt_bytes=0,
        context_bytes=0,
        latency_ms=(time.perf_counter() - started) * 1_000,
    )


def measure_pack_first(
    utterance: str,
    manager: PackManager,
    router: PackRouter,
    expected_pack_ids: frozenset[str] | None = None,
) -> PackRouteMeasurement:
    """Measure pure Pack-first routing using the complete compact catalog."""

    return _measure_router(
        "pack_first",
        utterance,
        manager,
        router,
        compact_pack_catalog(manager),
        expected_pack_ids,
    )


def measure_retrieval_assisted_pack_first(
    utterance: str,
    manager: PackManager,
    router: PackRouter,
    prefilter: PackPrefilter,
    expected_pack_ids: frozenset[str] | None = None,
) -> PackRouteMeasurement:
    """Measure Pack-first after a non-authoritative semantic Pack prefilter."""

    candidate_ids = tuple(prefilter(utterance, manager))[:10]
    return _measure_router(
        "retrieval_assisted_pack_first",
        utterance,
        manager,
        router,
        compact_pack_catalog(manager, candidate_ids),
        expected_pack_ids,
    )
