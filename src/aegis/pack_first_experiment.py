"""Isolated Pack-first cognition experiment.

This module deliberately stops at bounded candidate selection.  It does not
execute actions, install Packs, grant permissions, or participate in the
production interaction boundary.  Its output is evidence for comparing the
incumbent with two Pack-first variants.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .contracts import ActionCard
from .pack_lifecycle import PackManager, PackStatus
from .registry import CapabilityEmbedder


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


@dataclass(frozen=True)
class PackCase:
    utterance: str
    expected_pack_ids: frozenset[str] | None


OWNER_PACK_CORPUS: tuple[PackCase, ...] = (
    PackCase("we ran out of milk", frozenset({"dynamic-food"})),
    PackCase("what did I spend at the store", frozenset({"dynamic-finance"})),
    PackCase("is the media server reachable", frozenset({"dynamic-homelab"})),
    PackCase("what could I add safely", frozenset({"dynamic-forge"})),
    PackCase("groceris for tonight", frozenset({"dynamic-food"})),
    PackCase("show me the other one", None),
    PackCase("nah, tomorrow", None),
    PackCase("restart that instead", None),
    PackCase("tell me everything", None),
    PackCase("install this unknown Pack", frozenset()),
    PackCase("which pantry items are low", frozenset({"dynamic-food"})),
    PackCase("can I afford eighty dollars for groceries", frozenset({"dynamic-finance"})),
    PackCase("which services are unhealthy", frozenset({"dynamic-homelab"})),
    PackCase("show authorized network observations", frozenset({"dynamic-homelab"})),
    PackCase("what should I do about this unsupported capability", frozenset({"dynamic-forge"})),
    PackCase("make a report from the host inventory", frozenset({"dynamic-homelab"})),
    PackCase("add oat milk to my list", frozenset({"dynamic-food"})),
    PackCase("what did the hardware store cost last month", frozenset({"dynamic-finance"})),
    PackCase("is the Plex service healthy", frozenset({"dynamic-homelab"})),
    PackCase(
        "what can I safely investigate for this missing ability", frozenset({"dynamic-forge"})
    ),
    PackCase(
        "show grocery needs and whether eighty dollars is affordable",
        frozenset({"dynamic-food", "dynamic-finance"}),
    ),
    PackCase(
        "can I spend eighty dollars on groceries",
        frozenset({"dynamic-finance"}),
    ),
    PackCase(
        "can I keep tonight's grocery trip under eighty dollars based on what we need",
        frozenset({"dynamic-food", "dynamic-finance"}),
    ),
    PackCase("what groceries do we still need", frozenset({"dynamic-food"})),
    PackCase("what's in the pantry right now", frozenset({"dynamic-food"})),
    PackCase("did we already buy oat milk", frozenset({"dynamic-food"})),
    PackCase(
        "can I keep tonight's grocery trip under eighty dollars based on what we still need",
        frozenset({"dynamic-food", "dynamic-finance"}),
    ),
    PackCase(
        "check groceries then tell me if Plex is down",
        frozenset({"dynamic-food", "dynamic-homelab"}),
    ),
    PackCase("no, the other service", None),
    PackCase("enable that Pack for me", frozenset()),
)

_PACK_LIFECYCLE_LANGUAGE = re.compile(r"\b(?:install|enable|approve|grant)\b", re.IGNORECASE)


def is_pack_lifecycle_request(utterance: str) -> bool:
    """Recognize authority language for experimental fail-closed evaluation only."""

    return bool(_PACK_LIFECYCLE_LANGUAGE.search(utterance))


@dataclass(frozen=True)
class PackFollowUpContext:
    """Small, bounded context permitted for one experimental follow-up retry."""

    prior_pack_ids: tuple[str, ...] = ()
    canonical_result_type: str | None = None
    referent_ids: tuple[str, ...] = ()
    objective_requirements: tuple[str, ...] = ()


def route_with_one_context_retry(
    utterance: str,
    manager: PackManager,
    router: PackRouter,
    context: PackFollowUpContext | None = None,
) -> tuple[PackRouterResponse, int]:
    """Route with at most one bounded context retry; never grants authority."""

    if is_pack_lifecycle_request(utterance):
        return PackRouterResponse(status=PackRouterStatus.UNSUPPORTED), 0

    def route(prompt: str) -> PackRouterResponse:
        response = parse_router_response(router(prompt))
        validate_selected_packs(response, manager)
        return response

    response = route(router_prompt(utterance, compact_pack_catalog(manager, enabled_only=True)))
    calls = 1
    if response.status is PackRouterStatus.NEED_CONTEXT and context is not None:
        bounded = (
            "Bounded follow-up context (context only; no authority):\n"
            f"prior_pack_ids: {list(context.prior_pack_ids[:4])}\n"
            f"canonical_result_type: {context.canonical_result_type or ''}\n"
            f"referent_ids: {list(context.referent_ids[:8])}\n"
            f"objective_requirements: {list(context.objective_requirements[:8])}\n"
        )
        response = route(
            router_prompt(utterance, compact_pack_catalog(manager, enabled_only=True))
            + "\n"
            + bounded
        )
        calls += 1
    return response, calls


@dataclass(frozen=True)
class PackTournamentReport:
    measurements: tuple[PackRouteMeasurement, ...]
    metrics: dict[str, dict[str, float]]


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return ordered[index]


def _variant_metrics(measurements: Sequence[PackRouteMeasurement]) -> dict[str, float]:
    judged = [item for item in measurements if item.correct is not None]
    correct = sum(item.correct is True for item in judged)
    metrics = {
        "routing_recall": correct / len(judged) if judged else 0.0,
        "wrong_action_rate": (
            sum(item.valid and item.correct is False for item in judged) / len(judged)
            if judged
            else 0.0
        ),
        "clarification_rate": sum(
            item.status in {PackRouterStatus.NEED_CONTEXT, PackRouterStatus.UNSUPPORTED}
            for item in measurements
        )
        / len(measurements)
        if measurements
        else 0.0,
        "model_calls_per_case": sum(item.model_calls for item in measurements) / len(measurements)
        if measurements
        else 0.0,
        "prompt_bytes_per_case": sum(item.prompt_bytes for item in measurements) / len(measurements)
        if measurements
        else 0.0,
        "context_bytes_per_case": sum(item.context_bytes for item in measurements)
        / len(measurements)
        if measurements
        else 0.0,
        "p50_latency_ms": _percentile([item.latency_ms for item in measurements], 0.50),
        "p95_latency_ms": _percentile([item.latency_ms for item in measurements], 0.95),
    }
    metrics["invalid_measurement_rate"] = (
        sum(not item.valid for item in measurements) / len(measurements) if measurements else 0.0
    )
    failure_counts: dict[str, int] = {}
    for item in measurements:
        if item.failure_category:
            failure_counts[item.failure_category] = failure_counts.get(item.failure_category, 0) + 1
    for category, count in failure_counts.items():
        metrics[f"failure_category.{category}"] = float(count)
    return metrics


def run_pack_tournament(
    cases: Sequence[PackCase],
    manager: PackManager,
    incumbent: Callable[[str], Sequence[ActionCard]],
    router: PackRouter,
    prefilter: PackPrefilter,
) -> PackTournamentReport:
    """Run a deterministic corpus through all experimental variants."""

    measurements: list[PackRouteMeasurement] = []
    for case in cases:
        expected = case.expected_pack_ids
        measurements.extend(
            (
                measure_incumbent(case.utterance, incumbent, expected),
                measure_pack_first(case.utterance, manager, router, expected),
                measure_retrieval_assisted_pack_first(
                    case.utterance, manager, router, prefilter, expected
                ),
            )
        )
    grouped: dict[str, list[PackRouteMeasurement]] = {}
    for measurement in measurements:
        grouped.setdefault(measurement.variant, []).append(measurement)
    return PackTournamentReport(
        measurements=tuple(measurements),
        metrics={variant: _variant_metrics(values) for variant, values in grouped.items()},
    )


def compact_pack_catalog(
    manager: PackManager,
    pack_ids: Iterable[str] | None = None,
    *,
    enabled_only: bool = False,
) -> tuple[PackCatalogEntry, ...]:
    """Expose Pack metadata; router catalogs may be restricted to enabled Packs."""

    allowed = set(pack_ids) if pack_ids is not None else None
    entries: list[PackCatalogEntry] = []
    for bundle, status, _grants in manager.lifecycle_snapshot():
        if enabled_only and status is not PackStatus.ENABLED:
            continue
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


def semantic_pack_prefilter(
    utterance: str,
    manager: PackManager,
    embedder: CapabilityEmbedder,
    *,
    card_limit: int = 10,
) -> tuple[str, ...]:
    """Return a ranked Pack shortlist from existing enabled-card retrieval.

    Retrieval is only a non-authoritative narrowing hint.  The router still
    validates every selected Pack against lifecycle state before any later
    experimental stage could use it.
    """

    if not 1 <= card_limit <= 10:
        raise ValueError("semantic Pack prefilter card limit must be between one and ten")
    pack_ids: list[str] = []
    for card in manager.retrieve_semantic(utterance, embedder, limit=card_limit):
        pack_id = card.action.action_id.split(".", 1)[0]
        if pack_id not in pack_ids:
            pack_ids.append(pack_id)
    return tuple(pack_ids)


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
    if is_pack_lifecycle_request(utterance):
        return PackRouteMeasurement(
            variant=variant,
            utterance=utterance,
            status=PackRouterStatus.UNSUPPORTED,
            selected_pack_ids=(),
            valid=True,
            correct=(set() == set(expected_pack_ids)) if expected_pack_ids is not None else None,
            model_calls=0,
            prompt_bytes=0,
            context_bytes=0,
            latency_ms=(time.perf_counter() - started) * 1_000,
            failure_category="lifecycle_authority_request",
        )
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
        compact_pack_catalog(manager, enabled_only=True),
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
        compact_pack_catalog(manager, candidate_ids, enabled_only=True),
        expected_pack_ids,
    )
