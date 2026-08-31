"""Structured health/readiness results for operators and deployment checks."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from pydantic import Field

from .contracts import StrictModel


class ComponentHealth(StrictModel):
    name: str = Field(min_length=1)
    healthy: bool
    required: bool = True
    detail: str = Field(min_length=1)


class HealthReport(StrictModel):
    healthy: bool
    ready: bool
    components: tuple[ComponentHealth, ...]


class HealthService:
    def __init__(self, checks: Iterable[tuple[str, bool, bool, str]]) -> None:
        self.checks = tuple(checks)

    def report(self) -> HealthReport:
        components = tuple(
            ComponentHealth(name=name, healthy=healthy, required=required, detail=detail)
            for name, healthy, required, detail in self.checks
        )
        return HealthReport(
            healthy=all(component.healthy for component in components),
            ready=all(component.healthy for component in components if component.required),
            components=components,
        )

    @classmethod
    def from_callables(cls, checks: Iterable[tuple[str, bool, Callable[[], str]]]) -> HealthService:
        results: list[tuple[str, bool, bool, str]] = []
        for name, required, check in checks:
            try:
                results.append((name, True, required, check()))
            except Exception as exc:
                results.append((name, False, required, f"check failed: {type(exc).__name__}"))
        return cls(results)
