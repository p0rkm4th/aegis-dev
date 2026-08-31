"""Portable local-first model routing and compact evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Protocol

from .contracts import ModelRequest, ModelResponse


class ModelUnavailable(RuntimeError):
    """No provider satisfies availability and privacy policy."""


class ModelProvider(Protocol):
    provider_id: str
    local: bool

    def available(self) -> bool: ...
    def decide(self, request: ModelRequest) -> ModelResponse: ...


@dataclass(frozen=True)
class RoutingTrace:
    provider_id: str
    local: bool
    elapsed_ms: float
    attempted: tuple[str, ...]


class ConfiguredModelRouter:
    def __init__(self, providers: tuple[ModelProvider, ...], allow_cloud: bool = False) -> None:
        self.providers = providers
        self.allow_cloud = allow_cloud
        self.traces: list[RoutingTrace] = []

    def decide(self, request: ModelRequest) -> ModelResponse:
        attempted: list[str] = []
        for provider in self.providers:
            attempted.append(provider.provider_id)
            if not provider.local and not self.allow_cloud:
                continue
            if not provider.available():
                continue
            started = monotonic()
            response = provider.decide(request)
            self.traces.append(
                RoutingTrace(
                    provider_id=provider.provider_id,
                    local=provider.local,
                    elapsed_ms=(monotonic() - started) * 1000,
                    attempted=tuple(attempted),
                )
            )
            return response
        raise ModelUnavailable("no available provider satisfies configured privacy policy")


@dataclass
class BaselineMetrics:
    """Small evidence record; it does not decide correctness or authority."""

    cases: int = 0
    successes: int = 0
    schema_valid: int = 0
    false_completions: int = 0
    security_errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def record(
        self,
        *,
        success: bool,
        schema_valid: bool,
        false_completion: bool = False,
        security_error: bool = False,
        latency_ms: float = 0,
    ) -> None:
        self.cases += 1
        self.successes += int(success)
        self.schema_valid += int(schema_valid)
        self.false_completions += int(false_completion)
        self.security_errors += int(security_error)
        self.latencies_ms.append(latency_ms)

    def summary(self) -> dict[str, float | int]:
        average = sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0
        return {
            "cases": self.cases,
            "success_rate": self.successes / self.cases if self.cases else 0.0,
            "schema_valid_rate": self.schema_valid / self.cases if self.cases else 0.0,
            "false_completions": self.false_completions,
            "security_errors": self.security_errors,
            "average_latency_ms": average,
        }
