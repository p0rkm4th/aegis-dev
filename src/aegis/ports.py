"""Replaceable ports; implementations must not own Aegis authority."""

from __future__ import annotations

from typing import Protocol

from .contracts import (
    ActionCard,
    AuthorizationRequest,
    Decision,
    ExecutionRequest,
    ModelRequest,
    ModelResponse,
    Observation,
    PolicyDecision,
    VerificationContract,
    VerificationResult,
)


class ModelRouter(Protocol):
    def decide(self, request: ModelRequest) -> ModelResponse: ...


class DecisionDecoder(Protocol):
    def decode(self, response: ModelResponse, cards: tuple[ActionCard, ...]) -> Decision: ...


class Policy(Protocol):
    def authorize(self, request: AuthorizationRequest) -> PolicyDecision: ...


class Executor(Protocol):
    def execute(self, request: ExecutionRequest) -> Observation: ...


class Verifier(Protocol):
    def verify(
        self, observation: Observation, contract: VerificationContract
    ) -> VerificationResult: ...
