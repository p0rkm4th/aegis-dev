"""Untrusted-input-safe semantic contracts for the Aegis kernel."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DecisionKind(StrEnum):
    ACTION = "ACTION"
    ANSWER = "ANSWER"
    NEED_CONTEXT = "NEED_CONTEXT"
    CLARIFY = "CLARIFY"
    BLOCKED = "BLOCKED"


class ObjectiveState(StrEnum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    AUTHORIZED = "authorized"
    APPROVAL_REQUIRED = "approval_required"
    EXECUTING = "executing"
    OBSERVED = "observed"
    VERIFIED = "verified"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class Principal(StrictModel):
    id: str = Field(min_length=1)
    vault_id: str = Field(min_length=1)
    space_ids: tuple[str, ...] = ()


class IntentFrame(StrictModel):
    principal: Principal
    utterance: str = Field(min_length=1, max_length=20_000)
    correlation_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=utc_now)


class Context(StrictModel):
    values: dict[str, Any] = {}
    sources: tuple[str, ...] = ()


class DomainContract(StrictModel):
    name: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class WorkingSet(StrictModel):
    intent: IntentFrame
    context: Context = Field(default_factory=Context)
    domains: tuple[DomainContract, ...] = ()


class VerificationContract(StrictModel):
    kind: Literal["readback", "health", "custom"]
    expected: dict[str, Any] = {}


class ActionSpec(StrictModel):
    action_id: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    arguments: dict[str, Any] = {}
    required_permissions: tuple[str, ...] = ()
    verification: VerificationContract | None = None


class ActionCard(StrictModel):
    action: ActionSpec
    summary: str = Field(min_length=1)
    relevance: float = Field(ge=0, le=1)
    argument_keys: tuple[str, ...] = ()
    argument_descriptions: dict[str, str] = {}


class ProposedPlanStep(StrictModel):
    """Untrusted, candidate-bound proposal for one bounded plan step."""

    action_ref: str = Field(min_length=1)
    arguments: dict[str, Any] = {}
    depends_on: tuple[int, ...] = ()


class ProposedPlan(StrictModel):
    """Model proposal only; Core must bind every step to retrieved cards."""

    steps: tuple[ProposedPlanStep, ...] = Field(min_length=1, max_length=5)


class Decision(StrictModel):
    kind: DecisionKind
    answer: str | None = None
    action: ActionSpec | None = None
    action_ref: str | None = None
    action_arguments: dict[str, Any] = {}
    clarification: str | None = None
    reason: str | None = None
    semantic_mode: Literal["GENERATION", "READ", "ACTION", "CLARIFY"] | None = None
    context_focus: (
        Literal["canonical_items", "canonical_tasks", "canonical_obligations", "planning"] | None
    ) = None


class PolicyDecision(StrictModel):
    allowed: bool
    reason: str
    approval_required: bool = False


class AuthorizationRequest(StrictModel):
    principal: Principal
    objective_id: UUID
    action: ActionSpec


class ExecutionRequest(StrictModel):
    objective_id: UUID
    action_id: UUID
    action: ActionSpec
    idempotency_key: str = Field(min_length=1)


class ActionClaim(StrictModel):
    request: ExecutionRequest
    acquired: bool


class Observation(StrictModel):
    execution_id: UUID
    action_id: str | None = None
    evidence: dict[str, Any]
    command_succeeded: bool
    observed_at: datetime = Field(default_factory=utc_now)


class VerificationResult(StrictModel):
    verified: bool
    evidence: dict[str, Any]
    reason: str


class Result(StrictModel):
    objective_id: UUID
    state: ObjectiveState
    message: str
    evidence: dict[str, Any] = {}
    correlation_id: UUID
    retryable: bool = False


class RequestStatus(StrictModel):
    correlation_id: UUID
    state: ObjectiveState | Literal["unknown"]
    objective_id: UUID | None = None
    message: str | None = None
    retryable: bool | None = None


class Objective(StrictModel):
    id: UUID = Field(default_factory=uuid4)
    intent: IntentFrame
    state: ObjectiveState = ObjectiveState.PROPOSED
    action: ActionSpec | None = None
    correlation_id: UUID
    steps: tuple[ActionSpec, ...] = ()


class ModelRequest(StrictModel):
    working_set: WorkingSet
    action_cards: tuple[ActionCard, ...] = Field(max_length=10)
    allow_argument_proposals: bool = False
    routing_only: bool = False
    classification_only: bool = False


class ModelResponse(StrictModel):
    raw: Any
