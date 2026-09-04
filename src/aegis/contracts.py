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
    PLAN = "PLAN"
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
    semantic_scope: str | None = None


class ProposedPlanStep(StrictModel):
    """Untrusted, candidate-bound proposal for one bounded plan step."""

    action_ref: str = Field(min_length=1)
    arguments: dict[str, Any] = {}
    depends_on: tuple[int, ...] = ()


class ProposedPlan(StrictModel):
    """Model proposal only; Core must bind every step to retrieved cards."""

    steps: tuple[ProposedPlanStep, ...] = Field(min_length=1, max_length=5)


class ObjectiveRequirement(StrictModel):
    """Core-owned description of one bounded effect the user requested."""

    requirement_id: UUID = Field(default_factory=uuid4)
    action_ref: str = Field(min_length=1)
    arguments: dict[str, Any] = {}


class ObjectiveRequirementProposal(StrictModel):
    """Untrusted requirement proposal; Core assigns the durable identity."""

    action_ref: str = Field(min_length=1)
    arguments: dict[str, Any] = {}


class ObjectiveSpec(StrictModel):
    """Persisted objective meaning; never derived from plan exhaustion."""

    requirements: tuple[ObjectiveRequirement, ...] = Field(min_length=1, max_length=5)


class ObjectiveSpecProposal(StrictModel):
    """Model-facing objective meaning without model-controlled stable IDs."""

    requirements: tuple[ObjectiveRequirementProposal, ...] = Field(min_length=1, max_length=5)


class RequestedEffectResolution(StrEnum):
    """Core state for an utterance-grounded effect before capability mapping."""

    UNRESOLVED = "UNRESOLVED"
    MAPPED = "MAPPED"
    AMBIGUOUS = "AMBIGUOUS"
    UNSUPPORTED = "UNSUPPORTED"
    ALREADY_SATISFIED = "ALREADY_SATISFIED"


class RequestedEffect(StrictModel):
    """Core-owned, action-agnostic evidence of one requested outcome.

    This boundary deliberately has no action, permission, runtime, or plan
    authority.  ``effect_id`` is assigned by Core after span validation.
    """

    effect_id: UUID = Field(default_factory=uuid4)
    source_spans: tuple[tuple[int, int], ...] = Field(min_length=1, max_length=3)
    normalized_effect: str = Field(min_length=1, max_length=500)
    polarity: Literal["ACTIVE", "NEGATED", "SUPERSEDED"] = "ACTIVE"
    resolution: RequestedEffectResolution = RequestedEffectResolution.UNRESOLVED


class StructuralAnchor(StrictModel):
    """Untrusted parser evidence used only to test effect coverage."""

    source_span: tuple[int, int]
    kind: Literal["predicate", "object", "clause", "modifier", "negation"]


class StructuralCoverageSignal(StrictModel):
    """Parser output; it defines no meaning and grants no authority."""

    anchors: tuple[StructuralAnchor, ...] = Field(min_length=1, max_length=12)
    negation_spans: tuple[tuple[int, int], ...] = Field(default=(), max_length=12)


class ObjectiveFidelityVerdict(StrEnum):
    """Core-owned comparison result for an independently proposed objective."""

    COMPLETE = "COMPLETE"
    MISSING_REQUIREMENT = "MISSING_REQUIREMENT"
    EXTRA_REQUIREMENT = "EXTRA_REQUIREMENT"
    NEED_CLARIFICATION = "NEED_CLARIFICATION"


class ClarificationRecoveryOutcome(StrEnum):
    RESOLVED = "RESOLVED"
    NEED_USER = "NEED_USER"
    UNSUPPORTED = "UNSUPPORTED"


class ClarificationAmbiguityType(StrEnum):
    REFERENT = "REFERENT"
    ARGUMENT = "ARGUMENT"
    CAPABILITY = "CAPABILITY"


class ClarificationRecoveryProposal(StrictModel):
    """Untrusted, non-executable proposal for recovering a bounded clarification."""

    outcome: ClarificationRecoveryOutcome
    ambiguity_type: ClarificationAmbiguityType
    action_ref: str | None = None
    referent_ref: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict, max_length=8)
    clarification: str | None = Field(default=None, max_length=500)


class ProposalFailureKind(StrEnum):
    """Bounded Core diagnoses that may guide an untrusted proposal repair."""

    MISSING_EFFECT = "MISSING_EFFECT"
    EXTRA_EFFECT = "EXTRA_EFFECT"
    UNACCOUNTED_STRUCTURAL_ANCHOR = "UNACCOUNTED_STRUCTURAL_ANCHOR"
    NEGATED_EFFECT_ACTIVE = "NEGATED_EFFECT_ACTIVE"
    SUPERSEDED_EFFECT_ACTIVE = "SUPERSEDED_EFFECT_ACTIVE"
    BAD_SOURCE_SPAN = "BAD_SOURCE_SPAN"
    CAPABILITY_MISMATCH = "CAPABILITY_MISMATCH"
    CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"
    MISSING_ARGUMENT = "MISSING_ARGUMENT"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    UNKNOWN_ENTITY = "UNKNOWN_ENTITY"
    AMBIGUOUS_ENTITY = "AMBIGUOUS_ENTITY"
    CANONICAL_CONTRADICTION = "CANONICAL_CONTRADICTION"
    WRONG_SEMANTIC_MODE = "WRONG_SEMANTIC_MODE"
    DECODER_SCHEMA_FAILURE = "DECODER_SCHEMA_FAILURE"
    UNSUPPORTED_REQUIREMENT = "UNSUPPORTED_REQUIREMENT"


class ProposalFailureEvidence(StrictModel):
    """Small Core-owned diagnosis; it grants no authority to the repair model."""

    kind: ProposalFailureKind
    detail: str | None = Field(default=None, max_length=240)
    related_effect_ids: tuple[UUID, ...] = Field(default=(), max_length=5)
    related_source_spans: tuple[tuple[int, int], ...] = Field(default=(), max_length=5)
    structural_anchor_spans: tuple[tuple[int, int], ...] = Field(default=(), max_length=5)


class ValidatedPlanStep(StrictModel):
    """Core-bound plan step with stable identity and one requirement owner."""

    step_id: UUID
    requirement_id: UUID
    action: ActionSpec
    depends_on: tuple[UUID, ...] = ()


class ValidatedPlan(StrictModel):
    """Validated proposal; it grants no authority outside the ordinary Kernel."""

    objective_id: UUID
    steps: tuple[ValidatedPlanStep, ...] = Field(min_length=1, max_length=5)


class Decision(StrictModel):
    kind: DecisionKind
    answer: str | None = None
    action: ActionSpec | None = None
    action_ref: str | None = None
    action_arguments: dict[str, Any] = {}
    plan: ProposedPlan | None = None
    objective_spec: ObjectiveSpecProposal | None = None
    clarification: str | None = None
    reason: str | None = None
    semantic_mode: Literal["GENERATION", "READ", "ACTION", "CLARIFY"] | None = None
    knowledge_source: (
        Literal["general_model_knowledge", "external_evidence", "mixed_evidence"] | None
    ) = None
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
    objective_spec: ObjectiveSpec | None = None
    validated_plan: ValidatedPlan | None = None


class ModelRequest(StrictModel):
    working_set: WorkingSet
    action_cards: tuple[ActionCard, ...] = Field(max_length=10)
    allow_argument_proposals: bool = False
    routing_only: bool = False
    classification_only: bool = False
    allow_plan_proposals: bool = False
    capability_scoped: bool = False
    objective_interpretation_only: bool = False
    source_selection_only: bool = False
    objective_fidelity_only: bool = False
    objective_effect_only: bool = False
    clarification_recovery_only: bool = False
    clarification_reason: str | None = Field(default=None, max_length=500)
    proposal_repair_only: bool = False
    repair_validator_stage: str | None = Field(default=None, max_length=80)
    proposal_failure: ProposalFailureEvidence | None = None
    current_proposal: dict[str, Any] | None = None
    objective_spec_proposal: ObjectiveSpecProposal | None = None


class ModelResponse(StrictModel):
    raw: Any
