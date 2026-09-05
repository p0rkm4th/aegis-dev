"""Reusable AEGIS application interaction boundary for all clients."""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

from .audit import PostgresAuditLog
from .capability_retrieval import retrieve_action_cards
from .contracts import (
    ActionCard,
    ArgumentProvenanceKind,
    Context,
    Decision,
    DecisionKind,
    GroundingProposal,
    IntentFrame,
    Objective,
    ObjectiveSpec,
    ObjectiveSpecProposal,
    ObjectiveState,
    Principal,
    ProposedPlan,
    Result,
    StructuralCoverageSignal,
)
from .decoding import StrictDecisionDecoder
from .dispatch import (
    ActionExecutorDispatch as _ActionExecutorDispatch,  # noqa: F401
)
from .dispatch import (
    ActionVerifierDispatch as _ActionVerifierDispatch,  # noqa: F401
)
from .gateway_rpc import OpenClawWebSocketChannel
from .identity import PostgresSpacePolicy
from .interaction_cognition import decide_fallback
from .interaction_context import authorized_context_evidence
from .interaction_context import context_from_prior_result as _context_from_prior_result
from .interaction_context import with_continuation_context as _with_continuation_context
from .interaction_decisions import resolve_fallback_decision
from .kernel import Kernel, _FixedActionModel
from .ollama import OllamaHttpTransport, OllamaProvider
from .pack_lifecycle import PackManager, PostgresPackStore
from .pack_runtime import PackRuntimeRegistry
from .store import PostgresObjectiveStore
from .utterance import strip_context_reset


class InteractionInputError(ValueError):
    """A safe, actionable request-shape error from a client-facing selector."""


def _context_contains_canonical_ref(value: Any, reference: str, *, depth: int = 0) -> bool:
    """Find a bounded canonical identity in already-authorized context only."""

    if depth > 5:
        return False
    if isinstance(value, str):
        return value == reference
    if isinstance(value, dict):
        return any(
            _context_contains_canonical_ref(item, reference, depth=depth + 1)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(
            _context_contains_canonical_ref(item, reference, depth=depth + 1) for item in value[:20]
        )
    return False


def _argument_provenance_error(
    action: Any,
    utterance: str | None = None,
    *,
    card: ActionCard | None = None,
    context: Context | None = None,
) -> str | None:
    """Reject consequential arguments without Core-admissible provenance."""

    provenance = getattr(action, "argument_provenance", {})
    rules = getattr(card, "argument_grounding", {}) if card is not None else {}
    for key in action.arguments:
        evidence = provenance.get(key)
        if evidence is None:
            return f"argument {key!r} has no admissible provenance"
        rule = rules.get(key)
        if rule is not None:
            if evidence.kind not in rule.permitted_provenance:
                return f"argument {key!r} uses undeclared provenance {evidence.kind.value}"
            if (
                evidence.kind is ArgumentProvenanceKind.AUTHORIZED_CANONICAL_REFERENT
                and rule.canonical_source is None
            ):
                return f"argument {key!r} lacks a canonical source contract"
            if (
                evidence.kind is ArgumentProvenanceKind.DETERMINISTIC_DERIVATION
                and evidence.derivation not in rule.approved_derivations
            ):
                return f"argument {key!r} uses an unapproved deterministic derivation"
            if (
                evidence.kind is ArgumentProvenanceKind.APPROVED_DEFAULT
                and evidence.default_contract != rule.approved_default
            ):
                return f"argument {key!r} uses an unapproved default contract"
        if evidence.kind is ArgumentProvenanceKind.EXPLICIT_UTTERANCE:
            if not evidence.source_spans:
                return f"argument {key!r} lacks utterance evidence"
            if utterance is not None and not any(
                0 <= start < end <= len(utterance)
                and utterance[start:end].casefold() == str(action.arguments[key]).casefold()
                for start, end in evidence.source_spans
            ):
                return f"argument {key!r} does not match its utterance evidence"
        elif evidence.kind is ArgumentProvenanceKind.AUTHORIZED_CANONICAL_REFERENT:
            if not evidence.canonical_ref:
                return f"argument {key!r} lacks a canonical referent"
            if (
                rule is not None
                and context is not None
                and not _context_contains_canonical_ref(context.values, evidence.canonical_ref)
            ):
                return f"argument {key!r} references unavailable canonical evidence"
        elif evidence.kind is ArgumentProvenanceKind.DETERMINISTIC_DERIVATION:
            if not evidence.source_spans or not evidence.derivation:
                return f"argument {key!r} lacks deterministic derivation evidence"
        elif evidence.kind is ArgumentProvenanceKind.APPROVED_DEFAULT:
            if not evidence.default_contract:
                return f"argument {key!r} lacks an approved default contract"
    return None


def _apply_grounding_proposals(
    card: ActionCard, grounded: ActionCard | GroundingProposal | tuple[GroundingProposal, ...]
) -> ActionCard:
    """Apply Pack evidence without allowing a Pack to change the proposal value."""

    if isinstance(grounded, ActionCard):
        return grounded
    proposals = (grounded,) if isinstance(grounded, GroundingProposal) else grounded
    action = card.action
    provenance = dict(action.argument_provenance)
    for proposal in proposals:
        if proposal.argument_key not in action.arguments:
            raise ValueError(
                f"grounding proposal names undeclared argument {proposal.argument_key!r}"
            )
        if proposal.proposed_value != action.arguments[proposal.argument_key]:
            raise ValueError(f"grounding proposal changes argument {proposal.argument_key!r}")
        provenance[proposal.argument_key] = proposal.provenance
    return card.model_copy(
        update={"action": action.model_copy(update={"argument_provenance": provenance})}
    )


def _authorized_context_evidence(context: Context) -> dict[str, Any]:
    """Compatibility export for clients/tests; implementation lives in context ownership."""

    return authorized_context_evidence(context)


class InteractionDependencies:
    """Infrastructure callbacks supplied by the composition root."""

    def __init__(
        self,
        connect: Callable[[str], Any],
        required: Callable[[str], str],
        apply_migrations: Callable[[Any], None],
        ensure_local_identity: Callable[[Any, Principal], None],
        select_action: Callable[[str, PackManager], tuple[str, ActionCard]],
        openclaw_channel: Callable[[], OpenClawWebSocketChannel],
        local_identity: Callable[[], bool],
        model_provider: Callable[[], Any] | None = None,
        repair_model_provider: Callable[[], Any] | None = None,
        capability_retriever: Callable[[str, PackManager], tuple[ActionCard, ...]] | None = None,
        runtime_registry: PackRuntimeRegistry | None = None,
        pack_bundles: Callable[[], tuple[Any, ...]] | None = None,
        auto_enable_pack_ids: frozenset[str] = frozenset(),
        action_grounder: Callable[..., Any] | None = None,
        pre_model_resolver: Callable[..., Result | None] | None = None,
        fallback_card_selector: Callable[..., tuple[ActionCard, ...]] | None = None,
        plan_runner: Callable[..., Result | None] | None = None,
        decision_rewriter: Callable[..., Decision | Result | None] | None = None,
        fast_path_resolver: Callable[..., Result | None] | None = None,
        fallback_context_builder: Callable[..., Context] | None = None,
        runtime_resolver: Callable[..., Any] | None = None,
        safety_fast_path_resolver: Callable[..., Result | None] | None = None,
        research_answer: Callable[..., Result | None] | None = None,
        unresolved_requirement_investigator: Callable[..., Result | None] | None = None,
        structural_parser: Callable[[str], StructuralCoverageSignal] | None = None,
        reuse_classification_action_reference: bool = True,
    ) -> None:
        self.connect = connect
        self.required = required
        self.apply_migrations = apply_migrations
        self.ensure_local_identity = ensure_local_identity
        self.select_action = select_action
        self.openclaw_channel = openclaw_channel
        self.local_identity = local_identity
        self.model_provider = model_provider
        self.repair_model_provider = repair_model_provider
        self.capability_retriever = capability_retriever
        self.runtime_registry = runtime_registry
        self.pack_bundles = pack_bundles
        self.auto_enable_pack_ids = auto_enable_pack_ids
        self.action_grounder = action_grounder
        self.pre_model_resolver = pre_model_resolver
        self.fallback_card_selector = fallback_card_selector
        self.plan_runner = plan_runner
        self.decision_rewriter = decision_rewriter
        self.fast_path_resolver = fast_path_resolver
        self.fallback_context_builder = fallback_context_builder
        self.runtime_resolver = runtime_resolver
        self.safety_fast_path_resolver = safety_fast_path_resolver
        self.research_answer = research_answer
        self.unresolved_requirement_investigator = unresolved_requirement_investigator
        self.structural_parser = structural_parser
        self.reuse_classification_action_reference = reuse_classification_action_reference


class InteractionBoundary:
    """Canonical application interaction service used by every client."""

    def __init__(self, dependencies: InteractionDependencies) -> None:
        self.dependencies = dependencies

    def _fallback_cards(
        self, manager: PackManager, utterance: str, context: Context | None = None
    ) -> tuple[ActionCard, ...]:
        """Offer a bounded capability vocabulary; metadata remains Core-owned."""

        return retrieve_action_cards(self.dependencies, manager, utterance, context)

    def _fallback_decision(
        self, intent: IntentFrame, cards: tuple[ActionCard, ...], context: Context
    ) -> Decision | Result | None:
        return decide_fallback(self.dependencies, intent, cards, context)

    def _model(self) -> Any:
        if self.dependencies.model_provider is not None:
            return self.dependencies.model_provider()
        return OllamaProvider(
            os.environ.get("AEGIS_OLLAMA_MODEL", "qwen3:8b"),
            OllamaHttpTransport(self.dependencies.required("AEGIS_OLLAMA_URL")),
        )

    def _run_proposed_plan(
        self,
        intent: IntentFrame,
        proposal: ProposedPlan,
        cards: tuple[ActionCard, ...],
        connection: Any,
        principal: Principal,
        context: Context,
        objective_spec: ObjectiveSpec | ObjectiveSpecProposal | None = None,
    ) -> Result:
        """Run a candidate-bound plan through the existing per-step Kernel path."""

        if objective_spec is None:
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message=(
                    "I could not safely persist every requested change as an objective; "
                    "please clarify the request."
                ),
                correlation_id=intent.correlation_id,
            )
        if self.dependencies.runtime_registry is None:
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.FAILED,
                message="Plan runtime is unavailable; request can be retried",
                correlation_id=intent.correlation_id,
                retryable=True,
            )
        by_id = {card.action.action_id: card for card in cards}
        try:
            plan_cards = tuple(by_id[step.action_ref] for step in proposal.steps)
        except KeyError:
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message="A plan step is no longer an available capability",
                correlation_id=intent.correlation_id,
            )
        runtimes: dict[str, Any] = {}
        permissions: dict[str, frozenset[Any]] = {}
        cleanups: list[Callable[[], None]] = []
        try:
            grounded_steps = []
            for step, card in zip(proposal.steps, plan_cards, strict=True):
                runtime = self.dependencies.runtime_registry.resolve(card, connection, principal)
                runtimes[card.action.action_id] = runtime
                permissions.update(runtime.permissions)
                if runtime.cleanup is not None:
                    cleanups.append(runtime.cleanup)
                proposed_card = card.model_copy(
                    update={
                        "action": card.action.model_copy(
                            update={"arguments": step.arguments, "argument_provenance": {}}
                        )
                    }
                )
                grounder = (
                    getattr(runtime, "grounder", None)
                    if runtime is not None and getattr(runtime, "grounder", None) is not None
                    else self.dependencies.action_grounder
                )
                grounded = (
                    grounder(intent, proposed_card, connection, context)
                    if grounder is not None
                    else proposed_card
                )
                if isinstance(grounded, Result):
                    return grounded
                grounded = _apply_grounding_proposals(proposed_card, grounded)
                provenance_error = _argument_provenance_error(
                    grounded.action,
                    intent.utterance,
                    card=card,
                    context=context,
                )
                if provenance_error is not None:
                    return Result(
                        objective_id=uuid4(),
                        state=ObjectiveState.BLOCKED,
                        message=(
                            "I could not safely establish where a consequential plan "
                            f"argument came from ({provenance_error})."
                        ),
                        evidence={"failure": "CONSEQUENTIAL_ARGUMENT_PROVENANCE_UNAVAILABLE"},
                        correlation_id=intent.correlation_id,
                    )
                # A plan step owns only the arguments it proposed.  The shared
                # utterance may contain details for a neighboring step (for
                # example, an event date); allowing the domain grounder to add
                # those details here would silently cross-contaminate steps.
                grounded_arguments = {
                    key: grounded.action.arguments[key]
                    for key in step.arguments
                    if key in grounded.action.arguments
                }
                grounded_steps.append(step.model_copy(update={"arguments": grounded_arguments}))
            proposal = proposal.model_copy(update={"steps": tuple(grounded_steps)})
            kernel = Kernel(
                self._model(),
                StrictDecisionDecoder(),
                PostgresSpacePolicy(connection, permissions),
                _ActionExecutorDispatch(
                    {action_id: runtime.executor for action_id, runtime in runtimes.items()}
                ),
                _ActionVerifierDispatch(
                    {action_id: runtime.verifier for action_id, runtime in runtimes.items()}
                ),
                store=PostgresObjectiveStore(connection),
                audit=PostgresAuditLog(connection),
            )
            return kernel.run_proposed_plan(
                intent,
                proposal,
                plan_cards,
                context=context,
                objective_spec=objective_spec,
            )
        except ValueError as exc:
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message=f"I could not safely account for every requested change: {exc}",
                correlation_id=intent.correlation_id,
            )
        except (LookupError, PermissionError, RuntimeError):
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.FAILED,
                message="A plan capability is temporarily unavailable; request can be retried",
                correlation_id=intent.correlation_id,
                retryable=True,
            )
        finally:
            for cleanup in cleanups:
                cleanup()

    def run(
        self,
        utterance: str,
        principal: Principal,
        correlation_id: UUID | None = None,
        context_correlation_id: UUID | None = None,
    ) -> Result:
        connection = self.dependencies.connect(self.dependencies.required("AEGIS_DATABASE_URL"))
        channel: OpenClawWebSocketChannel | None = None
        runtime_cleanup: Callable[[], None] | None = None
        try:
            self.dependencies.apply_migrations(connection)
            if self.dependencies.local_identity():
                self.dependencies.ensure_local_identity(connection, principal)
            intent = IntentFrame(
                principal=principal,
                utterance=utterance,
                correlation_id=correlation_id or uuid4(),
            )
            objective_store = PostgresObjectiveStore(connection)
            context = _context_from_prior_result(objective_store, context_correlation_id, principal)
            normalized_utterance = strip_context_reset(utterance)
            normalized_input = " ".join(utterance.casefold().split())
            if normalized_utterance != normalized_input:
                # An explicit conversational reset revokes only the prior-turn
                # working context; canonical state remains available to the new
                # independent request through normal read paths.
                context = Context()
            recovered_plan = objective_store.get_objective_by_correlation(
                intent.correlation_id, principal
            )
            if recovered_plan is None and objective_store.correlation_bound(intent.correlation_id):
                return Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.BLOCKED,
                    message="request correlation is unavailable",
                    correlation_id=intent.correlation_id,
                )
            if recovered_plan is not None and recovered_plan.steps:
                prior_plan_result = objective_store.get_result(f"plan:{intent.correlation_id}")
                if (
                    prior_plan_result is not None
                    and prior_plan_result.state is ObjectiveState.COMPLETED
                    and not prior_plan_result.retryable
                ):
                    return prior_plan_result

            def persist_fast_result(result: Result) -> Result:
                result = _with_continuation_context(result, context)
                objective_store.save_objective(
                    Objective(
                        id=result.objective_id,
                        intent=intent,
                        correlation_id=intent.correlation_id,
                        state=result.state,
                    )
                )
                objective_store.save_result(f"interaction:{intent.correlation_id}", result)
                return result

            if recovered_plan is not None and not recovered_plan.steps:
                prior_interaction_result = objective_store.get_result(
                    f"interaction:{intent.correlation_id}"
                )
                if prior_interaction_result is not None and not prior_interaction_result.retryable:
                    return prior_interaction_result
            recovered_plan_actions = (
                recovered_plan.steps
                if recovered_plan is not None and recovered_plan.steps
                else None
            )
            if self.dependencies.safety_fast_path_resolver is not None:
                safety = self.dependencies.safety_fast_path_resolver
                arguments = (
                    intent,
                    recovered_plan_actions,
                    self.dependencies.model_provider is not None,
                )
                try:
                    inspect.signature(safety).bind(*arguments, context)
                except (TypeError, ValueError):
                    safety_result = safety(*arguments)
                else:
                    safety_result = safety(*arguments, context)
                if safety_result is not None:
                    return persist_fast_result(safety_result)
            if self.dependencies.pre_model_resolver is not None and recovered_plan_actions is None:
                pre_model_result = self.dependencies.pre_model_resolver(
                    intent, connection, principal
                )
                if pre_model_result is not None:
                    return persist_fast_result(pre_model_result)
            if self.dependencies.fast_path_resolver is not None:
                fast_result = self.dependencies.fast_path_resolver(
                    intent,
                    connection,
                    principal,
                    context,
                    recovered_plan_actions,
                    self.dependencies.required,
                    self.dependencies.model_provider is not None,
                )
                if fast_result is not None:
                    return persist_fast_result(fast_result)
            if (
                self.dependencies.model_provider is None
                and self.dependencies.plan_runner is None
                and self.dependencies.pre_model_resolver is None
                and self.dependencies.safety_fast_path_resolver is None
            ):
                return persist_fast_result(
                    Result(
                        objective_id=uuid4(),
                        state=ObjectiveState.BLOCKED,
                        message="No interaction runtime is configured for this request.",
                        correlation_id=intent.correlation_id,
                    )
                )
            manager = PackManager(store=PostgresPackStore(connection))
            if self.dependencies.pack_bundles is not None:
                manager.reconcile(
                    tuple(self.dependencies.pack_bundles()),
                    self.dependencies.auto_enable_pack_ids,
                )
            if self.dependencies.plan_runner is not None:
                # The runner is also the deterministic production seam for
                # bounded plans. Let it claim recognized plans before model
                # cognition, even when a model is configured; arbitrary
                # requests still fall through because the runner returns
                # None. This keeps ACTION and PLAN on the same grounding,
                # authorization, and Kernel path without adding a model call.
                plan_result = self.dependencies.plan_runner(
                    intent,
                    connection,
                    principal,
                    manager,
                    recovered_plan_actions,
                    context,
                    self._model(),
                    self.dependencies.runtime_registry,
                )
                if plan_result is not None:
                    return plan_result
            try:
                if self.dependencies.model_provider is not None:
                    raise InteractionInputError("semantic action resolution required")
                _domain, card = self.dependencies.select_action(utterance, manager)
            except InteractionInputError as exc:
                fallback_context = (
                    self.dependencies.fallback_context_builder(
                        context, connection, principal, utterance
                    )
                    if self.dependencies.fallback_context_builder is not None
                    else context
                )
                fallback_cards = self._fallback_cards(manager, utterance, fallback_context)
                fallback = self._fallback_decision(intent, fallback_cards, fallback_context)
                if isinstance(fallback, Result):
                    return persist_fast_result(fallback)
                if isinstance(fallback, Decision):
                    if self.dependencies.decision_rewriter is not None:
                        try:
                            inspect.signature(self.dependencies.decision_rewriter).bind(
                                intent, fallback, fallback_cards, fallback_context
                            )
                        except (TypeError, ValueError):
                            rewritten = self.dependencies.decision_rewriter(
                                intent, fallback, fallback_cards
                            )
                        else:
                            rewritten = self.dependencies.decision_rewriter(
                                intent, fallback, fallback_cards, fallback_context
                            )
                        if isinstance(rewritten, Result):
                            return persist_fast_result(rewritten)
                        if isinstance(rewritten, Decision):
                            fallback = rewritten
                    if fallback.kind is DecisionKind.PLAN and fallback.plan is not None:
                        return self._run_proposed_plan(
                            intent,
                            fallback.plan,
                            fallback_cards,
                            connection,
                            principal,
                            fallback_context,
                            fallback.objective_spec,
                        )
                    resolution = resolve_fallback_decision(
                        fallback,
                        intent,
                        fallback_context,
                        fallback_cards,
                        self.dependencies.unresolved_requirement_investigator,
                    )
                    if isinstance(resolution, Result):
                        return persist_fast_result(resolution)
                    card = resolution
                else:
                    return persist_fast_result(
                        Result(
                            objective_id=uuid4(),
                            state=ObjectiveState.BLOCKED,
                            message=str(exc),
                            correlation_id=intent.correlation_id,
                        )
                    )
            runtime = None
            if self.dependencies.runtime_registry is not None:
                try:
                    runtime = self.dependencies.runtime_registry.resolve(
                        card, connection, principal
                    )
                    runtime_cleanup = runtime.cleanup
                except (LookupError, PermissionError, RuntimeError):
                    return persist_fast_result(
                        Result(
                            objective_id=uuid4(),
                            state=ObjectiveState.FAILED,
                            message=(
                                "This action is temporarily unavailable because its execution "
                                "provider is not configured. You can retry it later."
                            ),
                            correlation_id=intent.correlation_id,
                            retryable=True,
                        )
                    )
            grounder = (
                getattr(runtime, "grounder", None)
                if runtime is not None and getattr(runtime, "grounder", None) is not None
                else self.dependencies.action_grounder
            )
            if grounder is not None:
                card = card.model_copy(
                    update={"action": card.action.model_copy(update={"argument_provenance": {}})}
                )
                grounded = grounder(intent, card, connection, context)
                if isinstance(grounded, Result):
                    return persist_fast_result(grounded)
                card = _apply_grounding_proposals(card, grounded)
            provenance_error = _argument_provenance_error(
                card.action,
                intent.utterance,
                card=card,
                context=context,
            )
            if provenance_error is not None:
                return persist_fast_result(
                    Result(
                        objective_id=uuid4(),
                        state=ObjectiveState.BLOCKED,
                        message=(
                            "I could not safely establish where a consequential argument "
                            f"came from ({provenance_error})."
                        ),
                        evidence={"failure": "CONSEQUENTIAL_ARGUMENT_PROVENANCE_UNAVAILABLE"},
                        correlation_id=intent.correlation_id,
                    )
                )
            try:
                if self.dependencies.runtime_registry is not None:
                    assert runtime is not None
                    executor = runtime.executor
                    verifier = runtime.verifier
                    permissions = runtime.permissions
                    runtime_cleanup = runtime.cleanup
                else:
                    if self.dependencies.runtime_resolver is None:
                        raise LookupError("no Pack runtime resolver configured")
                    runtime = self.dependencies.runtime_resolver(
                        card.action.action_id,
                        connection,
                        principal,
                        self.dependencies.openclaw_channel,
                    )
                    executor = runtime.executor
                    verifier = runtime.verifier
                    permissions = runtime.permissions
                    runtime_cleanup = runtime.cleanup
            except (LookupError, RuntimeError):
                return persist_fast_result(
                    Result(
                        objective_id=uuid4(),
                        state=ObjectiveState.FAILED,
                        message=(
                            "This action is temporarily unavailable because its execution "
                            "provider is not configured. You can retry it later."
                        ),
                        correlation_id=intent.correlation_id,
                        retryable=True,
                    )
                )
            kernel = Kernel(
                # The bounded model proposal was already decoded and
                # canonicalized above. Reuse it as a fixed proposal rather
                # than invoking cognition a second time with legacy decoder
                # settings or allowing the action to drift.
                _FixedActionModel(card.action),
                StrictDecisionDecoder(),
                PostgresSpacePolicy(connection, permissions),
                executor,
                verifier,
                store=PostgresObjectiveStore(connection),
                audit=PostgresAuditLog(connection),
            )
            return kernel.run(intent, (card,), context=context)
        finally:
            if runtime_cleanup is not None:
                runtime_cleanup()
            if channel is not None:
                channel.close()
            connection.close()
