"""Reusable AEGIS application interaction boundary for all clients."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

from .audit import PostgresAuditLog
from .capability_retrieval import retrieve_action_cards
from .contracts import (
    ActionCard,
    Context,
    Decision,
    DecisionKind,
    IntentFrame,
    Objective,
    ObjectiveState,
    Principal,
    Result,
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
from .interaction_context import authorized_context_evidence as _authorized_context_evidence
from .interaction_context import context_from_prior_result as _context_from_prior_result
from .interaction_context import with_continuation_context as _with_continuation_context
from .kernel import Kernel, _FixedActionModel
from .ollama import OllamaHttpTransport, OllamaProvider
from .pack_lifecycle import PackManager, PostgresPackStore
from .pack_runtime import PackRuntimeRegistry
from .store import PostgresObjectiveStore
from .utterance import has_multiple_question_clauses


class InteractionInputError(ValueError):
    """A safe, actionable request-shape error from a client-facing selector."""


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
                safety_result = self.dependencies.safety_fast_path_resolver(
                    intent,
                    recovered_plan_actions,
                    self.dependencies.model_provider is not None,
                )
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
                fallback = self._fallback_decision(
                    intent,
                    self._fallback_cards(manager, utterance, fallback_context),
                    fallback_context,
                )
                if isinstance(fallback, Result):
                    return persist_fast_result(fallback)
                if isinstance(fallback, Decision):
                    if fallback.kind is DecisionKind.ANSWER:
                        answer_evidence: dict[str, Any] = {
                            "provenance": "model_generated",
                            "authoritative": False,
                            "answer_mode": fallback.semantic_mode,
                        }
                        authorized_facts = _authorized_context_evidence(fallback_context)
                        if fallback.context_focus is not None:
                            focused = authorized_facts.get(fallback.context_focus)
                            authorized_facts = (
                                {fallback.context_focus: focused} if focused is not None else {}
                            )
                        if authorized_facts:
                            answer_evidence.update(authorized_facts)
                            answer_evidence["context_provenance"] = "authorized_working_set"
                        return persist_fast_result(
                            Result(
                                objective_id=uuid4(),
                                state=ObjectiveState.COMPLETED,
                                message=fallback.answer or "",
                                evidence=answer_evidence,
                                correlation_id=intent.correlation_id,
                            )
                        )
                    if fallback.kind is DecisionKind.CLARIFY:
                        clarification = fallback.clarification or "Please clarify your request."
                        if has_multiple_question_clauses(utterance):
                            # A model clarification must not import a domain
                            # from the authorized working set when the user
                            # asked multiple independent questions. Keep the
                            # safety result while making the next step useful.
                            clarification = (
                                "That request contains multiple independent questions. "
                                "Please ask one at a time so I can answer each from "
                                "authorized information."
                            )
                        return persist_fast_result(
                            Result(
                                objective_id=uuid4(),
                                state=ObjectiveState.BLOCKED,
                                message=clarification,
                                correlation_id=intent.correlation_id,
                            )
                        )
                    if fallback.kind is DecisionKind.NEED_CONTEXT:
                        return persist_fast_result(
                            Result(
                                objective_id=uuid4(),
                                state=ObjectiveState.BLOCKED,
                                message="I need more context to safely interpret that request.",
                                correlation_id=intent.correlation_id,
                            )
                        )
                    if fallback.kind is DecisionKind.ACTION and fallback.action is not None:
                        fallback_card = next(
                            (
                                candidate
                                for candidate in self._fallback_cards(
                                    manager,
                                    utterance,
                                    fallback_context,
                                )
                                if candidate.action.action_id == fallback.action.action_id
                            ),
                            None,
                        )
                        if fallback_card is None:
                            return persist_fast_result(
                                Result(
                                    objective_id=uuid4(),
                                    state=ObjectiveState.BLOCKED,
                                    message=(
                                        "I could not safely match that request to an "
                                        "available capability."
                                    ),
                                    correlation_id=intent.correlation_id,
                                    retryable=True,
                                )
                            )
                        # Keep the model's bounded arguments while restoring the
                        # canonical card metadata that the decoder validated.
                        card = fallback_card.model_copy(update={"action": fallback.action})
                    else:
                        return persist_fast_result(
                            Result(
                                objective_id=uuid4(),
                                state=ObjectiveState.BLOCKED,
                                message=fallback.reason or str(fallback.kind),
                                correlation_id=intent.correlation_id,
                            )
                        )
                else:
                    return persist_fast_result(
                        Result(
                            objective_id=uuid4(),
                            state=ObjectiveState.BLOCKED,
                            message=str(exc),
                            correlation_id=intent.correlation_id,
                        )
                    )
            if self.dependencies.action_grounder is not None:
                grounded = self.dependencies.action_grounder(intent, card, connection)
                if isinstance(grounded, Result):
                    return persist_fast_result(grounded)
                card = grounded
            if self.dependencies.runtime_registry is not None:
                runtime = self.dependencies.runtime_registry.resolve(card, connection, principal)
                executor = runtime.executor
                verifier = runtime.verifier
                permissions = runtime.permissions
                runtime_cleanup = runtime.cleanup
            else:
                if self.dependencies.runtime_resolver is None:
                    raise LookupError("no Pack runtime resolver configured")
                runtime = self.dependencies.runtime_resolver(
                    card.action.action_id, connection, principal, self.dependencies.openclaw_channel
                )
                executor = runtime.executor
                verifier = runtime.verifier
                permissions = runtime.permissions
                runtime_cleanup = runtime.cleanup
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
