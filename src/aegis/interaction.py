"""Reusable AEGIS application interaction boundary for all clients."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from .audit import PostgresAuditLog
from .contracts import (
    ActionCard,
    Context,
    Decision,
    DecisionKind,
    IntentFrame,
    ModelRequest,
    Objective,
    ObjectiveState,
    Principal,
    Result,
    WorkingSet,
)
from .decoding import InvalidDecision, StrictDecisionDecoder
from .dispatch import (
    ActionExecutorDispatch as _ActionExecutorDispatch,  # noqa: F401
)
from .dispatch import (
    ActionVerifierDispatch as _ActionVerifierDispatch,  # noqa: F401
)
from .gateway_rpc import OpenClawWebSocketChannel
from .identity import PostgresSpacePolicy
from .kernel import Kernel, _FixedActionModel
from .ollama import OllamaHttpTransport, OllamaProvider
from .pack_lifecycle import PackManager, PostgresPackStore
from .pack_runtime import PackRuntimeRegistry
from .planning import (
    ContextualMutationGuard,
    DomainClarificationFastPath,
    MultiActionFastPath,
)
from .reference_packs import reference_bundles
from .store import PostgresObjectiveStore
from .utterance import (
    has_multiple_question_clauses,
    is_mutation_request,
    is_question_request,
)

_MAX_CONTEXT_TURN_CHARS = 500
_MAX_CONTEXT_CANDIDATES = 10


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
        fallback_card_selector: Callable[[PackManager, str], tuple[ActionCard, ...]] | None = None,
        plan_runner: Callable[..., Result | None] | None = None,
        decision_rewriter: Callable[..., Decision | Result | None] | None = None,
        fast_path_resolver: Callable[..., Result | None] | None = None,
        fallback_context_builder: Callable[..., Context] | None = None,
        runtime_resolver: Callable[..., Any] | None = None,
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


def _compact_context_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Keep only bounded canonical facts for a follow-up model working set."""

    compact: dict[str, Any] = {}
    for key in (
        "canonical_items",
        "canonical_tasks",
        "canonical_chores",
        "canonical_obligations",
        "title",
        "item",
    ):
        value = evidence.get(key)
        if isinstance(value, (list, tuple)):
            if key == "canonical_items":
                compact[key] = list(dict.fromkeys(str(item) for item in value))[:20]
            elif key == "canonical_tasks":
                tasks = list(value)
                dated_open = [
                    item
                    for item in tasks
                    if isinstance(item, dict)
                    and item.get("status") == "open"
                    and isinstance(item.get("due_at"), str)
                ]
                remaining = [item for item in tasks if item not in dated_open]
                compact[key] = (dated_open + remaining)[:20]
            else:
                compact[key] = list(value[:20])
        elif isinstance(value, str):
            compact[key] = value
    planning = evidence.get("planning")
    if isinstance(planning, dict):
        compact_planning: dict[str, Any] = {}
        for key in ("open_tasks", "open_obligations", "memories", "affordability"):
            value = planning.get(key)
            if isinstance(value, (list, tuple)):
                compact_planning[key] = list(value[:10])
            elif isinstance(value, dict):
                compact_planning[key] = value
        if compact_planning:
            compact["planning"] = compact_planning
    return compact


def _authorized_context_evidence(context: Context) -> dict[str, Any]:
    """Carry only the authorized working-set facts needed by a later turn."""

    raw = context.values.get("canonical_facts")
    if not isinstance(raw, dict):
        return {}
    return _compact_context_evidence(raw)


def _grounded_context_answer(context: Context, raw: dict[str, Any]) -> Decision | None:
    """Recover an answer from one model-selected, authorized structured focus."""

    if raw.get("kind") != DecisionKind.ANSWER.value or raw.get("semantic_mode") != "READ":
        return None
    focus = raw.get("context_focus")
    facts = context.values.get("canonical_facts")
    if not isinstance(focus, str) or not isinstance(facts, dict):
        return None
    if focus not in {"canonical_items", "canonical_tasks", "canonical_obligations"}:
        return None
    value = facts.get(focus)
    if not isinstance(value, list) or not value:
        return None
    if focus == "canonical_items":
        answer = "Authorized groceries: " + ", ".join(str(item) for item in value)
    elif focus == "canonical_tasks":
        titles = [
            str(item.get("title"))
            for item in value
            if isinstance(item, dict) and isinstance(item.get("title"), str)
        ]
        if not titles:
            return None
        answer = "Authorized tasks: " + "; ".join(titles)
    elif focus == "canonical_obligations":
        titles = [
            str(item.get("title"))
            for item in value
            if isinstance(item, dict) and isinstance(item.get("title"), str)
        ]
        if not titles:
            return None
        answer = "Authorized obligations: " + "; ".join(titles)
    else:
        return None
    return Decision(
        kind=DecisionKind.ANSWER,
        answer=answer,
        semantic_mode="READ",
        context_focus=cast(
            Literal["canonical_items", "canonical_tasks", "canonical_obligations"], focus
        ),
    )


def _with_continuation_context(result: Result, context: Context) -> Result:
    """Trace a follow-up to authorized context without changing its identity."""

    prior_objective_id = context.values.get("prior_objective_id")
    if not isinstance(prior_objective_id, str):
        return result
    evidence = dict(result.evidence)
    evidence["continuation_of_objective_id"] = prior_objective_id
    evidence["continuation_context"] = "authorized_prior_result"
    return result.model_copy(update={"evidence": evidence})


def _context_from_prior_result(
    objective_store: Any, correlation_id: UUID | None, principal: Principal
) -> Context:
    """Resolve follow-up context only through the authorized canonical store."""

    if correlation_id is None:
        return Context()
    objective = objective_store.get_objective_by_correlation(correlation_id, principal)
    getter = getattr(objective_store, "get_result_for_correlation", None)
    if objective is None or not callable(getter):
        return Context()
    result = getter(correlation_id, principal)
    if result is None:
        return Context()
    evidence = _compact_context_evidence(result.evidence)
    if not evidence:
        return Context()
    referents: dict[str, Any] = {}
    for fact_key in ("canonical_items", "canonical_tasks", "canonical_obligations"):
        candidates = evidence.get(fact_key)
        if isinstance(candidates, list) and candidates:
            referents["those"] = {
                "source": "canonical_facts",
                "fact_key": fact_key,
                "candidates": candidates[:_MAX_CONTEXT_CANDIDATES],
            }
            break
    return Context(
        values={
            "prior_correlation_id": str(correlation_id),
            "prior_objective_id": str(objective.id),
            "prior_state": result.state.value,
            "recent_turns": [
                {
                    "role": "user",
                    "utterance": objective.intent.utterance[:_MAX_CONTEXT_TURN_CHARS],
                    "correlation_id": str(correlation_id),
                }
            ],
            "referents": referents,
            "canonical_facts": evidence,
        },
        sources=("authorized_canonical_result",),
    )


class InteractionBoundary:
    """Canonical application interaction service used by every client."""

    def __init__(self, dependencies: InteractionDependencies) -> None:
        self.dependencies = dependencies

    def _fallback_cards(
        self, manager: PackManager, utterance: str, context: Context | None = None
    ) -> tuple[ActionCard, ...]:
        """Offer a bounded capability vocabulary; metadata remains Core-owned."""

        facts = (context.values if context is not None else {}).get("canonical_facts", {})
        if isinstance(facts, dict):
            if (
                isinstance(facts.get("canonical_items"), list)
                and context is not None
                and "authorized_canonical_context" in context.sources
            ):
                # A contextual grocery question is answered from the
                # authorized projection; do not offer a mutation card that
                # could turn an unresolved reference into a write.
                return ()
            if (
                isinstance(facts.get("canonical_tasks"), list)
                and context is not None
                and "authorized_canonical_context" in context.sources
            ):
                return tuple(manager.retrieve("tasks"))[:10]
        if self.dependencies.capability_retriever is not None:
            try:
                semantic_cards = self.dependencies.capability_retriever(utterance, manager)
            except Exception:
                # Retrieval is an optimization; a provider outage must not
                # bypass the bounded model/decoder or change authority.
                semantic_cards = ()
            if semantic_cards:
                write_cards = tuple(
                    card
                    for card in semantic_cards
                    if any(
                        permission.endswith(".write")
                        for permission in card.action.required_permissions
                    )
                )
                if write_cards:
                    namespace = write_cards[0].action.action_id.split(".", 1)[0]
                    scoped_cards = tuple(
                        card
                        for card in semantic_cards
                        if card.action.action_id.split(".", 1)[0] == namespace
                    )
                    if scoped_cards:
                        return scoped_cards[:10]
                return tuple(semantic_cards)[:10]
        if self.dependencies.fallback_card_selector is not None:
            return self.dependencies.fallback_card_selector(manager, utterance)
        return tuple(manager.enabled_cards())[:10]

    def _fallback_decision(
        self, intent: IntentFrame, cards: tuple[ActionCard, ...], context: Context
    ) -> Decision | Result | None:
        if self.dependencies.model_provider is None:
            return None
        last_raw: dict[str, Any] | None = None
        focused_raw: dict[str, Any] | None = None
        try:
            provider = self.dependencies.model_provider()
            if cards:
                classification_values = {
                    key: value for key, value in context.values.items() if key != "canonical_facts"
                }
                classification_request = ModelRequest(
                    working_set=WorkingSet(
                        intent=intent,
                        context=context.model_copy(update={"values": classification_values}),
                    ),
                    # Classification may choose only a semantic mode, but it
                    # needs the bounded candidate vocabulary to distinguish a
                    # list destination from a request to create or complete.
                    # Core still decodes the later action proposal against the
                    # same cards and retains all authority gates.
                    action_cards=cards,
                    classification_only=True,
                )
                classification_response = provider.decide(classification_request)
                last_raw = (
                    classification_response.raw
                    if isinstance(classification_response.raw, dict)
                    else None
                )
                if last_raw is not None and last_raw.get("context_focus") is not None:
                    focused_raw = last_raw
                semantic_mode = (
                    classification_response.raw.get("semantic_mode")
                    if isinstance(classification_response.raw, dict)
                    else None
                )
                if semantic_mode == "ACTION" and isinstance(classification_response.raw, dict):
                    # Some providers return a bounded action reference even
                    # during the mode pass. Reuse it only when it exactly
                    # names a supplied ActionCard and its arguments are a
                    # declared subset. This prevents a second cognition pass
                    # from drifting a clear mutation into a nearby read;
                    # grounding, policy, execution, and verification remain
                    # below this proposal boundary.
                    action_ref = classification_response.raw.get("action_ref")
                    action_arguments = classification_response.raw.get("action_arguments", {})
                    selected_card = next(
                        (card for card in cards if card.action.action_id == action_ref), None
                    )
                    if selected_card is not None and isinstance(action_arguments, dict):
                        declared = set(selected_card.argument_keys)
                        if set(action_arguments).issubset(declared):
                            return Decision(
                                kind=DecisionKind.ACTION,
                                action=selected_card.action.model_copy(
                                    update={"arguments": dict(action_arguments)}
                                ),
                                semantic_mode="ACTION",
                            )
                if semantic_mode in {"GENERATION", "READ"}:
                    if semantic_mode == "GENERATION":
                        generated_answer = (
                            classification_response.raw.get("answer")
                            if isinstance(classification_response.raw, dict)
                            else None
                        )
                        if isinstance(generated_answer, str) and generated_answer.strip():
                            return Decision(
                                kind=DecisionKind.ANSWER,
                                answer=generated_answer,
                                semantic_mode="GENERATION",
                            )
                    answer_context = context
                    if semantic_mode == "GENERATION":
                        answer_context = context.model_copy(
                            update={
                                "values": {
                                    key: value
                                    for key, value in context.values.items()
                                    if key != "canonical_facts"
                                }
                            }
                        )
                    answer_request = ModelRequest(
                        working_set=WorkingSet(intent=intent, context=answer_context),
                        action_cards=(),
                    )
                    answer_response = provider.decide(answer_request)
                    last_raw = (
                        answer_response.raw if isinstance(answer_response.raw, dict) else None
                    )
                    if last_raw is not None and last_raw.get("context_focus") is not None:
                        focused_raw = last_raw
                    answer = StrictDecisionDecoder().decode(
                        answer_response, (), allow_argument_proposals=False
                    )
                    if answer.kind is DecisionKind.ANSWER:
                        return answer.model_copy(update={"semantic_mode": semantic_mode})
            routing_only = any(
                permission.endswith(".write")
                for card in cards
                for permission in card.action.required_permissions
            )
            routing_context = context
            if routing_only:
                routing_values = {
                    key: value for key, value in context.values.items() if key != "canonical_facts"
                }
                routing_context = context.model_copy(update={"values": routing_values})
            request = ModelRequest(
                working_set=WorkingSet(intent=intent, context=routing_context),
                action_cards=cards,
                allow_argument_proposals=True,
                routing_only=routing_only,
            )
            decoder = StrictDecisionDecoder()
            decision: Decision | None = None
            for attempt in range(2):
                response = provider.decide(request)
                last_raw = response.raw if isinstance(response.raw, dict) else None
                if last_raw is not None and last_raw.get("context_focus") is not None:
                    focused_raw = last_raw
                try:
                    decision = decoder.decode(
                        response, request.action_cards, allow_argument_proposals=True
                    )
                    break
                except InvalidDecision as error:
                    # Repair only an empty benign answer with no capability
                    # cards. This remains bounded cognition and cannot turn a
                    # malformed action into an executable proposal.
                    raw = response.raw
                    if not (
                        attempt == 0
                        and isinstance(raw, dict)
                        and raw.get("kind") == DecisionKind.ANSWER.value
                        and not isinstance(raw.get("answer"), str)
                    ):
                        raise error
                    request = request.model_copy(update={"action_cards": ()})
            if decision is None:
                raise InvalidDecision("model answer repair did not produce a decision")
            if routing_only and decision.kind is DecisionKind.ANSWER:
                reconsidered = decoder.decode(
                    provider.decide(request), request.action_cards, allow_argument_proposals=True
                )
                if reconsidered.kind is not decision.kind:
                    return Decision(
                        kind=DecisionKind.CLARIFY,
                        clarification=(
                            "I am not confident whether you want to change something or "
                            "ask about it. Please clarify the intended action."
                        ),
                    )
                if reconsidered.kind is DecisionKind.ANSWER:
                    decision = reconsidered
                else:
                    decision = reconsidered
            if routing_only and decision.kind is DecisionKind.ANSWER:
                request = request.model_copy(
                    update={
                        "working_set": WorkingSet(intent=intent, context=context),
                        "routing_only": False,
                    }
                )
                decision = decoder.decode(
                    provider.decide(request), request.action_cards, allow_argument_proposals=True
                )
            if decision.kind is DecisionKind.ACTION and decision.action is not None:
                if self.dependencies.decision_rewriter is not None:
                    rewritten = self.dependencies.decision_rewriter(intent, decision, cards)
                    if isinstance(rewritten, Result):
                        return rewritten
                    if rewritten is not None:
                        decision = rewritten
                action = decision.action
                if decision.kind is not DecisionKind.ACTION or action is None:
                    return decision
                selected_card = next(
                    (card for card in cards if card.action.action_id == action.action_id),
                    None,
                )
                if (
                    is_question_request(intent.utterance)
                    and selected_card is not None
                    and any(
                        permission.endswith(".write")
                        for permission in selected_card.action.required_permissions
                    )
                ):
                    read_cards = tuple(
                        card
                        for card in cards
                        if not any(
                            permission.endswith(".write")
                            for permission in card.action.required_permissions
                        )
                    )
                    if read_cards:
                        read_request = ModelRequest(
                            working_set=WorkingSet(intent=intent, context=context),
                            action_cards=read_cards,
                        )
                        read_decision = decoder.decode(
                            provider.decide(read_request),
                            read_cards,
                            allow_argument_proposals=False,
                        )
                        if (
                            read_decision.kind is not DecisionKind.ACTION
                            or read_decision.action is not None
                        ):
                            return read_decision
                action = decision.action
                if action is None:
                    return decision
                card = next(
                    (card for card in cards if card.action.action_id == action.action_id),
                    None,
                )
                if card is not None and card.argument_keys and not action.arguments:
                    # A small model can identify the capability but omit its
                    # object argument. Re-ask with the already-selected card;
                    # this remains bounded cognition, not phrase extraction.
                    focused = request.model_copy(update={"action_cards": (card,)})
                    decision = decoder.decode(
                        provider.decide(focused), (card,), allow_argument_proposals=True
                    )
            return decision
        except InvalidDecision as exc:
            if focused_raw is not None:
                grounded = _grounded_context_answer(context, focused_raw)
                if grounded is not None:
                    return grounded
            if is_question_request(intent.utterance) and not is_mutation_request(intent.utterance):
                try:
                    recovery_request = ModelRequest(
                        working_set=WorkingSet(intent=intent, context=context),
                        action_cards=(),
                    )
                    recovered = StrictDecisionDecoder().decode(
                        provider.decide(recovery_request), (), allow_argument_proposals=False
                    )
                    if recovered.kind is DecisionKind.ANSWER:
                        return recovered
                except Exception:
                    # Recovery is deliberately best-effort and answer-only;
                    # retain the original bounded failure if it cannot produce
                    # a valid non-authoritative answer.
                    pass
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message="I could not safely interpret that request. Please rephrase it.",
                evidence={
                    "provenance": "model_boundary",
                    "authoritative": False,
                    "failure_class": "invalid_model_decision",
                    "failure_reason": str(exc),
                },
                correlation_id=intent.correlation_id,
                retryable=True,
            )
        except Exception as exc:
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.FAILED,
                message="Model unavailable; request can be retried",
                evidence={
                    "provenance": "model_boundary",
                    "authoritative": False,
                    "error_type": type(exc).__name__,
                },
                correlation_id=intent.correlation_id,
                retryable=True,
            )

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
            if recovered_plan_actions is None:
                multi_action_result = MultiActionFastPath.resolve(intent)
                if multi_action_result is not None:
                    return persist_fast_result(multi_action_result)
            if self.dependencies.model_provider is None:
                domain_clarification = DomainClarificationFastPath.resolve(intent)
                if domain_clarification is not None:
                    return persist_fast_result(domain_clarification)
            contextual_mutation = ContextualMutationGuard.resolve(intent)
            if contextual_mutation is not None:
                return persist_fast_result(contextual_mutation)
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
            manager = PackManager(store=PostgresPackStore(connection))
            if self.dependencies.pack_bundles is not None:
                manager.reconcile(
                    tuple(self.dependencies.pack_bundles()),
                    self.dependencies.auto_enable_pack_ids,
                )
            else:
                # Compatibility path for older embedders; production clients
                # supply Pack bundles through the composition callback above.
                for bundle in reference_bundles():
                    try:
                        manager.status(bundle.manifest.pack_id)
                        installed_bundle = manager.bundle(bundle.manifest.pack_id)
                        if installed_bundle.model_dump(mode="json") != bundle.model_dump(
                            mode="json"
                        ):
                            manager.remove(bundle.manifest.pack_id)
                            manager.discover(bundle)
                    except KeyError:
                        manager.discover(bundle)
                manager.reconcile(tuple(reference_bundles()), frozenset(("tasks", "kitchen")))
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
                    domain_clarification = DomainClarificationFastPath.resolve(intent)
                    if domain_clarification is not None:
                        return persist_fast_result(domain_clarification)
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
