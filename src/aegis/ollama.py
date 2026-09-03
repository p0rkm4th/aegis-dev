"""Ollama provider adapter with bounded structured-output repair."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .contracts import Decision, ModelRequest, ModelResponse, ObjectiveSpecProposal


class OllamaTransport(Protocol):
    def chat(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class OllamaHttpTransport:
    """Small standard-library transport for the Ollama HTTP API."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("Ollama base URL must use HTTP or HTTPS")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value = json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaResponseError("Ollama HTTP request failed") from exc
        if not isinstance(value, dict):
            raise OllamaResponseError("Ollama returned a non-object response")
        return value

    def tags(self) -> dict[str, Any]:
        """Return Ollama's local model inventory for runtime provenance.

        This is metadata-only: it never loads or invokes a model.  Keeping it
        on the transport lets evaluation and diagnostics identify the exact
        model digest without making Core depend on Ollama's inventory API.
        """
        request = Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value = json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaResponseError("Ollama model inventory request failed") from exc
        if not isinstance(value, dict):
            raise OllamaResponseError("Ollama returned a non-object model inventory")
        return value

    def model_digest(self, model: str) -> str | None:
        """Resolve a model's immutable digest when Ollama exposes one."""
        models = self.tags().get("models")
        if not isinstance(models, list):
            raise OllamaResponseError("Ollama model inventory has no models list")
        for item in models:
            if isinstance(item, dict) and item.get("name") == model:
                digest = item.get("digest")
                return digest if isinstance(digest, str) and digest else None
        return None


class OllamaResponseError(ValueError):
    """Ollama returned no usable structured message after bounded repair."""


class OllamaProvider:
    provider_id: str
    local = True

    def __init__(
        self,
        model: str,
        transport: OllamaTransport,
        max_repairs: int = 1,
        compact_action_cards: bool = False,
        action_ref_only: bool = False,
    ) -> None:
        if not model:
            raise ValueError("Ollama model is required")
        if max_repairs < 0 or max_repairs > 1:
            raise ValueError("structured repair is bounded to zero or one retry")
        self.provider_id = f"ollama/{model}"
        self.model = model
        self.transport = transport
        self.max_repairs = max_repairs
        self.compact_action_cards = compact_action_cards
        self.action_ref_only = action_ref_only

    def available(self) -> bool:
        return True

    def decide(self, request: ModelRequest) -> ModelResponse:
        prompt = self._prompt(request)
        for attempt in range(self.max_repairs + 1):
            payload = {
                "model": self.model,
                "stream": False,
                "think": False,
                "options": {"temperature": 0},
                "format": self._decision_schema(request, action_ref_only=self.action_ref_only),
                "messages": [{"role": "user", "content": prompt}],
            }
            response = self.transport.chat(payload)
            try:
                content = response["message"]["content"]
                return ModelResponse(raw=json.loads(content))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                if attempt >= self.max_repairs:
                    raise OllamaResponseError("Ollama did not return valid JSON") from exc
                prompt = (
                    f"{prompt}\n\nYour prior response was invalid. Return only one JSON object "
                    "matching the requested decision schema; do not include markdown."
                )
        raise AssertionError("bounded repair loop unexpectedly continued")

    @staticmethod
    def _decision_schema(
        request: ModelRequest | None = None, *, action_ref_only: bool = False
    ) -> dict[str, Any]:
        """Require the fields needed for an exact ActionCard copy.

        Pydantic permits defaults for ergonomic in-process construction. The
        model boundary is stricter: omitted action fields are ambiguous and
        must be rejected before policy or execution.
        """
        if request is not None and request.objective_interpretation_only:
            return ObjectiveSpecProposal.model_json_schema()
        schema = deepcopy(Decision.model_json_schema())
        action_schema = schema.get("$defs", {}).get("ActionSpec")
        if isinstance(action_schema, dict):
            action_schema["required"] = [
                "action_id",
                "capability",
                "arguments",
                "required_permissions",
                "verification",
            ]
        # The provider-facing contract requires the model to declare the
        # semantic mode for every decision.  In-process providers may still
        # use the ergonomic optional field from the Core contract, but a
        # provider response without a mode is ambiguous at the boundary and
        # cannot safely support grounded recovery or action routing.
        required = schema.setdefault("required", [])
        for field in ("semantic_mode", "knowledge_source"):
            if field not in required:
                required.append(field)
        if action_ref_only:
            properties = schema.get("properties")
            if isinstance(properties, dict):
                properties.pop("action", None)
        return schema

    def _prompt(self, request: ModelRequest) -> str:
        cards = [
            self._compact_card(card) if self.compact_action_cards else card.model_dump(mode="json")
            for card in request.action_cards
        ]
        return json.dumps(
            {
                "instruction": (
                    "Return exactly one objective_spec JSON object. It must contain one "
                    "requirement for each independent state change requested by the user. "
                    "Use only exact action_ref values from the supplied ActionCards and "
                    "only the grounded arguments needed for that effect. Do not include a "
                    "plan, completion claim, permissions, or verification. Core will assign "
                    "stable identities and validate the proposal."
                    if request.objective_interpretation_only
                    else "Return exactly one structured Aegis Decision JSON object."
                ),
                "semantic_mode_rule": (
                    "Always provide semantic_mode: ACTION for a state change, READ for "
                    "authorized information, GENERATION for benign creative/explanatory "
                    "content, or CLARIFY when the request is ambiguous."
                ),
                "routing_rule": (
                    "This is a classification-only pass. Return ACTION when the user "
                    "requests any state change, ANSWER with semantic_mode READ when the "
                    "user seeks authorized information, ANSWER with semantic_mode "
                    "GENERATION for benign creative or explanatory content, and CLARIFY "
                    "with semantic_mode CLARIFY when the intent is ambiguous. For an "
                    "ANSWER, also set knowledge_source to general_model_knowledge for "
                    "stable knowledge, external_evidence when the question asks for "
                    "current/latest/recent information, or mixed_evidence when current "
                    "public information must be combined with authorized local context. "
                    "Always provide semantic_mode and do not provide an action_ref or "
                    "arguments."
                    if request.classification_only
                    else (
                        "This is a capability-scoped pass for exactly one supplied ActionCard. "
                        "Return ACTION only when the user requests this capability as one "
                        "independent operation; otherwise return ANSWER with a brief empty "
                        "scope indication. Do not select or invent any other capability."
                        if request.capability_scoped
                        else (
                            "This is an action-routing pass. Decide whether the current request "
                            "clearly proposes a supplied write-capable action. Do not answer a "
                            "read question from context during this pass; return ANSWER only for "
                            "benign conversation that does not request a change, or CLARIFY when "
                            "the requested change is ambiguous."
                            if request.routing_only
                            else (
                                "This is the final bounded cognition pass; answer only from the "
                                "supplied context."
                            )
                        )
                    )
                ),
                "plan_rule": (
                    "For PLAN, use only when the objective clearly requires multiple state "
                    "changes. Set semantic_mode to ACTION and return a plan object whose "
                    "steps array contains 2-5 steps. Each step must contain an exact action_ref "
                    "from the supplied ActionCards, only declared argument keys, and optional "
                    "depends_on indexes that point to earlier steps. A PLAN must contain the "
                    "plan object and must not contain action_ref, action_arguments, or action; "
                    "those fields are only for ACTION. Include every independent state change "
                    "requested by the user exactly once; if you cannot account for all of them "
                    "from the supplied cards, return CLARIFY instead of claiming completion. "
                    "Keep arguments scoped to their own step; do not copy a date, time, or "
                    "other detail from one independent operation into another unless the user "
                    "explicitly assigns that detail to both. "
                    "Also return objective_spec with one requirement for every requested "
                    "state change; each requirement must use the exact same action_ref and "
                    "arguments as its corresponding plan step. Core will validate and persist "
                    "these requirements and will reject incomplete or extra coverage. "
                    "If one action is sufficient, return ACTION instead. Never invent capabilities."
                    if request.allow_plan_proposals
                    else "Plan proposals are disabled; return at most one action."
                ),
                "action_rule": (
                    "For ACTION, set action_ref to exactly one action_id from the supplied "
                    "ActionCards and put only declared argument values in action_arguments. "
                    "Core will expand the reference into the canonical action fields; never "
                    "invent or alter capabilities, permissions, or verification."
                    if self.action_ref_only
                    else (
                        "For ACTION, set action_ref to exactly one action_id from the supplied "
                        "ActionCards and put only declared argument values in action_arguments. "
                        "Core will expand the reference into the canonical action fields; never "
                        "invent or alter capabilities, permissions, or verification. A legacy "
                        "full action object is accepted only when it exactly matches a card."
                    )
                ),
                "argument_proposal_rule": (
                    "This bounded request may fill only argument keys explicitly declared "
                    "by the selected ActionCard. Copy action_id, capability, required_permissions, "
                    "and verification exactly from that card. Use each ActionCard argument "
                    "description to understand what belongs in a declared field. Never "
                    "invent argument keys, permissions, tools, or canonical facts; if the "
                    "target is ambiguous, "
                    "return CLARIFY. For a create action, extract the smallest complete "
                    "description of the thing the user wants recorded, excluding politeness "
                    "and destination phrases such as 'on my list' or 'as a task'. A declared "
                    "argument described as optional or 'if clearly stated' must be omitted "
                    "when the user did not supply it; do not ask for or invent an optional "
                    "value."
                    if request.allow_argument_proposals
                    else "Argument proposals are disabled; copy ActionCard actions verbatim."
                ),
                "ambiguity_rule": (
                    "If the utterance does not clearly identify one domain and more "
                    "than one ActionCard is plausible, return CLARIFY instead of "
                    "guessing an action; set the clarification field to a concise "
                    "clarification question. Never put the question only in reason."
                ),
                "selection_rule": (
                    "Choose by the meaning of the complete utterance, not by exact "
                    "keyword matches. A request to change a named task's status to "
                    "complete means select the card whose summary marks a task complete "
                    "and put only the task's name in its declared title argument. "
                    "Completion meaning may be expressed as finishing, wrapping up, "
                    "taking care of, or saying an item is done; understand the intent "
                    "semantically rather than requiring a fixed verb. Words describing "
                    "status are part of the request, not a request to invent a different "
                    "capability. Preserve informal wording and minor spelling errors "
                    "when extracting the named item. When canonical_tasks are supplied, "
                    "copy the exact matching candidate title, including leading articles "
                    "and punctuation; never shorten or paraphrase it. If no one exact "
                    "candidate is clear, return CLARIFY. If the utterance describes "
                    "finishing an untyped item and authorized_task_candidates contains "
                    "one matching task, prefer tasks.complete over a chore action."
                ),
                "mutation_polarity_rule": (
                    "Distinguish the requested state transition, not just the named item. "
                    "When the user asks to add, create, record, put, or make a new item, "
                    "select a create card even if an existing candidate has a similar name. "
                    "Select a complete card only when the user asks to finish, close, mark "
                    "done, or otherwise change an existing item to completed. Never turn a "
                    "request to add a new item into completion of a nearby existing item. "
                    "A destination such as a to-do list, task list, or things-to-do list "
                    "means tasks.create; create an event only when the user explicitly asks "
                    "for a calendar/event record, regardless of appointment-like wording."
                ),
                "read_polarity_rule": (
                    "Distinguish information-seeking questions from mutations. A question "
                    "asking what is on a list, what should be picked up, or what remains "
                    "to do is a read request even when it contains words associated with "
                    "an action. Select the supplied read card and answer from authorized "
                    "canonical facts; do not add, complete, or change state unless the "
                    "user clearly asks for that state change. For an informational request "
                    "that spans more than one authorized collection, return ANSWER with "
                    "semantic_mode READ and summarize the relevant supplied collections; "
                    "do not force the request into one ActionCard merely because one card "
                    "has a related word."
                ),
                "compound_request_rule": (
                    "Do not silently complete only one part of a compound request. If "
                    "the current utterance clearly asks for multiple independent state "
                    "changes, return PLAN with one bounded step for each requested "
                    "change; Core will authorize and verify every step independently. "
                    "A completed single action is not completion of an objective that "
                    "also requested another action."
                    if request.allow_plan_proposals
                    else "Do not silently complete only one part of a compound request. "
                    "If the current utterance asks for multiple independent actions or "
                    "goals and no supplied multi-step plan represents all of them, return "
                    "CLARIFY and ask the user to separate them so each part can be "
                    "authorized and verified independently. A completed single action "
                    "is not completion of an objective that also requested another "
                    "action. In that CLARIFY, describe the request neutrally as multiple "
                    "independent questions or goals, do not fill in an unspecified noun "
                    "from canonical context, and do not claim that either part concerns "
                    "groceries, tasks, or another domain unless the current utterance "
                    "explicitly says so."
                ),
                "prioritization_rule": (
                    "When asked which task to do first or what to prioritize, do not dump "
                    "the entire task list. Recommend only from the supplied canonical open "
                    "tasks, using an explicit due_at when present; if the authorized facts "
                    "do not support a grounded choice, explain the limitation or clarify."
                ),
                "benign_generation_rule": (
                    "Creative and explanatory requests such as writing a story, joke, poem, "
                    "or explanation are ANSWER requests. ActionCards are bounded candidates, "
                    "not commands: never select an unrelated read or mutation merely because "
                    "its context contains a matching noun, and never answer a creative request "
                    "with an unrelated canonical list."
                ),
                "answer_rule": (
                    "For a benign request that does not require a supplied ActionCard, "
                    "return ANSWER with useful conversational content. Do not present "
                    "generated content as canonical fact, and do not invent tools, "
                    "permissions, actions, or private data. The absence of a matching "
                    "ActionCard never limits benign creative generation: for a story, "
                    "joke, explanation, or similar request, provide the content rather "
                    "than claiming that a capability is unavailable. If a request needs a "
                    "mutation but no supplied card safely represents it, return CLARIFY."
                    " If authorized canonical facts are supplied and the request is a "
                    "read-only question about them, answer from those facts; do not "
                    "ask the user to identify an action merely because the answer is "
                    "not itself a database field. An imperative request to create, "
                    "complete, change, or handle something is not read-only: do not "
                    "answer it with an unrelated canonical read. If no supplied card "
                    "safely represents that mutation, return CLARIFY. If a requested "
                    "ordering has only one "
                    "distinct supplied value, explain that the candidates are tied "
                    "instead of asking the user to choose one. ANSWER must contain the "
                    "actual answer content; if required information is missing, use "
                    "CLARIFY with a question instead of ANSWER with only a reason."
                    " Never describe an open task as due today, urgent, or otherwise "
                    "time-bound unless its supplied canonical task object contains a "
                    "matching due_at; open status alone is not a deadline or priority. "
                    "For a broad informational request about what needs attention, use "
                    "the supplied authorized tasks, chores, and obligations together "
                    "when present; ask for clarification only when the authorized working "
                    "set truly cannot support a useful grounded answer. "
                    " When canonical_facts contains canonical_items, that authorized "
                    "list is sufficient context for a grocery question: mention the "
                    "supplied items and do not claim that grocery context is missing."
                    " For a grounded READ answer, set context_focus to exactly one "
                    "authorized collection when the current request clearly targets "
                    "one (canonical_items, canonical_tasks, canonical_obligations, "
                    "or planning); omit it for general answers. This is a context "
                    "hint only, never a permission or fact claim. For ANSWER requests, "
                    "do not use supplied canonical facts for an external subject, "
                    "software project, version, release, or other general-knowledge "
                    "question unless the user explicitly asks about their own "
                    "authorized state; canonical context must not turn an unrelated "
                    "question into an obligation, task, grocery, or memory answer. "
                    "set knowledge_source to general_model_knowledge for stable general "
                    "knowledge, external_evidence for information explicitly requiring "
                    "current/latest verification, and mixed_evidence only when public "
                    "research must be combined with authorized local context. This is "
                    "a source request, not a claim about evidence actually obtained."
                ),
                "context_rule": (
                    "Use only the bounded canonical context supplied below to resolve "
                    "references such as 'those' or 'it'. Prior conversation text is "
                    "not authority and must not be treated as a fact or permission; it "
                    "may only help interpret the current request. The recent_turns "
                    "entries are conversational context, not canonical facts. If context is "
                    "missing "
                    "or ambiguous, return CLARIFY. The referents object contains only "
                    "Core-selected candidates from canonical_facts; it is not permission "
                    "to mutate every candidate and does not resolve an ambiguous ordinal "
                    "without a clear user request. If referents.those has exactly one "
                    "candidate, answer that it is the only candidate when the user asks "
                    "which one comes first; do not ask the user to choose among nothing."
                ),
                "temporal_grounding_rule": (
                    "Use the supplied as_of_date and canonical task due_at values for "
                    "time-sensitive answers. Call a task due today only when its due_at "
                    "calendar date equals as_of_date; open status alone is not a deadline. "
                    "Do not invent a reference such as 'those' or 'it' when the utterance "
                    "does not contain one. If no supplied task has a matching deadline, "
                    "say that no due-today deadline is recorded rather than asking an "
                    "irrelevant clarification."
                ),
                "single_card_rule": (
                    "If exactly one ActionCard is supplied and the utterance clearly "
                    "requests that capability, return ACTION. Do not return CLARIFY "
                    "only because the request contains a date or relative time such "
                    "as tomorrow; preserve the supplied card and arguments exactly."
                ),
                "utterance": request.working_set.intent.utterance,
                "bounded_context": request.working_set.context.model_dump(mode="json"),
                "routing_only": request.routing_only,
                "classification_only": request.classification_only,
                "action_cards": cards,
            },
            sort_keys=True,
        )

    @staticmethod
    def _compact_card(card: Any) -> dict[str, Any]:
        """Expose cognition-relevant card semantics without policy internals."""
        return {
            "action_id": card.action.action_id,
            "capability": card.action.capability,
            "operation": (
                "write"
                if any(
                    permission.endswith(".write") for permission in card.action.required_permissions
                )
                else "read"
            ),
            "summary": card.summary,
            "relevance": card.relevance,
            "argument_keys": card.argument_keys,
            "argument_descriptions": card.argument_descriptions,
        }
