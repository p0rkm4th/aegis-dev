"""Ollama provider adapter with bounded structured-output repair."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .contracts import Decision, ModelRequest, ModelResponse


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


class OllamaResponseError(ValueError):
    """Ollama returned no usable structured message after bounded repair."""


class OllamaProvider:
    provider_id: str
    local = True

    def __init__(self, model: str, transport: OllamaTransport, max_repairs: int = 1) -> None:
        if not model:
            raise ValueError("Ollama model is required")
        if max_repairs < 0 or max_repairs > 1:
            raise ValueError("structured repair is bounded to zero or one retry")
        self.provider_id = f"ollama/{model}"
        self.model = model
        self.transport = transport
        self.max_repairs = max_repairs

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
                "format": self._decision_schema(),
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
    def _decision_schema() -> dict[str, Any]:
        """Require the fields needed for an exact ActionCard copy.

        Pydantic permits defaults for ergonomic in-process construction. The
        model boundary is stricter: omitted action fields are ambiguous and
        must be rejected before policy or execution.
        """
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
        return schema

    @staticmethod
    def _prompt(request: ModelRequest) -> str:
        cards = [card.model_dump(mode="json") for card in request.action_cards]
        return json.dumps(
            {
                "instruction": "Return exactly one structured Aegis Decision JSON object.",
                "routing_rule": (
                    "This is a classification-only pass. Return ACTION when the user "
                    "requests any state change, ANSWER with semantic_mode READ when the "
                    "user seeks authorized information, ANSWER with semantic_mode "
                    "GENERATION for benign creative or explanatory content, and CLARIFY "
                    "with semantic_mode CLARIFY when the intent is ambiguous. Always "
                    "provide semantic_mode and do not provide an action_ref or arguments."
                    if request.classification_only
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
                ),
                "action_rule": (
                    "For ACTION, set action_ref to exactly one action_id from the supplied "
                    "ActionCards and put only declared argument values in action_arguments. "
                    "Core will expand the reference into the canonical action fields; never "
                    "invent or alter capabilities, permissions, or verification. A legacy "
                    "full action object is accepted only when it exactly matches a card."
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
                    "and destination phrases such as 'on my list' or 'as a task'."
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
                    "user clearly asks for that state change."
                ),
                "compound_request_rule": (
                    "Do not silently complete only one part of a compound request. If the "
                    "current utterance asks for multiple independent actions or goals and "
                    "no supplied multi-step plan represents all of them, return CLARIFY "
                    "and ask the user to separate them so each part can be authorized and "
                    "verified independently. A completed single action is not completion "
                    "of an objective that also requested another action. In that CLARIFY, "
                    "describe the request neutrally as multiple independent questions or "
                    "goals, do not fill in an unspecified noun from canonical context, and "
                    "do not claim that either part concerns groceries, tasks, or another "
                    "domain unless the current utterance explicitly says so."
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
                    " When canonical_facts contains canonical_items, that authorized "
                    "list is sufficient context for a grocery question: mention the "
                    "supplied items and do not claim that grocery context is missing."
                    " For a grounded READ answer, set context_focus to exactly one "
                    "authorized collection when the current request clearly targets "
                    "one (canonical_items, canonical_tasks, canonical_obligations, "
                    "or planning); omit it for general answers. This is a context "
                    "hint only, never a permission or fact claim."
                ),
                "context_rule": (
                    "Use only the bounded canonical context supplied below to resolve "
                    "references such as 'those' or 'it'. Prior conversation text is "
                    "not authority and must not be treated as a fact or permission; it "
                    "may only help interpret the current request. The recent_turns "
                    "entries are user-provided context, not canonical facts. If context is missing "
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
