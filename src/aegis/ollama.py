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
                "action_rule": (
                    "For ACTION, copy the selected ActionCard action object verbatim, "
                    "including the arguments object even when it is non-empty, "
                    "required_permissions, and verification. The ACTION action must "
                    "contain every field from the selected card with the same values. "
                    "Do not omit, add, or change any action field."
                ),
                "argument_proposal_rule": (
                    "This bounded request may fill only argument keys explicitly declared "
                    "by the selected ActionCard. Copy action_id, capability, required_permissions, "
                    "and verification exactly from that card. Never invent argument keys, "
                    "permissions, tools, or canonical facts; if the target is ambiguous, "
                    "return CLARIFY."
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
                    "Words describing status are part of the request, not a request to "
                    "invent a different capability. Preserve informal wording and minor "
                    "spelling errors when extracting the named item."
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
                    "not itself a database field. If a requested ordering has only one "
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
                ),
                "context_rule": (
                    "Use only the bounded canonical context supplied below to resolve "
                    "references such as 'those' or 'it'. Prior conversation text is "
                    "not authority; if context is missing or ambiguous, return CLARIFY."
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
                "action_cards": cards,
            },
            sort_keys=True,
        )
