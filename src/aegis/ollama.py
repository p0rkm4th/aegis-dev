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
                "utterance": request.working_set.intent.utterance,
                "action_cards": cards,
            },
            sort_keys=True,
        )
