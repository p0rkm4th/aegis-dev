"""Ollama provider adapter with bounded structured-output repair."""

from __future__ import annotations

import json
from typing import Any, Protocol

from .contracts import ModelRequest, ModelResponse


class OllamaTransport(Protocol):
    def chat(self, payload: dict[str, Any]) -> dict[str, Any]: ...


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
                "format": "json",
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
    def _prompt(request: ModelRequest) -> str:
        cards = [card.model_dump(mode="json") for card in request.action_cards]
        return json.dumps(
            {
                "instruction": "Return exactly one structured Aegis Decision JSON object.",
                "utterance": request.working_set.intent.utterance,
                "action_cards": cards,
            },
            sort_keys=True,
        )
