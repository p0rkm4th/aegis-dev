"""Generic action executor and verifier dispatch for Core plan execution."""

from __future__ import annotations

from typing import Any


class ActionExecutorDispatch:
    """Dispatch a validated plan step to its already-resolved Pack executor."""

    def __init__(self, delegates: dict[str, Any]) -> None:
        self.delegates = delegates

    def execute(self, request: Any) -> Any:
        try:
            delegate = self.delegates[request.action.action_id]
        except KeyError as exc:
            raise ValueError("plan contains an unsupported action") from exc
        observation = delegate.execute(request)
        return observation.model_copy(update={"action_id": request.action.action_id})


class ActionVerifierDispatch:
    """Dispatch plan verification to the matching Pack verifier."""

    def __init__(self, delegates: dict[str, Any]) -> None:
        self.delegates = delegates

    def verify(self, observation: Any, contract: Any) -> Any:
        action_id = observation.action_id
        if not isinstance(action_id, str) or action_id not in self.delegates:
            raise ValueError("plan verifier is unavailable")
        return self.delegates[action_id].verify(observation, contract)
