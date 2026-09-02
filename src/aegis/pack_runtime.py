"""Typed runtime bindings for Pack-provided actions.

Pack metadata describes what may be proposed. A runtime binding supplies the
separate executor, verifier, and policy relation used after Core validation.
The model and Pack metadata never manufacture this binding.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .contracts import ActionCard, Principal


@dataclass(frozen=True)
class ActionRuntime:
    executor: Any
    verifier: Any
    permissions: dict[str, frozenset[Any]]
    cleanup: Callable[[], None] | None = None


RuntimeFactory = Callable[[Any, Principal], ActionRuntime]


class PackRuntimeRegistry:
    """Resolve installed Pack action IDs without central domain knowledge."""

    def __init__(self) -> None:
        self._factories: dict[str, RuntimeFactory] = {}

    def register(self, action_id: str, factory: RuntimeFactory) -> None:
        if not action_id or action_id in self._factories:
            raise ValueError("runtime action ID must be non-empty and unique")
        self._factories[action_id] = factory

    def action_ids(self) -> tuple[str, ...]:
        """Expose registered bindings for diagnostics without leaking registry state."""

        return tuple(sorted(self._factories))

    def register_pack(
        self,
        cards: tuple[ActionCard, ...],
        factories: dict[str, RuntimeFactory],
    ) -> None:
        """Atomically register all runtimes declared by one Pack."""

        action_ids = tuple(card.action.action_id for card in cards)
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("Pack ActionCard IDs must be unique")
        if set(factories) != set(action_ids):
            raise ValueError("Pack runtime bindings must match its ActionCards")
        if set(action_ids) & self._factories.keys():
            raise ValueError("Pack runtime action IDs must be unique")
        self._factories.update(factories)

    def resolve(self, card: ActionCard, connection: Any, principal: Principal) -> ActionRuntime:
        try:
            runtime = self._factories[card.action.action_id](connection, principal)
        except KeyError as exc:
            raise LookupError(f"no runtime binding for {card.action.action_id}") from exc
        required = frozenset(card.action.required_permissions)
        provided = frozenset(runtime.permissions)
        if not required.issubset(provided):
            raise PermissionError(
                f"runtime binding does not cover {card.action.action_id} permissions"
            )
        return runtime
