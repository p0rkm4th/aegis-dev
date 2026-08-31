"""Deterministic handling for canonical reads and other safe operations."""

from __future__ import annotations

from typing import Protocol

from .contracts import IntentFrame, Result


class DeterministicFastPath(Protocol):
    def resolve(self, intent: IntentFrame) -> Result | None: ...


class NoopFastPath:
    def resolve(self, intent: IntentFrame) -> Result | None:
        return None
