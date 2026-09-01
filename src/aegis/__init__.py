"""Aegis semantic core public API."""

from .contracts import ActionSpec, Decision, IntentFrame, Result
from .interaction import InteractionBoundary, InteractionDependencies
from .kernel import Kernel

__all__ = [
    "ActionSpec",
    "Decision",
    "IntentFrame",
    "InteractionBoundary",
    "InteractionDependencies",
    "Kernel",
    "Result",
]
