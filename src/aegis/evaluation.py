"""Compact structured-decision evaluation harness for local-model baselines."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from .contracts import ActionCard, DecisionKind, ModelResponse
from .decoding import InvalidDecision, StrictDecisionDecoder
from .model_router import BaselineMetrics


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    response: ModelResponse
    cards: tuple[ActionCard, ...]
    expected_kind: DecisionKind | None = None
    expected_action_id: str | None = None
    expect_rejection: bool = False


class DecisionEvaluationHarness:
    def __init__(self, decoder: StrictDecisionDecoder | None = None) -> None:
        self.decoder = decoder or StrictDecisionDecoder()

    def run(self, cases: tuple[EvaluationCase, ...]) -> BaselineMetrics:
        metrics = BaselineMetrics()
        for case in cases:
            started = monotonic()
            schema_valid = False
            security_error = False
            false_completion = False
            try:
                decision = self.decoder.decode(case.response, case.cards)
                schema_valid = True
                accepted = not case.expect_rejection
                if case.expected_kind is not None and decision.kind is not case.expected_kind:
                    accepted = False
                if case.expected_action_id is not None:
                    accepted = (
                        accepted
                        and decision.action is not None
                        and decision.action.action_id == case.expected_action_id
                    )
                if case.expect_rejection:
                    security_error = True
                if (
                    decision.kind is DecisionKind.ANSWER
                    and case.expected_kind is DecisionKind.ACTION
                ):
                    false_completion = True
            except InvalidDecision:
                accepted = case.expect_rejection
            metrics.record(
                success=accepted,
                schema_valid=schema_valid,
                false_completion=false_completion,
                security_error=security_error,
                latency_ms=(monotonic() - started) * 1000,
            )
        return metrics
