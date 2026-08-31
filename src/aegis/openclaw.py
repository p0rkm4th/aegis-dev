"""Versioned OpenClaw execution port; Gateway remains the runtime authority."""

from __future__ import annotations

from typing import Protocol

from .contracts import ExecutionRequest, Observation

OPENCLAW_TESTED_RELEASE = "2026.8.1"


class RuntimePolicy(Protocol):
    def allows(self, request: ExecutionRequest) -> bool: ...


class Approval(Protocol):
    def required(self, request: ExecutionRequest) -> bool: ...
    def approved(self, request: ExecutionRequest) -> bool: ...


class GatewayClient(Protocol):
    def execute(self, request: ExecutionRequest) -> Observation: ...


class OpenClawExecutor:
    """Enforce runtime policy and approval before crossing the Gateway."""

    def __init__(
        self, client: GatewayClient, runtime_policy: RuntimePolicy, approval: Approval
    ) -> None:
        self.client = client
        self.runtime_policy = runtime_policy
        self.approval = approval

    def execute(self, request: ExecutionRequest) -> Observation:
        if not self.runtime_policy.allows(request):
            return Observation(
                execution_id=request.action_id,
                evidence={"runtime": "denied", "release": OPENCLAW_TESTED_RELEASE},
                command_succeeded=False,
            )
        if self.approval.required(request) and not self.approval.approved(request):
            return Observation(
                execution_id=request.action_id,
                evidence={"approval": "denied", "release": OPENCLAW_TESTED_RELEASE},
                command_succeeded=False,
            )
        return self.client.execute(request)
