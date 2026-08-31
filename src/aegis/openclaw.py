"""Versioned OpenClaw execution port; Gateway remains the runtime authority."""

from __future__ import annotations

from threading import Event
from typing import Protocol

from .contracts import ExecutionRequest, Observation

OPENCLAW_TESTED_RELEASE = "2026.8.1"


class GatewayDisconnected(RuntimeError):
    """The Gateway connection ended before an outcome was known."""


class RuntimePolicy(Protocol):
    def allows(self, request: ExecutionRequest) -> bool: ...


class Approval(Protocol):
    def required(self, request: ExecutionRequest) -> bool: ...
    def approved(self, request: ExecutionRequest) -> bool: ...


class GatewayClient(Protocol):
    def execute(self, request: ExecutionRequest) -> Observation: ...


class GatewayTransport(Protocol):
    def reconnect(self) -> None: ...
    def execute(self, request: ExecutionRequest) -> Observation: ...
    def retry_is_safe(self, request: ExecutionRequest) -> bool: ...
    def cancel(self, request: ExecutionRequest) -> None: ...


class ReconnectingGatewayClient:
    """Preserve correlation/idempotency across a reconnect without blind retries."""

    def __init__(self, transport: GatewayTransport) -> None:
        self.transport = transport

    def execute(self, request: ExecutionRequest, cancel_event: Event | None = None) -> Observation:
        if cancel_event is not None and cancel_event.is_set():
            self.transport.cancel(request)
            return Observation(
                execution_id=request.action_id,
                evidence={"transport": "cancelled", "idempotency_key": request.idempotency_key},
                command_succeeded=False,
            )
        try:
            return self.transport.execute(request)
        except GatewayDisconnected:
            if not self.transport.retry_is_safe(request):
                return Observation(
                    execution_id=request.action_id,
                    evidence={
                        "transport": "disconnected",
                        "outcome": "unknown",
                        "idempotency_key": request.idempotency_key,
                    },
                    command_succeeded=False,
                )
            self.transport.reconnect()
            return self.transport.execute(request)


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
