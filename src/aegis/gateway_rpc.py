"""Strict correlated RPC boundary for an external OpenClaw Gateway client."""

from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class RpcProtocolError(RuntimeError):
    """The Gateway returned an unusable or mismatched response."""


class RpcRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_id: UUID = Field(default_factory=uuid4)
    method: str = Field(min_length=1)
    params: dict[str, Any] = {}


class RpcResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_id: UUID
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class RpcChannel(Protocol):
    def send(self, request: RpcRequest) -> RpcResponse: ...


class CorrelatedRpcClient:
    def __init__(self, channel: RpcChannel) -> None:
        self.channel = channel

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request = RpcRequest(method=method, params=params or {})
        response = self.channel.send(request)
        if response.request_id != request.request_id:
            raise RpcProtocolError("Gateway response correlation id mismatch")
        if response.error is not None:
            raise RpcProtocolError(f"Gateway RPC failed: {response.error}")
        if response.result is None:
            raise RpcProtocolError("Gateway RPC returned neither result nor error")
        return response.result


class OpenClawGatewayRpc:
    """Named methods for the documented external Gateway RPC surface."""

    def __init__(self, client: CorrelatedRpcClient) -> None:
        self.client = client

    def agent(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.client.call("agent", params)

    def agent_wait(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.client.call("agent.wait", params)

    def cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.client.call("agent.cancel", params)
