"""Strict correlated RPC boundary for an external OpenClaw Gateway client."""

from __future__ import annotations

import json
from typing import Any, Protocol, cast
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


class OpenClawWebSocketChannel:
    """Minimal protocol-4 channel for a loopback OpenClaw Gateway.

    The channel deliberately owns only transport framing and authentication.
    Semantic authorization and action verification remain in AEGIS Core.
    """

    def __init__(
        self, url: str, token: str, timeout: float = 10.0, persistent: bool = False
    ) -> None:
        self.url = url
        self.token = token
        self.timeout = timeout
        self.persistent = persistent
        self._socket: Any | None = None

    def send(self, request: RpcRequest) -> RpcResponse:
        try:
            import websocket
        except ImportError as exc:
            raise RpcProtocolError(
                "websocket-client is required for live Gateway transport"
            ) from exc

        try:
            new_socket = self._socket is None
            socket = self._socket or websocket.create_connection(self.url, timeout=self.timeout)
            if new_socket and self.persistent:
                self._socket = socket
            try:
                if new_socket:
                    self._connect(socket)
                socket.send(json.dumps({
                    "type": "req",
                    "id": str(request.request_id),
                    "method": request.method,
                    "params": request.params,
                }))
                response = self._receive_response(socket, str(request.request_id))
                if response.get("ok"):
                    payload = response.get("payload")
                    if not isinstance(payload, dict):
                        raise RpcProtocolError("Gateway RPC payload is not an object")
                    return RpcResponse(request_id=request.request_id, result=payload)
                error = response.get("error")
                return RpcResponse(
                    request_id=request.request_id,
                    error=error if isinstance(error, dict) else {"message": str(error)},
                )
            finally:
                if not self.persistent:
                    socket.close()
        except RpcProtocolError:
            raise
        except Exception as exc:
            raise RpcProtocolError(f"Gateway transport failed: {exc}") from exc

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _connect(self, socket: Any) -> None:
        challenge = json.loads(socket.recv())
        if challenge.get("event") != "connect.challenge":
            raise RpcProtocolError("Gateway did not send connect.challenge")
        connect_id = str(uuid4())
        socket.send(json.dumps({
            "type": "req",
            "id": connect_id,
            "method": "connect",
            "params": {
                "minProtocol": 4,
                "maxProtocol": 4,
                "client": {
                    "id": "gateway-client",
                    "version": "0.1.0-dev",
                    "platform": "linux",
                    "mode": "backend",
                },
                "role": "operator",
                "scopes": ["operator.read", "operator.write", "operator.admin"],
                "caps": [],
                "commands": [],
                "permissions": {},
                "auth": {"token": self.token},
                "locale": "en-US",
                "userAgent": "aegis-core/0.1.0-dev",
            },
        }))
        hello = self._receive_response(socket, connect_id)
        if not hello.get("ok"):
            raise RpcProtocolError(f"Gateway handshake failed: {hello.get('error')}")

    @staticmethod
    def _receive_response(socket: Any, request_id: str) -> dict[str, Any]:
        while True:
            frame = cast(dict[str, Any], json.loads(socket.recv()))
            if frame.get("type") == "res" and frame.get("id") == request_id:
                return frame


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
