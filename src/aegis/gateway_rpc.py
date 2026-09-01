"""Strict correlated RPC boundary for an external OpenClaw Gateway client."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class RpcProtocolError(RuntimeError):
    """The Gateway returned an unusable or mismatched response."""


def _safe_gateway_code(error: Any) -> str:
    """Keep remote Gateway errors diagnostic without copying remote text."""

    if isinstance(error, dict):
        code = error.get("code")
        if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", code):
            return code
    return "gateway_error"


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
        self,
        url: str,
        token: str,
        timeout: float = 10.0,
        persistent: bool = False,
        device_id: str | None = None,
        device_token: str | None = None,
        private_key_pem: str | None = None,
        public_key_pem: str | None = None,
    ) -> None:
        self.url = url
        self.token = token
        self.timeout = timeout
        self.persistent = persistent
        self.device_id = device_id
        self.device_token = device_token
        self.private_key_pem = private_key_pem
        self.public_key_pem = public_key_pem
        self._socket: Any | None = None
        self._pending_events: list[dict[str, Any]] = []

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
                socket.send(
                    json.dumps(
                        {
                            "type": "req",
                            "id": str(request.request_id),
                            "method": request.method,
                            "params": request.params,
                        }
                    )
                )
                response = self._receive_response(socket, str(request.request_id))
                if response.get("ok"):
                    payload = response.get("payload")
                    if not isinstance(payload, dict):
                        raise RpcProtocolError("Gateway RPC payload is not an object")
                    return RpcResponse(request_id=request.request_id, result=payload)
                error = response.get("error")
                return RpcResponse(
                    request_id=request.request_id,
                    error={"code": _safe_gateway_code(error)},
                )
            finally:
                if not self.persistent:
                    socket.close()
        except RpcProtocolError:
            if self.persistent:
                self._discard_socket()
            raise
        except Exception as exc:
            if self.persistent:
                self._discard_socket()
            raise RpcProtocolError(f"Gateway transport failed: {type(exc).__name__}") from exc

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _discard_socket(self) -> None:
        socket = self._socket
        self._socket = None
        self._pending_events.clear()
        if socket is not None:
            try:
                socket.close()
            except Exception:
                pass

    def receive_event(self, event_name: str) -> dict[str, Any]:
        """Receive the next named Gateway event on a persistent connection."""
        if self._socket is None:
            raise RpcProtocolError("Gateway event requires a persistent connection")
        try:
            for index, frame in enumerate(self._pending_events):
                if frame.get("event") == event_name:
                    self._pending_events.pop(index)
                    payload = frame.get("payload")
                    if not isinstance(payload, dict):
                        raise RpcProtocolError("Gateway event payload is not an object")
                    return payload
            while True:
                frame = cast(dict[str, Any], json.loads(self._socket.recv()))
                if frame.get("type") == "event" and frame.get("event") == event_name:
                    payload = frame.get("payload")
                    if not isinstance(payload, dict):
                        raise RpcProtocolError("Gateway event payload is not an object")
                    return payload
                if frame.get("type") == "event":
                    self._pending_events.append(frame)
        except RpcProtocolError:
            self._discard_socket()
            raise
        except Exception as exc:
            self._discard_socket()
            raise RpcProtocolError(f"Gateway event transport failed: {type(exc).__name__}") from exc

    def _connect(self, socket: Any) -> None:
        challenge = json.loads(socket.recv())
        if challenge.get("event") != "connect.challenge":
            raise RpcProtocolError("Gateway did not send connect.challenge")
        connect_id = str(uuid4())
        scopes = ["operator.read", "operator.write", "operator.admin", "operator.approvals"]
        auth: dict[str, str] = {"token": self.token}
        device: dict[str, Any] | None = None
        if all((self.device_id, self.device_token, self.private_key_pem, self.public_key_pem)):
            try:
                import base64

                from cryptography.hazmat.primitives import serialization
                from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                    Ed25519PrivateKey,
                    Ed25519PublicKey,
                )

                if not self.private_key_pem or not self.public_key_pem:
                    raise ValueError("missing OpenClaw device key")
                private_key = serialization.load_pem_private_key(
                    self.private_key_pem.encode(), password=None
                )
                public_key = serialization.load_pem_public_key(self.public_key_pem.encode())
                if not isinstance(private_key, Ed25519PrivateKey) or not isinstance(
                    public_key, Ed25519PublicKey
                ):
                    raise ValueError("OpenClaw device identity must use Ed25519")
                raw_public_key = public_key.public_bytes(
                    serialization.Encoding.Raw, serialization.PublicFormat.Raw
                )
                signed_at = int(challenge["payload"]["ts"])
                signed_payload = "|".join(
                    [
                        "v3",
                        self.device_id or "",
                        "gateway-client",
                        "backend",
                        "operator",
                        ",".join(scopes),
                        str(signed_at),
                        self.token,
                        challenge["payload"]["nonce"],
                        "linux",
                        "",
                    ]
                )

                def encode(value: bytes) -> str:
                    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()

                device = {
                    "id": self.device_id,
                    "publicKey": encode(raw_public_key),
                    "signature": encode(private_key.sign(signed_payload.encode())),
                    "signedAt": signed_at,
                    "nonce": challenge["payload"]["nonce"],
                }
                auth = {"token": self.token, "deviceToken": self.device_token or ""}
            except (KeyError, TypeError, ValueError) as exc:
                raise RpcProtocolError("invalid OpenClaw device identity") from exc
        socket.send(
            json.dumps(
                {
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
                        "scopes": scopes,
                        "caps": [],
                        "commands": [],
                        "permissions": {},
                        "auth": auth,
                        "locale": "en-US",
                        "userAgent": "aegis-core/0.1.0-dev",
                        **({"device": device} if device is not None else {}),
                    },
                }
            )
        )
        hello = self._receive_response(socket, connect_id)
        if not hello.get("ok"):
            raise RpcProtocolError(
                f"Gateway handshake failed: {_safe_gateway_code(hello.get('error'))}"
            )

    def _receive_response(self, socket: Any, request_id: str) -> dict[str, Any]:
        try:
            while True:
                frame = cast(dict[str, Any], json.loads(socket.recv()))
                if frame.get("type") == "res" and frame.get("id") == request_id:
                    return frame
                if frame.get("type") == "event":
                    self._pending_events.append(frame)
        except RpcProtocolError:
            raise
        except Exception as exc:
            raise RpcProtocolError(f"Gateway transport failed: {type(exc).__name__}") from exc


class CorrelatedRpcClient:
    def __init__(self, channel: RpcChannel) -> None:
        self.channel = channel

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request = RpcRequest(method=method, params=params or {})
        response = self.channel.send(request)
        if response.request_id != request.request_id:
            raise RpcProtocolError("Gateway response correlation id mismatch")
        if response.error is not None:
            raise RpcProtocolError(f"Gateway RPC failed: {_safe_gateway_code(response.error)}")
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

    def notify(self, params: dict[str, Any]) -> dict[str, Any]:
        """Queue a Gateway system event; it is not an authorization decision."""
        if not isinstance(params.get("text"), str) or not params["text"].strip():
            raise ValueError("Gateway notification requires non-empty text")
        return self.client.call("system-event", params)

    def terminal_open(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.client.call("terminal.open", params or {})

    def terminal_input(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.client.call("terminal.input", params)

    def terminal_close(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.client.call("terminal.close", params)
