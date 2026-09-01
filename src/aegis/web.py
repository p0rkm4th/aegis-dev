"""Small browser adapter over the existing interaction and state boundaries."""

from __future__ import annotations

import json
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse
from uuid import UUID, uuid4

from .contracts import Principal
from .health import HealthReport

Interaction = Callable[[str, Principal, UUID], str | dict[str, Any]]
ConstellationState = Callable[[Principal], dict[str, Any]]
PrincipalProvider = Callable[[], Principal]
HealthProvider = Callable[[], HealthReport | dict[str, Any]]
_MAX_BODY_BYTES = 20_000


_INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>AEGIS Constellation</title>
<style>body{font:16px system-ui;margin:2rem;max-width:70rem}
main{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(12rem,1fr))}
.node{border:1px solid #bbb;border-radius:.6rem;padding:1rem}.muted{color:#666}
form{display:flex;gap:.5rem;margin:2rem 0}input{flex:1;padding:.6rem}button{padding:.6rem}</style>
</head><body><h1>AEGIS Constellation</h1>
<p class="muted">Canonical state and conversation from AEGIS Core.</p>
<p id="health" aria-live="polite">Checking readiness…</p>
<form id="chat"><input id="utterance" autocomplete="off"
placeholder="Ask AEGIS..."><button>Send</button></form>
<p id="answer" aria-live="polite"></p><p id="detail" class="muted"></p>
<h2>Conversation</h2><ol id="conversation" aria-live="polite"></ol>
<main id="nodes"><p>Loading state…</p></main>
<h2>Relationships</h2><ul id="edges"><li>Loading relationships…</li></ul>
<script>
const nodes = document.getElementById('nodes');
const edges = document.getElementById('edges');
let pendingCorrelationId = null;
async function loadHealth() {
  const response = await fetch('/api/health'); const report = await response.json();
  const required = (report.components || []).filter(component => component.required);
  const ready = report.ready ? 'READY' : 'NOT READY';
  document.getElementById('health').textContent =
    `Runtime: ${ready} · ${required.filter(component => component.healthy).length}` +
    `/${required.length} required checks OK`;
}
async function loadState() {
  const response = await fetch('/api/constellation'); const state = await response.json();
  if (!response.ok) throw new Error(state.error || 'State is unavailable.');
  nodes.replaceChildren(...(state.nodes || []).map(node => {
    const card = document.createElement('button'); card.className = 'node'; card.type = 'button';
    const title = document.createElement('h2'); title.textContent = node.label;
    const detail = document.createElement('p'); detail.textContent = node.detail || '';
    card.addEventListener('click', () => {
      document.getElementById('detail').textContent =
        `${node.label}: ${node.detail || 'No detail'}`;
    });
    card.append(title, detail); return card;
  }));
  const labels = Object.fromEntries((state.nodes || []).map(node => [node.id, node.label]));
  edges.replaceChildren(...(state.edges || []).map(edge => {
    const item = document.createElement('li');
    item.textContent =
      `${labels[edge.source] || 'Authorized node'} → ${labels[edge.target] || 'Authorized node'}`;
    return item;
  }));
}
loadHealth().catch(() => {
  document.getElementById('health').textContent = 'Runtime status unavailable.';
});
loadState().catch(error => { nodes.textContent = error.message; });
document.getElementById('chat').addEventListener('submit', async event => {
  event.preventDefault(); const input = document.getElementById('utterance');
  const send = event.currentTarget.querySelector('button');
  const utterance = input.value.trim(); if (!utterance || send.disabled) return;
  const conversation = document.getElementById('conversation');
  const correlationId = pendingCorrelationId || crypto.randomUUID();
  if (!pendingCorrelationId) {
    const userLine = document.createElement('li'); userLine.textContent = `You: ${utterance}`;
    conversation.append(userLine);
  }
  send.disabled = true; input.disabled = true;
  document.getElementById('detail').textContent = 'Status: working';
  try {
    const response = await fetch('/api/message', {method:'POST',
      headers:{'content-type':'application/json'},
      body:JSON.stringify({utterance, correlation_id:correlationId})});
    const result = await response.json();
    const answer = result.message || result.error || 'No response';
    document.getElementById('answer').textContent = answer;
    const assistantLine = document.createElement('li');
    assistantLine.textContent = `AEGIS: ${answer}`; conversation.append(assistantLine);
    if (result.state) document.getElementById('detail').textContent = `Status: ${result.state}`;
    if (response.ok) {
      pendingCorrelationId = null; send.textContent = 'Send';
      if (result.state === 'completed') loadState().catch(() => {});
    } else {
      pendingCorrelationId = correlationId; send.textContent = 'Retry';
    }
  } catch (_) {
    document.getElementById('answer').textContent = 'AEGIS is unavailable.';
    document.getElementById('detail').textContent = 'Status: unavailable';
    pendingCorrelationId = correlationId; send.textContent = 'Retry';
  } finally {
    send.disabled = false; input.disabled = false;
  }
});
</script></body></html>"""


class BrowserApp:
    """HTTP presentation adapter; callbacks retain semantic and state ownership."""

    def __init__(
        self,
        principal: Principal | PrincipalProvider,
        interaction: Interaction,
        state: ConstellationState,
        health: HealthProvider | None = None,
    ) -> None:
        self.principal_provider = principal if callable(principal) else lambda: principal
        self.interaction = interaction
        self.state = state
        self.health = health

    def dispatch(self, method: str, path: str, body: bytes = b"") -> tuple[int, str, bytes]:
        route = urlparse(path).path
        if method == "GET" and route == "/":
            return HTTPStatus.OK, "text/html; charset=utf-8", _INDEX_HTML.encode()
        if method == "GET" and route == "/api/health":
            if self.health is None:
                return self._json(HTTPStatus.OK, {"healthy": True, "ready": True, "components": []})
            report = self.health()
            payload = report.model_dump(mode="json") if isinstance(report, HealthReport) else report
            return self._json(HTTPStatus.OK, payload)
        if route.startswith("/api/"):
            try:
                principal = self.principal_provider()
            except (OSError, RuntimeError, ValueError, PermissionError):
                return self._json(HTTPStatus.UNAUTHORIZED, {"error": "identity unavailable"})
        else:
            return self._json(HTTPStatus.NOT_FOUND, {"error": "route not found"})
        if method == "GET" and route == "/api/constellation":
            try:
                state = self.state(principal)
            except PermissionError:
                return self._json(HTTPStatus.FORBIDDEN, {"error": "state access denied"})
            except (OSError, RuntimeError):
                return self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "state unavailable"})
            return self._json(HTTPStatus.OK, state)
        if method == "POST" and route == "/api/message":
            if len(body) > _MAX_BODY_BYTES:
                return self._json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"}
                )
            try:
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                utterance = payload["utterance"]
                if not isinstance(utterance, str) or not utterance.strip():
                    raise ValueError("utterance must be a non-empty string")
                correlation_value = payload.get("correlation_id")
                if correlation_value is None:
                    correlation_id = uuid4()
                elif isinstance(correlation_value, str):
                    correlation_id = UUID(correlation_value)
                else:
                    raise ValueError("correlation_id must be a UUID string")
                message = self.interaction(utterance, principal, correlation_id)
            except (ValueError, KeyError, json.JSONDecodeError, TypeError) as exc:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except PermissionError:
                return self._json(HTTPStatus.FORBIDDEN, {"error": "request denied"})
            except (OSError, RuntimeError):
                return self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "request unavailable"})
            if isinstance(message, dict):
                message.setdefault("correlation_id", str(correlation_id))
                return self._json(HTTPStatus.OK, message)
            return self._json(
                HTTPStatus.OK,
                {"message": message, "correlation_id": str(correlation_id)},
            )
        return self._json(HTTPStatus.NOT_FOUND, {"error": "route not found"})

    @staticmethod
    def _json(status: HTTPStatus, payload: Any) -> tuple[int, str, bytes]:
        return status, "application/json", json.dumps(payload).encode()


def serve(
    host: str,
    port: int,
    principal: Principal | PrincipalProvider,
    interaction: Interaction,
    state: ConstellationState,
    health: HealthProvider | None = None,
) -> None:
    """Serve the proof using callbacks supplied by the Core/client composition root."""

    app = BrowserApp(principal, interaction, state, health)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._respond(app.dispatch("GET", self.path))

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("content-length", "0"))
            except ValueError:
                self._respond(
                    app._json(HTTPStatus.BAD_REQUEST, {"error": "invalid content length"})
                )
                return
            if length < 0:
                self._respond(
                    app._json(HTTPStatus.BAD_REQUEST, {"error": "invalid content length"})
                )
                return
            if length > _MAX_BODY_BYTES:
                self._respond(
                    app._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
                )
                return
            self._respond(app.dispatch("POST", self.path, self.rfile.read(length)))

        def _respond(self, response: tuple[int, str, bytes]) -> None:
            status, content_type, payload = response
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Pragma", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    ThreadingHTTPServer((host, port), Handler).serve_forever()
