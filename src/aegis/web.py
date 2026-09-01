"""Small browser adapter over the existing interaction and state boundaries."""

from __future__ import annotations

import json
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .contracts import Principal

Interaction = Callable[[str, Principal], str | dict[str, Any]]
ConstellationState = Callable[[Principal], dict[str, Any]]


_INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>AEGIS Constellation</title>
<style>body{font:16px system-ui;margin:2rem;max-width:70rem}
main{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(12rem,1fr))}
.node{border:1px solid #bbb;border-radius:.6rem;padding:1rem}.muted{color:#666}
form{display:flex;gap:.5rem;margin:2rem 0}input{flex:1;padding:.6rem}button{padding:.6rem}</style>
</head><body><h1>AEGIS Constellation</h1>
<p class="muted">Canonical state and conversation from AEGIS Core.</p>
<form id="chat"><input id="utterance" autocomplete="off"
placeholder="Ask AEGIS..."><button>Send</button></form>
<p id="answer" aria-live="polite"></p><p id="detail" class="muted"></p>
<main id="nodes"><p>Loading state…</p></main>
<h2>Relationships</h2><ul id="edges"><li>Loading relationships…</li></ul>
<script>
const nodes = document.getElementById('nodes');
const edges = document.getElementById('edges');
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
loadState().catch(error => { nodes.textContent = error.message; });
document.getElementById('chat').addEventListener('submit', async event => {
  event.preventDefault(); const input = document.getElementById('utterance');
  const response = await fetch('/api/message', {method:'POST',
    headers:{'content-type':'application/json'},
    body:JSON.stringify({utterance:input.value})});
  const result = await response.json();
  document.getElementById('answer').textContent = result.message || result.error;
  if (result.state) document.getElementById('detail').textContent = `Status: ${result.state}`;
  if (response.ok) {
    input.value = ''; if (result.state === 'completed') loadState().catch(() => {});
  }
});
</script></body></html>"""


class BrowserApp:
    """HTTP presentation adapter; callbacks retain semantic and state ownership."""

    def __init__(
        self, principal: Principal, interaction: Interaction, state: ConstellationState
    ) -> None:
        self.principal = principal
        self.interaction = interaction
        self.state = state

    def dispatch(self, method: str, path: str, body: bytes = b"") -> tuple[int, str, bytes]:
        route = urlparse(path).path
        if method == "GET" and route == "/":
            return HTTPStatus.OK, "text/html; charset=utf-8", _INDEX_HTML.encode()
        if method == "GET" and route == "/api/constellation":
            try:
                state = self.state(self.principal)
            except PermissionError:
                return self._json(HTTPStatus.FORBIDDEN, {"error": "state access denied"})
            except (OSError, RuntimeError):
                return self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "state unavailable"})
            return self._json(HTTPStatus.OK, state)
        if method == "POST" and route == "/api/message":
            try:
                payload = json.loads(body)
                utterance = payload["utterance"]
                if not isinstance(utterance, str) or not utterance.strip():
                    raise ValueError("utterance must be a non-empty string")
                message = self.interaction(utterance, self.principal)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except PermissionError:
                return self._json(HTTPStatus.FORBIDDEN, {"error": "request denied"})
            except (OSError, RuntimeError):
                return self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "request unavailable"})
            if isinstance(message, dict):
                return self._json(HTTPStatus.OK, message)
            return self._json(HTTPStatus.OK, {"message": message})
        return self._json(HTTPStatus.NOT_FOUND, {"error": "route not found"})

    @staticmethod
    def _json(status: HTTPStatus, payload: Any) -> tuple[int, str, bytes]:
        return status, "application/json", json.dumps(payload).encode()


def serve(
    host: str, port: int, principal: Principal, interaction: Interaction, state: ConstellationState
) -> None:
    """Serve the proof using callbacks supplied by the Core/client composition root."""

    app = BrowserApp(principal, interaction, state)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._respond(app.dispatch("GET", self.path))

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("content-length", "0"))
            self._respond(app.dispatch("POST", self.path, self.rfile.read(length)))

        def _respond(self, response: tuple[int, str, bytes]) -> None:
            status, content_type, payload = response
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    ThreadingHTTPServer((host, port), Handler).serve_forever()
