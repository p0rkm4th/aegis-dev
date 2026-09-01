"""Small browser adapter over the existing interaction and state boundaries."""

from __future__ import annotations

import json
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .contracts import Principal, RequestStatus
from .health import HealthReport

Interaction = Callable[[str, Principal, UUID], str | dict[str, Any]]
ConstellationState = Callable[[Principal], dict[str, Any]]
PrincipalProvider = Callable[[], Principal]
HealthProvider = Callable[[], HealthReport | dict[str, Any]]
RequestStatusProvider = Callable[[Principal, UUID], RequestStatus | dict[str, Any]]
_MAX_BODY_BYTES = 20_000
_RETRY_AFTER_SECONDS = 5


def _write_response_payload(writer: Any, payload: bytes) -> None:
    """Ignore a client disconnect after the response has been safely computed."""

    try:
        writer.write(payload)
    except (BrokenPipeError, ConnectionResetError):
        return


class BrowserMessage(BaseModel):
    """Stable presentation envelope for one interaction response."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    message: str
    state: str | None = None
    detail: str | None = None
    objective_id: UUID | None = None
    correlation_id: UUID
    retryable: bool | None = None


class ConstellationNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    detail: str = ""


class ConstellationEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)


class ConstellationProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    nodes: tuple[ConstellationNode, ...] = ()
    edges: tuple[ConstellationEdge, ...] = ()
    details: dict[str, Any] = {}


_INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>AEGIS Constellation</title>
<style>body{font:16px system-ui;margin:2rem;max-width:70rem}
main{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(12rem,1fr))}
.node{border:1px solid #bbb;border-radius:.6rem;padding:1rem}
.node[aria-pressed="true"]{border-color:#2457a6;box-shadow:0 0 0 .15rem #c9dcff}
.muted{color:#666}
form{display:flex;gap:.5rem;margin:2rem 0}input{flex:1;padding:.6rem}button{padding:.6rem}
#detail{border:1px solid #ddd;border-radius:.6rem;padding:1rem;min-height:2rem}
#detail dl{display:grid;grid-template-columns:minmax(8rem,14rem) 1fr;gap:.35rem .8rem}
#detail dt{font-weight:600}#detail dd{margin:0}
#detail ul{margin:.25rem 0;padding-left:1.25rem}</style>
</head><body><h1>AEGIS Constellation</h1>
<p class="muted">Canonical state and conversation from AEGIS Core.</p>
<p id="health" aria-live="polite">Checking readiness…</p>
<ul id="health-details" class="muted" aria-live="polite"></ul>
<form id="chat"><input id="utterance" autocomplete="off"
placeholder="Ask AEGIS..."><button>Send</button></form>
<p id="answer" aria-live="polite"></p><div id="detail" class="muted" role="region"
aria-live="polite" aria-label="Selected node details"></div>
<h2>Conversation</h2><ol id="conversation" aria-live="polite"></ol>
<p><button id="refresh" type="button">Refresh state</button></p>
<p id="state-status" class="muted" aria-live="polite"></p>
<main id="nodes"><p>Loading state…</p></main>
<h2>Relationships</h2><ul id="edges"><li>Loading relationships…</li></ul>
<script>
const nodes = document.getElementById('nodes');
const edges = document.getElementById('edges');
const refresh = document.getElementById('refresh');
const messageTimeoutMs = 120000;
const refreshRequestTimeoutMs = 10000;
const pendingStorageKey = 'aegis.pending-request';
const recoveryPollMs = 5000;
const recoveryRequestTimeoutMs = 10000;
const maxRecoveryPolls = 60;
let pendingCorrelationId = null;
let selectedNode = null;
let recoveryPollScheduled = false;
let recoveryPollAttempts = 0;
const retryableCodes = new Set([
  'identity_unavailable', 'state_unavailable', 'request_unavailable', 'request_timeout'
]);
function persistPendingRequest(utterance, correlationId) {
  try {
    sessionStorage.setItem(pendingStorageKey, JSON.stringify(
      {utterance, correlation_id: correlationId}));
  } catch (_) { /* session storage is optional; Core correlation remains authoritative. */ }
}
function clearPendingRequest() {
  try { sessionStorage.removeItem(pendingStorageKey); } catch (_) { /* optional storage */ }
}
function restorePendingRequest() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(pendingStorageKey) || 'null');
    if (!saved || typeof saved.utterance !== 'string' || !saved.utterance.trim() ||
        typeof saved.correlation_id !== 'string' ||
        !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
          .test(saved.correlation_id)) {
      clearPendingRequest(); return;
    }
    pendingCorrelationId = saved.correlation_id;
    document.getElementById('utterance').value = saved.utterance;
    document.getElementById('detail').textContent =
      'A previous request may still be in progress. Retry uses the same correlation.';
    document.querySelector('#chat button').textContent = 'Retry';
  } catch (_) { clearPendingRequest(); }
}
function scheduleRecoveryPoll() {
  if (recoveryPollScheduled || !pendingCorrelationId || recoveryPollAttempts >= maxRecoveryPolls) {
    if (pendingCorrelationId && recoveryPollAttempts >= maxRecoveryPolls) {
      document.getElementById('detail').textContent =
        'Status checks paused after five minutes. Retry remains explicit.';
    }
    return;
  }
  recoveryPollAttempts += 1;
  recoveryPollScheduled = true;
  setTimeout(() => {
    recoveryPollScheduled = false;
    recoverPendingRequest();
  }, recoveryPollMs);
}
async function recoverPendingRequest() {
  if (!pendingCorrelationId) return;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), recoveryRequestTimeoutMs);
  try {
    const response = await fetch(
      `/api/request-status?correlation_id=${pendingCorrelationId}`, {signal: controller.signal});
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) {
        clearAuthorizedDisplays();
        document.getElementById('state-status').textContent =
          'Authorization lost; authorized state cleared.';
      } else {
        scheduleRecoveryPoll();
      }
      return;
    }
    const status = await response.json();
    if (status.state === 'unknown') { scheduleRecoveryPoll(); return; }
    const inProgressStates = new Set([
      'proposed', 'validated', 'authorized', 'executing', 'observed'
    ]);
    if (inProgressStates.has(status.state)) {
      document.getElementById('detail').textContent =
        `Request status recovered: ${status.state}. Retry remains explicit.`;
      scheduleRecoveryPoll();
      return;
    }
    const recoveredCorrelationId = pendingCorrelationId;
    recoveryPollAttempts = 0;
    document.getElementById('answer').textContent = status.message || 'Request status recovered.';
    document.getElementById('detail').textContent = `Status: ${status.state}`;
    if (status.retryable === true) {
      document.querySelector('#chat button').textContent = 'Retry';
      persistPendingRequest(document.getElementById('utterance').value, recoveredCorrelationId);
    } else {
      pendingCorrelationId = null; clearPendingRequest();
      document.querySelector('#chat button').textContent = 'Send';
    }
    if (status.state === 'completed') refreshState();
  } catch (_) {
    scheduleRecoveryPoll();
  } finally {
    clearTimeout(timeout);
  }
}
async function loadHealth() {
  const response = await fetchWithTimeout('/api/health'); const report = await response.json();
  if (!response.ok) {
    const error = new Error(report.error || 'Runtime status unavailable.');
    error.code = report.code || 'health_unavailable'; throw error;
  }
  const required = (report.components || []).filter(component => component.required);
  const ready = report.ready ? 'READY' : 'NOT READY';
  document.getElementById('health').textContent =
    `Runtime: ${ready} · ${required.filter(component => component.healthy).length}` +
    `/${required.length} required checks OK`;
  const healthDetails = document.getElementById('health-details');
  healthDetails.replaceChildren(...(report.components || []).map(component => {
    const item = document.createElement('li');
    const requirement = component.required ? 'required' : 'optional';
    const status = component.healthy ? 'OK' : 'FAIL';
    item.textContent = `${component.name}: ${status} (${requirement}) — ${component.detail || ''}`;
    return item;
  }));
}
function renderDetailValue(value) {
  if (value === null) {
    const empty = document.createElement('span'); empty.textContent = '—'; return empty;
  }
  if (Array.isArray(value)) {
    const list = document.createElement('ul');
    if (!value.length) {
      const item = document.createElement('li'); item.textContent = 'None'; list.append(item);
    }
    value.forEach(item => {
      const row = document.createElement('li');
      row.append(renderDetailValue(item)); list.append(row);
    });
    return list;
  }
  if (typeof value === 'object') {
    const definition = document.createElement('dl');
    Object.entries(value).forEach(([key, item]) => {
      const term = document.createElement('dt'); term.textContent = key;
      const description = document.createElement('dd'); description.append(renderDetailValue(item));
      definition.append(term, description);
    });
    return definition;
  }
  const text = document.createElement('span'); text.textContent = String(value); return text;
}
function clearAuthorizedDisplays() {
  nodes.replaceChildren(); edges.replaceChildren();
  document.getElementById('detail').replaceChildren();
  document.getElementById('answer').textContent = '';
  document.getElementById('conversation').replaceChildren();
  pendingCorrelationId = null;
  recoveryPollAttempts = 0;
  clearPendingRequest();
  document.querySelector('#chat button').textContent = 'Send';
}
async function fetchWithTimeout(resource, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), refreshRequestTimeoutMs);
  try {
    return await fetch(resource, {...options, signal: controller.signal});
  } finally {
    clearTimeout(timeout);
  }
}
async function loadState() {
  const response = await fetchWithTimeout('/api/constellation');
  const state = await response.json();
  if (!response.ok) {
    const error = new Error(state.error || 'State is unavailable.');
    error.code = state.code || 'state_unavailable'; throw error;
  }
  document.getElementById('state-status').textContent = '';
  const details = state.details || {};
  selectedNode = null;
  document.getElementById('detail').replaceChildren();
  nodes.replaceChildren(...(state.nodes || []).map(node => {
    const card = document.createElement('button'); card.className = 'node'; card.type = 'button';
    card.setAttribute('aria-pressed', 'false');
    card.setAttribute('aria-label', `${node.label}: ${node.detail || 'No detail'}`);
    const title = document.createElement('h2'); title.textContent = node.label;
    const detail = document.createElement('p'); detail.textContent = node.detail || '';
    card.addEventListener('click', () => {
      if (selectedNode) selectedNode.setAttribute('aria-pressed', 'false');
      selectedNode = card; card.setAttribute('aria-pressed', 'true');
      const panel = document.getElementById('detail'); panel.replaceChildren();
      const heading = document.createElement('p');
      heading.textContent = `${node.label}: ${node.detail || 'No detail'}`;
      panel.append(heading);
      if (Object.prototype.hasOwnProperty.call(details, node.id)) {
        panel.append(renderDetailValue(details[node.id]));
      }
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
async function refreshState() {
  refresh.disabled = true;
  try {
    try {
      await loadHealth();
    } catch (error) {
      const code = error.code || 'health_unavailable';
      document.getElementById('health').textContent =
        `Runtime status unavailable (${code}).`;
    }
    try {
      await loadState();
    } catch (error) {
      const code = error.code || 'state_unavailable';
      document.getElementById('state-status').textContent =
        `State refresh failed (${code}). Use Refresh state to try again.`;
      if (code === 'identity_unavailable' || code === 'state_access_denied') {
        clearAuthorizedDisplays();
      }
    }
  } finally {
    refresh.disabled = false;
  }
}
refresh.addEventListener('click', refreshState);
refreshState().catch(() => {
  document.getElementById('health').textContent = 'Runtime status unavailable.';
});
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
  persistPendingRequest(utterance, correlationId);
  document.getElementById('detail').textContent = 'Status: working';
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), messageTimeoutMs);
  try {
    const response = await fetch('/api/message', {method:'POST',
      headers:{'content-type':'application/json'},
      body:JSON.stringify({utterance, correlation_id:correlationId}), signal:controller.signal});
    const result = await response.json();
    const answer = result.message || result.error || 'No response';
    document.getElementById('answer').textContent = answer;
    const assistantLine = document.createElement('li');
    assistantLine.textContent = `AEGIS: ${answer}`; conversation.append(assistantLine);
    if (result.state) document.getElementById('detail').textContent =
      `Status: ${result.state}${result.detail ? ` · ${result.detail}` : ''}`;
    else if (!response.ok) document.getElementById('detail').textContent =
      `Status: ${result.code || 'request_failed'}`;
    if (response.ok) {
      if (result.retryable === true) {
        pendingCorrelationId = correlationId; send.textContent = 'Retry';
        persistPendingRequest(utterance, correlationId);
      } else {
        pendingCorrelationId = null; send.textContent = 'Send';
        clearPendingRequest();
      }
      if (result.state === 'completed') refreshState();
    } else {
      const authorizationLost = result.code === 'identity_unavailable' ||
        result.code === 'state_access_denied';
      if (authorizationLost) {
        clearAuthorizedDisplays();
        document.getElementById('state-status').textContent =
          'Authorization lost; authorized state cleared.';
        send.textContent = 'Send';
      } else {
        pendingCorrelationId = retryableCodes.has(result.code) ? correlationId : null;
        if (!pendingCorrelationId) clearPendingRequest();
        send.textContent = pendingCorrelationId ? 'Retry' : 'Send';
      }
    }
  } catch (error) {
    const timedOut = error && error.name === 'AbortError';
    document.getElementById('answer').textContent = timedOut
      ? 'AEGIS did not respond in time. The outcome is unknown.'
      : 'AEGIS is unavailable.';
    document.getElementById('detail').textContent = timedOut
      ? 'Status: request_timeout · Retry uses the same correlation.'
      : 'Status: unavailable';
    pendingCorrelationId = correlationId; send.textContent = 'Retry';
    persistPendingRequest(utterance, correlationId);
  } finally {
    clearTimeout(timeout);
    send.disabled = false; input.disabled = false;
  }
});
restorePendingRequest();
recoverPendingRequest();
</script></body></html>"""


class BrowserApp:
    """HTTP presentation adapter; callbacks retain semantic and state ownership."""

    def __init__(
        self,
        principal: Principal | PrincipalProvider,
        interaction: Interaction,
        state: ConstellationState,
        health: HealthProvider | None = None,
        request_status: RequestStatusProvider | None = None,
    ) -> None:
        self.principal_provider = principal if callable(principal) else lambda: principal
        self.interaction = interaction
        self.state = state
        self.health = health
        self.request_status = request_status

    def dispatch(self, method: str, path: str, body: bytes = b"") -> tuple[int, str, bytes]:
        route = urlparse(path).path
        if method == "GET" and route == "/":
            return HTTPStatus.OK, "text/html; charset=utf-8", _INDEX_HTML.encode()
        if method == "GET" and route in {"/api/health", "/api/ready"}:
            if self.health is None:
                payload: dict[str, Any] = {"healthy": True, "ready": True, "components": []}
                return self._json(HTTPStatus.OK, payload)
            try:
                report = self.health()
            except Exception:
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "health_unavailable",
                    "runtime status unavailable",
                )
            try:
                payload = HealthReport.model_validate(report).model_dump(mode="json")
            except (TypeError, ValueError, ValidationError):
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "health_unavailable",
                    "runtime status unavailable",
                )
            if route == "/api/ready" and payload["ready"] is not True:
                return self._json(HTTPStatus.SERVICE_UNAVAILABLE, payload)
            return self._json(HTTPStatus.OK, payload)
        if route.startswith("/api/"):
            try:
                principal = self.principal_provider()
            except Exception:
                return self._error(
                    HTTPStatus.UNAUTHORIZED, "identity_unavailable", "identity unavailable"
                )
        else:
            return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
        if method == "GET" and route == "/api/constellation":
            try:
                state = ConstellationProjection.model_validate(self.state(principal))
                payload = state.model_dump(mode="json")
            except PermissionError:
                return self._error(
                    HTTPStatus.FORBIDDEN, "state_access_denied", "state access denied"
                )
            except Exception:
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "state_unavailable", "state unavailable"
                )
            return self._json(HTTPStatus.OK, payload)
        if method == "GET" and route == "/api/request-status":
            if self.request_status is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            query = parse_qs(urlparse(path).query, keep_blank_values=True)
            if set(query) != {"correlation_id"} or len(query["correlation_id"]) != 1:
                return self._error(HTTPStatus.BAD_REQUEST, "invalid_request", "invalid request")
            try:
                correlation_id = UUID(query["correlation_id"][0])
                status = self.request_status(principal, correlation_id)
                payload = (
                    status.model_dump(mode="json")
                    if isinstance(status, RequestStatus)
                    else RequestStatus.model_validate(status).model_dump(
                        mode="json", exclude_none=True
                    )
                )
            except (ValueError, TypeError, ValidationError):
                return self._error(HTTPStatus.BAD_REQUEST, "invalid_request", "invalid request")
            except PermissionError:
                return self._error(
                    HTTPStatus.FORBIDDEN, "state_access_denied", "state access denied"
                )
            except Exception:
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "state_unavailable", "state unavailable"
                )
            return self._json(HTTPStatus.OK, payload)
        if method == "POST" and route == "/api/message":
            if len(body) > _MAX_BODY_BYTES:
                return self._error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_too_large", "request too large"
                )
            try:
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                unknown_fields = set(payload) - {"utterance", "correlation_id"}
                if unknown_fields:
                    raise ValueError("request contains undocumented fields")
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
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._error(HTTPStatus.BAD_REQUEST, "invalid_request", "invalid request")
            except (ValueError, KeyError, TypeError) as exc:
                return self._error(HTTPStatus.BAD_REQUEST, "invalid_request", str(exc))
            try:
                message = self.interaction(utterance, principal, correlation_id)
            except PermissionError:
                return self._error(HTTPStatus.FORBIDDEN, "request_denied", "request denied")
            except Exception:
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "request_unavailable", "request unavailable"
                )
            try:
                response = BrowserMessage.model_validate(
                    {"message": message, "correlation_id": correlation_id}
                    if isinstance(message, str)
                    else {**message, "correlation_id": correlation_id}
                )
            except (TypeError, ValidationError):
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "request_unavailable",
                    "request unavailable",
                )
            return self._json(HTTPStatus.OK, response.model_dump(mode="json", exclude_none=True))
        return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")

    @staticmethod
    def _json(status: HTTPStatus, payload: Any) -> tuple[int, str, bytes]:
        try:
            serialized = json.dumps(payload)
        except (TypeError, ValueError):
            status = HTTPStatus.SERVICE_UNAVAILABLE
            serialized = json.dumps(
                {"code": "response_unavailable", "error": "response unavailable"}
            )
        return status, "application/json", serialized.encode()

    @classmethod
    def _error(cls, status: HTTPStatus, code: str, message: str) -> tuple[int, str, bytes]:
        return cls._json(status, {"code": code, "error": message})


def serve(
    host: str,
    port: int,
    principal: Principal | PrincipalProvider,
    interaction: Interaction,
    state: ConstellationState,
    health: HealthProvider | None = None,
    request_status: RequestStatusProvider | None = None,
) -> None:
    """Serve the proof using callbacks supplied by the Core/client composition root."""

    app = BrowserApp(principal, interaction, state, health, request_status)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._respond(app.dispatch("GET", self.path))

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("content-length", "0"))
            except ValueError:
                self._respond(
                    app._error(
                        HTTPStatus.BAD_REQUEST, "invalid_content_length", "invalid content length"
                    )
                )
                return
            if length < 0:
                self._respond(
                    app._error(
                        HTTPStatus.BAD_REQUEST, "invalid_content_length", "invalid content length"
                    )
                )
                return
            if length > _MAX_BODY_BYTES:
                self._respond(
                    app._error(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        "request_too_large",
                        "request too large",
                    )
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
            self.send_header("X-Frame-Options", "DENY")
            if status == HTTPStatus.SERVICE_UNAVAILABLE:
                self.send_header("Retry-After", str(_RETRY_AFTER_SECONDS))
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            )
            self.end_headers()
            _write_response_payload(self.wfile, payload)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return
    finally:
        server.server_close()
