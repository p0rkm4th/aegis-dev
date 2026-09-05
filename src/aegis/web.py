"""Small browser adapter over the existing interaction and state boundaries."""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .contracts import ObjectiveState, Principal, RequestStatus
from .health import HealthReport

Interaction = Callable[[str, Principal, UUID], str | dict[str, Any]]
ContextualInteraction = Callable[[str, Principal, UUID, UUID | None], str | dict[str, Any]]
ConstellationState = Callable[[Principal], dict[str, Any]]
PrincipalProvider = Callable[[], Principal]
HealthProvider = Callable[[], HealthReport | dict[str, Any]]
RequestStatusProvider = Callable[[Principal, UUID], RequestStatus | dict[str, Any]]
FeedbackRecorder = Callable[[Principal, UUID, str, str | None], None]
_MAX_BODY_BYTES = 20_000
_MAX_RESPONSE_BYTES = 1_000_000
_RETRY_AFTER_SECONDS = 5


def _write_response_payload(writer: Any, payload: bytes) -> None:
    """Ignore a client disconnect after the response has been safely computed."""

    try:
        writer.write(payload)
    except (BrokenPipeError, ConnectionResetError):
        return


class BrowserStep(BaseModel):
    """Bounded canonical step status for presentation clients."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    index: int = Field(ge=0)
    action_id: str = Field(min_length=1)
    state: ObjectiveState
    objective_id: UUID
    correlation_id: UUID
    message: str


class BrowserSource(BaseModel):
    """Bounded non-authoritative source metadata for research answers."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=500)
    url: str = Field(min_length=1, max_length=2_000)
    retrieved_at: str = Field(min_length=1, max_length=100)


class BrowserMessage(BaseModel):
    """Stable presentation envelope for one interaction response."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    message: str
    code: str | None = None
    session_id: UUID
    state: ObjectiveState | None = None
    detail: str | None = None
    objective_id: UUID | None = None
    correlation_id: UUID
    retryable: bool | None = None
    steps: tuple[BrowserStep, ...] | None = Field(default=None, max_length=5)
    sources: tuple[BrowserSource, ...] | None = Field(default=None, max_length=5)


class ConstellationNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    detail: str = ""
    category: str | None = Field(default=None, min_length=1)
    detail_view: str | None = Field(default=None, min_length=1)


class ConstellationEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)


class ConstellationProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    nodes: tuple[ConstellationNode, ...] = ()
    edges: tuple[ConstellationEdge, ...] = ()
    details: dict[str, Any] = {}

    @model_validator(mode="after")
    def validate_relationship_integrity(self) -> "ConstellationProjection":
        node_ids = {node.id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("Constellation node IDs must be unique")
        if any(edge.source not in node_ids or edge.target not in node_ids for edge in self.edges):
            raise ValueError("Constellation edge references an unknown node")
        return self


_INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta name="aegis-session-token" content="__AEGIS_SESSION_TOKEN__"><title>AEGIS · Personal intelligence</title>
<style>
:root{color-scheme:dark;--bg:#0e1117;--panel:#171c25;--panel-raised:#1d2430;--border:#2b3442;--text:#edf2f7;--muted:#9aa8b8;--accent:#8dc7ff;--shadow:0 1rem 3rem #0005}
:root[data-theme="light"]{color-scheme:light;--bg:#f4f6f8;--panel:#fff;--panel-raised:#f8fafc;--border:#d7dee7;--text:#18212b;--muted:#536273;--accent:#155ea8;--shadow:0 .75rem 2rem #18212b18}
*{box-sizing:border-box}body{font:16px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:var(--bg);color:var(--text)}
.app-shell{max-width:76rem;margin:auto;padding:1rem clamp(1rem,4vw,3rem) 4rem}.topbar{display:flex;align-items:center;justify-content:space-between;padding:.5rem 0 2rem}.brand{display:flex;gap:.75rem;align-items:center}.brand-mark{display:grid;place-items:center;width:2.25rem;height:2.25rem;border:1px solid var(--border);border-radius:.75rem;color:var(--accent);background:var(--panel);font-weight:700}.brand h1{font-size:1.25rem;letter-spacing:.03em;margin:0}.eyebrow{font-size:.72rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase}.workspace{max-width:54rem;margin:auto}.conversation-panel{background:var(--panel);border:1px solid var(--border);border-radius:1.25rem;padding:clamp(1rem,3vw,2rem);box-shadow:var(--shadow)}
.intro{padding:.5rem 0 1.25rem}.intro h2{font-size:clamp(1.65rem,4vw,2.35rem);letter-spacing:-.03em;line-height:1.1;margin:0 0 .55rem}.intro p{max-width:38rem;color:var(--muted);margin:0}.muted{color:var(--muted)}.health-line{display:flex;align-items:center;justify-content:space-between;gap:1rem;margin:0 0 1rem;color:var(--muted);font-size:.9rem}.health-line strong{color:var(--text);font-weight:600}.status-dot{display:inline-block;width:.5rem;height:.5rem;border-radius:50%;background:#6fd18a;margin-right:.45rem}details summary{cursor:pointer;color:var(--muted);font-size:.85rem}#health-details{margin:.5rem 0 0;padding-left:1.2rem;font-size:.82rem}#health-details:empty{display:none}
#chat{display:flex;align-items:flex-end;gap:.7rem;margin:1rem 0 1.5rem}#utterance{flex:1;min-width:0;min-height:3.25rem;max-height:11rem;resize:none;overflow-y:hidden;padding:.85rem 1rem;border:1px solid var(--border);border-radius:.8rem;background:var(--bg);color:var(--text);font:inherit;line-height:1.45}#composer-hint{font-size:.78rem;margin:-1rem 0 1rem;color:var(--muted)}button{padding:.75rem 1rem;border:1px solid var(--border);border-radius:.7rem;background:var(--panel-raised);color:var(--text);font:inherit;cursor:pointer}button:hover{border-color:var(--accent)}button:focus-visible,input:focus-visible,textarea:focus-visible{outline:3px solid color-mix(in srgb,var(--accent) 55%,transparent);outline-offset:2px}button:disabled,input:disabled,textarea:disabled{cursor:wait;opacity:.65}
#answer{margin:0}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}#activity,#step-status{font-size:.85rem;margin:.35rem 0}.research-sources{font-size:.8rem;color:var(--muted);margin:.4rem 0 0}#conversation{display:flex;flex-direction:column;gap:.8rem;list-style:none;max-height:min(60vh,42rem);overflow-y:auto;padding:.25rem .35rem .5rem 0;margin:1.5rem 0 0;scroll-behavior:smooth}#conversation li{max-width:88%;padding:.7rem .9rem;border-radius:.85rem;white-space:pre-wrap;overflow-wrap:anywhere}#conversation li.owner-message{align-self:flex-end;background:color-mix(in srgb,var(--accent) 16%,var(--panel))}#conversation li.aegis-message{align-self:flex-start;background:var(--panel-raised)}#conversation li.conversation-empty{max-width:none;color:var(--muted);text-align:center;border:1px dashed var(--border);background:transparent}
.secondary{margin-top:1.5rem;border-top:1px solid var(--border);padding-top:1rem}.secondary>summary{font-weight:600;color:var(--muted);padding:.35rem 0}.state-tools{display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap;margin-top:1rem}.secondary h2{font-size:.9rem;color:var(--muted);font-weight:600}#detail{border:1px solid var(--border);border-radius:.8rem;padding:1rem;min-height:2rem;background:var(--panel-raised)}#detail dl{display:grid;grid-template-columns:minmax(8rem,14rem) 1fr;gap:.35rem .8rem}#detail dt{font-weight:600}#detail dd{margin:0}#nodes{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(12rem,1fr))}.node{text-align:left;width:100%}.node[aria-pressed="true"]{border-color:var(--accent);box-shadow:0 0 0 .15rem color-mix(in srgb,var(--accent) 25%,transparent)}
@media(max-width:36rem){#chat{align-items:stretch;flex-direction:column}#chat button{width:100%}#conversation li{max-width:100%}}
</style>
<style>.status-badge{display:inline-flex;align-items:center;gap:.4rem;width:max-content;margin:.2rem 0 .35rem;padding:.3rem .65rem;border:1px solid var(--border);border-radius:999px;color:var(--muted);font-size:.78rem;font-weight:600}.status-badge[data-state="completed"]{border-color:#4f9b68;color:#9be2ae}.status-badge[data-state="blocked"],.status-badge[data-state="failed"]{border-color:#a56a6a;color:#f0b0b0}</style>
</head><body><div class="app-shell"><header class="topbar"><div class="brand"><span class="brand-mark" aria-hidden="true">A</span><div><div class="eyebrow">Personal intelligence</div><h1>AEGIS</h1></div></div><button id="theme-toggle" type="button" aria-label="Switch color theme">Light mode</button></header>
<div class="workspace"><section class="conversation-panel" aria-label="Conversation with AEGIS"><div class="health-line"><span><span class="status-dot" aria-hidden="true"></span><strong id="health" aria-live="polite">Checking readiness…</strong></span><details><summary>Runtime details</summary><ul id="health-details" class="muted" aria-live="polite"></ul></details></div><div class="intro"><h2>What can I help you with?</h2><p>Ask naturally. I’ll keep track of your authorized information and tell you clearly what happened.</p></div>
<form id="chat"><label class="sr-only" for="utterance">Message AEGIS</label><textarea id="utterance" rows="2" autocomplete="off"
placeholder="Talk to AEGIS…" aria-describedby="composer-hint"></textarea><button>Send</button></form>
<p id="composer-hint" class="muted">Enter to send · Shift+Enter for a new line</p>
<p id="answer" class="sr-only" aria-live="polite"></p><p id="status-badge" class="status-badge" data-state="idle" aria-live="polite">Ready</p><p id="step-status" class="muted"
aria-live="polite"></p><div id="detail" class="muted" role="region"
aria-live="polite" aria-label="Selected node details"></div>
<p id="feedback" hidden>Was this useful?
<button type="button" data-feedback="helpful">Helpful</button>
<button type="button" data-feedback="not_helpful">Not helpful</button>
<span id="feedback-status" class="muted" aria-live="polite"></span></p>
<p id="activity" class="muted" aria-live="polite" aria-atomic="true"></p>
<p id="research-sources" class="research-sources" aria-live="polite"></p>
<h2 class="sr-only">Conversation</h2><ol id="conversation" role="log" aria-live="polite" aria-relevant="additions text"><li class="conversation-empty">Your conversation will appear here.</li></ol></section>
<details class="secondary" aria-label="Canonical state"><summary>Canonical state</summary><div class="state-tools"><h2 class="sr-only">Canonical state</h2><button id="refresh" type="button">Refresh state</button></div>
<p id="state-status" class="muted" aria-live="polite"></p>
<label for="node-filter">Find a node <input id="node-filter" type="search"
autocomplete="off" aria-describedby="node-filter-status"
placeholder="Filter authorized nodes"></label>
<p id="node-filter-status" class="muted" aria-live="polite" aria-atomic="true"></p>
<main id="nodes"><p>Loading state…</p></main>
<h2>Relationships</h2><ul id="edges"><li>Loading relationships…</li></ul></details></div></div>
<script>
const nodes = document.getElementById('nodes');
const edges = document.getElementById('edges');
const refresh = document.getElementById('refresh');
const nodeFilter = document.getElementById('node-filter');
const nodeFilterStatus = document.getElementById('node-filter-status');
const messageTimeoutMs = 120000;
const refreshRequestTimeoutMs = 10000;
const pendingStorageKey = 'aegis.pending-request';
const sessionStorageKey = 'aegis.session-id';
const contextStorageKey = 'aegis.context-correlation';
const themeStorageKey = 'aegis.theme';
const recoveryPollMs = 5000;
const recoveryRequestTimeoutMs = 10000;
const maxRecoveryPolls = 60;
let pendingCorrelationId = null;
let conversationSessionId = null;
let conversationContextCorrelationId = null;
let selectedNode = null;
let renderedNodeCards = new Map();
let renderedNodeText = new Map();
let renderedEdgeRows = [];
let authorizedProjectionLoaded = false;
let recoveryPollScheduled = false;
let recoveryPollAttempts = 0;
const themeToggle = document.getElementById('theme-toggle');
function applyTheme(theme) {
  const selected = theme === 'light' ? 'light' : 'dark';
  document.documentElement.dataset.theme = selected;
  themeToggle.textContent = selected === 'dark' ? 'Light mode' : 'Dark mode';
  themeToggle.setAttribute('aria-label', `Switch to ${selected === 'dark' ? 'light' : 'dark'} mode`);
  try { localStorage.setItem(themeStorageKey, selected); } catch (_) { /* optional */ }
}
let initialTheme = 'dark';
try {
  const savedTheme = localStorage.getItem(themeStorageKey);
  initialTheme = savedTheme || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
} catch (_) { /* optional */ }
applyTheme(initialTheme);
themeToggle.addEventListener('click', () =>
  applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
try {
  const savedSession = sessionStorage.getItem(sessionStorageKey);
  if (savedSession &&
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(savedSession))
    conversationSessionId = savedSession;
  else {
    conversationSessionId = crypto.randomUUID();
    sessionStorage.setItem(sessionStorageKey, conversationSessionId);
  }
} catch (_) { conversationSessionId = crypto.randomUUID(); }
try {
  const savedContext = localStorage.getItem(contextStorageKey);
  if (savedContext &&
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(savedContext))
    conversationContextCorrelationId = savedContext;
} catch (_) { /* local storage is optional */ }
function persistConversationContext(correlationId) {
  conversationContextCorrelationId = correlationId;
  try { localStorage.setItem(contextStorageKey, correlationId); } catch (_) { /* optional */ }
}
function clearConversationContext() {
  conversationContextCorrelationId = null;
  try { localStorage.removeItem(contextStorageKey); } catch (_) { /* optional */ }
}
const retryableCodes = new Set([
  'identity_unavailable', 'state_unavailable', 'request_unavailable', 'request_timeout'
]);
const errorLabels = Object.freeze({
  identity_unavailable: 'Identity unavailable', state_access_denied: 'Authorization denied',
  state_unavailable: 'State unavailable', health_unavailable: 'Runtime status unavailable',
  request_denied: 'Request denied', request_unavailable: 'Request unavailable',
  response_unavailable: 'Response unavailable', request_timeout: 'Request timed out',
  invalid_request: 'Invalid request', request_too_large: 'Request too large'
});
function errorLabel(code) { return errorLabels[code] || 'Request failed'; }
const lifecycleLabels = Object.freeze({
  proposed: 'Proposed', validated: 'Validated', authorized: 'Authorized',
  approval_required: 'Approval required', executing: 'Executing', observed: 'Observed',
  verified: 'Verified', completed: 'Completed', failed: 'Failed', blocked: 'Blocked',
  unknown: 'Outcome unknown'
});
function lifecycleLabel(state) { return lifecycleLabels[state] || state; }
function setOutcomeStatus(state) {
  const badge = document.getElementById('status-badge');
  const normalized = state || 'idle';
  badge.dataset.state = normalized;
  badge.textContent = normalized === 'idle' ? 'Ready' : lifecycleLabel(normalized);
}
function persistPendingRequest(utterance, correlationId) {
  try {
    sessionStorage.setItem(pendingStorageKey, JSON.stringify(
    {utterance, correlation_id: correlationId, session_id: conversationSessionId}));
  } catch (_) { /* session storage is optional; Core correlation remains authoritative. */ }
}
function clearPendingRequest() {
  try { sessionStorage.removeItem(pendingStorageKey); } catch (_) { /* optional storage */ }
}
function resizeComposer() {
  const input = document.getElementById('utterance');
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 176)}px`;
  input.style.overflowY = input.scrollHeight > 176 ? 'auto' : 'hidden';
}
function appendConversationMessage(kind, text) {
  const conversation = document.getElementById('conversation');
  conversation.querySelector('.conversation-empty')?.remove();
  const wasNearBottom = conversation.scrollHeight - conversation.scrollTop - conversation.clientHeight < 80;
  const line = document.createElement('li');
  line.className = kind;
  line.textContent = text;
  conversation.append(line);
  if (wasNearBottom || kind === 'aegis-message')
    line.scrollIntoView({block: 'nearest', behavior: 'smooth'});
}
const composer = document.getElementById('utterance');
composer.addEventListener('input', resizeComposer);
composer.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    if (!document.querySelector('#chat button').disabled)
      document.getElementById('chat').requestSubmit();
  }
});
resizeComposer();
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
    document.getElementById('activity').textContent =
      'A previous request may still be in progress. Retry uses the same correlation.';
    document.querySelector('#chat button').textContent = 'Retry';
  } catch (_) { clearPendingRequest(); }
  resizeComposer();
}
function scheduleRecoveryPoll() {
  if (recoveryPollScheduled || !pendingCorrelationId || recoveryPollAttempts >= maxRecoveryPolls) {
    if (pendingCorrelationId && recoveryPollAttempts >= maxRecoveryPolls) {
      document.getElementById('activity').textContent =
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
    const response = await apiFetch(
      `/api/request-status?correlation_id=${pendingCorrelationId}`, {signal: controller.signal});
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) {
        clearAuthorizedDisplays();
        document.getElementById('state-status').textContent =
          'Authorization lost; authorized state cleared.';
      } else {
        document.getElementById('activity').textContent =
          'Status check unavailable; retry remains explicit.';
        scheduleRecoveryPoll();
      }
      return;
    }
    const status = await response.json();
    if (status.state === 'unknown') {
      document.getElementById('activity').textContent =
        'Outcome unknown; checking canonical status. Retry remains explicit.';
      scheduleRecoveryPoll(); return;
    }
    const inProgressStates = new Set([
      'proposed', 'validated', 'authorized', 'executing', 'observed'
    ]);
    if (inProgressStates.has(status.state)) {
      document.getElementById('activity').textContent =
        `Request status recovered: ${lifecycleLabel(status.state)}. Retry remains explicit.`;
      scheduleRecoveryPoll();
      return;
    }
    const recoveredCorrelationId = pendingCorrelationId;
    recoveryPollAttempts = 0;
    document.getElementById('answer').textContent = status.message || 'Request status recovered.';
    document.getElementById('activity').textContent = `Status: ${lifecycleLabel(status.state)}`;
    if (status.retryable === true) {
      document.querySelector('#chat button').textContent = 'Retry';
      persistPendingRequest(document.getElementById('utterance').value, recoveredCorrelationId);
    } else {
      pendingCorrelationId = null; clearPendingRequest();
      document.querySelector('#chat button').textContent = 'Send';
    }
    if (status.state === 'completed') refreshState();
  } catch (_) {
    document.getElementById('activity').textContent =
      'Status check unavailable; retry remains explicit.';
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
      const item = document.createElement('li');
      item.textContent = 'No canonical records available';
      list.append(item);
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
      const term = document.createElement('dt');
      const label = key.replaceAll('_', ' ');
      term.textContent = label.replace(/\\b\\w/g, character => character.toUpperCase());
      const description = document.createElement('dd'); description.append(renderDetailValue(item));
      definition.append(term, description);
    });
    return definition;
  }
  const text = document.createElement('span'); text.textContent = String(value); return text;
}
function clearAuthorizedDisplays() {
  nodes.replaceChildren(); edges.replaceChildren();
  renderedNodeCards = new Map();
  renderedNodeText = new Map();
  renderedEdgeRows = [];
  authorizedProjectionLoaded = false;
  nodeFilter.value = '';
  document.getElementById('detail').replaceChildren();
  document.getElementById('activity').textContent = '';
  document.getElementById('step-status').textContent = '';
  nodeFilterStatus.textContent = 'Authorized nodes unavailable.';
  document.getElementById('answer').textContent = '';
  document.getElementById('conversation').replaceChildren();
  const empty = document.createElement('li');
  empty.className = 'conversation-empty';
  empty.textContent = 'Your conversation will appear here.';
  document.getElementById('conversation').append(empty);
  clearConversationContext();
  pendingCorrelationId = null;
  recoveryPollAttempts = 0;
  clearPendingRequest();
  document.querySelector('#chat button').textContent = 'Send';
}
function clearHealthDetails() {
  document.getElementById('health-details').replaceChildren();
}
function applyNodeFilter() {
  if (!authorizedProjectionLoaded) {
    nodeFilterStatus.textContent = 'Authorized nodes unavailable.';
    return;
  }
  const query = nodeFilter.value.trim().toLowerCase();
  let visibleCount = 0;
  renderedNodeCards.forEach((card, nodeId) => {
    card.hidden = Boolean(query && !renderedNodeText.get(nodeId).includes(query));
    if (!card.hidden) visibleCount += 1;
  });
  renderedEdgeRows.forEach(({item, edge}) => {
    const sourceMatches = !query || renderedNodeText.get(edge.source).includes(query);
    const targetMatches = !query || renderedNodeText.get(edge.target).includes(query);
    item.hidden = Boolean(query && !sourceMatches && !targetMatches);
  });
  nodeFilterStatus.textContent = query
    ? (visibleCount
      ? `Showing ${visibleCount} of ${renderedNodeCards.size} authorized nodes.`
      : `No authorized nodes match “${nodeFilter.value.trim()}”.`)
    : `${renderedNodeCards.size} authorized nodes.`;
}
nodeFilter.addEventListener('input', applyNodeFilter);
async function apiFetch(resource, options = {}) {
  const token = document.querySelector('meta[name="aegis-session-token"]')?.content;
  const headers = new Headers(options.headers || {});
  if (token) headers.set('x-aegis-session', token);
  const response = await fetch(resource, {...options, headers});
  if (response.status !== 401 || resource === '/') return response;
  // A normal owner-service token rotation should not discard the persistent
  // conversation. Refresh the transport token once, then let the endpoint's
  // normal authorization response decide whether the request is allowed.
  const root = await fetch('/');
  if (!root.ok) return response;
  const html = await root.text();
  const refreshed = html.match(/<meta name="aegis-session-token" content="([^"]*)"/);
  if (!refreshed || !refreshed[1]) return response;
  const meta = document.querySelector('meta[name="aegis-session-token"]');
  if (meta) meta.content = refreshed[1];
  const retryHeaders = new Headers(options.headers || {});
  retryHeaders.set('x-aegis-session', refreshed[1]);
  return fetch(resource, {...options, headers: retryHeaders});
}
async function fetchWithTimeout(resource, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), refreshRequestTimeoutMs);
  try {
    return await apiFetch(resource, {...options, signal: controller.signal});
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
  nodeFilterStatus.textContent = '';
  const details = state.details || {};
  selectedNode = null;
  document.getElementById('detail').replaceChildren();
  const nodeCards = new Map();
  const selectNode = (node, card) => {
    if (selectedNode) selectedNode.setAttribute('aria-pressed', 'false');
    selectedNode = card; card.setAttribute('aria-pressed', 'true');
    const panel = document.getElementById('detail'); panel.replaceChildren();
    const heading = document.createElement('p');
    heading.textContent = `${node.label}: ${node.detail || 'No detail'}`;
    panel.append(heading);
    if (Object.prototype.hasOwnProperty.call(details, node.id)) {
      panel.append(renderDetailValue(details[node.id]));
    }
  };
  renderedNodeCards = new Map();
  renderedNodeText = new Map();
  renderedEdgeRows = [];
  nodes.replaceChildren(...(state.nodes || []).map(node => {
    const card = document.createElement('button'); card.className = 'node'; card.type = 'button';
    card.setAttribute('aria-pressed', 'false');
    card.setAttribute('aria-label', `${node.label}: ${node.detail || 'No detail'}`);
    const title = document.createElement('h2'); title.textContent = node.label;
    const detail = document.createElement('p'); detail.textContent = node.detail || '';
    card.addEventListener('click', () => selectNode(node, card));
    nodeCards.set(node.id, card);
    renderedNodeText.set(node.id, `${node.label} ${node.detail || ''}`.toLowerCase());
    card.append(title, detail); return card;
  }));
  renderedNodeCards = nodeCards;
  authorizedProjectionLoaded = true;
  const labels = Object.fromEntries((state.nodes || []).map(node => [node.id, node.label]));
  edges.replaceChildren(...(state.edges || []).map(edge => {
    const item = document.createElement('li');
    const link = document.createElement('button'); link.type = 'button';
    link.textContent =
      `${labels[edge.source] || 'Authorized node'} → ${labels[edge.target] || 'Authorized node'}`;
    link.setAttribute(
      'aria-label', `Open relationship to ${labels[edge.target] || 'authorized node'}`);
    link.addEventListener('click', () => {
      const target = nodeCards.get(edge.target);
      if (target) { target.focus(); target.click(); }
    });
    item.append(link);
    renderedEdgeRows.push({item, edge});
    return item;
  }));
  applyNodeFilter();
}
async function refreshState() {
  refresh.disabled = true;
  nodes.setAttribute('aria-busy', 'true');
  try {
    try {
      await loadHealth();
    } catch (error) {
      const code = error.code || 'health_unavailable';
      clearHealthDetails();
      document.getElementById('health').textContent =
        `${errorLabel(code)} (${code}).`;
    }
    try {
      await loadState();
    } catch (error) {
      const code = error.code || 'state_unavailable';
      document.getElementById('state-status').textContent =
        `State refresh failed — ${errorLabel(code)} (${code}). Use Refresh state to try again.`;
      if (code === 'identity_unavailable' || code === 'state_access_denied') {
        clearAuthorizedDisplays();
      }
    }
  } finally {
    nodes.setAttribute('aria-busy', 'false');
    refresh.disabled = false;
  }
}
refresh.addEventListener('click', refreshState);
refreshState().catch(() => {
  clearHealthDetails();
  document.getElementById('health').textContent = 'Runtime status unavailable.';
});
document.getElementById('chat').addEventListener('submit', async event => {
  event.preventDefault(); const form = event.currentTarget;
  const input = document.getElementById('utterance');
  const send = form.querySelector('button');
  const utterance = input.value.trim(); if (!utterance || send.disabled) return;
  const correlationId = pendingCorrelationId || crypto.randomUUID();
  if (!pendingCorrelationId) appendConversationMessage('owner-message', `You: ${utterance}`);
  send.disabled = true; input.disabled = true;
  form.setAttribute('aria-busy', 'true');
  persistPendingRequest(utterance, correlationId);
  document.getElementById('activity').textContent = 'Status: working';
  setOutcomeStatus('executing');
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), messageTimeoutMs);
  try {
    const requestBody = {utterance, correlation_id:correlationId, session_id:conversationSessionId};
    if (!pendingCorrelationId && conversationContextCorrelationId)
      requestBody.context_correlation_id = conversationContextCorrelationId;
    const response = await apiFetch('/api/message', {method:'POST',
      headers:{'content-type':'application/json'},
      body:JSON.stringify(requestBody), signal:controller.signal});
    const result = await response.json();
    const answer = result.message || result.error || 'No response';
    document.getElementById('answer').textContent = answer;
    appendConversationMessage('aegis-message', `AEGIS: ${answer}`);
    const sources = document.getElementById('research-sources');
    sources.textContent = result.sources && result.sources.length
      ? `Sources · ${result.sources.map(source => `${source.title} (${source.url})`).join(' · ')}` : '';
    if (result.steps && result.steps.length) document.getElementById('step-status').textContent =
      result.steps.map(step =>
        `${step.action_id}: ${lifecycleLabel(step.state)} · ${step.message}`).join(' | ');
    else document.getElementById('step-status').textContent = '';
    if (result.state) {
      setOutcomeStatus(result.state);
      document.getElementById('activity').textContent =
      `Status: ${lifecycleLabel(result.state)}${result.detail ? ` · ${result.detail}` : ''}`;
    } else if (!response.ok) {
      setOutcomeStatus('failed');
      document.getElementById('activity').textContent =
      `Status: ${errorLabel(result.code)} (${result.code || 'request_failed'})`;
    }
    if (response.ok) {
      const feedback = document.getElementById('feedback');
      feedback.hidden = !result.correlation_id;
      feedback.dataset.correlationId = result.correlation_id || '';
      document.getElementById('feedback-status').textContent = '';
      if (result.retryable === true) {
        pendingCorrelationId = correlationId; send.textContent = 'Retry';
        persistPendingRequest(utterance, correlationId);
      } else {
        pendingCorrelationId = null; send.textContent = 'Send';
        clearPendingRequest();
        if (result.state === 'completed' && result.correlation_id)
          persistConversationContext(result.correlation_id);
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
    setOutcomeStatus('unknown');
    document.getElementById('activity').textContent = timedOut
      ? 'Status: request_timeout · Retry uses the same correlation.'
      : 'Status: unavailable';
    pendingCorrelationId = correlationId; send.textContent = 'Retry';
    persistPendingRequest(utterance, correlationId);
  } finally {
    clearTimeout(timeout);
    form.setAttribute('aria-busy', 'false');
    send.disabled = false; input.disabled = false;
  }
});
document.querySelectorAll('[data-feedback]').forEach(button =>
  button.addEventListener('click', async event => {
  const feedback = document.getElementById('feedback');
  const correlationId = feedback.dataset.correlationId;
  if (!correlationId) return;
  for (const item of document.querySelectorAll('[data-feedback]')) item.disabled = true;
  try {
    const response = await apiFetch('/api/feedback', {method:'POST',
      headers:{'content-type':'application/json'}, body:JSON.stringify({
        correlation_id:correlationId, outcome:event.currentTarget.dataset.feedback})});
    const result = await response.json();
    document.getElementById('feedback-status').textContent = response.ok ? 'Recorded.' :
      (result.error || 'Feedback unavailable.');
  } catch (_) {
    document.getElementById('feedback-status').textContent = 'Feedback unavailable.';
  }
}));
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
        contextual_interaction: ContextualInteraction | None = None,
        feedback: FeedbackRecorder | None = None,
        session_token: str | None = None,
    ) -> None:
        self.principal_provider = principal if callable(principal) else lambda: principal
        self.interaction = interaction
        self.state = state
        self.health = health
        self.request_status = request_status
        self.contextual_interaction = contextual_interaction
        self.feedback = feedback
        self.session_token = session_token

    def dispatch(
        self,
        method: str,
        path: str,
        body: bytes = b"",
        headers: Mapping[str, str] | None = None,
    ) -> tuple[int, str, bytes]:
        route = urlparse(path).path
        if method == "GET" and route == "/":
            html = _INDEX_HTML.replace("__AEGIS_SESSION_TOKEN__", self.session_token or "")
            return HTTPStatus.OK, "text/html; charset=utf-8", html.encode()
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
            session_header = (
                next(
                    (
                        value
                        for key, value in headers.items()
                        if key.casefold() == "x-aegis-session"
                    ),
                    None,
                )
                if headers is not None
                else None
            )
            if self.session_token is not None and (session_header != self.session_token):
                return self._error(
                    HTTPStatus.UNAUTHORIZED, "identity_unavailable", "identity unavailable"
                )
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
            except (ValueError, TypeError):
                return self._error(HTTPStatus.BAD_REQUEST, "invalid_request", "invalid request")
            try:
                status = self.request_status(principal, correlation_id)
            except PermissionError:
                return self._error(
                    HTTPStatus.FORBIDDEN, "state_access_denied", "state access denied"
                )
            except Exception:
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "state_unavailable", "state unavailable"
                )
            try:
                payload = (
                    status.model_dump(mode="json")
                    if isinstance(status, RequestStatus)
                    else RequestStatus.model_validate(status).model_dump(
                        mode="json", exclude_none=True
                    )
                )
            except (TypeError, ValidationError):
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "state_unavailable", "state unavailable"
                )
            except Exception:
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "state_unavailable", "state unavailable"
                )
            return self._json(HTTPStatus.OK, payload)
        if method == "POST" and route == "/api/feedback":
            if self.feedback is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            if len(body) > _MAX_BODY_BYTES:
                return self._error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_too_large", "request too large"
                )
            try:
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                if set(payload) - {"correlation_id", "outcome", "reason"}:
                    raise ValueError("request contains undocumented fields")
                correlation_id = UUID(str(payload["correlation_id"]))
                outcome = payload["outcome"]
                reason = payload.get("reason")
                if outcome not in {"helpful", "not_helpful"}:
                    raise ValueError("invalid outcome")
                if reason is not None and reason not in {
                    "objective_failed",
                    "incorrect",
                    "unclear",
                    "other",
                }:
                    raise ValueError("invalid reason")
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError, KeyError):
                return self._error(HTTPStatus.BAD_REQUEST, "invalid_request", "invalid request")
            try:
                self.feedback(principal, correlation_id, outcome, reason)
            except PermissionError:
                return self._error(HTTPStatus.FORBIDDEN, "feedback_denied", "feedback denied")
            except Exception:
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "feedback_unavailable", "feedback unavailable"
                )
            return self._json(
                HTTPStatus.OK, {"recorded": True, "correlation_id": str(correlation_id)}
            )
        if method == "POST" and route == "/api/message":
            if len(body) > _MAX_BODY_BYTES:
                return self._error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_too_large", "request too large"
                )
            try:
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                unknown_fields = set(payload) - {
                    "utterance",
                    "correlation_id",
                    "context_correlation_id",
                    "session_id",
                }
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
                session_value = payload.get("session_id")
                if session_value is None:
                    session_id = uuid4()
                elif isinstance(session_value, str):
                    session_id = UUID(session_value)
                else:
                    raise ValueError("session_id must be a UUID string")
                context_value = payload.get("context_correlation_id")
                if context_value is None:
                    context_correlation_id = None
                elif isinstance(context_value, str):
                    context_correlation_id = UUID(context_value)
                else:
                    raise ValueError("context_correlation_id must be a UUID string")
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._error(HTTPStatus.BAD_REQUEST, "invalid_request", "invalid request")
            except (ValueError, KeyError, TypeError) as exc:
                # Keep the one useful user-facing validation hint, but never
                # expose parser/implementation details such as UUID errors.
                detail = str(exc)
                safe_messages = {
                    "utterance must be a non-empty string",
                    "request contains undocumented fields",
                }
                validation_message = detail if detail in safe_messages else "invalid request"
                return self._error(HTTPStatus.BAD_REQUEST, "invalid_request", validation_message)
            try:
                if self.contextual_interaction is not None:
                    message = self.contextual_interaction(
                        utterance, principal, correlation_id, context_correlation_id
                    )
                else:
                    message = self.interaction(utterance, principal, correlation_id)
            except PermissionError:
                return self._error(HTTPStatus.FORBIDDEN, "request_denied", "request denied")
            except Exception:
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "request_unavailable", "request unavailable"
                )
            try:
                response = BrowserMessage.model_validate(
                    {"message": message, "correlation_id": correlation_id, "session_id": session_id}
                    if isinstance(message, str)
                    else {**message, "correlation_id": correlation_id, "session_id": session_id}
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
            serialized = json.dumps(payload).encode()
        except (TypeError, ValueError):
            status = HTTPStatus.SERVICE_UNAVAILABLE
            serialized = json.dumps(
                {"code": "response_unavailable", "error": "response unavailable"}
            ).encode()
        if len(serialized) > _MAX_RESPONSE_BYTES:
            status = HTTPStatus.SERVICE_UNAVAILABLE
            serialized = json.dumps(
                {"code": "response_unavailable", "error": "response unavailable"}
            ).encode()
        return status, "application/json", serialized

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
    contextual_interaction: ContextualInteraction | None = None,
    feedback: FeedbackRecorder | None = None,
) -> None:
    """Serve the proof using callbacks supplied by the Core/client composition root."""

    app = BrowserApp(
        principal,
        interaction,
        state,
        health,
        request_status,
        contextual_interaction,
        feedback,
        session_token=secrets.token_urlsafe(32),
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._respond(
                app.dispatch(
                    "GET",
                    self.path,
                    headers={key.lower(): value for key, value in self.headers.items()},
                )
            )

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
            self._respond(
                app.dispatch(
                    "POST",
                    self.path,
                    self.rfile.read(length),
                    headers={key.lower(): value for key, value in self.headers.items()},
                )
            )

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
