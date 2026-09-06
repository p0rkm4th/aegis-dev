"""Small browser adapter over the existing interaction and state boundaries."""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .contracts import ObjectiveState, Principal, RequestStatus
from .health import HealthReport

Interaction = Callable[[str, Principal, UUID], str | dict[str, Any]]
ContextualInteraction = Callable[[str, Principal, UUID, UUID | None], str | dict[str, Any]]
ConstellationState = Callable[[Principal], dict[str, Any]]
WorkspaceState = Callable[[Principal], dict[str, Any]]
WorkspaceFile = Callable[[Principal, str, str], dict[str, Any]]
WorkspaceCreate = Callable[[Principal, dict[str, Any]], dict[str, Any]]
CompositionState = Callable[[Principal], dict[str, Any]]
PackState = Callable[[Principal], dict[str, Any]]
PackEnable = Callable[[Principal, dict[str, Any]], dict[str, Any]]
CalendarState = Callable[[Principal], dict[str, Any]]
DeviceState = Callable[[Principal], dict[str, Any]]
SystemsState = Callable[[Principal], dict[str, Any]]
WeatherState = Callable[[Principal], dict[str, Any]]
AirQualityState = Callable[[Principal], dict[str, Any]]

_AEGIS_CSS = (Path(__file__).with_name("static") / "aegis.css").read_text(encoding="utf-8")
_AEGIS_JS = (Path(__file__).with_name("static") / "aegis.js").read_text(encoding="utf-8")
TodayState = Callable[[Principal], dict[str, Any]]
ObjectivesState = Callable[[Principal], dict[str, Any]]
CommunicationsState = Callable[[Principal], dict[str, Any]]
DocumentsState = Callable[[Principal], dict[str, Any]]
DocumentFile = Callable[[Principal, str], dict[str, Any]]
DailyDriverState = Callable[[Principal], dict[str, Any]]
ResearchState = Callable[[Principal], dict[str, Any]]
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


class WorkspaceProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    workspaces: tuple[dict[str, Any], ...] = Field(default=(), max_length=50)


class CompositionProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    compositions: tuple[dict[str, Any], ...] = Field(default=(), max_length=50)


class PackProjection(BaseModel):
    """Bounded owner view of Pack lifecycle and declared capability metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    packs: tuple[dict[str, Any], ...] = Field(default=(), max_length=50)


class PackEnableRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pack_id: str = Field(min_length=1, max_length=100)
    permissions: tuple[str, ...] = Field(default=(), max_length=50)
    confirm: bool


_INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta name="aegis-session-token" content="__AEGIS_SESSION_TOKEN__"><link rel="stylesheet" href="/static/aegis.css"><script src="/static/aegis.js" defer></script><title>AEGIS · Personal intelligence</title>
</head><body><div class="app-shell"><header class="topbar"><div class="brand"><span class="brand-mark" aria-hidden="true">A</span><div><div class="eyebrow">Personal intelligence</div><h1>AEGIS</h1></div></div><button id="theme-toggle" type="button" aria-label="Switch color theme">Light mode</button></header>
<nav class="product-nav" aria-label="AEGIS views">
<div class="nav-group nav-primary" aria-label="Everyday">
<span class="nav-group-label">Everyday</span>
<button type="button" data-view="home" aria-current="page">Today</button>
<button type="button" data-view="tasks">Tasks</button>
<button type="button" data-view="calendar">Calendar</button>
<button type="button" data-view="household">Household</button>
<button type="button" data-view="systems">Systems</button>
<button type="button" data-view="documents">Documents</button>
</div>
<div class="nav-group nav-advanced" aria-label="Explore">
<span class="nav-group-label">Explore</span>
<button type="button" data-view="weather">Weather</button>
<button type="button" data-view="air-quality">Air quality</button>
<button type="button" data-view="devices">Devices</button>
<button type="button" data-view="communications">Communications</button>
<button type="button" data-view="daily-driver">Daily driver</button>
<button type="button" data-view="research">Research</button>
<button type="button" data-view="packs">Packs</button>
<button type="button" data-view="objectives">Objectives</button>
<button type="button" data-view="constellation">Constellation</button>
<button type="button" data-view="workspace">Workspace</button>
<button type="button" data-view="compositions">Compositions</button>
</div>
</nav>
<div class="workspace"><section class="conversation-panel" aria-label="Conversation with AEGIS"><div class="health-line"><span><span class="status-dot" aria-hidden="true"></span><strong id="health" aria-live="polite">Checking readiness…</strong></span><details><summary>Runtime details</summary><ul id="health-details" class="muted" aria-live="polite"></ul></details></div><div class="intro"><h2>What can I help you with?</h2><p>Ask naturally. I’ll keep track of your authorized information and tell you clearly what happened.</p></div>
<div class="view-summary"><h2 id="view-title">Today</h2><p id="view-description">Your conversation and authorized world at a glance.</p></div>
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
<details id="research-panel" class="research-sources" hidden><summary>External research evidence</summary><ul id="research-sources" aria-live="polite"></ul></details>
<h2 class="sr-only">Conversation</h2><ol id="conversation" role="log" aria-live="polite" aria-relevant="additions text"><li class="conversation-empty">Your conversation will appear here.</li></ol></section>
<details class="secondary" aria-label="Canonical state"><summary>Canonical state</summary><div class="state-tools"><h2 class="sr-only">Canonical state</h2><button id="refresh" type="button">Refresh state</button></div>
<p id="state-status" class="muted" aria-live="polite"></p>
<label for="node-filter">Find a node <input id="node-filter" type="search"
autocomplete="off" aria-describedby="node-filter-status"
placeholder="Filter authorized nodes"></label>
<p id="node-filter-status" class="muted" aria-live="polite" aria-atomic="true"></p>
<h2>Constellation</h2><p class="muted">AEGIS is the root of an authorized semantic map. Select a domain or capability for its conventional detail.</p><main id="nodes" role="group" aria-label="Authorized AEGIS constellation"><p>Loading state…</p></main>
<h2>Relationships</h2><ul id="edges"><li>Loading relationships…</li></ul></details></div></div>
</body></html>"""


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
        workspace_state: WorkspaceState | None = None,
        workspace_file: WorkspaceFile | None = None,
        workspace_create: WorkspaceCreate | None = None,
        composition_state: CompositionState | None = None,
        pack_state: PackState | None = None,
        pack_enable: PackEnable | None = None,
        calendar_state: CalendarState | None = None,
        device_state: DeviceState | None = None,
        systems_state: SystemsState | None = None,
        weather_state: WeatherState | None = None,
        air_quality_state: AirQualityState | None = None,
        today_state: TodayState | None = None,
        objectives_state: ObjectivesState | None = None,
        communications_state: CommunicationsState | None = None,
        documents_state: DocumentsState | None = None,
        document_file: DocumentFile | None = None,
        daily_driver_state: DailyDriverState | None = None,
        research_state: ResearchState | None = None,
    ) -> None:
        self.principal_provider = principal if callable(principal) else lambda: principal
        self.interaction = interaction
        self.state = state
        self.health = health
        self.request_status = request_status
        self.contextual_interaction = contextual_interaction
        self.feedback = feedback
        self.session_token = session_token
        self.workspace_state = workspace_state
        self.workspace_file = workspace_file
        self.workspace_create = workspace_create
        self.composition_state = composition_state
        self.pack_state = pack_state
        self.pack_enable = pack_enable
        self.calendar_state = calendar_state
        self.device_state = device_state
        self.systems_state = systems_state
        self.weather_state = weather_state
        self.air_quality_state = air_quality_state
        self.today_state = today_state
        self.objectives_state = objectives_state
        self.communications_state = communications_state
        self.documents_state = documents_state
        self.document_file = document_file
        self.daily_driver_state = daily_driver_state
        self.research_state = research_state

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
        if method == "GET" and route == "/static/aegis.css":
            return HTTPStatus.OK, "text/css; charset=utf-8", _AEGIS_CSS.encode()
        if method == "GET" and route == "/static/aegis.js":
            return HTTPStatus.OK, "text/javascript; charset=utf-8", _AEGIS_JS.encode()
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
        if method == "GET" and route == "/api/workspace":
            if self.workspace_state is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            try:
                projection = WorkspaceProjection.model_validate(self.workspace_state(principal))
            except PermissionError:
                return self._error(
                    HTTPStatus.FORBIDDEN, "state_access_denied", "state access denied"
                )
            except (TypeError, ValueError, ValidationError):
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "workspace_unavailable", "workspace unavailable"
                )
            return self._json(HTTPStatus.OK, projection.model_dump(mode="json"))
        if method == "GET" and route == "/api/workspace/file":
            if self.workspace_file is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            query = parse_qs(urlparse(path).query, keep_blank_values=True)
            if set(query) != {"workspace_id", "path"} or any(
                len(values) != 1 for values in query.values()
            ):
                return self._error(
                    HTTPStatus.BAD_REQUEST, "invalid_request", "invalid workspace file request"
                )
            try:
                payload = self.workspace_file(principal, query["workspace_id"][0], query["path"][0])
            except PermissionError:
                return self._error(
                    HTTPStatus.FORBIDDEN, "state_access_denied", "state access denied"
                )
            except (KeyError, TypeError, ValueError, OSError):
                return self._error(
                    HTTPStatus.NOT_FOUND, "file_unavailable", "workspace file unavailable"
                )
            return self._json(HTTPStatus.OK, payload)
        if method == "GET" and route == "/api/compositions":
            if self.composition_state is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            try:
                composition_projection = CompositionProjection.model_validate(
                    self.composition_state(principal)
                )
            except PermissionError:
                return self._error(
                    HTTPStatus.FORBIDDEN, "state_access_denied", "state access denied"
                )
            except (TypeError, ValueError, ValidationError):
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "state_unavailable",
                    "composition state unavailable",
                )
            return self._json(HTTPStatus.OK, composition_projection.model_dump(mode="json"))
        if method == "GET" and route == "/api/packs":
            if self.pack_state is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            try:
                pack_projection = PackProjection.model_validate(self.pack_state(principal))
            except PermissionError:
                return self._error(
                    HTTPStatus.FORBIDDEN, "state_access_denied", "state access denied"
                )
            except (TypeError, ValueError, ValidationError):
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "state_unavailable", "Pack state unavailable"
                )
            return self._json(HTTPStatus.OK, pack_projection.model_dump(mode="json"))
        if method == "GET" and route == "/api/calendar":
            if self.calendar_state is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            try:
                calendar_projection = self.calendar_state(principal)
                if not isinstance(calendar_projection, dict):
                    raise ValueError("calendar state must be an object")
            except PermissionError:
                return self._error(
                    HTTPStatus.FORBIDDEN, "state_access_denied", "state access denied"
                )
            except (TypeError, ValueError):
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "state_unavailable",
                    "calendar state unavailable",
                )
            return self._json(HTTPStatus.OK, calendar_projection)
        if method == "GET" and route == "/api/devices":
            if self.device_state is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            try:
                device_projection = self.device_state(principal)
                if not isinstance(device_projection, dict):
                    raise ValueError("device state must be an object")
            except PermissionError:
                return self._error(
                    HTTPStatus.FORBIDDEN, "state_access_denied", "state access denied"
                )
            except (TypeError, ValueError):
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "state_unavailable", "device state unavailable"
                )
            return self._json(HTTPStatus.OK, device_projection)
        if method == "GET" and route == "/api/systems":
            if self.systems_state is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            try:
                systems_projection = self.systems_state(principal)
                if not isinstance(systems_projection, dict):
                    raise ValueError("systems state must be an object")
            except PermissionError:
                return self._error(
                    HTTPStatus.FORBIDDEN, "state_access_denied", "state access denied"
                )
            except (TypeError, ValueError):
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "state_unavailable",
                    "systems state unavailable",
                )
            return self._json(HTTPStatus.OK, systems_projection)
        if method == "GET" and route == "/api/today":
            if self.today_state is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            try:
                today_projection = self.today_state(principal)
                if not isinstance(today_projection, dict):
                    raise ValueError("Today state must be an object")
            except PermissionError:
                return self._error(
                    HTTPStatus.FORBIDDEN, "state_access_denied", "state access denied"
                )
            except (TypeError, ValueError):
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "state_unavailable",
                    "Today state unavailable",
                )
            return self._json(HTTPStatus.OK, today_projection)
        if method == "GET" and route == "/api/weather":
            if self.weather_state is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            try:
                weather_projection = self.weather_state(principal)
                if not isinstance(weather_projection, dict):
                    raise ValueError("weather state must be an object")
            except PermissionError:
                return self._error(
                    HTTPStatus.FORBIDDEN, "state_access_denied", "state access denied"
                )
            except (TypeError, ValueError):
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "state_unavailable", "weather unavailable"
                )
            return self._json(HTTPStatus.OK, weather_projection)
        if method == "GET" and route == "/api/air-quality":
            if self.air_quality_state is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            try:
                air_quality_projection = self.air_quality_state(principal)
                if not isinstance(air_quality_projection, dict):
                    raise ValueError("air-quality state must be an object")
            except PermissionError:
                return self._error(
                    HTTPStatus.FORBIDDEN, "state_access_denied", "state access denied"
                )
            except (TypeError, ValueError):
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "state_unavailable", "air quality unavailable"
                )
            return self._json(HTTPStatus.OK, air_quality_projection)
        if method == "GET" and route == "/api/objectives":
            if self.objectives_state is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            try:
                objectives_projection = self.objectives_state(principal)
                if not isinstance(objectives_projection, dict):
                    raise ValueError("objectives state must be an object")
            except PermissionError:
                return self._error(
                    HTTPStatus.FORBIDDEN, "state_access_denied", "state access denied"
                )
            except (TypeError, ValueError):
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE, "state_unavailable", "objectives unavailable"
                )
            return self._json(HTTPStatus.OK, objectives_projection)
        if method == "GET" and route == "/api/communications":
            if self.communications_state is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            try:
                communications_projection = self.communications_state(principal)
                if not isinstance(communications_projection, dict):
                    raise ValueError("communications state must be an object")
            except PermissionError:
                return self._error(
                    HTTPStatus.FORBIDDEN, "state_access_denied", "state access denied"
                )
            except (TypeError, ValueError):
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "state_unavailable",
                    "communications unavailable",
                )
            return self._json(HTTPStatus.OK, communications_projection)
        if method == "GET" and route == "/api/documents":
            if self.documents_state is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            try:
                documents_projection = self.documents_state(principal)
                if not isinstance(documents_projection, dict):
                    raise ValueError("documents state must be an object")
            except PermissionError:
                return self._error(
                    HTTPStatus.FORBIDDEN, "state_access_denied", "state access denied"
                )
            except (TypeError, ValueError):
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "state_unavailable",
                    "documents unavailable",
                )
            return self._json(HTTPStatus.OK, documents_projection)
        if method == "GET" and route == "/api/documents/file":
            if self.document_file is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            query = parse_qs(urlparse(path).query, keep_blank_values=True)
            if set(query) != {"document_id"} or len(query["document_id"]) != 1:
                return self._error(
                    HTTPStatus.BAD_REQUEST, "invalid_request", "invalid document request"
                )
            try:
                payload = self.document_file(principal, query["document_id"][0])
            except PermissionError:
                return self._error(
                    HTTPStatus.FORBIDDEN, "state_access_denied", "state access denied"
                )
            except (KeyError, TypeError, ValueError, OSError):
                return self._error(
                    HTTPStatus.NOT_FOUND, "document_unavailable", "document unavailable"
                )
            return self._json(HTTPStatus.OK, payload)
        if method == "GET" and route == "/api/daily-driver":
            if self.daily_driver_state is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            try:
                daily_driver_projection = self.daily_driver_state(principal)
                if not isinstance(daily_driver_projection, dict):
                    raise ValueError("daily-driver state must be an object")
            except PermissionError:
                return self._error(
                    HTTPStatus.FORBIDDEN, "state_access_denied", "state access denied"
                )
            except (TypeError, ValueError):
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "state_unavailable",
                    "daily-driver status unavailable",
                )
            return self._json(HTTPStatus.OK, daily_driver_projection)
        if method == "GET" and route == "/api/research":
            if self.research_state is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            try:
                research_projection = self.research_state(principal)
                if not isinstance(research_projection, dict):
                    raise ValueError("research state must be an object")
            except PermissionError:
                return self._error(
                    HTTPStatus.FORBIDDEN, "state_access_denied", "state access denied"
                )
            except (TypeError, ValueError):
                return self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "state_unavailable",
                    "research state unavailable",
                )
            return self._json(HTTPStatus.OK, research_projection)
        if method == "POST" and route == "/api/packs/enable":
            if self.pack_enable is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            try:
                request = PackEnableRequest.model_validate(json.loads(body.decode("utf-8")))
                if request.confirm is not True:
                    raise PermissionError("explicit Pack enablement confirmation is required")
                result = self.pack_enable(principal, request.model_dump(mode="json"))
                if not isinstance(result, dict):
                    raise ValueError("Pack enablement result must be an object")
            except PermissionError as exc:
                return self._error(HTTPStatus.FORBIDDEN, "action_denied", str(exc))
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
                ValidationError,
            ):
                return self._error(
                    HTTPStatus.BAD_REQUEST, "invalid_request", "invalid Pack enablement request"
                )
            return self._json(HTTPStatus.OK, result)
        if method == "POST" and route == "/api/workspace":
            if self.workspace_create is None:
                return self._error(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            try:
                request = json.loads(body.decode("utf-8"))
                if not isinstance(request, dict) or not request:
                    raise ValueError("workspace request must be an object")
                result = self.workspace_create(principal, request)
                if not isinstance(result, dict):
                    raise ValueError("workspace result must be an object")
            except PermissionError:
                return self._error(HTTPStatus.FORBIDDEN, "action_denied", "workspace action denied")
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                return self._error(
                    HTTPStatus.BAD_REQUEST, "invalid_request", "invalid workspace request"
                )
            return self._json(HTTPStatus.OK, result)
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
    workspace_state: WorkspaceState | None = None,
    workspace_file: WorkspaceFile | None = None,
    workspace_create: WorkspaceCreate | None = None,
    composition_state: CompositionState | None = None,
    pack_state: PackState | None = None,
    pack_enable: PackEnable | None = None,
    calendar_state: CalendarState | None = None,
    device_state: DeviceState | None = None,
    systems_state: SystemsState | None = None,
    weather_state: WeatherState | None = None,
    air_quality_state: AirQualityState | None = None,
    today_state: TodayState | None = None,
    objectives_state: ObjectivesState | None = None,
    communications_state: CommunicationsState | None = None,
    documents_state: DocumentsState | None = None,
    document_file: DocumentFile | None = None,
    daily_driver_state: DailyDriverState | None = None,
    research_state: ResearchState | None = None,
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
        workspace_state=workspace_state,
        workspace_file=workspace_file,
        workspace_create=workspace_create,
        composition_state=composition_state,
        pack_state=pack_state,
        pack_enable=pack_enable,
        calendar_state=calendar_state,
        device_state=device_state,
        systems_state=systems_state,
        weather_state=weather_state,
        air_quality_state=air_quality_state,
        today_state=today_state,
        objectives_state=objectives_state,
        communications_state=communications_state,
        documents_state=documents_state,
        document_file=document_file,
        daily_driver_state=daily_driver_state,
        research_state=research_state,
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
                "default-src 'none'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
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
