"""Human-facing alpha adapter for the existing AEGIS semantic pipeline."""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

import psycopg

from .contracts import ActionCard, Principal, RequestStatus, Result
from .finance import PostgresFinanceSnapshotStore
from .gateway_rpc import OpenClawWebSocketChannel
from .health import ComponentHealth, HealthReport
from .homelab import PostgresHomelabStore
from .household import (
    PostgresHouseholdStore,
)
from .identity import (
    KeycloakIdentityProvider,
    KeycloakOIDCClient,
    PostgresExternalPrincipalResolver,
)
from .interaction import InteractionBoundary, InteractionDependencies
from .network import PostgresNetworkStore
from .pack_lifecycle import PackManager, PostgresPackStore
from .personal import PostgresPersonalStateStore
from .reference_packs import reference_bundles
from .store import PostgresObjectiveStore
from .tasks import PostgresTaskStore
from .web import serve


class _RuntimePolicy:
    def allows(self, request: Any) -> bool:
        return bool(request.action.action_id == "kitchen.groceries.add")


class _NoApproval:
    def required(self, request: Any) -> bool:
        return False

    def approved(self, request: Any) -> bool:
        return True


class _ReadOnlyHomelabRuntime:
    def restart(self, service: Any) -> bool:
        return False

    def health(self, service: Any) -> bool:
        return False


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def _distribution_version() -> str:
    try:
        return version("aegis-core")
    except PackageNotFoundError:
        return "source-checkout"


def _load_env_file(path: str) -> None:
    """Load simple AEGIS configuration without evaluating shell syntax."""

    env_path = Path(path)
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"unable to read env file {path!r}: {type(exc).__name__}") from exc
    seen: set[str] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"invalid env file line {line_number}; expected AEGIS_NAME=value")
        name, value = (part.strip() for part in line.split("=", 1))
        if not re.fullmatch(r"AEGIS_[A-Z0-9_]+", name):
            raise ValueError(f"invalid env file key on line {line_number}; use AEGIS_* keys")
        if name in seen:
            raise ValueError(f"duplicate env file key on line {line_number}: {name}")
        seen.add(name)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(name, value)


def _initialize_env_file(path: str) -> None:
    """Create a private placeholder configuration without overwriting files."""

    target = Path(path)
    template = files("aegis").joinpath("aegis.env.example").read_text(encoding="utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            descriptor = -1
            output.write(template)
    finally:
        if descriptor != -1:
            os.close(descriptor)


def _runtime_report() -> HealthReport:
    """Check operator-facing prerequisites without creating or changing state."""

    components: list[ComponentHealth] = []
    database_url = os.environ.get("AEGIS_DATABASE_URL")
    components.append(_postgres_health(database_url))

    ollama_url = os.environ.get("AEGIS_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    ollama_model = os.environ.get("AEGIS_OLLAMA_MODEL", "qwen3:8b")
    components.append(_ollama_health(ollama_url, ollama_model))

    openclaw_names = (
        "AEGIS_OPENCLAW_GATEWAY_URL",
        "AEGIS_OPENCLAW_TOKEN",
        "AEGIS_OPENCLAW_DEVICE_TOKEN",
        "AEGIS_OPENCLAW_IDENTITY_DB",
    )
    configured_openclaw = [name for name in openclaw_names if os.environ.get(name)]
    if not configured_openclaw:
        openclaw_detail = "not configured (optional until an external mutation is requested)"
        openclaw_healthy = True
    elif len(configured_openclaw) != len(openclaw_names):
        missing = ", ".join(name for name in openclaw_names if not os.environ.get(name))
        openclaw_detail = f"incomplete configuration; missing {missing}"
        openclaw_healthy = False
    else:
        openclaw_detail = "configuration present"
        openclaw_healthy = True
    components.append(
        ComponentHealth(
            name="openclaw",
            healthy=openclaw_healthy,
            required=False,
            detail=openclaw_detail,
        )
    )

    identity_healthy, identity_detail = _identity_health()
    components.append(
        ComponentHealth(
            name="identity", healthy=identity_healthy, required=True, detail=identity_detail
        )
    )
    return HealthReport(
        healthy=all(component.healthy for component in components),
        ready=all(component.healthy for component in components if component.required),
        components=tuple(components),
    )


def _identity_health() -> tuple[bool, str]:
    """Validate configured bearer identity without exposing token/provider details."""

    token = os.environ.get("AEGIS_KEYCLOAK_ACCESS_TOKEN")
    issuer = os.environ.get("AEGIS_KEYCLOAK_ISSUER")
    if not token and not issuer:
        return True, "local development identity mode"
    if not token or not issuer:
        return (
            False,
            "incomplete bearer identity configuration; set both "
            "AEGIS_KEYCLOAK_ISSUER and AEGIS_KEYCLOAK_ACCESS_TOKEN",
        )
    if not os.environ.get("AEGIS_DATABASE_URL"):
        return False, "bearer identity requires AEGIS_DATABASE_URL for canonical subject mapping"
    try:
        _principal()
    except (OSError, RuntimeError, ValueError, PermissionError, psycopg.Error):
        return (
            False,
            "bearer identity is unavailable; verify the token, Keycloak issuer, "
            "and canonical subject mapping",
        )
    return True, "validated bearer-token identity and canonical subject mapping"


def _safe_endpoint(value: str) -> str:
    """Render a diagnostic endpoint without exposing URL userinfo."""

    try:
        parsed = urlsplit(value)
    except ValueError:
        return "configured Ollama endpoint"
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "configured Ollama endpoint"
    try:
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = host
        if parsed.port is not None:
            netloc += f":{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))
    except ValueError:
        return "configured Ollama endpoint"


def _postgres_health(database_url: str | None) -> ComponentHealth:
    if not database_url:
        return ComponentHealth(
            name="postgres",
            healthy=False,
            required=True,
            detail="set AEGIS_DATABASE_URL",
        )
    if any(placeholder in database_url for placeholder in ("USER", "PASSWORD", "DBNAME")):
        return ComponentHealth(
            name="postgres",
            healthy=False,
            required=True,
            detail="configuration contains template placeholders; replace AEGIS_DATABASE_URL",
        )
    try:
        connection = psycopg.connect(database_url, connect_timeout=2)
        try:
            row = connection.execute(
                "SELECT to_regclass(%s), to_regclass(%s), to_regclass(%s)",
                ("public.objectives", "public.results", "public.space_memberships"),
            ).fetchone()
        finally:
            connection.close()
        if not row or any(value is None for value in row):
            return ComponentHealth(
                name="postgres",
                healthy=False,
                required=True,
                detail=(
                    "connection succeeded but the canonical schema is incomplete; "
                    "apply migrations before starting AEGIS"
                ),
            )
        return ComponentHealth(
            name="postgres", healthy=True, required=True, detail="connection and schema succeeded"
        )
    except psycopg.ProgrammingError:
        return ComponentHealth(
            name="postgres",
            healthy=False,
            required=True,
            detail="invalid database configuration; verify AEGIS_DATABASE_URL",
        )
    except psycopg.Error as exc:
        return ComponentHealth(
            name="postgres",
            healthy=False,
            required=True,
            detail=(
                f"connection failed: {type(exc).__name__}; verify AEGIS_DATABASE_URL "
                "and ensure PostgreSQL is running"
            ),
        )


def _ollama_health(url: str, model: str) -> ComponentHealth:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ComponentHealth(
            name="ollama",
            healthy=False,
            required=True,
            detail="invalid URL; set AEGIS_OLLAMA_URL to an http:// or https:// endpoint",
        )
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=2) as response:
            if response.status != 200:
                raise urllib.error.URLError(f"HTTP {response.status}")
            payload = json.loads(response.read())
        if not isinstance(payload, dict):
            raise ValueError("invalid Ollama model response")
        model_names = {
            str(item.get("name"))
            for item in payload.get("models", [])
            if isinstance(item, dict) and item.get("name")
        }
        if model not in model_names:
            return ComponentHealth(
                name="ollama",
                healthy=False,
                required=True,
                detail=(
                    f"model {model!r} is not installed; run 'ollama pull {model}' "
                    "or set AEGIS_OLLAMA_MODEL"
                ),
            )
        return ComponentHealth(
            name="ollama",
            healthy=True,
            required=True,
            detail=f"API responded; model {model} is available",
        )
    except (OSError, urllib.error.URLError, ValueError) as exc:
        endpoint = _safe_endpoint(url)
        return ComponentHealth(
            name="ollama",
            healthy=False,
            required=True,
            detail=(
                f"API unavailable at {endpoint}: {type(exc).__name__}; "
                f"check `curl {endpoint}/api/tags`, start Ollama, or set "
                "AEGIS_OLLAMA_URL to its reachable address"
            ),
        )


def _print_runtime_report(report: HealthReport, as_json: bool) -> int:
    if as_json:
        print(report.model_dump_json())
    else:
        print(f"AEGIS runtime: {'READY' if report.ready else 'NOT READY'}")
        for component in report.components:
            state = "OK" if component.healthy else "FAIL"
            requirement = "required" if component.required else "optional"
            print(f"{component.name}: {state} ({requirement}) — {component.detail}")
    return 0 if report.ready else 1


def _print_json_error(code: str, message: str) -> None:
    print(json.dumps({"code": code, "error": message, "state": "failed"}))


def _constellation_state(principal: Principal) -> dict[str, Any]:
    """Build a small authorized view from canonical stores for the browser adapter."""

    connection = psycopg.connect(_required("AEGIS_DATABASE_URL"))
    try:
        _apply_migrations(connection)
        household = PostgresHouseholdStore(connection).read_snapshot(principal)
        tasks = PostgresTaskStore(connection).list(principal)
        groceries = cast(tuple[str, ...], household.get("groceries", ()))
        personal = PostgresPersonalStateStore(connection, principal.vault_id).load_for_principal(
            principal
        )
        finance = PostgresFinanceSnapshotStore(connection).load(principal.id)
        network = PostgresNetworkStore(connection).load(principal)
        homelab = PostgresHomelabStore(connection).load(principal, _ReadOnlyHomelabRuntime())
        persisted = {
            bundle.manifest.pack_id: (bundle, status)
            for bundle, status, _grants in PostgresPackStore(connection).load()
        }
        nodes: list[dict[str, str]] = [
            {"id": "aegis", "label": "AEGIS", "detail": "central hub"},
        ]
        edges: list[dict[str, str]] = []
        available = {bundle.manifest.pack_id: bundle for bundle in reference_bundles()}
        available.update(
            {pack_id: item[0] for pack_id, item in persisted.items() if pack_id not in available}
        )
        for pack_id, bundle in sorted(available.items()):
            ui = bundle.manifest.ui
            label = ui.label if ui is not None else pack_id.replace("-", " ").title()
            status = persisted.get(pack_id, (None, "available"))[1]
            status_text = status.value if hasattr(status, "value") else str(status)
            node_id = f"pack-{pack_id}"
            detail = f"{ui.category if ui else 'domain'} · {status_text}"
            if pack_id == "tasks":
                detail += f" · {len(tasks)} tasks"
            elif pack_id == "kitchen":
                detail += f" · {len(groceries)} groceries"
            nodes.append(
                {
                    "id": node_id,
                    "label": label,
                    "detail": detail,
                }
            )
            edges.append({"source": "aegis", "target": node_id})
        domain_summaries = (
            (
                "personal",
                "Personal",
                f"{len(personal.projects)} projects · {len(personal.goals)} goals · "
                f"{len(personal.memories)} memories",
            ),
            (
                "household",
                "Household",
                f"{len(cast(tuple[object, ...], household.get('chores', ())))} chores · "
                f"{len(cast(tuple[object, ...], household.get('events', ())))} events",
            ),
            (
                "finance",
                "Finance",
                "private snapshot available" if finance is not None else "no private snapshot",
            ),
            (
                "homelab",
                "Infrastructure",
                f"{len(homelab.hosts)} hosts · {len(homelab.services)} services",
            ),
            (
                "network",
                "Network",
                f"{len(network.devices)} devices · {len(network.scopes)} authorized scopes",
            ),
        )
        for domain_id, label, detail in domain_summaries:
            node_id = f"domain-{domain_id}"
            nodes.append({"id": node_id, "label": label, "detail": detail})
            edges.append({"source": "aegis", "target": node_id})
        # Record lists stay in authorized detail views. The graph is intentionally
        # bounded to hubs and semantic context rather than growing with rows.
        details: dict[str, Any] = {
            "domain-personal": {
                "projects": [
                    {"name": project.name, "created_at": project.created_at.isoformat()}
                    for project in personal.projects.values()
                ],
                "goals": [
                    {
                        "description": goal.description,
                        "project_id": str(goal.project_id) if goal.project_id else None,
                    }
                    for goal in personal.goals.values()
                ],
            },
            "domain-household": {
                "groceries": list(groceries),
                "chores": [
                    {
                        "title": chore.title,
                        "assignee_id": chore.assignee_id,
                        "completed": chore.completed,
                    }
                    for chore in cast(tuple[Any, ...], household.get("chores", ()))
                ],
                "events": [
                    {"title": event.title, "starts_at": event.starts_at.isoformat()}
                    for event in cast(tuple[Any, ...], household.get("events", ()))
                ],
            },
            "domain-finance": {"snapshot_available": finance is not None},
            "domain-homelab": {
                "hosts": [{"hostname": host.hostname} for host in homelab.hosts.values()],
                "services": [{"name": service.name} for service in homelab.services.values()],
            },
            "domain-network": {
                "devices": [
                    {
                        "name": device.hostname or device.address,
                        "address": device.address,
                    }
                    for device in network.devices.values()
                ],
                "authorized_scopes": [scope.scope_id for scope in network.scopes.values()],
            },
            "pack-tasks": {
                "tasks": [{"title": task.title, "status": task.status.value} for task in tasks]
            },
            "pack-kitchen": {"groceries": list(groceries)},
        }
        return {"nodes": nodes, "edges": edges, "details": details}
    finally:
        connection.close()


def _browser_interaction(
    utterance: str, principal: Principal, correlation_id: UUID | None = None
) -> dict[str, Any]:
    result = run_interaction(utterance, principal, correlation_id)
    response: dict[str, Any] = {
        "message": _format(result),
        "state": result.state.value,
        "detail": result.message,
        "objective_id": str(result.objective_id),
        "correlation_id": str(result.correlation_id),
    }
    if result.retryable:
        response["retryable"] = True
    return response


def _browser_request_status(principal: Principal, correlation_id: UUID) -> RequestStatus:
    connection = psycopg.connect(_required("AEGIS_DATABASE_URL"))
    try:
        _apply_migrations(connection)
        status = PostgresObjectiveStore(connection).get_request_status(correlation_id, principal)
        if status is None:
            return RequestStatus(correlation_id=correlation_id, state="unknown")
        objective_id, state, message, retryable = status
        return RequestStatus(
            correlation_id=correlation_id,
            objective_id=objective_id,
            state=state.value,
            message=message,
            retryable=retryable or None,
        )
    finally:
        connection.close()


def _principal() -> Principal:
    token = os.environ.get("AEGIS_KEYCLOAK_ACCESS_TOKEN")
    if token:
        issuer = _required("AEGIS_KEYCLOAK_ISSUER")
        resolver = PostgresExternalPrincipalResolver(
            psycopg.connect, _required("AEGIS_DATABASE_URL")
        )
        return KeycloakOIDCClient(issuer, principal_resolver=resolver).principal_from_access_token(
            token
        )
    # Local alpha mode still crosses the identity mapping boundary. It uses an
    # explicitly configured development identity rather than pretending to be a
    # bearer-token session.
    claims = {
        "sub": os.environ.get("AEGIS_PRINCIPAL_ID", "alice"),
        "aegis_vault_id": os.environ.get("AEGIS_VAULT_ID", "alice-vault"),
        "aegis_space_ids": [os.environ.get("AEGIS_SPACE_ID", "apartment")],
    }
    return KeycloakIdentityProvider().principal_from_claims(claims)


def _apply_migrations(connection: Any) -> None:
    if os.environ.get("AEGIS_AUTO_MIGRATE", "1").lower() in {"0", "false", "no"}:
        return
    packaged_migrations = files("aegis").joinpath("migrations")
    if packaged_migrations.is_dir():
        migration_texts = [
            migration.read_text(encoding="utf-8")
            for migration in sorted(packaged_migrations.iterdir(), key=lambda path: path.name)
            if migration.name.endswith(".sql")
        ]
    else:
        root = Path(__file__).resolve().parents[2]
        migration_texts = [
            migration.read_text(encoding="utf-8")
            for migration in sorted((root / "migrations").glob("*.sql"))
        ]
    for migration_text in migration_texts:
        connection.execute(migration_text)
    connection.commit()


def _ensure_local_identity(connection: Any, principal: Principal) -> None:
    space_id = principal.space_ids[0]
    connection.execute(
        "INSERT INTO aegis_principals (id, external_subject) VALUES (%s, %s) "
        "ON CONFLICT (id) DO NOTHING",
        (principal.id, principal.id),
    )
    connection.execute(
        "INSERT INTO vaults (id, owner_principal_id) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
        (principal.vault_id, principal.id),
    )
    connection.execute(
        "INSERT INTO spaces (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
        (space_id, os.environ.get("AEGIS_SPACE_NAME", "Apartment")),
    )
    connection.execute(
        "INSERT INTO space_memberships (principal_id, space_id, role, active) "
        "VALUES (%s, %s, 'owner', TRUE) ON CONFLICT (principal_id, space_id) "
        "DO NOTHING",
        (principal.id, space_id),
    )
    connection.commit()


def _prepare_local_web_runtime(principal: Principal) -> None:
    """Prepare first-run local state without restoring revoked membership."""

    connection = psycopg.connect(_required("AEGIS_DATABASE_URL"))
    try:
        _apply_migrations(connection)
        _ensure_local_identity(connection, principal)
    finally:
        connection.close()


def _openclaw_channel() -> OpenClawWebSocketChannel:
    identity_db = _required("AEGIS_OPENCLAW_IDENTITY_DB")
    row = (
        sqlite3.connect(identity_db)
        .execute(
            "SELECT device_id, private_key_pem, public_key_pem FROM device_identities "
            "WHERE identity_key='primary'"
        )
        .fetchone()
    )
    if row is None:
        raise RuntimeError("OpenClaw primary device identity was not found")
    return OpenClawWebSocketChannel(
        _required("AEGIS_OPENCLAW_GATEWAY_URL"),
        _required("AEGIS_OPENCLAW_TOKEN"),
        timeout=15,
        persistent=True,
        device_id=str(row[0]),
        device_token=_required("AEGIS_OPENCLAW_DEVICE_TOKEN"),
        private_key_pem=str(row[1]),
        public_key_pem=str(row[2]),
    )


def _domain_and_action(utterance: str, manager: PackManager) -> tuple[str, ActionCard]:
    text = utterance.lower()
    if "task" in text:
        domain = "tasks"
        if "event" in text:
            action_id = "tasks.events.create"
        elif "chore" in text:
            action_id = (
                "tasks.chores.list"
                if text.startswith(("what", "show", "list"))
                else "tasks.chores.create"
            )
        else:
            action_id = (
                "tasks.list" if text.startswith(("what", "show", "list")) else "tasks.create"
            )
    elif "chore" in text:
        domain = "tasks"
        action_id = (
            "tasks.chores.list"
            if text.startswith(("what", "show", "list"))
            else "tasks.chores.create"
        )
    elif "event" in text or "inspection" in text:
        domain = "tasks"
        action_id = "tasks.events.create"
    elif any(word in text for word in ("grocery", "groceries", "rice", "food")):
        domain = "kitchen"
        action_id = (
            "kitchen.groceries.list"
            if text.startswith(("what", "show", "list"))
            else "kitchen.groceries.add"
        )
    else:
        raise ValueError("alpha supports groceries and tasks; try one of the four demo requests")
    cards = manager.retrieve(domain)
    card = next((item for item in cards if item.action.action_id == action_id), None)
    if card is None:
        raise RuntimeError(f"enabled Pack did not provide ActionCard {action_id}")
    action = card.action
    if action_id == "kitchen.groceries.add":
        match = re.search(r"add\s+(.+?)\s+to\s+(?:the\s+)?grocer(?:y|ies)\b", text)
        if match is None:
            raise ValueError("tell AEGIS what to add, for example: Add rice to groceries.")
        action = action.model_copy(update={"arguments": {"item": match.group(1).strip()}})
    elif action_id == "tasks.create":
        match = re.search(r"(?:create\s+)?(?:a\s+)?task\s+(?:to\s+)?(.+)$", text)
        if match is None:
            raise ValueError("tell AEGIS the task, for example: Create a task to buy cat food.")
        action = action.model_copy(update={"arguments": {"title": match.group(1).strip()}})
    elif action_id == "tasks.chores.create":
        match = re.search(r"(?:create|add)\s+(?:a\s+)?chore\s+(?:to\s+)?(.+)$", text)
        if match is None:
            raise ValueError(
                "tell AEGIS the chore, for example: Create a chore to clean the kitchen."
            )
        action = action.model_copy(update={"arguments": {"title": match.group(1).strip()}})
    elif action_id == "tasks.events.create":
        match = re.search(r"(?:create|add)\s+(?:an?\s+)?event\s+(?:for\s+)?(.+)$", text)
        if match is None:
            raise ValueError(
                "tell AEGIS the event, for example: Create an event for apartment inspection."
            )
        title = match.group(1).strip()
        if title.endswith(" tomorrow"):
            title = title.removesuffix(" tomorrow").strip()
            starts_at = datetime.now(timezone.utc) + timedelta(days=1)
        else:
            starts_at = datetime.now(timezone.utc)
        action = action.model_copy(
            update={"arguments": {"title": title, "starts_at": starts_at.isoformat()}}
        )
    return domain, ActionCard(action=action, summary=card.summary, relevance=card.relevance)


def _format(result: Any) -> str:
    if result.state.value != "completed":
        return f"Not completed — {result.message}"
    evidence = result.evidence
    if evidence.get("canonical_items") is not None:
        items = evidence["canonical_items"]
        return "Groceries: " + (", ".join(items) if items else "(empty)")
    if evidence.get("canonical_tasks") is not None:
        tasks = evidence["canonical_tasks"]
        listing = "; ".join(f"{item['title']} ({item['status']})" for item in tasks)
        return "Tasks: " + (listing if tasks else "(empty)")
    if evidence.get("memories") is not None:
        memories = evidence["memories"]
        if not memories:
            return "Memories: (none found)"
        return "Memories: " + "; ".join(
            f"{item['content']} [{item['provenance']}]" for item in memories
        )
    if evidence.get("projects") is not None:
        projects = evidence["projects"]
        return "Projects: " + (
            "; ".join(item["name"] for item in projects) if projects else "(none)"
        )
    if evidence.get("goals") is not None:
        goals = evidence["goals"]
        return "Goals: " + (
            "; ".join(
                f"{item['description']}" + (f" [{item['project']}]" if item["project"] else "")
                for item in goals
            )
            if goals
            else "(none)"
        )
    if evidence.get("obligations") is not None:
        obligations = evidence["obligations"]
        outstanding = [item for item in obligations if not item["settled"]]
        return "Outstanding obligations: " + (
            "; ".join(f"{item['title']} ({item['responsible_id']})" for item in outstanding)
            if outstanding
            else "(none)"
        )
    if evidence.get("chores") is not None:
        chores = evidence["chores"]
        return "Chores: " + (
            "; ".join(
                f"{item['title']} ({item['assignee_id']})"
                for item in chores
                if not item["completed"]
            )
            if chores
            else "(none)"
        )
    if evidence.get("events") is not None:
        events = evidence["events"]
        return "Events: " + ("; ".join(item["title"] for item in events) if events else "(none)")
    if evidence.get("affordable") is not None:
        status = "yes" if evidence["affordable"] else "no"
        return (
            f"Affordable: {status} (purchase ${evidence['purchase_cents'] / 100:.2f}; "
            f"shared obligations ${evidence['shared_obligations_cents'] / 100:.2f})"
        )
    if evidence.get("collection") == "chores" and evidence.get("title"):
        return f"Done — created chore: {evidence['title']}"
    if evidence.get("collection") == "events" and evidence.get("title"):
        return f"Done — created event: {evidence['title']}"
    if evidence.get("title"):
        return f"Done — created task: {evidence['title']}"
    if evidence.get("item"):
        return f"Done — added {evidence['item']} to groceries"
    return f"Done — {result.message}"


def run_interaction(
    utterance: str, principal: Principal, correlation_id: UUID | None = None
) -> Result:
    """Compatibility composition root for CLI and browser clients."""

    boundary = InteractionBoundary(
        InteractionDependencies(
            connect=psycopg.connect,
            required=_required,
            apply_migrations=_apply_migrations,
            ensure_local_identity=_ensure_local_identity,
            select_action=_domain_and_action,
            openclaw_channel=_openclaw_channel,
            local_identity=lambda: not bool(os.environ.get("AEGIS_KEYCLOAK_ACCESS_TOKEN")),
        )
    )
    return boundary.run(utterance, principal, correlation_id)


def handle(utterance: str, principal: Principal) -> str:
    """Preserve the human CLI presentation over the shared interaction result."""

    return _format(run_interaction(utterance, principal))


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="aegis",
        description="Run a request through the AEGIS interaction boundary.",
        epilog=(
            "Interactive mode reads requests until quit or exit. "
            "All state-changing requests still require the normal Core policy "
            "and verification gates."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_distribution_version()}",
        help="show the installed AEGIS package version and exit",
    )
    parser.add_argument(
        "--once",
        metavar="REQUEST",
        help="handle one natural-language request and exit (useful for scripts)",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="suppress the interactive startup banner",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check configuration and runtime readiness, then exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON (valid with --check or --once)",
    )
    parser.add_argument(
        "--env-file",
        metavar="PATH",
        help="load AEGIS_* settings from a simple KEY=value file before startup",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="create a private placeholder .env file and exit (refuses to overwrite)",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="serve the minimal Constellation browser client on loopback",
    )
    parser.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=_port_value, default=8080, help="browser client port")
    args = parser.parse_args()
    if args.json and not (args.check or args.once is not None):
        parser.error("--json requires --check or --once")
    if args.web and (args.check or args.once is not None):
        parser.error("--web cannot be combined with --check or --once")
    if args.init and (args.check or args.once is not None or args.web):
        parser.error("--init cannot be combined with --check, --once, or --web")
    if args.init:
        target = args.env_file or ".env"
        try:
            _initialize_env_file(target)
        except FileExistsError:
            print(f"Not completed — configuration file already exists: {target}")
            return 1
        except OSError:
            print(f"Not completed — unable to create configuration file: {target}")
            return 1
        print(f"Created private configuration template: {target}")
        print("Replace its placeholders, then run './scripts/aegis --check'.")
        return 0
    env_file = args.env_file
    if env_file is None and Path(".env").is_file():
        env_file = ".env"
    if env_file:
        try:
            _load_env_file(env_file)
        except ValueError as exc:
            if args.json:
                _print_json_error("configuration_invalid", "configuration file is invalid")
            else:
                print(f"Not completed — invalid configuration: {exc}")
            return 1
    if args.check:
        return _print_runtime_report(_runtime_report(), args.json)
    try:
        principal = _principal()
    except (RuntimeError, ValueError, OSError, PermissionError, psycopg.Error) as exc:
        if args.json:
            _print_json_error("identity_unavailable", "identity unavailable")
            return 1
        print(f"Not completed — unable to initialize identity: {exc}")
        return 1
    if args.web:
        try:
            if not os.environ.get("AEGIS_KEYCLOAK_ACCESS_TOKEN"):
                try:
                    _prepare_local_web_runtime(principal)
                except (RuntimeError, psycopg.Error):
                    # The loopback shell can still expose readiness diagnostics when
                    # canonical storage is unavailable. State and interaction routes
                    # fail closed until the operator repairs the reported dependency.
                    print(
                        "AEGIS runtime is not ready; the browser will show diagnostics. "
                        "Run './scripts/aegis --check' to see remediation."
                    )
            print(f"AEGIS Constellation available at http://{args.host}:{args.port}")
            serve(
                args.host,
                args.port,
                _principal,
                _browser_interaction,
                _constellation_state,
                _runtime_report,
                _browser_request_status,
            )
        except OSError as exc:
            print(f"Not completed — {_browser_startup_error(exc, args.port)}")
            return 1
        except (RuntimeError, ValueError, PermissionError) as exc:
            print(f"Not completed — unable to start browser: {exc}")
            return 1
        return 0
    if args.once is not None:
        try:
            if args.json:
                result = run_interaction(args.once, principal)
                print(result.model_dump_json())
                return 0 if result.state.value == "completed" else 1
            else:
                print(handle(args.once, principal))
        except PermissionError:
            if args.json:
                _print_json_error("request_denied", "request denied")
            else:
                print("Not completed — request denied")
            return 1
        except psycopg.Error:
            if args.json:
                _print_json_error("request_unavailable", "request unavailable")
            else:
                print(
                    "Not completed — request unavailable; run "
                    "'./scripts/aegis --check' and verify AEGIS_DATABASE_URL"
                )
            return 1
        except (RuntimeError, OSError):
            if args.json:
                _print_json_error("request_unavailable", "request unavailable")
            else:
                print(
                    "Not completed — request unavailable; run "
                    "'./scripts/aegis --check' and verify configured services"
                )
            return 1
        except ValueError as exc:
            if args.json:
                _print_json_error("request_unavailable", "request unavailable")
            else:
                print(f"Not completed — {exc}")
            return 1
        return 0
    if not args.no_banner:
        print("AEGIS alpha. Type a request, or 'quit' to exit.")
    while True:
        try:
            utterance = input("aegis> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if utterance.lower() in {"quit", "exit"}:
            return 0
        if not utterance:
            continue
        try:
            print(_format_error(handle(utterance, principal)))
        except PermissionError:
            print("Not completed — request denied")
        except psycopg.Error:
            print(
                "Not completed — request unavailable; run "
                "'./scripts/aegis --check' and verify AEGIS_DATABASE_URL"
            )
        except (RuntimeError, OSError):
            print(
                "Not completed — request unavailable; run "
                "'./scripts/aegis --check' and verify configured services"
            )
        except ValueError as exc:
            print(f"Not completed — {exc}")


def _format_error(message: str) -> str:
    return message


def _port_value(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer from 1 to 65535") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be an integer from 1 to 65535")
    return port


def _browser_startup_error(exc: OSError, port: int) -> str:
    if exc.errno == errno.EADDRINUSE:
        return f"browser port {port} is already in use; choose another with --port"
    return f"unable to start browser: {exc}"


if __name__ == "__main__":
    raise SystemExit(main())
