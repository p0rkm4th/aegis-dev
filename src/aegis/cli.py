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
from collections.abc import Callable
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid4

import psycopg

from .audit import PostgresAuditLog
from .contracts import (
    ActionCard,
    Context,
    IntentFrame,
    ModelRequest,
    ObjectiveState,
    Principal,
    RequestStatus,
    Result,
    WorkingSet,
)
from .embeddings import OllamaEmbeddingProvider
from .feedback_triage import harvest_defect_candidates
from .finance import PostgresFinanceSnapshotStore
from .gateway_rpc import OpenClawWebSocketChannel
from .health import ComponentHealth, HealthReport, RuntimeIdentity
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
from .ollama import OllamaHttpTransport, OllamaProvider
from .pack_lifecycle import PostgresPackStore
from .pack_runtime import PackRuntimeRegistry
from .personal import PostgresPersonalStateStore
from .reference_interaction import (
    build_reference_fallback_context_runtime,
    ground_reference_action_runtime,
    reference_constellation_state,
    reference_domain_and_action,
    reference_fallback_cards,
    reference_format_result,
    resolve_reference_fast_paths,
    resolve_reference_pre_model,
    resolve_reference_safety_fast_paths,
    rewrite_reference_decision,
    run_reference_plan,
)
from .reference_packs import (
    reference_bundles,
)
from .reference_runtime import default_runtime_registry, legacy_runtime
from .release_truth import runtime_release_sha
from .research import (
    DocumentFetcher,
    ResearchService,
    ResearchUnavailable,
    SearchRequest,
    SearxngSearchProvider,
    TrafilaturaContentExtractor,
)
from .store import PostgresObjectiveStore
from .structural import SpacyStructuralParser, StructuralParserUnavailable
from .tasks import PostgresTaskStore
from .utterance import is_task_destination_request
from .web import serve

# Backward-compatible import for callers of the alpha's legacy helper.  The
# implementation belongs to the reference-Pack composition module; CLI is only
# a transport/composition adapter.
_domain_and_action = reference_domain_and_action
_format = reference_format_result


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


def _default_runtime_registry(
    openclaw_channel: Callable[[], OpenClawWebSocketChannel],
) -> PackRuntimeRegistry:
    return default_runtime_registry(openclaw_channel)


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
    components.append(_structural_health(os.environ.get("AEGIS_STRUCTURAL_MODEL")))

    identity_healthy, identity_detail = _identity_health()
    components.append(
        ComponentHealth(
            name="identity", healthy=identity_healthy, required=True, detail=identity_detail
        )
    )
    release_sha = runtime_release_sha(__file__)
    return HealthReport(
        healthy=all(component.healthy for component in components),
        ready=all(component.healthy for component in components if component.required),
        components=tuple(components),
        runtime=RuntimeIdentity(
            package_version=_distribution_version(),
            execution_mode=("installed-release" if release_sha else "source-checkout"),
            release_sha=release_sha,
            provider="ollama",
            model=ollama_model,
            model_digest=_ollama_model_digest(ollama_url, ollama_model),
            endpoint=_safe_endpoint(ollama_url),
        ),
    )


@lru_cache(maxsize=4)
def _cached_structural_parser(model: str) -> Callable[[str], Any]:
    """Load one configured parser per model for the life of the process."""

    return SpacyStructuralParser(model_path=model).parse


def _structural_health(model: str | None) -> ComponentHealth:
    """Report optional structural evidence without changing required readiness."""

    if not model:
        return ComponentHealth(
            name="structural_parser",
            healthy=True,
            required=False,
            detail="not configured (compound coverage remains fail-closed)",
        )
    try:
        _cached_structural_parser(model)
    except StructuralParserUnavailable:
        return ComponentHealth(
            name="structural_parser",
            healthy=False,
            required=False,
            detail="configured parser/model unavailable",
        )
    return ComponentHealth(
        name="structural_parser",
        healthy=True,
        required=False,
        detail=f"configured ({model})",
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
    except Exception:
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
    except Exception:
        return ComponentHealth(
            name="postgres",
            healthy=False,
            required=True,
            detail=(
                "database health check failed; verify AEGIS_DATABASE_URL and ensure "
                "PostgreSQL is running"
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
    except Exception:
        endpoint = _safe_endpoint(url)
        return ComponentHealth(
            name="ollama",
            healthy=False,
            required=True,
            detail=(
                f"Ollama health check failed at {endpoint}; check `curl {endpoint}/api/tags`, "
                "start Ollama, or set AEGIS_OLLAMA_URL to its reachable address"
            ),
        )


def _ollama_model_digest(url: str, model: str) -> str | None:
    """Read the configured model digest without loading or invoking it."""
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=2) as response:
            if response.status != 200:
                return None
            payload = json.loads(response.read())
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            return None
        for item in models:
            if isinstance(item, dict) and item.get("name") == model:
                digest = item.get("digest")
                return digest if isinstance(digest, str) and digest else None
    except Exception:
        return None
    return None


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
    """Adapt the reference-Pack projection to the browser callback contract."""

    return reference_constellation_state(
        principal,
        psycopg.connect,
        _required,
        _apply_migrations,
        household_store_factory=PostgresHouseholdStore,
        task_store_factory=PostgresTaskStore,
        personal_store_factory=PostgresPersonalStateStore,
        finance_store_factory=PostgresFinanceSnapshotStore,
        network_store_factory=PostgresNetworkStore,
        homelab_store_factory=PostgresHomelabStore,
        pack_store_factory=PostgresPackStore,
    )


def _browser_interaction(
    utterance: str,
    principal: Principal,
    correlation_id: UUID | None = None,
    context_correlation_id: UUID | None = None,
    runtime_registry: PackRuntimeRegistry | None = None,
) -> dict[str, Any]:
    if context_correlation_id is None:
        if runtime_registry is None:
            result = run_interaction(utterance, principal, correlation_id)
        else:
            result = run_interaction(
                utterance, principal, correlation_id, runtime_registry=runtime_registry
            )
    else:
        if runtime_registry is None:
            result = run_interaction(
                utterance, principal, correlation_id, context_correlation_id=context_correlation_id
            )
        else:
            result = run_interaction(
                utterance,
                principal,
                correlation_id,
                context_correlation_id=context_correlation_id,
                runtime_registry=runtime_registry,
            )
    response: dict[str, Any] = {
        "message": _format(result),
        "code": (
            str(result.evidence["code"])
            if isinstance(result.evidence.get("code"), str)
            else result.state.value
        ),
        "state": result.state.value,
        "detail": result.message,
        "objective_id": str(result.objective_id),
        "correlation_id": str(result.correlation_id),
    }
    raw_steps = result.evidence.get("steps")
    if isinstance(raw_steps, (list, tuple)):
        response["steps"] = tuple(
            {
                "index": step["index"],
                "action_id": step["action_id"],
                "state": step["state"],
                "objective_id": step["objective_id"],
                "correlation_id": step["correlation_id"],
                "message": step["message"],
            }
            for step in raw_steps[:5]
            if isinstance(step, dict)
        )
    research = result.evidence.get("research")
    if isinstance(research, dict) and isinstance(research.get("sources"), list):
        response["sources"] = tuple(
            {
                "source_id": item.get("source_id"),
                "title": item.get("title"),
                "url": item.get("url"),
                "retrieved_at": item.get("retrieved_at"),
            }
            for item in research["sources"][:5]
            if isinstance(item, dict)
            and all(
                isinstance(item.get(key), str)
                for key in ("source_id", "title", "url", "retrieved_at")
            )
        )
    if result.retryable:
        response["retryable"] = True
    return response


def browser_interaction(
    runtime_registry: PackRuntimeRegistry | None = None,
) -> Callable[[str, Principal, UUID, UUID | None], dict[str, Any]]:
    """Bind a client-neutral browser handler to optional Pack runtimes."""

    def handler(
        utterance: str,
        principal: Principal,
        correlation_id: UUID,
        context_correlation_id: UUID | None = None,
    ) -> dict[str, Any]:
        return _browser_interaction(
            utterance,
            principal,
            correlation_id,
            context_correlation_id,
            runtime_registry,
        )

    return handler


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
            state=state,
            message=message,
            retryable=retryable or None,
        )
    finally:
        connection.close()


def _browser_feedback(
    principal: Principal, correlation_id: UUID, outcome: str, reason: str | None
) -> None:
    connection = psycopg.connect(_required("AEGIS_DATABASE_URL"))
    try:
        _apply_migrations(connection)
        store = PostgresObjectiveStore(connection)
        result = store.get_result_for_correlation(correlation_id, principal)
        if result is None:
            raise PermissionError("feedback correlation is unavailable")
        PostgresAuditLog(connection).append(
            "owner.feedback",
            principal.id,
            {
                "correlation_id": str(correlation_id),
                "outcome": outcome,
                "reason": reason,
                "result_state": result.state.value,
                "retryable": result.retryable,
            },
            objective_id=result.objective_id,
        )
    finally:
        connection.close()


def _owner_feedback_report(principal: Principal, limit: int = 20) -> list[dict[str, Any]]:
    """Return bounded owner feedback metadata for defect triage."""

    connection = psycopg.connect(_required("AEGIS_DATABASE_URL"))
    try:
        _apply_migrations(connection)
        rows = connection.execute(
            """SELECT id, objective_id, payload, created_at
               FROM audit_events
               WHERE principal_id = %s AND event_type = 'owner.feedback'
               ORDER BY created_at DESC, id DESC
               LIMIT %s""",
            (principal.id, min(max(limit, 1), 100)),
        ).fetchall()
        report: list[dict[str, Any]] = []
        for event_id, objective_id, payload, created_at in rows:
            values = payload if isinstance(payload, dict) else {}
            report.append(
                {
                    "event_id": str(event_id),
                    "objective_id": str(objective_id) if objective_id else None,
                    "correlation_id": values.get("correlation_id"),
                    "created_at": (
                        created_at.isoformat()
                        if hasattr(created_at, "isoformat")
                        else str(created_at)
                    ),
                    "outcome": values.get("outcome"),
                    "reason": values.get("reason"),
                    "result_state": values.get("result_state"),
                    "retryable": values.get("retryable"),
                }
            )
        return report
    finally:
        connection.close()


def _print_owner_feedback(
    report: list[dict[str, Any]], as_json: bool, harvest: bool = False
) -> int:
    if harvest:
        defects = harvest_defect_candidates(report)
        if as_json:
            print(json.dumps({"defects": defects}, sort_keys=True))
            return 0
        if not defects:
            print("No owner feedback defect candidates recorded.")
            return 0
        print(f"Owner feedback defect candidates ({len(defects)}):")
        for item in defects:
            print(
                f"- {item['classification']}: objective={item['objective_id']} "
                f"event={item['event_id']} (reproduction required; no replay)"
            )
        return 0
    if as_json:
        print(json.dumps({"feedback": report}, sort_keys=True))
        return 0
    if not report:
        print("No owner feedback recorded.")
        return 0
    print(f"Recent owner feedback ({len(report)}):")
    for item in report:
        reason = f"/{item['reason']}" if item["reason"] else ""
        print(
            f"- {item['outcome']}{reason}: result={item['result_state']} "
            f"objective={item['objective_id']} event={item['event_id']}"
        )
    return 0


def _principal() -> Principal:
    token = os.environ.get("AEGIS_KEYCLOAK_ACCESS_TOKEN")
    issuer = os.environ.get("AEGIS_KEYCLOAK_ISSUER")
    if token or issuer:
        if not token or not issuer:
            raise RuntimeError(
                "incomplete bearer identity configuration; set both "
                "AEGIS_KEYCLOAK_ISSUER and AEGIS_KEYCLOAK_ACCESS_TOKEN"
            )
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


def run_interaction(
    utterance: str,
    principal: Principal,
    correlation_id: UUID | None = None,
    context_correlation_id: UUID | None = None,
    runtime_registry: PackRuntimeRegistry | None = None,
) -> Result:
    """Compose the shared boundary, optionally with Pack runtime bindings."""

    if runtime_registry is None:
        runtime_registry = _default_runtime_registry(_openclaw_channel)

    def research_answer(intent: IntentFrame, context: Context, source_kind: str) -> Result:
        endpoint = os.environ.get("AEGIS_SEARCH_ENDPOINT")
        if not endpoint:
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.FAILED,
                message=(
                    "I couldn't verify current information right now because research is "
                    "not configured."
                ),
                evidence={"source_kind": source_kind, "authoritative": False},
                correlation_id=intent.correlation_id,
                retryable=True,
            )
        try:
            service = ResearchService(
                SearxngSearchProvider(endpoint),
                DocumentFetcher(),
                TrafilaturaContentExtractor(),
            )
            evidence = service.collect(SearchRequest(intent.utterance))
            evidence_values = {
                "query": evidence.query,
                "provider_id": evidence.provider_id,
                "retrieved_at": evidence.retrieved_at.isoformat(),
                "sources": [
                    {
                        "source_id": item.source_id,
                        "url": item.final_url,
                        "title": item.title,
                        "text": item.text,
                        "retrieved_at": item.retrieved_at.isoformat(),
                    }
                    for item in evidence.evidence
                ],
            }
            synthesis_values: dict[str, Any] = {"research_evidence": evidence_values}
            if source_kind == "mixed_evidence":
                # Local synthesis may use authorized context only after public
                # retrieval. This value is never passed to SearchProvider.
                synthesis_values["authorized_local_context"] = context.values
            synthesis_context = Context(values=synthesis_values)
            provider = OllamaProvider(
                os.environ.get("AEGIS_OLLAMA_MODEL", "qwen3:8b"),
                OllamaHttpTransport(_required("AEGIS_OLLAMA_URL")),
            )
            response = provider.decide(
                ModelRequest(
                    working_set=WorkingSet(intent=intent, context=synthesis_context),
                    action_cards=(),
                )
            )
            from .decoding import StrictDecisionDecoder

            decision = StrictDecisionDecoder().decode(response, (), allow_argument_proposals=False)
            if decision.kind.value != "ANSWER" or not decision.answer:
                raise ResearchUnavailable("research synthesis returned no answer")
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.COMPLETED,
                message=decision.answer,
                evidence={
                    "source_kind": source_kind,
                    "authoritative": False,
                    "research": evidence_values,
                    "local_context_sources": (
                        list(context.sources) if source_kind == "mixed_evidence" else []
                    ),
                },
                correlation_id=intent.correlation_id,
            )
        except Exception:
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.FAILED,
                message=(
                    "I couldn't verify current information right now. The research provider "
                    "was unavailable."
                ),
                evidence={"source_kind": source_kind, "authoritative": False},
                correlation_id=intent.correlation_id,
                retryable=True,
            )

    def retrieve_reference_capabilities(query: str, manager: Any) -> tuple[ActionCard, ...]:
        if is_task_destination_request(query):
            return reference_fallback_cards(manager, query)
        return cast(
            tuple[ActionCard, ...],
            manager.retrieve_semantic(
                query,
                OllamaEmbeddingProvider(
                    os.environ.get("AEGIS_EMBEDDING_MODEL", "nomic-embed-text"),
                    _required("AEGIS_OLLAMA_URL"),
                ),
                limit=10,
            ),
        )

    structural_model = os.environ.get("AEGIS_STRUCTURAL_MODEL")
    try:
        structural_parser = (
            _cached_structural_parser(structural_model) if structural_model else None
        )
    except StructuralParserUnavailable:
        # Compound objectives fail closed when independent structural evidence
        # is unavailable; ordinary reads and answers retain their existing paths.
        structural_parser = None

    boundary = InteractionBoundary(
        InteractionDependencies(
            connect=psycopg.connect,
            required=_required,
            apply_migrations=_apply_migrations,
            ensure_local_identity=_ensure_local_identity,
            select_action=reference_domain_and_action,
            openclaw_channel=_openclaw_channel,
            local_identity=lambda: not bool(os.environ.get("AEGIS_KEYCLOAK_ACCESS_TOKEN")),
            model_provider=lambda: OllamaProvider(
                os.environ.get("AEGIS_OLLAMA_MODEL", "qwen3:8b"),
                OllamaHttpTransport(_required("AEGIS_OLLAMA_URL")),
            ),
            capability_retriever=retrieve_reference_capabilities,
            runtime_registry=runtime_registry,
            pack_bundles=reference_bundles,
            auto_enable_pack_ids=frozenset(("tasks", "kitchen")),
            action_grounder=ground_reference_action_runtime,
            pre_model_resolver=resolve_reference_pre_model,
            fallback_card_selector=reference_fallback_cards,
            plan_runner=run_reference_plan,
            decision_rewriter=rewrite_reference_decision,
            fast_path_resolver=resolve_reference_fast_paths,
            fallback_context_builder=build_reference_fallback_context_runtime,
            runtime_resolver=legacy_runtime,
            safety_fast_path_resolver=resolve_reference_safety_fast_paths,
            research_answer=research_answer,
            structural_parser=structural_parser,
        )
    )
    return boundary.run(
        utterance,
        principal,
        correlation_id,
        context_correlation_id=context_correlation_id,
    )


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
        "--feedback",
        action="store_true",
        help="show recent owner feedback metadata for defect triage, then exit",
    )
    parser.add_argument(
        "--harvest",
        action="store_true",
        help="with --feedback, emit safe defect candidates requiring reproduction",
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
    if args.json and not (args.check or args.once is not None or args.feedback):
        parser.error("--json requires --check, --once, or --feedback")
    if args.web and (args.check or args.once is not None):
        parser.error("--web cannot be combined with --check or --once")
    if args.feedback and (args.check or args.once is not None or args.web or args.init):
        parser.error("--feedback cannot be combined with --check, --once, --web, or --init")
    if args.harvest and not args.feedback:
        parser.error("--harvest requires --feedback")
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
    except Exception:
        if args.json:
            _print_json_error("identity_unavailable", "identity unavailable")
            return 1
        print(
            "Not completed — identity unavailable; run './scripts/aegis --check' and "
            "verify identity configuration"
        )
        return 1
    if args.feedback:
        try:
            return _print_owner_feedback(_owner_feedback_report(principal), args.json, args.harvest)
        except (RuntimeError, psycopg.Error):
            if args.json:
                _print_json_error("feedback_unavailable", "feedback unavailable")
            else:
                print(
                    "Not completed — feedback unavailable; run "
                    "'./scripts/aegis --check' and verify AEGIS_DATABASE_URL"
                )
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
            print(f"Starting AEGIS Constellation at http://{args.host}:{args.port}")
            browser_handler = cast(
                Callable[[str, Principal, UUID], dict[str, Any]], browser_interaction()
            )
            contextual_browser_handler = browser_interaction()
            serve(
                args.host,
                args.port,
                _principal,
                browser_handler,
                _constellation_state,
                _runtime_report,
                _browser_request_status,
                contextual_browser_handler,
                _browser_feedback,
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
        except Exception:
            if args.json:
                _print_json_error("request_unavailable", "request unavailable")
            else:
                print(
                    "Not completed — request unavailable; run "
                    "'./scripts/aegis --check' and verify configured services"
                )
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
        except Exception:
            print(
                "Not completed — request unavailable; run "
                "'./scripts/aegis --check' and verify configured services"
            )


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
