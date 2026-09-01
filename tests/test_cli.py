import json
from datetime import datetime
from uuid import uuid4

import pytest

from aegis import InteractionBoundary, InteractionDependencies
from aegis.cli import _domain_and_action, _ensure_local_identity
from aegis.contracts import IntentFrame, ObjectiveState, Principal, Result
from aegis.pack_lifecycle import PackManager
from aegis.reference_packs import reference_bundles, reference_packs
from aegis.tasks import Task, TaskReadFastPath, TaskStatus
from aegis.web import BrowserApp


def test_interaction_boundary_is_public_without_live_runtime():
    assert InteractionBoundary.__module__ == "aegis.interaction"
    assert InteractionDependencies.__module__ == "aegis.interaction"


def manager_with_reference_cards() -> PackManager:
    manager = PackManager()
    for pack in reference_packs():
        from aegis.pack_lifecycle import PackBundle, PackManifest

        manager.discover(
            PackBundle(
                manifest=PackManifest(
                    pack_id=pack.pack_id,
                    version=pack.version,
                    permissions=(
                        "tasks.write",
                        "tasks.read",
                    )
                    if pack.pack_id == "tasks"
                    else ("kitchen.write", "kitchen.read")
                    if pack.pack_id == "kitchen"
                    else ("network.read",)
                    if pack.pack_id == "network"
                    else ("homelab.service.restart",),
                ),
                cards=pack.cards,
            )
        )
        manager.install(
            pack.pack_id,
            frozenset(manager._bundles[pack.pack_id].manifest.permissions),
        )
        manager.enable(pack.pack_id)
    return manager


def test_cli_routes_task_before_food_keyword() -> None:
    domain, card = _domain_and_action(
        "Create a task to buy cat food.", manager_with_reference_cards()
    )

    assert domain == "tasks"
    assert card.action.action_id == "tasks.create"
    assert card.action.arguments == {"title": "buy cat food."}


def test_cli_prepares_grocery_action_card_arguments() -> None:
    domain, card = _domain_and_action("Add rice to groceries.", manager_with_reference_cards())

    assert domain == "kitchen"
    assert card.action.action_id == "kitchen.groceries.add"
    assert card.action.arguments == {"item": "rice"}


def test_cli_retrieves_read_cards() -> None:
    manager = manager_with_reference_cards()

    _, groceries = _domain_and_action("What's on my grocery list?", manager)
    _, tasks = _domain_and_action("Show my tasks.", manager)

    assert groceries.action.action_id == "kitchen.groceries.list"
    assert tasks.action.action_id == "tasks.list"


def test_local_identity_bootstrap_does_not_reactivate_revoked_membership():
    class Connection:
        def __init__(self) -> None:
            self.queries: list[tuple[str, tuple[object, ...]]] = []

        def execute(self, query: str, params: tuple[object, ...] = ()) -> None:
            self.queries.append((query, params))

        def commit(self) -> None:
            pass

    connection = Connection()
    _ensure_local_identity(
        connection,
        Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    membership_query = next(
        query for query, _ in connection.queries if "INSERT INTO space_memberships" in query
    )
    assert "DO NOTHING" in membership_query
    assert "active = TRUE" not in membership_query.split("DO NOTHING")[1]


def test_cli_help_is_available_without_runtime_configuration(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--help"])

    with pytest.raises(SystemExit) as raised:
        cli.main()

    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "--once REQUEST" in output
    assert "--no-banner" in output


def test_cli_env_file_loads_aegis_settings_without_overriding_shell(monkeypatch, tmp_path):
    from aegis import cli

    env_file = tmp_path / "aegis.env"
    env_file.write_text(
        "AEGIS_DATABASE_URL=postgresql://file\nAEGIS_PRINCIPAL_ID=file-user\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AEGIS_DATABASE_URL", "postgresql://shell")
    monkeypatch.delenv("AEGIS_PRINCIPAL_ID", raising=False)

    cli._load_env_file(str(env_file))

    assert cli.os.environ["AEGIS_DATABASE_URL"] == "postgresql://shell"
    assert cli.os.environ["AEGIS_PRINCIPAL_ID"] == "file-user"


def test_cli_env_file_rejects_shell_syntax_and_non_aegis_keys(tmp_path):
    from aegis import cli

    env_file = tmp_path / "aegis.env"
    env_file.write_text("export AEGIS_DATABASE_URL=secret\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid env file key"):
        cli._load_env_file(str(env_file))


def test_cli_env_file_json_failure_is_machine_safe(monkeypatch, capsys, tmp_path):
    from aegis import cli

    env_file = tmp_path / "aegis.env"
    env_file.write_text("not-a-setting\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["aegis", "--check", "--json", "--env-file", str(env_file)])

    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "code": "configuration_invalid",
        "error": "configuration file is invalid",
        "state": "failed",
    }


def test_cli_once_routes_through_handle(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--once", "Show my tasks."])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(cli, "handle", lambda utterance, principal: f"handled: {utterance}")

    cli.main()

    assert capsys.readouterr().out == "handled: Show my tasks.\n"


def test_cli_once_returns_failure_status_for_handled_error(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--once", "Show my tasks."])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )

    def fail(_utterance, _principal):
        raise ValueError("request is ambiguous")

    monkeypatch.setattr(cli, "handle", fail)

    assert cli.main() == 1
    assert capsys.readouterr().out == "Not completed — request is ambiguous\n"


def test_cli_once_json_serializes_canonical_result(monkeypatch, capsys):
    from aegis import cli

    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.COMPLETED,
        message="task list read",
        evidence={"canonical_tasks": []},
        correlation_id=uuid4(),
    )
    monkeypatch.setattr("sys.argv", ["aegis", "--once", "Show my tasks", "--json"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(cli, "run_interaction", lambda *_: result)

    assert cli.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "completed"
    assert payload["objective_id"] == str(result.objective_id)
    assert payload["evidence"] == {"canonical_tasks": []}


def test_cli_once_json_returns_failure_for_non_completed_result(monkeypatch, capsys):
    from aegis import cli

    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.BLOCKED,
        message="authorization denied",
        correlation_id=uuid4(),
    )
    monkeypatch.setattr("sys.argv", ["aegis", "--once", "do it", "--json"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(cli, "run_interaction", lambda *_: result)

    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out)["state"] == "blocked"


def test_cli_once_json_returns_stable_error_for_request_failure(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--once", "do it", "--json"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(
        cli,
        "run_interaction",
        lambda *_: (_ for _ in ()).throw(RuntimeError("secret implementation detail")),
    )

    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "code": "request_unavailable",
        "error": "request unavailable",
        "state": "failed",
    }


def test_cli_once_json_returns_stable_error_for_denied_request(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--once", "private read", "--json"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(
        cli,
        "run_interaction",
        lambda *_: (_ for _ in ()).throw(PermissionError("private Vault")),
    )

    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "code": "request_denied",
        "error": "request denied",
        "state": "failed",
    }


def test_cli_once_json_returns_stable_error_for_identity_failure(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--once", "do it", "--json"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: (_ for _ in ()).throw(RuntimeError("secret identity detail")),
    )

    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "code": "identity_unavailable",
        "error": "identity unavailable",
        "state": "failed",
    }


def test_cli_json_requires_check_or_once(monkeypatch):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--json"])
    with pytest.raises(SystemExit) as raised:
        cli.main()

    assert raised.value.code == 2


def test_cli_rejects_invalid_browser_port(monkeypatch):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--web", "--port", "70000"])
    with pytest.raises(SystemExit) as raised:
        cli.main()

    assert raised.value.code == 2


def test_cli_web_reports_bootstrap_failure(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--web", "--port", "8081"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(
        cli,
        "_prepare_local_web_runtime",
        lambda _principal: (_ for _ in ()).throw(RuntimeError("missing database")),
    )

    assert cli.main() == 1
    assert capsys.readouterr().out == "Not completed — unable to start browser: missing database\n"


def test_cli_check_reports_missing_required_configuration(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--check"])
    for name in ("AEGIS_DATABASE_URL", "AEGIS_OLLAMA_URL"):
        monkeypatch.delenv(name, raising=False)

    assert cli.main() == 1
    output = capsys.readouterr().out
    assert "AEGIS runtime: NOT READY" in output
    assert "postgres: FAIL (required) — set AEGIS_DATABASE_URL" in output
    assert "ollama: FAIL (required)" in output
    assert "openclaw: OK (optional)" in output


def test_cli_check_json_is_machine_readable(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--check", "--json"])
    monkeypatch.delenv("AEGIS_DATABASE_URL", raising=False)
    monkeypatch.delenv("AEGIS_OLLAMA_URL", raising=False)

    assert cli.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ready"] is False
    assert {item["name"] for item in report["components"]} == {
        "postgres",
        "ollama",
        "openclaw",
        "identity",
    }


def test_cli_check_rejects_missing_ollama_model(monkeypatch, capsys):
    from aegis import cli

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"models": [{"name": "another-model"}]}'

    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr("sys.argv", ["aegis", "--check"])
    monkeypatch.delenv("AEGIS_DATABASE_URL", raising=False)
    monkeypatch.setenv("AEGIS_OLLAMA_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("AEGIS_OLLAMA_MODEL", "qwen3:8b")

    assert cli.main() == 1
    output = capsys.readouterr().out
    assert "ollama: FAIL (required)" in output
    assert "ollama pull qwen3:8b" in output


def test_cli_check_explains_unreachable_ollama_endpoint(monkeypatch, capsys):
    from aegis import cli

    def unavailable(*_args, **_kwargs):
        raise cli.urllib.error.URLError("connection refused")

    monkeypatch.setattr(cli.urllib.request, "urlopen", unavailable)
    monkeypatch.setattr("sys.argv", ["aegis", "--check"])
    monkeypatch.delenv("AEGIS_DATABASE_URL", raising=False)
    monkeypatch.setenv("AEGIS_OLLAMA_URL", "http://127.0.0.1:11434")

    assert cli.main() == 1
    output = capsys.readouterr().out
    assert (
        "API unavailable at http://127.0.0.1:11434: URLError; "
        "check `curl http://127.0.0.1:11434/api/tags`" in output
    )


def test_safe_endpoint_does_not_render_url_credentials():
    from aegis.cli import _safe_endpoint

    assert _safe_endpoint("http://secret:password@example.test:11434/api") == (
        "http://example.test:11434/api"
    )


def test_browser_app_uses_core_callbacks_for_state_and_messages():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    seen: list[tuple[str, str]] = []
    app = BrowserApp(
        principal,
        lambda utterance, current, _correlation_id: (
            seen.append((utterance, current.id)) or "canonical answer"
        ),
        lambda current: {"nodes": [{"id": "tasks", "label": "Tasks", "detail": current.id}]},
    )

    status, content_type, payload = app.dispatch("GET", "/api/constellation")
    assert status == 200
    assert content_type == "application/json"
    assert json.loads(payload)["nodes"][0]["detail"] == "alice"

    status, _, payload = app.dispatch(
        "POST",
        "/api/message",
        b'{"utterance":"Show my tasks.","correlation_id":"00000000-0000-4000-8000-000000000001"}',
    )
    assert status == 200
    assert json.loads(payload) == {
        "message": "canonical answer",
        "correlation_id": "00000000-0000-4000-8000-000000000001",
    }
    assert seen == [("Show my tasks.", "alice")]


def test_browser_app_rejects_empty_messages_and_unknown_routes():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(principal, lambda *_: "unused", lambda _: {"nodes": []})

    status, _, payload = app.dispatch("POST", "/api/message", b'{"utterance":" "}')
    assert status == 400
    assert "non-empty" in json.loads(payload)["error"]

    status, _, _ = app.dispatch("GET", "/missing")
    assert status == 404


def test_browser_app_fails_closed_when_state_is_not_authorized():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(principal, lambda *_: "unused", lambda _: _deny())

    status, _, payload = app.dispatch("GET", "/api/constellation")
    assert status == 403
    assert json.loads(payload) == {
        "code": "state_access_denied",
        "error": "state access denied",
    }


def test_browser_app_rejects_malformed_constellation_projection():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(principal, lambda *_: "unused", lambda _: {"nodes": [{"id": "missing label"}]})

    status, _, payload = app.dispatch("GET", "/api/constellation")

    assert status == 503
    assert json.loads(payload) == {"code": "state_unavailable", "error": "state unavailable"}


def test_browser_api_resolves_identity_for_each_request():
    principals = iter(
        (
            Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
            Principal(id="bob", vault_id="bob-vault", space_ids=("apartment",)),
        )
    )
    seen: list[str] = []
    app = BrowserApp(
        lambda: next(principals),
        lambda _utterance, current, _correlation_id: seen.append(current.id) or "answer",
        lambda current: {"nodes": [{"id": current.id, "label": current.id}]},
    )

    app.dispatch("GET", "/api/constellation")
    app.dispatch("POST", "/api/message", b'{"utterance":"show tasks"}')

    assert seen == ["bob"]


def test_browser_api_rejects_unavailable_identity():
    app = BrowserApp(
        lambda: (_ for _ in ()).throw(ValueError("malformed claims")),
        lambda *_: "unreachable",
        lambda _: {"nodes": []},
    )

    status, _, payload = app.dispatch("GET", "/api/constellation")
    assert status == 401
    assert json.loads(payload) == {
        "code": "identity_unavailable",
        "error": "identity unavailable",
    }


def test_browser_reuses_correlation_id_for_retry_safe_delivery():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    seen: list[str] = []
    app = BrowserApp(
        principal,
        lambda _utterance, _principal, correlation_id: (
            seen.append(str(correlation_id)) or {"message": "completed", "state": "completed"}
        ),
        lambda _: {"nodes": []},
    )
    body = (
        b'{"utterance":"Add rice to groceries.",'
        b'"correlation_id":"00000000-0000-4000-8000-000000000007"}'
    )

    first_status, _, first_payload = app.dispatch("POST", "/api/message", body)
    second_status, _, second_payload = app.dispatch("POST", "/api/message", body)

    assert first_status == second_status == 200
    assert seen == [
        "00000000-0000-4000-8000-000000000007",
        "00000000-0000-4000-8000-000000000007",
    ]
    assert (
        json.loads(first_payload)["correlation_id"] == json.loads(second_payload)["correlation_id"]
    )


def test_browser_rejects_malformed_correlation_id_before_core():
    called = False

    def interaction(*_args):
        nonlocal called
        called = True
        return "unreachable"

    app = BrowserApp(
        Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
        interaction,
        lambda _: {"nodes": []},
    )
    status, _, payload = app.dispatch(
        "POST", "/api/message", b'{"utterance":"show tasks","correlation_id":"bad"}'
    )

    assert status == 400
    assert "badly formed" in json.loads(payload)["error"]
    assert called is False


def test_browser_rejects_undocumented_request_fields_before_core():
    called = False

    def interaction(*_args):
        nonlocal called
        called = True
        return "unreachable"

    app = BrowserApp(
        Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
        interaction,
        lambda _: {"nodes": []},
    )
    status, _, payload = app.dispatch(
        "POST", "/api/message", b'{"utterance":"show tasks","private_debug":true}'
    )

    assert status == 400
    assert json.loads(payload) == {
        "code": "invalid_request",
        "error": "request contains undocumented fields",
    }
    assert called is False


def test_browser_rejects_undocumented_interaction_fields():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: {"message": "answer", "private_debug": "secret"},
        lambda _: {"nodes": []},
    )

    status, _, payload = app.dispatch("POST", "/api/message", b'{"utterance":"show tasks"}')

    assert status == 503
    assert json.loads(payload) == {
        "code": "request_unavailable",
        "error": "request unavailable",
    }


def test_browser_health_uses_structured_readiness_without_identity():
    from aegis.health import ComponentHealth, HealthReport

    app = BrowserApp(
        lambda: (_ for _ in ()).throw(ValueError("identity unavailable")),
        lambda *_: "unreachable",
        lambda _: {"nodes": []},
        lambda: HealthReport(
            healthy=False,
            ready=False,
            components=(
                ComponentHealth(
                    name="postgres", healthy=False, required=True, detail="unavailable"
                ),
            ),
        ),
    )

    status, _, payload = app.dispatch("GET", "/api/health")
    assert status == 200
    assert json.loads(payload)["ready"] is False


def test_browser_health_provider_failure_is_generic_and_recoverable():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: "unreachable",
        lambda _: {"nodes": []},
        lambda: (_ for _ in ()).throw(RuntimeError("database password")),
    )

    status, _, payload = app.dispatch("GET", "/api/health")

    assert status == 503
    assert json.loads(payload) == {
        "code": "health_unavailable",
        "error": "runtime status unavailable",
    }


def test_browser_rejects_oversized_request_body():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(principal, lambda *_: "unreachable", lambda _: {"nodes": []})

    status, _, payload = app.dispatch("POST", "/api/message", b"x" * 20_001)
    assert status == 413
    assert json.loads(payload) == {
        "code": "request_too_large",
        "error": "request too large",
    }


def test_browser_surface_has_transcript_and_duplicate_submission_guard():
    from aegis.web import _INDEX_HTML

    assert 'id="conversation"' in _INDEX_HTML
    assert 'id="health-details"' in _INDEX_HTML
    assert "component.detail" in _INDEX_HTML
    assert "send.disabled = true" in _INDEX_HTML
    assert "input.disabled = true" in _INDEX_HTML
    assert "conversation.append(assistantLine)" in _INDEX_HTML
    assert "renderDetailValue(details[node.id])" in _INDEX_HTML
    assert "textContent = key" in _INDEX_HTML
    assert "retryableCodes.has(result.code)" in _INDEX_HTML
    assert "new AbortController()" in _INDEX_HTML
    assert "request_timeout" in _INDEX_HTML
    assert "outcome is unknown" in _INDEX_HTML
    assert 'id="refresh"' in _INDEX_HTML
    assert "refreshState()" in _INDEX_HTML
    assert "state_access_denied" in _INDEX_HTML
    assert "State refresh failed" in _INDEX_HTML
    assert "response.ok" in _INDEX_HTML
    assert "clearAuthorizedDisplays()" in _INDEX_HTML
    assert "conversation').replaceChildren()" in _INDEX_HTML


def test_browser_transport_disables_caching_and_referrer_disclosure():
    from pathlib import Path

    source = Path(__file__).parents[1].joinpath("src/aegis/web.py").read_text()

    assert 'self.send_header("Cache-Control", "no-store")' in source
    assert 'self.send_header("Pragma", "no-cache")' in source
    assert 'self.send_header("Referrer-Policy", "no-referrer")' in source


def test_task_read_fast_path_returns_membership_checked_canonical_tasks():
    task = Task(uuid4(), "apartment", "replace filter", "alice", status=TaskStatus.OPEN)

    class Store:
        def list(self, principal):
            assert principal.id == "alice"
            return (task,)

    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    intent = IntentFrame(
        principal=principal,
        utterance="Show my tasks",
        correlation_id=uuid4(),
    )

    result = TaskReadFastPath(Store()).resolve(intent)

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.evidence["canonical_tasks"] == [
        {"task_id": str(task.task_id), "title": "replace filter", "status": "open"}
    ]
    assert not TaskReadFastPath.matches("Create a task to replace filter")


def test_reference_pack_ui_metadata_is_optional_and_non_authoritative():
    bundles = reference_bundles()

    assert {bundle.manifest.ui.label for bundle in bundles if bundle.manifest.ui} == {
        "Tasks",
        "Kitchen",
        "Homelab",
        "Network",
    }
    assert all(bundle.manifest.permissions for bundle in bundles)


def test_browser_interaction_exposes_canonical_result_status(monkeypatch):
    from aegis import cli

    result = Result(
        objective_id=uuid4(),
        state=ObjectiveState.BLOCKED,
        message="authorization denied",
        correlation_id=uuid4(),
    )
    monkeypatch.setattr(cli, "run_interaction", lambda *_: result)

    payload = cli._browser_interaction(
        "Add rice to groceries.",
        Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )

    assert payload["state"] == "blocked"
    assert payload["message"] == "Not completed — authorization denied"
    assert payload["detail"] == "authorization denied"
    assert payload["objective_id"] == str(result.objective_id)


def test_constellation_state_keeps_current_pack_ui_metadata(monkeypatch):
    from aegis import cli
    from aegis.network import HomelabInventory
    from aegis.personal import PersonalState

    class Connection:
        def close(self):
            pass

    monkeypatch.setenv("AEGIS_DATABASE_URL", "postgresql://example")
    monkeypatch.setattr(cli.psycopg, "connect", lambda *_args, **_kwargs: Connection())
    monkeypatch.setattr(cli, "_apply_migrations", lambda _connection: None)
    monkeypatch.setattr(
        cli.PostgresHouseholdStore,
        "read_snapshot",
        lambda _store, _principal: {"groceries": ()},
    )
    monkeypatch.setattr(cli.PostgresTaskStore, "list", lambda _store, _principal: ())
    monkeypatch.setattr(cli.PostgresPackStore, "load", lambda _store: ())
    monkeypatch.setattr(
        cli.PostgresPersonalStateStore,
        "load_for_principal",
        lambda _store, _principal: PersonalState(),
    )
    monkeypatch.setattr(cli.PostgresFinanceSnapshotStore, "load", lambda _store, _owner: None)
    monkeypatch.setattr(
        cli.PostgresNetworkStore, "load", lambda _store, _principal: HomelabInventory()
    )
    monkeypatch.setattr(
        cli.PostgresHomelabStore,
        "load",
        lambda _store, _principal, _runtime: type("Pack", (), {"hosts": {}, "services": {}})(),
    )

    state = cli._constellation_state(
        Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    )

    assert [node["label"] for node in state["nodes"][:5]] == [
        "AEGIS",
        "Homelab",
        "Kitchen",
        "Network",
        "Tasks",
    ]


def test_constellation_graph_keeps_record_rows_in_detail_views(monkeypatch):
    from aegis import cli
    from aegis.network import HomelabInventory
    from aegis.personal import PersonalState, Project

    class Connection:
        def close(self):
            pass

    project = Project(uuid4(), "Private project", datetime(2026, 1, 1))
    task = Task(uuid4(), "apartment", "Private task", "alice")
    personal = PersonalState(projects={project.project_id: project})
    monkeypatch.setenv("AEGIS_DATABASE_URL", "postgresql://example")
    monkeypatch.setattr(cli.psycopg, "connect", lambda *_args, **_kwargs: Connection())
    monkeypatch.setattr(cli, "_apply_migrations", lambda _connection: None)
    monkeypatch.setattr(
        cli.PostgresHouseholdStore,
        "read_snapshot",
        lambda _store, _principal: {"groceries": ()},
    )
    monkeypatch.setattr(cli.PostgresTaskStore, "list", lambda _store, _principal: (task,))
    monkeypatch.setattr(cli.PostgresPackStore, "load", lambda _store: ())
    monkeypatch.setattr(
        cli.PostgresPersonalStateStore,
        "load_for_principal",
        lambda _store, _principal: personal,
    )
    monkeypatch.setattr(cli.PostgresFinanceSnapshotStore, "load", lambda _store, _owner: None)
    monkeypatch.setattr(
        cli.PostgresNetworkStore, "load", lambda _store, _principal: HomelabInventory()
    )
    monkeypatch.setattr(
        cli.PostgresHomelabStore,
        "load",
        lambda _store, _principal, _runtime: type("Pack", (), {"hosts": {}, "services": {}})(),
    )

    state = cli._constellation_state(
        Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    )
    node_ids = {node["id"] for node in state["nodes"]}

    assert f"project-{project.project_id}" not in node_ids
    assert f"task-{task.task_id}" not in node_ids
    assert state["details"]["domain-personal"]["projects"][0]["name"] == "Private project"
    assert state["details"]["pack-tasks"]["tasks"][0]["title"] == "Private task"


def _deny() -> dict[str, object]:
    raise PermissionError("revoked")
