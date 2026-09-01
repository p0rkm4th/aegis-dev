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
    assert "--init" in output


def test_cli_version_is_available_without_runtime_configuration(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--version"])
    with pytest.raises(SystemExit) as raised:
        cli.main()

    assert raised.value.code == 0
    assert capsys.readouterr().out.startswith("aegis ")


def test_cli_migration_source_fallback_executes_sql(monkeypatch):
    from aegis import cli

    class Connection:
        def __init__(self):
            self.statements = []
            self.committed = False

        def execute(self, statement):
            self.statements.append(statement)

        def commit(self):
            self.committed = True

    connection = Connection()
    monkeypatch.delenv("AEGIS_AUTO_MIGRATE", raising=False)

    cli._apply_migrations(connection)

    assert len(connection.statements) == 14
    assert connection.statements[0].startswith("-- PostgreSQL canonical schema")
    assert connection.committed is True


def test_cli_init_creates_private_non_overwriting_template(monkeypatch, capsys, tmp_path):
    from aegis import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["aegis", "--init"])

    assert cli.main() == 0
    target = tmp_path / ".env"
    assert target.is_file()
    assert target.stat().st_mode & 0o777 == 0o600
    assert "AEGIS_DATABASE_URL=" in target.read_text(encoding="utf-8")
    assert "Created private configuration template: .env" in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", ["aegis", "--init"])
    assert cli.main() == 1
    assert "configuration file already exists" in capsys.readouterr().out


def test_cli_init_refuses_symlink_target(monkeypatch, capsys, tmp_path):
    from aegis import cli

    monkeypatch.chdir(tmp_path)
    target = tmp_path / ".env"
    existing = tmp_path / "existing.env"
    existing.write_text("AEGIS_DATABASE_URL=untouched\n", encoding="utf-8")
    target.symlink_to(existing)
    monkeypatch.setattr("sys.argv", ["aegis", "--init"])

    assert cli.main() == 1
    assert "configuration file already exists" in capsys.readouterr().out
    assert existing.read_text(encoding="utf-8") == "AEGIS_DATABASE_URL=untouched\n"


def test_cli_init_reads_packaged_template(monkeypatch, capsys, tmp_path):
    from importlib.resources import files

    from aegis import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["aegis", "--init"])

    assert cli.main() == 0
    assert (tmp_path / ".env").read_text(encoding="utf-8") == files("aegis").joinpath(
        "aegis.env.example"
    ).read_text(encoding="utf-8")
    capsys.readouterr()


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


def test_cli_env_file_rejects_duplicate_keys(tmp_path):
    from aegis import cli

    env_file = tmp_path / "aegis.env"
    env_file.write_text(
        "AEGIS_DATABASE_URL=postgresql://first\nAEGIS_DATABASE_URL=postgresql://second\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate env file key"):
        cli._load_env_file(str(env_file))


def test_cli_auto_discovers_local_env_file(monkeypatch, capsys, tmp_path):
    from aegis import cli
    from aegis.health import HealthReport

    (tmp_path / ".env").write_text("AEGIS_PRINCIPAL_ID=discovered-user\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AEGIS_PRINCIPAL_ID", raising=False)
    monkeypatch.setattr("sys.argv", ["aegis", "--check", "--json"])

    def report():
        assert cli.os.environ["AEGIS_PRINCIPAL_ID"] == "discovered-user"
        return HealthReport(healthy=True, ready=True, components=())

    monkeypatch.setattr(cli, "_runtime_report", report)

    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["ready"] is True


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


def test_cli_once_hides_database_failure_details(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--once", "Show my tasks."])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(
        cli,
        "run_interaction",
        lambda *_: (_ for _ in ()).throw(cli.psycopg.OperationalError("password=private-secret")),
    )

    assert cli.main() == 1
    output = capsys.readouterr().out
    assert output == (
        "Not completed — request unavailable; run "
        "'./scripts/aegis --check' and verify AEGIS_DATABASE_URL\n"
    )
    assert "private-secret" not in output


def test_cli_once_hides_unexpected_request_failure_details(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--once", "Show my tasks."])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(
        cli,
        "handle",
        lambda *_: (_ for _ in ()).throw(Exception("provider private detail")),
    )

    assert cli.main() == 1
    output = capsys.readouterr().out
    assert output == (
        "Not completed — request unavailable; run "
        "'./scripts/aegis --check' and verify configured services\n"
    )
    assert "provider private detail" not in output


def test_interactive_cli_contains_database_failure(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--no-banner"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(
        cli,
        "handle",
        lambda *_: (_ for _ in ()).throw(cli.psycopg.OperationalError("password=private-secret")),
    )
    prompts = iter(["Show my tasks.", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(prompts))

    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "request unavailable; run './scripts/aegis --check'" in output
    assert "private-secret" not in output


def test_interactive_cli_hides_runtime_failure_details(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--no-banner"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(
        cli,
        "handle",
        lambda *_: (_ for _ in ()).throw(RuntimeError("gateway-password=private-secret")),
    )
    prompts = iter(["Show my tasks.", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(prompts))

    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "request unavailable; run './scripts/aegis --check'" in output
    assert "private-secret" not in output


def test_interactive_cli_hides_unexpected_request_failure_details(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--no-banner"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(
        cli,
        "handle",
        lambda *_: (_ for _ in ()).throw(Exception("provider private detail")),
    )
    prompts = iter(["Show my tasks.", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(prompts))

    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "request unavailable; run './scripts/aegis --check'" in output
    assert "provider private detail" not in output


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


def test_cli_once_json_contains_value_error_details(monkeypatch, capsys):
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
        lambda *_: (_ for _ in ()).throw(ValueError("private parser detail")),
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


def test_cli_human_identity_errors_are_generic(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--once", "Show my tasks."])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: (_ for _ in ()).throw(
            cli.psycopg.OperationalError("password=private-secret host=internal-db")
        ),
    )

    assert cli.main() == 1
    output = capsys.readouterr().out
    assert output == (
        "Not completed — identity unavailable; run './scripts/aegis --check' and "
        "verify identity configuration\n"
    )
    assert "private-secret" not in output
    assert "internal-db" not in output


def test_cli_human_unexpected_identity_errors_are_generic(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--once", "Show my tasks."])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: (_ for _ in ()).throw(Exception("provider private detail")),
    )

    assert cli.main() == 1
    output = capsys.readouterr().out
    assert output == (
        "Not completed — identity unavailable; run './scripts/aegis --check' and "
        "verify identity configuration\n"
    )
    assert "provider private detail" not in output


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
    monkeypatch.setattr(cli, "serve", lambda *_args: None)

    assert cli.main() == 0
    assert capsys.readouterr().out == (
        "AEGIS runtime is not ready; the browser will show diagnostics. "
        "Run './scripts/aegis --check' to see remediation.\n"
        "Starting AEGIS Constellation at http://127.0.0.1:8081\n"
    )


def test_cli_web_reports_port_conflict_with_remediation(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--web", "--port", "8081"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(cli, "_prepare_local_web_runtime", lambda _principal: None)
    monkeypatch.setattr(
        cli,
        "serve",
        lambda *_args: (_ for _ in ()).throw(OSError(cli.errno.EADDRINUSE, "address occupied")),
    )

    assert cli.main() == 1
    assert capsys.readouterr().out == (
        "Starting AEGIS Constellation at http://127.0.0.1:8081\n"
        "Not completed — browser port 8081 is already in use; choose another with --port\n"
    )


def test_cli_web_hides_database_failure_details(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--web", "--port", "8082"])
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )
    monkeypatch.setattr(
        cli,
        "_prepare_local_web_runtime",
        lambda _principal: (_ for _ in ()).throw(
            cli.psycopg.OperationalError("password=private-secret")
        ),
    )
    monkeypatch.setattr(cli, "serve", lambda *_args: None)

    assert cli.main() == 0
    output = capsys.readouterr().out
    assert output == (
        "AEGIS runtime is not ready; the browser will show diagnostics. "
        "Run './scripts/aegis --check' to see remediation.\n"
        "Starting AEGIS Constellation at http://127.0.0.1:8082\n"
    )
    assert "private-secret" not in output


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


def test_identity_health_rejects_incomplete_bearer_configuration(monkeypatch):
    from aegis import cli

    monkeypatch.setenv("AEGIS_KEYCLOAK_ACCESS_TOKEN", "token")
    monkeypatch.delenv("AEGIS_KEYCLOAK_ISSUER", raising=False)

    assert cli._identity_health() == (
        False,
        "incomplete bearer identity configuration; set both AEGIS_KEYCLOAK_ISSUER and "
        "AEGIS_KEYCLOAK_ACCESS_TOKEN",
    )


@pytest.mark.parametrize("configured", ["token", "issuer"])
def test_principal_does_not_fall_back_to_local_identity_for_incomplete_bearer_config(
    monkeypatch, configured
):
    from aegis import cli

    monkeypatch.delenv("AEGIS_KEYCLOAK_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("AEGIS_KEYCLOAK_ISSUER", raising=False)
    monkeypatch.setenv(
        "AEGIS_KEYCLOAK_ACCESS_TOKEN" if configured == "token" else "AEGIS_KEYCLOAK_ISSUER",
        "token" if configured == "token" else "https://keycloak.example/realms/aegis",
    )

    with pytest.raises(RuntimeError, match="incomplete bearer identity configuration"):
        cli._principal()


def test_identity_health_hides_bearer_mapping_failure(monkeypatch):
    from aegis import cli

    monkeypatch.setenv("AEGIS_KEYCLOAK_ACCESS_TOKEN", "token")
    monkeypatch.setenv("AEGIS_KEYCLOAK_ISSUER", "https://keycloak.example/realms/aegis")
    monkeypatch.setenv("AEGIS_DATABASE_URL", "postgresql://db")
    monkeypatch.setattr(
        cli,
        "_principal",
        lambda: (_ for _ in ()).throw(RuntimeError("password=private-secret")),
    )

    healthy, detail = cli._identity_health()

    assert healthy is False
    assert detail == (
        "bearer identity is unavailable; verify the token, Keycloak issuer, and "
        "canonical subject mapping"
    )
    assert "private-secret" not in detail


def test_postgres_health_reports_partial_canonical_schema(monkeypatch):
    from aegis import cli

    class Cursor:
        def fetchone(self):
            return ("objectives", None, "space_memberships")

    class Connection:
        def execute(self, *_args, **_kwargs):
            return Cursor()

        def close(self):
            pass

    monkeypatch.setattr(cli.psycopg, "connect", lambda *_args, **_kwargs: Connection())

    component = cli._postgres_health("postgresql://operator:secret@example.test/aegis")

    assert component.healthy is False
    assert component.detail == (
        "connection succeeded but the canonical schema is incomplete; apply migrations before "
        "starting AEGIS"
    )


def test_postgres_health_connection_failure_has_safe_remediation(monkeypatch):
    from aegis import cli

    monkeypatch.setattr(
        cli.psycopg,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            cli.psycopg.OperationalError("password=private-secret")
        ),
    )

    component = cli._postgres_health("postgresql://operator:secret@example.test/aegis")

    assert component.healthy is False
    assert component.detail == (
        "connection failed: OperationalError; verify AEGIS_DATABASE_URL and ensure PostgreSQL is "
        "running"
    )


def test_postgres_health_contains_unexpected_driver_failure(monkeypatch):
    from aegis import cli

    monkeypatch.setattr(
        cli.psycopg,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(Exception("password=private-secret")),
    )

    component = cli._postgres_health("postgresql://operator:secret@example.test/aegis")

    assert component.healthy is False
    assert component.detail == (
        "database health check failed; verify AEGIS_DATABASE_URL and ensure PostgreSQL is running"
    )
    assert "private-secret" not in component.detail


def test_ollama_health_contains_unexpected_client_failure(monkeypatch):
    from aegis import cli

    monkeypatch.setattr(
        cli.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(Exception("token=private-secret")),
    )

    component = cli._ollama_health("http://ollama.example:11434", "qwen3:8b")

    assert component.healthy is False
    assert component.detail == (
        "Ollama health check failed at http://ollama.example:11434; check "
        "`curl http://ollama.example:11434/api/tags`, start Ollama, or set "
        "AEGIS_OLLAMA_URL to its reachable address"
    )
    assert "private-secret" not in component.detail
    assert "private-secret" not in component.detail


def test_postgres_health_rejects_malformed_connection_configuration():
    from aegis import cli

    component = cli._postgres_health("not-a-postgresql-url")

    assert component.healthy is False
    assert component.detail == "invalid database configuration; verify AEGIS_DATABASE_URL"


def test_postgres_health_detects_generated_template_placeholder():
    from aegis import cli

    component = cli._postgres_health("postgresql://USER:PASSWORD@127.0.0.1:5432/DBNAME")

    assert component.healthy is False
    assert component.detail == (
        "configuration contains template placeholders; replace AEGIS_DATABASE_URL"
    )


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


def test_cli_check_rejects_invalid_ollama_endpoint(monkeypatch, capsys):
    from aegis import cli

    monkeypatch.setattr("sys.argv", ["aegis", "--check"])
    monkeypatch.delenv("AEGIS_DATABASE_URL", raising=False)
    monkeypatch.setenv("AEGIS_OLLAMA_URL", "ftp://ollama.example.invalid")

    assert cli.main() == 1
    output = capsys.readouterr().out
    assert "ollama: FAIL (required) — invalid URL" in output
    assert "set AEGIS_OLLAMA_URL to an http:// or https:// endpoint" in output


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


def test_browser_app_rejects_invalid_utf8_without_decoder_details():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(principal, lambda *_: "unreachable", lambda _: {"nodes": []})

    status, _, payload = app.dispatch("POST", "/api/message", b"\xff")

    assert status == 400
    assert json.loads(payload) == {
        "code": "invalid_request",
        "error": "invalid request",
    }


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


@pytest.mark.parametrize(
    "projection",
    [
        {
            "nodes": [
                {"id": "tasks", "label": "Tasks"},
                {"id": "tasks", "label": "Duplicate"},
            ]
        },
        {
            "nodes": [{"id": "tasks", "label": "Tasks"}],
            "edges": [{"source": "aegis", "target": "tasks"}],
        },
    ],
)
def test_browser_rejects_ambiguous_constellation_relationships(projection):
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(principal, lambda *_: "unused", lambda _: projection)

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


def test_browser_exposes_core_retryability_without_inventing_it():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: {
            "message": "Model unavailable; request can be retried",
            "state": "failed",
            "retryable": True,
        },
        lambda _: {"nodes": []},
    )

    status, _, payload = app.dispatch("POST", "/api/message", b'{"utterance":"show tasks"}')

    assert status == 200
    assert json.loads(payload)["retryable"] is True


def test_browser_request_status_is_a_bounded_canonical_projection():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    correlation_id = "00000000-0000-4000-8000-000000000008"
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        request_status=lambda _principal, _correlation: {
            "correlation_id": correlation_id,
            "objective_id": "00000000-0000-4000-8000-000000000009",
            "state": "completed",
            "message": "canonical readback verified",
        },
    )

    status, _, payload = app.dispatch("GET", f"/api/request-status?correlation_id={correlation_id}")

    assert status == 200
    assert json.loads(payload) == {
        "correlation_id": correlation_id,
        "objective_id": "00000000-0000-4000-8000-000000000009",
        "state": "completed",
        "message": "canonical readback verified",
    }


def test_browser_request_status_rejects_bad_query_and_contains_provider_failure():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        request_status=lambda *_: (_ for _ in ()).throw(Exception("private state")),
    )

    status, _, payload = app.dispatch("GET", "/api/request-status?correlation_id=bad")
    assert status == 400
    assert json.loads(payload) == {"code": "invalid_request", "error": "invalid request"}

    status, _, payload = app.dispatch(
        "GET",
        "/api/request-status?correlation_id=00000000-0000-4000-8000-000000000008",
    )
    assert status == 503
    assert json.loads(payload) == {"code": "state_unavailable", "error": "state unavailable"}


@pytest.mark.parametrize(
    "request_status",
    [
        lambda *_: (_ for _ in ()).throw(ValueError("provider implementation detail")),
        lambda *_: {"correlation_id": "00000000-0000-4000-8000-000000000008", "state": "bogus"},
    ],
)
def test_browser_request_status_contains_provider_validation_failures(request_status):
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": []},
        request_status=request_status,
    )

    status, _, payload = app.dispatch(
        "GET",
        "/api/request-status?correlation_id=00000000-0000-4000-8000-000000000008",
    )

    assert status == 503
    assert json.loads(payload) == {"code": "state_unavailable", "error": "state unavailable"}


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
    assert json.loads(payload) == {"code": "invalid_request", "error": "invalid request"}
    assert called is False


def test_browser_rejects_malformed_json_with_generic_error_before_core():
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
    status, _, payload = app.dispatch("POST", "/api/message", b'{"utterance":')

    assert status == 400
    assert json.loads(payload) == {"code": "invalid_request", "error": "invalid request"}
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


def test_browser_rejects_unknown_interaction_lifecycle_state():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: {"message": "answer", "state": "invented_state"},
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

    status, _, payload = app.dispatch("GET", "/api/ready")
    assert status == 503
    assert json.loads(payload)["ready"] is False


def test_browser_contains_unexpected_identity_provider_failure():
    app = BrowserApp(
        lambda: (_ for _ in ()).throw(Exception("identity database password")),
        lambda *_: "unreachable",
        lambda _: {"nodes": []},
    )

    status, _, payload = app.dispatch("GET", "/api/constellation")

    assert status == 401
    assert json.loads(payload) == {
        "code": "identity_unavailable",
        "error": "identity unavailable",
    }


def test_browser_ready_is_healthy_for_ready_report_without_identity():
    from aegis.health import HealthReport

    app = BrowserApp(
        lambda: (_ for _ in ()).throw(ValueError("identity unavailable")),
        lambda *_: "unreachable",
        lambda _: {"nodes": []},
        lambda: HealthReport(healthy=True, ready=True, components=()),
    )

    status, _, payload = app.dispatch("GET", "/api/ready")
    assert status == 200
    assert json.loads(payload)["ready"] is True


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


def test_browser_rejects_malformed_health_payload_without_server_exception():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: "unreachable",
        lambda _: {"nodes": []},
        lambda: ["not a health report"],
    )

    status, _, payload = app.dispatch("GET", "/api/health")

    assert status == 503
    assert json.loads(payload) == {
        "code": "health_unavailable",
        "error": "runtime status unavailable",
    }


def test_browser_contains_unexpected_service_exceptions_at_the_boundary():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))

    app = BrowserApp(
        principal,
        lambda *_: (_ for _ in ()).throw(Exception("database password")),
        lambda _: (_ for _ in ()).throw(Exception("private row")),
        lambda: (_ for _ in ()).throw(Exception("provider secret")),
    )

    status, _, payload = app.dispatch("GET", "/api/health")
    assert status == 503
    assert json.loads(payload) == {
        "code": "health_unavailable",
        "error": "runtime status unavailable",
    }

    status, _, payload = app.dispatch("GET", "/api/constellation")
    assert status == 503
    assert json.loads(payload) == {"code": "state_unavailable", "error": "state unavailable"}

    status, _, payload = app.dispatch("POST", "/api/message", b'{"utterance":"show tasks"}')
    assert status == 503
    assert json.loads(payload) == {
        "code": "request_unavailable",
        "error": "request unavailable",
    }


def test_browser_does_not_treat_core_value_error_as_client_validation():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: (_ for _ in ()).throw(ValueError("database password=private-secret")),
        lambda _: {"nodes": []},
    )

    status, _, payload = app.dispatch("POST", "/api/message", b'{"utterance":"show tasks"}')

    assert status == 503
    assert json.loads(payload) == {
        "code": "request_unavailable",
        "error": "request unavailable",
    }
    assert b"private-secret" not in payload


def test_browser_contains_non_json_provider_payloads():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": [], "details": {"bad": object()}},
    )

    status, _, payload = app.dispatch("GET", "/api/constellation")

    assert status == 503
    assert json.loads(payload) == {
        "code": "state_unavailable",
        "error": "state unavailable",
    }


def test_browser_bounds_serialized_response_size():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    app = BrowserApp(
        principal,
        lambda *_: "unused",
        lambda _: {"nodes": [], "details": {"oversized": "x" * 1_000_001}},
    )

    status, _, payload = app.dispatch("GET", "/api/constellation")

    assert status == 503
    assert json.loads(payload) == {
        "code": "response_unavailable",
        "error": "response unavailable",
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
    assert 'id="activity"' in _INDEX_HTML
    assert 'id="health-details"' in _INDEX_HTML
    assert 'id="node-filter"' in _INDEX_HTML
    assert "component.detail" in _INDEX_HTML
    assert "send.disabled = true" in _INDEX_HTML
    assert "input.disabled = true" in _INDEX_HTML
    assert "event.currentTarget.setAttribute('aria-busy', 'true')" in _INDEX_HTML
    assert "event.currentTarget.setAttribute('aria-busy', 'false')" in _INDEX_HTML
    assert "nodes.setAttribute('aria-busy', 'true')" in _INDEX_HTML
    assert "nodes.setAttribute('aria-busy', 'false')" in _INDEX_HTML
    assert "conversation.append(assistantLine)" in _INDEX_HTML
    assert "renderDetailValue(details[node.id])" in _INDEX_HTML
    assert "aria-pressed" in _INDEX_HTML
    assert 'role="region"' in _INDEX_HTML
    assert 'aria-label="Selected node details"' in _INDEX_HTML
    assert "card.setAttribute('aria-label'" in _INDEX_HTML
    assert "selectedNode" in _INDEX_HTML
    assert "const nodeCards = new Map()" in _INDEX_HTML
    assert "const selectNode = (node, card)" in _INDEX_HTML
    assert "Open relationship to" in _INDEX_HTML
    assert "target.focus(); target.click()" in _INDEX_HTML
    assert "function applyNodeFilter()" in _INDEX_HTML
    assert "renderedEdgeRows" in _INDEX_HTML
    assert (
        "selectedNode = null;\n  document.getElementById('detail').replaceChildren();"
        in _INDEX_HTML
    )
    assert "No canonical records available" in _INDEX_HTML
    assert "key.replaceAll('_', ' ')" in _INDEX_HTML
    assert "retryableCodes.has(result.code)" in _INDEX_HTML
    assert "const lifecycleLabels = Object.freeze" in _INDEX_HTML
    assert "function lifecycleLabel(state)" in _INDEX_HTML
    assert "lifecycleLabel(result.state)" in _INDEX_HTML
    assert "lifecycleLabel(status.state)" in _INDEX_HTML
    assert "Request status recovered" in _INDEX_HTML
    assert "new AbortController()" in _INDEX_HTML
    assert "request_timeout" in _INDEX_HTML
    assert "refreshRequestTimeoutMs = 10000" in _INDEX_HTML
    assert "async function fetchWithTimeout(resource, options = {})" in _INDEX_HTML
    assert "setTimeout(() => controller.abort(), refreshRequestTimeoutMs)" in _INDEX_HTML
    assert "fetchWithTimeout('/api/health')" in _INDEX_HTML
    assert "fetchWithTimeout('/api/constellation')" in _INDEX_HTML
    assert "outcome is unknown" in _INDEX_HTML
    assert 'id="refresh"' in _INDEX_HTML
    assert "refreshState()" in _INDEX_HTML
    assert "state_access_denied" in _INDEX_HTML
    assert "State refresh failed" in _INDEX_HTML
    assert "Runtime status unavailable (${code})." in _INDEX_HTML
    assert "await loadHealth();" in _INDEX_HTML
    assert "await loadState();" in _INDEX_HTML
    assert "response.ok" in _INDEX_HTML
    assert "clearAuthorizedDisplays()" in _INDEX_HTML
    assert "document.querySelector('#chat button').textContent = 'Send';" in _INDEX_HTML
    assert "Authorization lost; authorized state cleared." in _INDEX_HTML
    assert "conversation').replaceChildren()" in _INDEX_HTML
    assert "sessionStorage" in _INDEX_HTML
    assert "A previous request may still be in progress" in _INDEX_HTML
    assert "persistPendingRequest(utterance, correlationId)" in _INDEX_HTML
    assert "/api/request-status?correlation_id=" in _INDEX_HTML
    assert "recoverPendingRequest();" in _INDEX_HTML
    assert "maxRecoveryPolls = 60" in _INDEX_HTML
    assert "recoveryRequestTimeoutMs = 10000" in _INDEX_HTML
    assert "setTimeout(() => controller.abort(), recoveryRequestTimeoutMs)" in _INDEX_HTML
    assert "signal: controller.signal" in _INDEX_HTML
    assert "Status checks paused after five minutes." in _INDEX_HTML
    assert "Outcome unknown; checking canonical status. Retry remains explicit." in _INDEX_HTML
    assert "Status check unavailable; retry remains explicit." in _INDEX_HTML
    assert "inProgressStates.has(status.state)" in _INDEX_HTML
    assert "Retry remains explicit." in _INDEX_HTML
    assert "scheduleRecoveryPoll();" in _INDEX_HTML
    assert "recoveryPollMs = 5000" in _INDEX_HTML
    assert "if (result.state === 'completed') refreshState();" in _INDEX_HTML
    assert "if (status.state === 'completed') refreshState();" in _INDEX_HTML
    assert "loadState().catch(() => {})" not in _INDEX_HTML
    assert "Status: ${result.code || 'request_failed'}" in _INDEX_HTML
    assert "result.retryable === true" in _INDEX_HTML
    assert "status.retryable === true" in _INDEX_HTML


def test_browser_transport_disables_caching_and_referrer_disclosure():
    from pathlib import Path

    source = Path(__file__).parents[1].joinpath("src/aegis/web.py").read_text()

    assert 'self.send_header("Cache-Control", "no-store")' in source
    assert 'self.send_header("Pragma", "no-cache")' in source
    assert 'self.send_header("Referrer-Policy", "no-referrer")' in source


def test_browser_transport_ignores_expected_client_disconnect():
    from aegis.web import _write_response_payload

    class DisconnectedWriter:
        def write(self, _payload):
            raise BrokenPipeError

    _write_response_payload(DisconnectedWriter(), b"response")


def test_browser_transport_restricts_embedding_and_resource_execution():
    from pathlib import Path

    source = Path(__file__).parents[1].joinpath("src/aegis/web.py").read_text()

    assert 'self.send_header("X-Frame-Options", "DENY")' in source
    assert "Content-Security-Policy" in source
    assert "default-src 'none'" in source
    assert "connect-src 'self'" in source
    assert "frame-ancestors 'none'" in source
    assert 'self.send_header("Retry-After", str(_RETRY_AFTER_SECONDS))' in source


def test_browser_server_closes_cleanly_on_keyboard_interrupt(monkeypatch):
    from aegis import web

    instances = []

    class FakeServer:
        def __init__(self, _address, _handler):
            self.closed = False
            instances.append(self)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            self.closed = True

    monkeypatch.setattr(web, "ThreadingHTTPServer", FakeServer)

    web.serve("127.0.0.1", 18099, lambda: None, lambda *_: "unused", lambda _: {})

    assert instances[0].closed is True


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
