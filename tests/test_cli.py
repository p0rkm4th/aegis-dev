import json
from uuid import uuid4

import pytest

from aegis.cli import _domain_and_action, _ensure_local_identity
from aegis.contracts import ObjectiveState, Principal, Result
from aegis.pack_lifecycle import PackManager
from aegis.reference_packs import reference_bundles, reference_packs
from aegis.web import BrowserApp


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


def test_browser_app_uses_core_callbacks_for_state_and_messages():
    principal = Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",))
    seen: list[tuple[str, str]] = []
    app = BrowserApp(
        principal,
        lambda utterance, current: seen.append((utterance, current.id)) or "canonical answer",
        lambda current: {"nodes": [{"id": "tasks", "label": "Tasks", "detail": current.id}]},
    )

    status, content_type, payload = app.dispatch("GET", "/api/constellation")
    assert status == 200
    assert content_type == "application/json"
    assert json.loads(payload)["nodes"][0]["detail"] == "alice"

    status, _, payload = app.dispatch("POST", "/api/message", b'{"utterance":"Show my tasks."}')
    assert status == 200
    assert json.loads(payload) == {"message": "canonical answer"}
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
    assert json.loads(payload) == {"error": "state access denied"}


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
        lambda _utterance, current: seen.append(current.id) or "answer",
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
    assert json.loads(payload) == {"error": "identity unavailable"}


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
    monkeypatch.setattr(cli, "_run_interaction", lambda *_: result)

    payload = cli._browser_interaction(
        "Add rice to groceries.",
        Principal(id="alice", vault_id="alice-vault", space_ids=("apartment",)),
    )

    assert payload["state"] == "blocked"
    assert payload["message"] == "Not completed — authorization denied"
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
    monkeypatch.setattr(cli.PostgresPersonalStateStore, "load", lambda _store: PersonalState())
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


def _deny() -> dict[str, object]:
    raise PermissionError("revoked")
