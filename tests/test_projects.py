from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aegis.contracts import Principal
from aegis.projects import ProjectRegistry, ProjectRegistryError
from aegis.web import BrowserApp


def _write_registry(path: Path, repository: Path, principal: str = "alice") -> None:
    path.write_text(
        json.dumps(
            {
                "projects": [
                    {
                        "project_id": "aegis",
                        "name": "AEGIS",
                        "repository": str(repository),
                        "remote": "git@github.com:p0rkm4th/aegis-dev.git",
                        "default_branch": "aegis-dev",
                        "allowed_paths": ["src", "tests"],
                        "principal_ids": [principal],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_project_registry_filters_principal_and_hides_repository_path(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    config = tmp_path / "projects.json"
    _write_registry(config, repository)

    visible = ProjectRegistry(config).for_principal("alice")
    assert visible[0] == {
        "project_id": "aegis",
        "name": "AEGIS",
        "remote": "git@github.com:p0rkm4th/aegis-dev.git",
        "default_branch": "aegis-dev",
        "allowed_paths": ("src", "tests"),
        "repository_state": "ready",
    }
    assert str(repository) not in json.dumps(visible)
    assert ProjectRegistry(config).for_principal("bob") == ()


def test_project_registry_reports_missing_repository_without_exposing_path(tmp_path: Path) -> None:
    config = tmp_path / "projects.json"
    _write_registry(config, tmp_path / "missing")

    project = ProjectRegistry(config).for_principal("alice")[0]
    assert project["repository_state"] == "unavailable"
    assert str(tmp_path / "missing") not in json.dumps(project)


def test_project_registry_inspects_only_registered_relative_text(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    (repository / "src").mkdir(parents=True)
    (repository / "src" / "main.py").write_text("print('safe')", encoding="utf-8")
    (repository / "secret.txt").write_text("private", encoding="utf-8")
    config = tmp_path / "projects.json"
    _write_registry(config, repository)
    registry = ProjectRegistry(config)

    inventory = registry.inspect_for_principal("alice", "aegis")
    assert inventory["files"] == ("src/main.py",)
    file_projection = registry.inspect_for_principal("alice", "aegis", "src/main.py")
    assert file_projection["content"] == "print('safe')"
    with pytest.raises(PermissionError):
        registry.inspect_for_principal("alice", "aegis", "secret.txt")
    with pytest.raises(ValueError):
        registry.inspect_for_principal("alice", "aegis", "../secret.txt")


def test_project_registry_rejects_unscoped_or_invalid_registration(tmp_path: Path) -> None:
    config = tmp_path / "projects.json"
    config.write_text(json.dumps({"projects": [{"project_id": "aegis"}]}), encoding="utf-8")

    with pytest.raises(ProjectRegistryError):
        ProjectRegistry(config).for_principal("alice")


def test_browser_projects_route_preserves_owner_boundary() -> None:
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_: "unused",
        lambda _: {"nodes": []},
        project_state=lambda principal: {"projects": [{"name": principal.id}]},
        session_token="session-secret",
    )

    status, _, payload = app.dispatch(
        "GET", "/api/projects", headers={"X-Aegis-Session": "session-secret"}
    )
    assert status == 200
    assert json.loads(payload) == {"projects": [{"name": "alice"}]}


def test_browser_project_inspection_route_is_bounded() -> None:
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_: "unused",
        lambda _: {"nodes": []},
        project_inspection=lambda current, project_id, path: {
            "project_id": project_id,
            "path": path,
            "principal": current.id,
        },
        session_token="session-secret",
    )
    status, _, payload = app.dispatch(
        "GET",
        "/api/projects/aegis/inspect?path=src%2Fmain.py",
        headers={"X-Aegis-Session": "session-secret"},
    )
    assert status == 200
    assert json.loads(payload) == {
        "project_id": "aegis",
        "path": "src/main.py",
        "principal": "alice",
    }


def test_browser_developer_inspection_route_is_explicit_and_bounded() -> None:
    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_: "unused",
        lambda _: {"nodes": []},
        developer_inspect=lambda current, project_id, question: {
            "project_id": project_id,
            "question": question,
            "principal": current.id,
            "authority": "untrusted worker evidence; no repository mutation",
        },
        session_token="session-secret",
    )
    status, _, payload = app.dispatch(
        "POST",
        "/api/projects/aegis/inspect",
        b'{"question":"Where is conversation persistence implemented?"}',
        headers={"X-Aegis-Session": "session-secret"},
    )
    assert status == 200
    assert json.loads(payload)["authority"] == "untrusted worker evidence; no repository mutation"


def test_browser_developer_modify_route_requires_explicit_confirmation() -> None:
    seen: list[tuple[str, bool]] = []

    def modify(_current: Any, project_id: str, objective: str, confirm: bool) -> dict[str, object]:
        assert objective == "Fix the bug"
        seen.append((project_id, confirm))
        return {"state": "approval_required"}

    app = BrowserApp(
        Principal(id="alice", vault_id="vault"),
        lambda *_: "unused",
        lambda _: {"nodes": []},
        developer_modify=modify,
        session_token="session-secret",
    )
    status, _, payload = app.dispatch(
        "POST",
        "/api/projects/aegis/modify",
        b'{"objective":"Fix the bug","confirm":false}',
        headers={"X-Aegis-Session": "session-secret"},
    )
    assert status == 200
    assert json.loads(payload) == {"state": "approval_required"}
    assert seen == [("aegis", False)]
