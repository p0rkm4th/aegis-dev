from __future__ import annotations

import json
from pathlib import Path

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
