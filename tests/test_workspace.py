import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from aegis.workspace import ScopedWorkspace, WorkspaceError, WorkspaceManager


def test_workspace_writes_reads_and_lists_only_scoped_files(tmp_path: Path) -> None:
    workspace = ScopedWorkspace(tmp_path / "owner")
    workspace.write("site/index.html", "<h1>AEGIS</h1>")
    assert workspace.read("site/index.html") == "<h1>AEGIS</h1>"
    assert workspace.list_files() == ("site/index.html",)
    assert (tmp_path / "owner").stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("relative", ["../escape.txt", "/tmp/escape.txt"])
def test_workspace_rejects_host_path_escape(tmp_path: Path, relative: str) -> None:
    workspace = ScopedWorkspace(tmp_path / "owner")
    with pytest.raises(WorkspaceError):
        workspace.write(relative, "no")


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is not installed")
def test_workspace_runs_allowlisted_command_without_network(tmp_path: Path) -> None:
    workspace = ScopedWorkspace(tmp_path / "owner", allowed_commands=("python3",))
    workspace.write("index.html", "<h1>ok</h1>")
    result = workspace.run(
        ("python3", "-c", "from pathlib import Path; print(Path('index.html').read_text())"),
        uuid4(),
    )
    assert result.returncode == 0
    assert "<h1>ok</h1>" in result.stdout


def test_workspace_rejects_unallowlisted_command_and_symlink(tmp_path: Path) -> None:
    workspace = ScopedWorkspace(tmp_path / "owner", allowed_commands=("python3",))
    with pytest.raises(WorkspaceError):
        workspace.run(("sh", "-c", "echo escape"), uuid4())
    link = workspace.root / "secret"
    link.symlink_to("/etc/passwd")
    with pytest.raises(WorkspaceError):
        workspace.read("secret")


def test_workspace_manager_is_stable_and_principal_scoped(tmp_path: Path) -> None:
    objective_id = uuid4()
    manager = WorkspaceManager(tmp_path / "work")
    first = manager.for_objective("alice", objective_id)
    second = manager.for_objective("alice", objective_id)
    other = manager.for_objective("bob", objective_id)
    assert first.root == second.root
    assert first.root != other.root
    first.write("index.html", "ok")
    assert manager.list_for_principal("alice")[0]["files"] == ("index.html",)
    assert manager.list_for_principal("bob")[0]["files"] == ()
    with pytest.raises(WorkspaceError):
        manager.for_objective("alice/other", objective_id)
