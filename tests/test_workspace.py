import shutil
import socket
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


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is not installed")
def test_workspace_network_namespace_cannot_reach_parent_loopback(tmp_path: Path) -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    workspace = ScopedWorkspace(tmp_path / "owner", allowed_commands=("python3",))
    script = (
        "import socket; s=socket.socket(); s.settimeout(1); "
        f"\ntry: s.connect(('127.0.0.1', {port})); print('connected') "
        "\nexcept OSError: print('blocked')"
    )
    try:
        result = workspace.run(("python3", "-c", script), uuid4())
    finally:
        listener.close()
    assert result.returncode == 0
    assert result.stdout.strip() == "blocked"


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is not installed")
@pytest.mark.skipif(shutil.which("prlimit") is None, reason="prlimit is not installed")
def test_workspace_run_applies_resource_limits_inside_sandbox(tmp_path: Path) -> None:
    workspace = ScopedWorkspace(
        tmp_path / "owner",
        max_cpu_seconds=3,
        max_memory_bytes=128 * 1024 * 1024,
        max_processes=17,
        max_open_files=29,
        max_file_bytes=4096,
    )
    script = (
        "import resource; names=('RLIMIT_CPU','RLIMIT_AS','RLIMIT_NPROC',"
        "'RLIMIT_NOFILE','RLIMIT_FSIZE'); print([resource.getrlimit(getattr(resource,n))[0] "
        "for n in names])"
    )
    result = workspace.run(("python3", "-c", script), uuid4())
    assert result.returncode == 0
    assert result.stdout.strip() == "[3, 134217728, 17, 29, 4096]"


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is not installed")
def test_workspace_run_timeout_cleans_up_the_sandbox_process_group(tmp_path: Path) -> None:
    workspace = ScopedWorkspace(tmp_path / "owner", timeout_seconds=0.2)
    result = workspace.run(("python3", "-c", "import time; time.sleep(5)"), uuid4())
    assert result.timed_out is True
    assert result.returncode == 124


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is not installed")
def test_workspace_run_stops_file_abuse(tmp_path: Path) -> None:
    workspace = ScopedWorkspace(tmp_path / "owner", max_workspace_files=2, max_workspace_bytes=1024)
    script = "from pathlib import Path; [Path(f'abuse-{i}').write_text('x') for i in range(20)]"
    result = workspace.run(("python3", "-c", script), uuid4())
    assert result.returncode == 122
    assert "file-count limit exceeded" in result.stderr


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


def test_workspace_artifact_requires_independent_validation(tmp_path: Path) -> None:
    workspace = ScopedWorkspace(tmp_path / "owner")
    result = workspace.write_artifact(
        {"index.html": "<!doctype html><html><body>HckrSlsh</body></html>"},
        uuid4(),
        lambda current: None if "<html" in current.read("index.html") else "missing html",
    )
    assert result.validated is True
    assert result.files == ("index.html",)
    with pytest.raises(WorkspaceError, match="validation failed"):
        workspace.write_artifact({"bad.txt": "not a site"}, uuid4(), lambda _: "bad output")


def test_workspace_artifact_supports_bounded_multi_file_composition(tmp_path: Path) -> None:
    workspace = ScopedWorkspace(tmp_path / "owner")
    files = {
        "index.html": "<!doctype html><link rel=stylesheet href=style.css>",
        "style.css": "body { color: navy; }",
    }
    result = workspace.write_artifact(
        files,
        uuid4(),
        lambda current: (
            None
            if all(current.read(path) == content for path, content in files.items())
            else "readback mismatch"
        ),
    )
    assert result.files == ("index.html", "style.css")
