import shutil
import socket
import subprocess
import textwrap
import time
from pathlib import Path
from uuid import uuid4

import pytest

from aegis.workspace import ScopedWorkspace, WorkspaceError, WorkspaceManager


@pytest.fixture
def required_sandbox_tools() -> None:
    missing = [tool for tool in ("bwrap", "prlimit") if shutil.which(tool) is None]
    if missing:
        pytest.fail("sandbox attestation requires: " + ", ".join(missing))


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


def test_workspace_runs_allowlisted_command_without_network(
    tmp_path: Path, required_sandbox_tools: None
) -> None:
    workspace = ScopedWorkspace(tmp_path / "owner", allowed_commands=("python3",))
    workspace.write("index.html", "<h1>ok</h1>")
    result = workspace.run(
        ("python3", "-c", "from pathlib import Path; print(Path('index.html').read_text())"),
        uuid4(),
    )
    assert result.returncode == 0
    assert "<h1>ok</h1>" in result.stdout


def test_workspace_network_namespace_cannot_reach_parent_loopback(
    tmp_path: Path, required_sandbox_tools: None
) -> None:
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


def test_workspace_pid_namespace_hides_and_protects_host_sentinel(
    tmp_path: Path, required_sandbox_tools: None
) -> None:
    sentinel = subprocess.Popen(("python3", "-c", "import time; time.sleep(10)"))
    try:
        workspace = ScopedWorkspace(tmp_path / "owner", allowed_commands=("python3",))
        script = textwrap.dedent(
            f"""
            from pathlib import Path
            import os

            host_pid = {sentinel.pid}
            visible = Path('/proc') / str(host_pid)
            enumerated = any(
                entry.name == str(host_pid)
                for entry in Path('/proc').iterdir()
                if entry.name.isdigit()
            )
            print(
                'sentinel-hidden'
                if not visible.exists() and not enumerated
                else 'sentinel-visible'
            )
            try:
                os.kill(host_pid, 0)
            except ProcessLookupError:
                print('sentinel-signal-blocked')
            except PermissionError:
                print('sentinel-signal-denied')
            else:
                print('sentinel-signal-visible')
            """
        )
        result = workspace.run(("python3", "-c", script), uuid4())
        assert result.returncode == 0
        assert "sentinel-hidden" in result.stdout
        assert "sentinel-signal-blocked" in result.stdout
        assert sentinel.poll() is None
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=5)


def test_hostile_validator_is_disposable_and_cannot_reach_host_boundary(
    tmp_path: Path, required_sandbox_tools: None
) -> None:
    sentinel = tmp_path / "host-sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    host_home = tmp_path / "host-home"
    host_home.mkdir()
    (host_home / "secret").write_text("home-secret", encoding="utf-8")
    host_ssh = host_home / ".ssh"
    host_ssh.mkdir()
    (host_ssh / "id_ed25519").write_text("ssh-secret", encoding="utf-8")
    host_codex = tmp_path / "host-codex"
    host_codex.mkdir()
    (host_codex / "secret").write_text("codex-secret", encoding="utf-8")
    host_unexpected = tmp_path / "unexpected-host-file"
    host_repo = tmp_path / "host-repo"
    host_repo.mkdir()
    (host_repo / "tracked.txt").write_text("unchanged", encoding="utf-8")
    subprocess.run(("git", "init", "--quiet", str(host_repo)), check=True)
    before_git = subprocess.check_output(
        ("git", "-C", str(host_repo), "status", "--porcelain"), text=True
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    workspace = ScopedWorkspace(tmp_path / "disposable", allowed_commands=("python3",))
    script = textwrap.dedent(
        f"""
        from pathlib import Path
        import os
        import socket
        import subprocess
        import time

        host = Path({str(sentinel)!r})
        ssh = Path({str(host_ssh)!r})
        unexpected = Path({str(host_unexpected)!r})
        repo = Path({str(host_repo)!r})
        print('home-secret' if os.environ.get('HOME') and
              Path(os.environ['HOME']).exists() else 'home-hidden')
        print('ssh-secret' if ssh.exists() and any(ssh.iterdir()) else 'ssh-hidden')
        print('codex-secret' if os.environ.get('CODEX_HOME') else 'codex-hidden')
        Path('validator-mutation.txt').write_text('only disposable')
        try:
            host.write_text('changed')
        except OSError:
            print('host-file-blocked')
        try:
            unexpected.write_text('created')
        except OSError:
            print('unexpected-file-blocked')
        subprocess.run(['git', '-C', str(repo), 'commit', '-am', 'bad'], check=False)
        subprocess.run(['git', '-C', str(repo), 'push'], check=False)
        s = socket.socket()
        s.settimeout(0.2)
        try:
            s.connect(('127.0.0.1', {port}))
            print('network-open')
        except OSError:
            print('network-closed')
        child_code = (
            "from pathlib import Path; import os, sys, time; "
            "parent = int(sys.argv[1]); "
            "[time.sleep(0.01) for _ in iter("
            "lambda: 0 if os.getppid() == parent else 1, 1)]; "
            "Path('child-escaped.txt').write_text('escaped')"
        )
        child = subprocess.Popen(['python3', '-c', child_code, str(os.getpid())])
        Path('child.pid').write_text(str(child.pid))
        time.sleep(0.1)
        """
    )
    try:
        result = workspace.run(("python3", "-c", script), uuid4())
    finally:
        listener.close()
    assert result.returncode == 0
    assert "home-hidden" in result.stdout
    assert "ssh-hidden" in result.stdout
    assert "codex-hidden" in result.stdout
    assert "host-file-blocked" in result.stdout
    assert "unexpected-file-blocked" in result.stdout
    assert "home-secret" not in result.stdout
    assert "ssh-secret" not in result.stdout
    assert "codex-secret" not in result.stdout
    assert "network-closed" in result.stdout
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert not host_unexpected.exists()
    assert host_repo.joinpath("tracked.txt").read_text() == "unchanged"
    assert (
        subprocess.check_output(("git", "-C", str(host_repo), "status", "--porcelain"), text=True)
        == before_git
    )
    assert (workspace.root / "validator-mutation.txt").read_text() == "only disposable"
    time.sleep(0.2)
    assert not (workspace.root / "child-escaped.txt").exists()

    nonzero = workspace.run(
        (
            "python3",
            "-c",
            "from pathlib import Path; Path('nonzero.txt').write_text('kept'); raise SystemExit(7)",
        ),
        uuid4(),
    )
    zero = workspace.run(
        ("python3", "-c", "from pathlib import Path; Path('zero.txt').write_text('kept')"),
        uuid4(),
    )
    assert nonzero.returncode == 7
    assert zero.returncode == 0
    assert (workspace.root / "nonzero.txt").read_text() == "kept"
    assert (workspace.root / "zero.txt").read_text() == "kept"


def test_workspace_run_applies_resource_limits_inside_sandbox(
    tmp_path: Path, required_sandbox_tools: None
) -> None:
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


def test_workspace_run_timeout_cleans_up_the_sandbox_process_group(
    tmp_path: Path, required_sandbox_tools: None
) -> None:
    workspace = ScopedWorkspace(tmp_path / "owner", timeout_seconds=0.2)
    result = workspace.run(("python3", "-c", "import time; time.sleep(5)"), uuid4())
    assert result.timed_out is True
    assert result.returncode == 124


def test_workspace_run_stops_file_abuse(tmp_path: Path, required_sandbox_tools: None) -> None:
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
