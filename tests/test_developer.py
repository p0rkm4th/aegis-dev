from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aegis.developer import CodexInspectWorker, CodexModifyWorker, DeveloperWorkerError
from aegis.projects import RegisteredProject


def _project(tmp_path: Path) -> RegisteredProject:
    repository = tmp_path / "repo"
    (repository / "src").mkdir(parents=True)
    (repository / "src" / "main.py").write_text("print('safe')", encoding="utf-8")
    (repository / "secret.txt").write_text("not in scope", encoding="utf-8")
    return RegisteredProject(
        "aegis", "AEGIS", repository, "git@example/aegis.git", "main", ("src",), ("alice",)
    )


def test_codex_inspect_worker_uses_read_only_snapshot_and_sanitized_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        seen["command"] = command
        seen["env"] = kwargs["env"]
        output = Path(command[command.index("-o") + 1])
        snapshot = Path(command[command.index("--cd") + 1])
        seen["snapshot_files"] = sorted(
            str(item.relative_to(snapshot)) for item in snapshot.rglob("*") if item.is_file()
        )
        output.write_text("src/main.py contains the entry point.", encoding="utf-8")

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.setattr("aegis.developer.subprocess.run", fake_run)
    monkeypatch.setenv("AEGIS_DATABASE_URL", "postgresql://secret")
    result = CodexInspectWorker().inspect(_project(tmp_path), "Where is the entry point?")

    command = seen["command"]
    assert isinstance(command, list)
    assert "--sandbox" in command and command[command.index("--sandbox") + 1] == "read-only"
    assert "--ask-for-approval" in command
    assert seen["snapshot_files"] == ["src/main.py"]
    assert isinstance(seen["env"], dict)
    assert "AEGIS_DATABASE_URL" not in seen["env"]
    assert result["authority"] == "untrusted worker evidence; no repository mutation"


def test_codex_inspect_worker_rejects_empty_question(tmp_path: Path) -> None:
    with pytest.raises(DeveloperWorkerError, match="question is required"):
        CodexInspectWorker().inspect(_project(tmp_path), " ")


def test_codex_modify_requires_confirmation_without_running_worker(tmp_path: Path) -> None:
    project = _project(tmp_path)
    result = CodexModifyWorker().modify(project, "Fix the bug", False)
    assert result["state"] == "approval_required"
    assert (project.repository / "src" / "main.py").read_text() == "print('safe')"


def test_codex_modify_applies_only_allowlisted_change_and_runs_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        calls.append(command)

        class Completed:
            returncode = 0
            stdout = ""

        if command[:4] == ["git", "-C", str(project.repository), "status"]:
            return Completed()
        if "exec" in command:
            snapshot = Path(command[command.index("--cd") + 1])
            (snapshot / "src" / "main.py").write_text("print('changed')", encoding="utf-8")
            output = Path(command[command.index("-o") + 1])
            output.write_text("changed entry point", encoding="utf-8")
            return Completed()
        assert command == ["bash", "scripts/validate.sh"]
        return Completed()

    monkeypatch.setattr("aegis.developer.subprocess.run", fake_run)
    result = CodexModifyWorker().modify(project, "Change the entry point", True)

    assert result["state"] == "modified"
    assert result["changed_paths"] == ("src/main.py",)
    assert result["tests"] == "scripts/validate.sh passed"
    assert (project.repository / "src" / "main.py").read_text() == "print('changed')"
    assert any(command == ["bash", "scripts/validate.sh"] for command in calls)
