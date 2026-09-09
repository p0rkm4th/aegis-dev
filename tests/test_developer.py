from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import aegis.developer as developer
from aegis.developer import (
    _PROPOSALS,
    MAX_CHANGED_FILE_BYTES,
    MAX_CHANGED_TOTAL_BYTES,
    MAX_DIFF_BYTES,
    MAX_MODIFY_FILES,
    MAX_SOURCE_FILE_BYTES,
    MAX_SOURCE_FILES,
    MAX_SOURCE_TOTAL_BYTES,
    CodexInspectWorker,
    CodexModifyWorker,
    DeveloperWorkerError,
)
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
    assert "sandbox_workspace_write.network_access=false" in command
    assert "--ignore-user-config" in command and "--ignore-rules" in command
    assert "--ask-for-approval" in command
    assert seen["snapshot_files"] == ["src/main.py"]
    assert isinstance(seen["env"], dict)
    assert "AEGIS_DATABASE_URL" not in seen["env"]
    assert result["authority"] == "untrusted worker evidence; no repository mutation"


def test_codex_inspect_worker_rejects_empty_question(tmp_path: Path) -> None:
    with pytest.raises(DeveloperWorkerError, match="question is required"):
        CodexInspectWorker().inspect(_project(tmp_path), " ")


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("MAX_SOURCE_FILES", 2, "file-count"),
        ("MAX_SOURCE_FILE_BYTES", 4, "size bound"),
        ("MAX_SOURCE_TOTAL_BYTES", 4, "byte bound"),
    ],
)
def test_allowlist_rejects_incomplete_source_snapshot_before_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    value: int,
    message: str,
) -> None:
    project = _project(tmp_path)
    for name, content in (("a.py", "1234"), ("b.py", "5678"), ("c.py", "90ab")):
        (project.repository / "src" / name).write_text(content, encoding="utf-8")
    monkeypatch.setattr(developer, setting, value)
    if setting == "MAX_SOURCE_TOTAL_BYTES":
        monkeypatch.setattr(developer, "MAX_SOURCE_FILE_BYTES", 100)
    destination = tmp_path / "snapshot"
    with pytest.raises(DeveloperWorkerError, match=message):
        CodexInspectWorker._copy_allowlist(project, destination)
    assert not destination.exists()


def test_changed_payload_accepts_exact_bounds_and_rejects_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(developer, "MAX_MODIFY_FILES", 2)
    monkeypatch.setattr(developer, "MAX_CHANGED_FILE_BYTES", 4)
    monkeypatch.setattr(developer, "MAX_CHANGED_TOTAL_BYTES", 8)
    monkeypatch.setattr(developer, "MAX_DIFF_BYTES", 4)
    CodexModifyWorker._validate_changed_payload({"a": b"1234", "b": b"5678"}, (), "1234")
    with pytest.raises(DeveloperWorkerError, match="too many files"):
        CodexModifyWorker._validate_changed_payload({"a": b"1", "b": b"2", "c": b"3"}, (), "1")
    with pytest.raises(DeveloperWorkerError, match="changed file exceeds"):
        CodexModifyWorker._validate_changed_payload({"a": b"12345"}, (), "1")
    monkeypatch.setattr(developer, "MAX_CHANGED_TOTAL_BYTES", 7)
    with pytest.raises(DeveloperWorkerError, match="too many bytes"):
        CodexModifyWorker._validate_changed_payload({"a": b"1234", "b": b"5678"}, (), "1")
    with pytest.raises(DeveloperWorkerError, match="diff exceeds"):
        CodexModifyWorker._validate_changed_payload({"a": b"1"}, (), "12345")


def test_diff_is_rejected_as_a_complete_oversized_representation() -> None:
    before = {"main.py": b"old\n"}
    after = {"main.py": b"new\n"}
    diff = CodexModifyWorker._diff(before, after, Path("."), ["main.py"])
    assert diff.startswith("--- a/main.py")
    assert len(diff.encode()) <= MAX_DIFF_BYTES
    assert MAX_SOURCE_FILES > MAX_MODIFY_FILES
    assert MAX_SOURCE_FILE_BYTES >= MAX_CHANGED_FILE_BYTES
    assert MAX_SOURCE_TOTAL_BYTES > MAX_CHANGED_TOTAL_BYTES


def test_codex_modify_requires_confirmation_without_running_worker(tmp_path: Path) -> None:
    project = _project(tmp_path)
    result = CodexModifyWorker().modify(project, "Fix the bug", False)
    assert result["state"] == "approval_required"
    assert (project.repository / "src" / "main.py").read_text() == "print('safe')"


def test_codex_modify_proposes_then_applies_only_exact_allowlisted_change(
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
        if command[:4] == ["git", "-C", str(project.repository), "rev-parse"]:
            Completed.stdout = "base-sha\n"
            return Completed()
        if "exec" in command:
            snapshot = Path(command[command.index("--cd") + 1])
            (snapshot / "src" / "main.py").write_text("print('changed')", encoding="utf-8")
            output = Path(command[command.index("-o") + 1])
            output.write_text("changed entry point", encoding="utf-8")
            return Completed()
        return Completed()

    monkeypatch.setattr("aegis.developer.subprocess.run", fake_run)
    monkeypatch.setattr(CodexModifyWorker, "_validate_candidate", lambda *args: None)
    worker = CodexModifyWorker()
    result = worker.modify(project, "Change the entry point", True)

    assert result["state"] == "approval_required"
    assert result["changed_paths"] == ("src/main.py",)
    assert result["authority"] == "candidate only; no repository mutation"
    assert (project.repository / "src" / "main.py").read_text() == "print('safe')"
    with pytest.raises(DeveloperWorkerError, match="digest"):
        CodexModifyWorker().apply_approved(project, str(result["proposal_id"]), "wrong-digest")
    assert worker.last_proposal is not None
    persisted = worker.last_proposal.record()
    _PROPOSALS.pop(str(result["proposal_id"]), None)
    applied = CodexModifyWorker().apply_approved(
        project,
        str(result["proposal_id"]),
        str(result["diff_digest"]),
        persisted_record=persisted,
    )
    assert applied["state"] == "modified"
    assert (project.repository / "src" / "main.py").read_text() == "print('changed')"
    assert all(command != ["bash", "scripts/validate.sh"] for command in calls)
