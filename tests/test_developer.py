from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aegis.developer import CodexInspectWorker, DeveloperWorkerError
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
