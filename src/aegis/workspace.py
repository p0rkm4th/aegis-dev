"""Bounded, owner-scoped workspace operations for general-purpose artifacts.

This module is a provider boundary, not a shell escape hatch.  Callers receive
only a workspace directory and an allowlisted argv operation; Core still owns
intent, authorization, observation, and verification around consequential work.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from uuid import UUID, uuid5


class WorkspaceError(ValueError):
    """A requested workspace operation is outside its authority envelope."""


@dataclass(frozen=True)
class WorkspaceRun:
    correlation_id: UUID
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True)
class WorkspaceArtifact:
    """Verified inventory returned after a bounded artifact operation."""

    correlation_id: UUID
    files: tuple[str, ...]
    validated: bool
    validation_detail: str


def _bounded_output(value: str | bytes | None, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value[:limit].decode("utf-8", errors="replace")
    return value[:limit]


class ScopedWorkspace:
    """A confined workspace with bounded file and process operations."""

    def __init__(
        self,
        root: Path,
        *,
        allowed_commands: tuple[str, ...] = ("python3",),
        max_file_bytes: int = 200_000,
        max_output_bytes: int = 100_000,
        timeout_seconds: float = 10.0,
    ) -> None:
        if max_file_bytes <= 0 or max_output_bytes <= 0 or timeout_seconds <= 0:
            raise WorkspaceError("workspace bounds must be positive")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.allowed_commands = frozenset(allowed_commands)
        self.max_file_bytes = max_file_bytes
        self.max_output_bytes = max_output_bytes
        self.timeout_seconds = timeout_seconds

    def _path(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise WorkspaceError("workspace path escapes the scoped root")
        return candidate

    def write(self, relative: str, content: str) -> None:
        data = content.encode("utf-8")
        if len(data) > self.max_file_bytes:
            raise WorkspaceError("workspace file exceeds size bound")
        path = self._path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.is_symlink():
            raise WorkspaceError("workspace symlinks are not writable")
        path.write_bytes(data)

    def read(self, relative: str) -> str:
        path = self._path(relative)
        if not path.is_file() or path.is_symlink():
            raise WorkspaceError("workspace file is unavailable")
        if path.stat().st_size > self.max_file_bytes:
            raise WorkspaceError("workspace file exceeds size bound")
        return path.read_text(encoding="utf-8")

    def list_files(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                str(path.relative_to(self.root)) for path in self.root.rglob("*") if path.is_file()
            )
        )

    def run(self, argv: tuple[str, ...], correlation_id: UUID) -> WorkspaceRun:
        if (
            not argv
            or argv[0] not in self.allowed_commands
            or any(not isinstance(item, str) or not item for item in argv)
        ):
            raise WorkspaceError("command is not allowlisted")
        if shutil.which(argv[0]) is None:
            raise WorkspaceError("allowlisted command is unavailable")
        bubblewrap = shutil.which("bwrap")
        if bubblewrap is None:
            raise WorkspaceError("workspace sandbox is unavailable")
        command = [
            bubblewrap,
            "--die-with-parent",
            "--unshare-net",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/bin",
            "/bin",
            "--ro-bind",
            "/lib",
            "/lib",
            "--ro-bind",
            "/lib64",
            "/lib64",
            "--bind",
            str(self.root),
            "/workspace",
            "--chdir",
            "/workspace",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            "--",
            *argv,
        ]
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                env={},
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            return WorkspaceRun(
                correlation_id,
                124,
                _bounded_output(exc.stdout, self.max_output_bytes),
                _bounded_output(exc.stderr, self.max_output_bytes),
                True,
            )
        return WorkspaceRun(
            correlation_id,
            completed.returncode,
            completed.stdout[: self.max_output_bytes],
            completed.stderr[: self.max_output_bytes],
            timed_out,
        )

    def write_artifact(
        self,
        files: dict[str, str],
        correlation_id: UUID,
        validator: Callable[["ScopedWorkspace"], str | None],
    ) -> WorkspaceArtifact:
        """Materialize files, then require an independent bounded validator."""
        if not files or len(files) > 50:
            raise WorkspaceError("artifact file count is outside bounds")
        for relative, content in files.items():
            self.write(relative, content)
        detail = validator(self)
        if detail is not None:
            raise WorkspaceError(f"artifact validation failed: {detail}")
        return WorkspaceArtifact(
            correlation_id=correlation_id,
            files=self.list_files(),
            validated=True,
            validation_detail="bounded validator accepted artifact",
        )


class WorkspaceManager:
    """Create stable workspaces scoped by principal and objective identity."""

    _namespace = UUID("8b9f4b1f-2a83-4a0b-98cf-6f3d5f5c2e42")

    def __init__(
        self,
        root: Path,
        *,
        allowed_commands: tuple[str, ...] = ("python3",),
        max_file_bytes: int = 200_000,
        max_output_bytes: int = 100_000,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.allowed_commands = allowed_commands
        self.max_file_bytes = max_file_bytes
        self.max_output_bytes = max_output_bytes
        self.timeout_seconds = timeout_seconds

    def for_objective(self, principal_id: str, objective_id: UUID) -> ScopedWorkspace:
        if not principal_id or "/" in principal_id or "\\" in principal_id:
            raise WorkspaceError("principal identity is not a valid workspace scope")
        stable_id = uuid5(self._namespace, f"{principal_id}:{objective_id}")
        return ScopedWorkspace(
            self.root / principal_id / str(stable_id),
            allowed_commands=self.allowed_commands,
            max_file_bytes=self.max_file_bytes,
            max_output_bytes=self.max_output_bytes,
            timeout_seconds=self.timeout_seconds,
        )

    def list_for_principal(self, principal_id: str) -> tuple[dict[str, object], ...]:
        """Return bounded artifact metadata without exposing host paths."""
        if not principal_id or "/" in principal_id or "\\" in principal_id:
            raise WorkspaceError("principal identity is not a valid workspace scope")
        owner_root = self.root / principal_id
        if not owner_root.is_dir() or owner_root.is_symlink():
            return ()
        workspaces: list[dict[str, object]] = []
        for path in sorted(owner_root.iterdir()):
            if not path.is_dir() or path.is_symlink():
                continue
            files = tuple(
                sorted(
                    str(item.relative_to(path))
                    for item in path.rglob("*")
                    if item.is_file() and not item.is_symlink()
                )
            )
            workspaces.append({"workspace_id": path.name, "files": files})
        return tuple(workspaces)
