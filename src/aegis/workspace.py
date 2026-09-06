"""Bounded, owner-scoped workspace operations for general-purpose artifacts.

This module is a provider boundary, not a shell escape hatch.  Callers receive
only a workspace directory and an allowlisted argv operation; Core still owns
intent, authorization, observation, and verification around consequential work.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import time
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
    """Executor-local artifact sanity evidence, not Core postcondition truth."""

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
        max_cpu_seconds: int = 5,
        max_memory_bytes: int = 256 * 1024 * 1024,
        max_processes: int = 32,
        max_open_files: int = 64,
        max_workspace_files: int = 50,
        max_workspace_bytes: int = 10_000_000,
    ) -> None:
        if (
            max_file_bytes <= 0
            or max_output_bytes <= 0
            or timeout_seconds <= 0
            or max_cpu_seconds <= 0
            or max_memory_bytes <= 0
            or max_processes <= 0
            or max_open_files <= 0
            or max_workspace_files <= 0
            or max_workspace_bytes <= 0
        ):
            raise WorkspaceError("workspace bounds must be positive")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.allowed_commands = frozenset(allowed_commands)
        self.max_file_bytes = max_file_bytes
        self.max_output_bytes = max_output_bytes
        self.timeout_seconds = timeout_seconds
        self.max_cpu_seconds = max_cpu_seconds
        self.max_memory_bytes = max_memory_bytes
        self.max_processes = max_processes
        self.max_open_files = max_open_files
        self.max_workspace_files = max_workspace_files
        self.max_workspace_bytes = max_workspace_bytes

    def _path(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise WorkspaceError("workspace path escapes the scoped root")
        return candidate

    def verify_expected_files(self, expected: dict[str, str]) -> tuple[bool, str]:
        """Independently inspect promised bytes in this already-scoped workspace."""
        if not expected or len(expected) > 50:
            return False, "expected workspace file set is outside bounds"
        for relative, expected_hash in expected.items():
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                return False, "invalid workspace postcondition"
            try:
                resolved = self._path(relative)
            except WorkspaceError as exc:
                return False, str(exc)
            candidate = self.root / relative
            # Check the lexical path as well as the resolved path: resolve()
            # alone would make a symlink look like an ordinary file.
            current = self.root
            for part in candidate.relative_to(self.root).parts:
                current = current / part
                if current.is_symlink():
                    return False, f"workspace path is a symlink: {relative}"
            if resolved != candidate or not candidate.is_file():
                return False, f"workspace file is unavailable: {relative}"
            try:
                actual_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
            except OSError:
                return False, f"workspace file cannot be read: {relative}"
            if actual_hash != expected_hash:
                return False, f"workspace hash mismatch: {relative}"
        return True, "independent workspace postcondition matched"

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

    def _workspace_usage(self) -> tuple[int, int]:
        file_count = 0
        byte_count = 0
        for path in self.root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            file_count += 1
            try:
                byte_count += path.stat().st_size
            except OSError:
                return self.max_workspace_files + 1, self.max_workspace_bytes + 1
        return file_count, byte_count

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
        prlimit = shutil.which("prlimit")
        if prlimit is None:
            raise WorkspaceError("workspace resource limiter is unavailable")
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
            prlimit,
            f"--cpu={self.max_cpu_seconds}",
            f"--as={self.max_memory_bytes}",
            f"--nproc={self.max_processes}",
            f"--nofile={self.max_open_files}",
            f"--fsize={self.max_file_bytes}",
            "--",
            *argv,
        ]

        def establish_process_group() -> None:
            """Give timeout cleanup a process-group boundary around bwrap."""

            os.setsid()

        timed_out = False
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={},
            preexec_fn=establish_process_group,
            start_new_session=False,
        )
        deadline = time.monotonic() + self.timeout_seconds
        resource_exceeded = False
        resource_detail = ""
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, self.timeout_seconds)
                try:
                    stdout, stderr = process.communicate(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    files, bytes_used = self._workspace_usage()
                    if files > self.max_workspace_files:
                        resource_exceeded = True
                        resource_detail = "workspace file-count limit exceeded"
                        break
                    if bytes_used > self.max_workspace_bytes:
                        resource_exceeded = True
                        resource_detail = "workspace byte limit exceeded"
                        break
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            # The bwrap process is the process-group leader; make timeout
            # cleanup include children that may have been forked by the tool.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()
            stdout, stderr = process.communicate()
            return WorkspaceRun(
                correlation_id,
                124,
                _bounded_output(stdout or exc.stdout, self.max_output_bytes),
                _bounded_output(stderr or exc.stderr, self.max_output_bytes),
                True,
            )
        if resource_exceeded:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()
            stdout, stderr = process.communicate()
            return WorkspaceRun(
                correlation_id,
                122,
                _bounded_output(stdout, self.max_output_bytes),
                _bounded_output(f"{stderr or ''}{resource_detail}\n", self.max_output_bytes),
                False,
            )
        files, bytes_used = self._workspace_usage()
        if files > self.max_workspace_files or bytes_used > self.max_workspace_bytes:
            detail = (
                "workspace file-count limit exceeded"
                if files > self.max_workspace_files
                else "workspace byte limit exceeded"
            )
            return WorkspaceRun(
                correlation_id,
                122,
                _bounded_output(stdout, self.max_output_bytes),
                _bounded_output(f"{detail}\n", self.max_output_bytes),
                False,
            )
        return WorkspaceRun(
            correlation_id,
            process.returncode,
            _bounded_output(stdout, self.max_output_bytes),
            _bounded_output(stderr, self.max_output_bytes),
            timed_out,
        )

    def write_artifact(
        self,
        files: dict[str, str],
        correlation_id: UUID,
        validator: Callable[["ScopedWorkspace"], str | None],
    ) -> WorkspaceArtifact:
        """Materialize files, then run an executor-local bounded sanity check."""
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

    def for_workspace_id(self, principal_id: str, workspace_id: str) -> ScopedWorkspace:
        """Resolve an inventory ID only within the current Principal's scope."""

        if not principal_id or "/" in principal_id or "\\" in principal_id:
            raise WorkspaceError("principal identity is not a valid workspace scope")
        try:
            requested = UUID(workspace_id)
        except (AttributeError, ValueError) as exc:
            raise WorkspaceError("workspace identity is invalid") from exc
        owner_root = self.root / principal_id
        if not owner_root.is_dir() or owner_root.is_symlink():
            raise WorkspaceError("workspace is not available for this Principal")
        candidate = owner_root / str(requested)
        if not candidate.is_dir() or candidate.is_symlink():
            raise WorkspaceError("workspace is not available for this Principal")
        return ScopedWorkspace(
            candidate,
            allowed_commands=self.allowed_commands,
            max_file_bytes=self.max_file_bytes,
            max_output_bytes=self.max_output_bytes,
            timeout_seconds=self.timeout_seconds,
        )


def workspace_expected_postcondition(
    principal_id: str, objective_id: UUID, files: dict[str, str]
) -> dict[str, object]:
    """Create a durable, content-minimal expectation before workspace mutation."""
    if not files or len(files) > 50:
        raise WorkspaceError("expected workspace file set is outside bounds")
    hashes: dict[str, str] = {}
    for relative, content in files.items():
        if not isinstance(relative, str) or not isinstance(content, str):
            raise WorkspaceError("workspace postcondition requires text files")
        hashes[relative] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return {
        "version": 1,
        "principal_id": principal_id,
        "objective_id": str(objective_id),
        "files": hashes,
    }
