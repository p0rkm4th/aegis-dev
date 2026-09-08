"""Principal-scoped, read-only registered project context for the owner harness."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


class ProjectRegistryError(ValueError):
    """The configured project registry is invalid or unavailable."""


@dataclass(frozen=True)
class RegisteredProject:
    project_id: str
    name: str
    repository: Path
    remote: str | None
    default_branch: str
    allowed_paths: tuple[str, ...]
    principal_ids: tuple[str, ...]


class ProjectRegistry:
    """Load explicit project registrations without granting filesystem authority."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path

    def for_principal(self, principal_id: str) -> tuple[dict[str, Any], ...]:
        projects = self._load()
        result: list[dict[str, Any]] = []
        for project in projects:
            if principal_id not in project.principal_ids:
                continue
            repository_exists = project.repository.is_dir() and not project.repository.is_symlink()
            result.append(
                {
                    "project_id": project.project_id,
                    "name": project.name,
                    "remote": project.remote,
                    "default_branch": project.default_branch,
                    "allowed_paths": project.allowed_paths,
                    "repository_state": "ready" if repository_exists else "unavailable",
                }
            )
        return tuple(result)

    def inspect_for_principal(
        self, principal_id: str, project_id: str, relative_path: str | None = None
    ) -> dict[str, Any]:
        """Return bounded read-only inventory or text from an authorized project."""

        project = next(
            (
                item
                for item in self._load()
                if item.project_id == project_id and principal_id in item.principal_ids
            ),
            None,
        )
        if project is None:
            raise PermissionError("project is not registered for principal")
        if not project.repository.is_dir() or project.repository.is_symlink():
            raise ProjectRegistryError("registered project repository is unavailable")
        if relative_path is None or not relative_path.strip():
            files: list[str] = []
            for allowed in project.allowed_paths or (".",):
                root = project.repository / allowed
                if root.is_file() and not root.is_symlink():
                    files.append(str(root.relative_to(project.repository)))
                elif root.is_dir() and not root.is_symlink():
                    files.extend(
                        str(path.relative_to(project.repository))
                        for path in sorted(root.rglob("*"))
                        if path.is_file() and not path.is_symlink()
                    )
                if len(files) >= 500:
                    break
            return {
                "project_id": project.project_id,
                "name": project.name,
                "files": tuple(sorted(set(files))[:500]),
                "truncated": len(files) > 500,
            }
        safe_path = Path(relative_path)
        if safe_path.is_absolute() or ".." in safe_path.parts or "\\" in relative_path:
            raise ProjectRegistryError("project path is outside the registered scope")
        requested = project.repository / safe_path
        if not requested.is_file() or requested.is_symlink():
            raise ProjectRegistryError("project file is unavailable")
        allowed_roots = tuple(project.repository / item for item in project.allowed_paths)
        if allowed_roots and not any(
            requested == root or root in requested.parents for root in allowed_roots
        ):
            raise PermissionError("project path is outside the registered scope")
        try:
            content = requested.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ProjectRegistryError("project file is not readable text") from exc
        bounded = content[:100_000]
        return {
            "project_id": project.project_id,
            "name": project.name,
            "path": str(safe_path),
            "content": bounded,
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "truncated": len(content) > len(bounded),
        }

    def _load(self) -> tuple[RegisteredProject, ...]:
        try:
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return ()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProjectRegistryError("project registry is unavailable") from exc
        entries = raw.get("projects") if isinstance(raw, dict) else raw
        if not isinstance(entries, list) or len(entries) > 50:
            raise ProjectRegistryError("project registry must contain at most 50 projects")
        projects: list[RegisteredProject] = []
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ProjectRegistryError("project registration must be an object")
            project_id = entry.get("project_id")
            name = entry.get("name")
            repository = entry.get("repository")
            principal_ids = entry.get("principal_ids")
            if (
                not isinstance(project_id, str)
                or not _PROJECT_ID.fullmatch(project_id)
                or project_id in seen
                or not isinstance(name, str)
                or not name.strip()
                or len(name) > 200
                or not isinstance(repository, str)
                or not Path(repository).is_absolute()
                or not isinstance(principal_ids, list)
                or not principal_ids
                or any(not isinstance(item, str) or not item.strip() for item in principal_ids)
            ):
                raise ProjectRegistryError("project registration is invalid")
            default_branch = entry.get("default_branch", "main")
            remote = entry.get("remote")
            allowed_paths = entry.get("allowed_paths", [])
            if (
                not isinstance(default_branch, str)
                or not default_branch.strip()
                or len(default_branch) > 200
                or (remote is not None and (not isinstance(remote, str) or len(remote) > 2_000))
                or not isinstance(allowed_paths, list)
                or len(allowed_paths) > 100
                or any(
                    not isinstance(item, str) or not item.strip() or item.startswith("/")
                    for item in allowed_paths
                )
            ):
                raise ProjectRegistryError("project registration is invalid")
            seen.add(project_id)
            projects.append(
                RegisteredProject(
                    project_id,
                    name.strip(),
                    Path(repository).resolve(),
                    remote.strip() if isinstance(remote, str) and remote.strip() else None,
                    default_branch.strip(),
                    tuple(item.strip() for item in allowed_paths),
                    tuple(dict.fromkeys(item.strip() for item in principal_ids)),
                )
            )
        return tuple(projects)
