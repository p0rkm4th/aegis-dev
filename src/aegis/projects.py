"""Principal-scoped, read-only registered project context for the owner harness."""

from __future__ import annotations

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
