"""Canonical personal intelligence state with provenance and temporal queries."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4


class Provenance(StrEnum):
    EXPLICIT_USER = "explicit_user"
    OBSERVED = "observed"
    INFERRED = "inferred"
    IMPORTED = "imported"
    DERIVED = "derived"
    CORRECTED = "corrected"


@dataclass(frozen=True)
class Entity:
    entity_id: UUID
    canonical_name: str
    aliases: tuple[str, ...] = ()


@dataclass
class MemoryRecord:
    memory_id: UUID
    content: str
    occurred_at: datetime
    provenance: Provenance
    entity_ids: tuple[UUID, ...] = ()
    superseded_by: UUID | None = None


@dataclass(frozen=True)
class Project:
    project_id: UUID
    name: str
    created_at: datetime


@dataclass(frozen=True)
class Goal:
    goal_id: UUID
    project_id: UUID | None
    description: str
    created_at: datetime


@dataclass
class PersonalState:
    entities: dict[UUID, Entity] = field(default_factory=dict)
    memories: dict[UUID, MemoryRecord] = field(default_factory=dict)
    projects: dict[UUID, Project] = field(default_factory=dict)
    goals: dict[UUID, Goal] = field(default_factory=dict)

    schema_version = 1

    def add_entity(self, canonical_name: str, aliases: tuple[str, ...] = ()) -> Entity:
        if not canonical_name.strip():
            raise ValueError("entity name is required")
        entity = Entity(uuid4(), canonical_name, aliases)
        self.entities[entity.entity_id] = entity
        return entity

    def add_project(self, name: str, created_at: datetime) -> Project:
        if not name.strip():
            raise ValueError("project name is required")
        project = Project(uuid4(), name, created_at)
        self.projects[project.project_id] = project
        return project

    def add_goal(
        self, description: str, created_at: datetime, project_id: UUID | None = None
    ) -> Goal:
        if not description.strip():
            raise ValueError("goal description is required")
        if project_id is not None and project_id not in self.projects:
            raise ValueError("goal references an unknown project")
        goal = Goal(uuid4(), project_id, description, created_at)
        self.goals[goal.goal_id] = goal
        return goal

    def resolve_entity(self, reference: str) -> Entity | None:
        normalized = reference.casefold().strip()
        return next(
            (
                entity
                for entity in self.entities.values()
                if entity.canonical_name.casefold() == normalized
                or normalized in {alias.casefold() for alias in entity.aliases}
            ),
            None,
        )

    def add_memory(
        self,
        content: str,
        occurred_at: datetime,
        provenance: Provenance,
        entity_ids: tuple[UUID, ...] = (),
    ) -> MemoryRecord:
        if not content.strip():
            raise ValueError("memory content is required")
        if any(entity_id not in self.entities for entity_id in entity_ids):
            raise ValueError("memory references an unknown entity")
        memory = MemoryRecord(uuid4(), content, occurred_at, provenance, entity_ids)
        self.memories[memory.memory_id] = memory
        return memory

    def correct_memory(self, memory_id: UUID, content: str, corrected_at: datetime) -> MemoryRecord:
        original = self.memories.get(memory_id)
        if original is None or original.superseded_by is not None:
            raise ValueError("memory is missing or already superseded")
        corrected = self.add_memory(
            content, corrected_at, Provenance.CORRECTED, original.entity_ids
        )
        original.superseded_by = corrected.memory_id
        return corrected

    def memories_between(
        self, start: datetime, end: datetime, entity_id: UUID | None = None
    ) -> tuple[MemoryRecord, ...]:
        return tuple(
            sorted(
                (
                    memory
                    for memory in self.memories.values()
                    if memory.superseded_by is None
                    and start <= memory.occurred_at <= end
                    and (entity_id is None or entity_id in memory.entity_ids)
                ),
                key=lambda memory: memory.occurred_at,
            )
        )

    def to_json(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "entities": [
                {
                    "entity_id": str(entity.entity_id),
                    "canonical_name": entity.canonical_name,
                    "aliases": list(entity.aliases),
                }
                for entity in self.entities.values()
            ],
            "memories": [
                {
                    "memory_id": str(memory.memory_id),
                    "content": memory.content,
                    "occurred_at": memory.occurred_at.isoformat(),
                    "provenance": memory.provenance.value,
                    "entity_ids": [str(entity_id) for entity_id in memory.entity_ids],
                    "superseded_by": str(memory.superseded_by) if memory.superseded_by else None,
                }
                for memory in self.memories.values()
            ],
            "projects": [
                {
                    "project_id": str(project.project_id),
                    "name": project.name,
                    "created_at": project.created_at.isoformat(),
                }
                for project in self.projects.values()
            ],
            "goals": [
                {
                    "goal_id": str(goal.goal_id),
                    "project_id": str(goal.project_id) if goal.project_id else None,
                    "description": goal.description,
                    "created_at": goal.created_at.isoformat(),
                }
                for goal in self.goals.values()
            ],
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, encoded: str) -> PersonalState:
        try:
            payload = json.loads(encoded)
            if payload["schema_version"] != cls.schema_version:
                raise ValueError("unsupported personal state schema version")
            entities = {
                UUID(item["entity_id"]): Entity(
                    UUID(item["entity_id"]), item["canonical_name"], tuple(item["aliases"])
                )
                for item in payload["entities"]
            }
            memories = {
                UUID(item["memory_id"]): MemoryRecord(
                    UUID(item["memory_id"]),
                    item["content"],
                    datetime.fromisoformat(item["occurred_at"]),
                    Provenance(item["provenance"]),
                    tuple(UUID(entity_id) for entity_id in item["entity_ids"]),
                    UUID(item["superseded_by"]) if item["superseded_by"] else None,
                )
                for item in payload["memories"]
            }
            projects = {
                UUID(item["project_id"]): Project(
                    UUID(item["project_id"]),
                    item["name"],
                    datetime.fromisoformat(item["created_at"]),
                )
                for item in payload.get("projects", [])
            }
            goals = {
                UUID(item["goal_id"]): Goal(
                    UUID(item["goal_id"]),
                    UUID(item["project_id"]) if item["project_id"] else None,
                    item["description"],
                    datetime.fromisoformat(item["created_at"]),
                )
                for item in payload.get("goals", [])
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid personal state document") from exc
        if any(
            entity_id not in entities
            for memory in memories.values()
            for entity_id in memory.entity_ids
        ):
            raise ValueError("memory references an entity missing from state")
        if any(
            goal.project_id is not None and goal.project_id not in projects
            for goal in goals.values()
        ):
            raise ValueError("goal references a project missing from state")
        return cls(entities=entities, memories=memories, projects=projects, goals=goals)
