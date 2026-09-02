"""Canonical personal intelligence state with provenance and temporal queries."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol, cast
from uuid import UUID, uuid4

from .contracts import Context, IntentFrame, ObjectiveState, Principal, Result
from .embeddings import EmbeddingProvider, MemoryVectorIndex
from .utterance import is_mutation_request


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


class PersonalStateConnection(Protocol):
    def execute(self, query: str, params: tuple[object, ...] = ()) -> Any: ...

    def commit(self) -> None: ...


class PostgresPersonalStateStore:
    """Persist personal state under one canonical private Vault."""

    def __init__(self, connection: PersonalStateConnection, vault_id: str) -> None:
        if not vault_id:
            raise ValueError("personal state requires a Vault")
        self.connection = connection
        self.vault_id = vault_id

    def save(self, state: PersonalState) -> None:
        for entity in state.entities.values():
            self.connection.execute(
                """INSERT INTO personal_entities (id, vault_id, canonical_name, aliases)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET canonical_name = EXCLUDED.canonical_name,
                     aliases = EXCLUDED.aliases""",
                (
                    str(entity.entity_id),
                    self.vault_id,
                    entity.canonical_name,
                    json.dumps(entity.aliases),
                ),
            )
        for project in state.projects.values():
            self.connection.execute(
                """INSERT INTO personal_projects (id, vault_id, name, created_at)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name,
                     created_at = EXCLUDED.created_at""",
                (str(project.project_id), self.vault_id, project.name, project.created_at),
            )
        for goal in state.goals.values():
            self.connection.execute(
                """INSERT INTO personal_goals (id, vault_id, project_id, description, created_at)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET project_id = EXCLUDED.project_id,
                     description = EXCLUDED.description, created_at = EXCLUDED.created_at""",
                (
                    str(goal.goal_id),
                    self.vault_id,
                    str(goal.project_id) if goal.project_id else None,
                    goal.description,
                    goal.created_at,
                ),
            )
        for memory in state.memories.values():
            self.connection.execute(
                """INSERT INTO personal_memories
                   (id, vault_id, content, occurred_at, provenance, entity_ids, superseded_by)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content,
                     provenance = EXCLUDED.provenance, entity_ids = EXCLUDED.entity_ids,
                     superseded_by = EXCLUDED.superseded_by""",
                (
                    str(memory.memory_id),
                    self.vault_id,
                    memory.content,
                    memory.occurred_at,
                    memory.provenance.value,
                    json.dumps([str(entity_id) for entity_id in memory.entity_ids]),
                    None,
                ),
            )
        for memory in state.memories.values():
            if memory.superseded_by is not None:
                self.connection.execute(
                    "UPDATE personal_memories SET superseded_by = %s "
                    "WHERE id = %s AND vault_id = %s",
                    (str(memory.superseded_by), str(memory.memory_id), self.vault_id),
                )
        self.connection.commit()

    def load(self) -> PersonalState:
        entities: dict[UUID, Entity] = {}
        rows = self.connection.execute(
            "SELECT id, canonical_name, aliases FROM personal_entities WHERE vault_id = %s",
            (self.vault_id,),
        ).fetchall()
        for row in rows:
            aliases = row[2] if isinstance(row[2], list) else json.loads(str(row[2]))
            entity_id = UUID(str(row[0]))
            entities[entity_id] = Entity(entity_id, str(row[1]), tuple(str(a) for a in aliases))
        projects: dict[UUID, Project] = {}
        rows = self.connection.execute(
            "SELECT id, name, created_at FROM personal_projects WHERE vault_id = %s",
            (self.vault_id,),
        ).fetchall()
        for row in rows:
            project_id = UUID(str(row[0]))
            projects[project_id] = Project(project_id, str(row[1]), cast(datetime, row[2]))
        goals: dict[UUID, Goal] = {}
        rows = self.connection.execute(
            "SELECT id, project_id, description, created_at "
            "FROM personal_goals WHERE vault_id = %s",
            (self.vault_id,),
        ).fetchall()
        for row in rows:
            goal_id = UUID(str(row[0]))
            goals[goal_id] = Goal(
                goal_id,
                UUID(str(row[1])) if row[1] else None,
                str(row[2]),
                cast(datetime, row[3]),
            )
        memories: dict[UUID, MemoryRecord] = {}
        rows = self.connection.execute(
            """SELECT id, content, occurred_at, provenance, entity_ids, superseded_by
               FROM personal_memories WHERE vault_id = %s""",
            (self.vault_id,),
        ).fetchall()
        for row in rows:
            entity_ids = row[4] if isinstance(row[4], list) else json.loads(str(row[4]))
            memory_id = UUID(str(row[0]))
            memories[memory_id] = MemoryRecord(
                memory_id,
                str(row[1]),
                cast(datetime, row[2]),
                Provenance(str(row[3])),
                tuple(UUID(str(entity_id)) for entity_id in entity_ids),
                UUID(str(row[5])) if row[5] else None,
            )
        state = PersonalState(entities, memories, projects, goals)
        if any(
            goal.project_id is not None and goal.project_id not in projects
            for goal in goals.values()
        ):
            raise ValueError("stored goal references a missing project")
        if any(
            entity_id not in entities
            for memory in memories.values()
            for entity_id in memory.entity_ids
        ):
            raise ValueError("stored memory references a missing entity")
        return state

    def load_for_principal(self, principal: Principal) -> PersonalState:
        """Load only when the canonical Vault owner matches the principal."""

        row = self.connection.execute(
            "SELECT 1 FROM vaults WHERE id = %s AND owner_principal_id = %s",
            (self.vault_id, principal.id),
        ).fetchone()
        if row is None:
            raise PermissionError("personal Vault access denied")
        return self.load()


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

    def search_memories(
        self,
        query: str,
        limit: int = 10,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> tuple[MemoryRecord, ...]:
        """Return current memories ranked by deterministic query-term matches.

        This is a small fast path for local queries. It is deliberately not a
        truth or vector-index authority: superseded memories remain excluded,
        and callers still receive the original provenance-bearing records.
        A future semantic index can sit behind the same retrieval boundary.
        """
        if not query.strip():
            return ()
        if limit < 1:
            raise ValueError("memory search limit must be positive")
        if (start is None) != (end is None):
            raise ValueError("memory search requires both temporal bounds")
        if start is not None and end is not None and start > end:
            raise ValueError("memory search start must not be after end")
        terms = tuple(dict.fromkeys(query.casefold().split()))
        ranked: list[tuple[int, datetime, MemoryRecord]] = []
        for memory in self.memories.values():
            if memory.superseded_by is not None:
                continue
            if start is not None and end is not None and not start <= memory.occurred_at <= end:
                continue
            haystack = memory.content.casefold()
            for entity_id in memory.entity_ids:
                entity = self.entities.get(entity_id)
                if entity is not None:
                    haystack += " " + entity.canonical_name.casefold()
                    haystack += " " + " ".join(alias.casefold() for alias in entity.aliases)
            score = sum(term in haystack for term in terms)
            if score:
                ranked.append((score, memory.occurred_at, memory))
        ranked.sort(key=lambda item: (-item[0], -item[1].timestamp()))
        return tuple(item[2] for item in ranked[:limit])

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


class PersonalMemoryFastPath:
    """Deterministic read adapter for grounded personal-context questions."""

    _TRIGGERS = (
        "memory",
        "remember",
        "recall",
        "working on",
        "working with",
        "what did i",
        "what was i",
        "what do i know",
        "tell me about",
        "tell me more",
        "what about",
    )
    _PROJECT_TRIGGERS = ("project", "projects")
    _GOAL_TRIGGERS = ("goal", "goals")
    _STOPWORDS = frozenset(
        {
            "a",
            "an",
            "and",
            "about",
            "that",
            "did",
            "do",
            "from",
            "i",
            "me",
            "more",
            "my",
            "know",
            "on",
            "tell",
            "the",
            "was",
            "what",
            "with",
            "you",
            "last",
            "night",
            "yesterday",
            "remember",
            "recall",
        }
    )
    _NORMALIZED_TERMS = {"working": "work", "worked": "work"}

    def __init__(
        self,
        state: PersonalState,
        now: datetime | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_index: MemoryVectorIndex | None = None,
        vault_id: str | None = None,
    ) -> None:
        self.state = state
        self.now = now or datetime.now().astimezone()
        if self.now.tzinfo is None:
            raise ValueError("personal retrieval clock must be timezone-aware")
        if (embedding_provider is None) != (vector_index is None):
            raise ValueError("semantic retrieval requires both an embedding provider and index")
        if embedding_provider is not None and not vault_id:
            raise ValueError("semantic retrieval requires a Vault")
        self.embedding_provider = embedding_provider
        self.vector_index = vector_index
        self.vault_id = vault_id

    def resolve(self, intent: IntentFrame, context: Context | None = None) -> Result | None:
        text = intent.utterance.casefold()
        if is_mutation_request(text):
            return None
        if any(trigger in text for trigger in self._PROJECT_TRIGGERS):
            return self._projects_result(intent)
        if any(trigger in text for trigger in self._GOAL_TRIGGERS):
            return self._goals_result(intent)
        if not any(trigger in text for trigger in self._TRIGGERS):
            return None
        query = " ".join(
            self._NORMALIZED_TERMS.get(word.strip(".,!?;:"), word.strip(".,!?;:"))
            for word in text.split()
            if word.strip(".,!?;:") not in self._STOPWORDS and word.strip(".,!?;:")
        )
        start, end = self._temporal_window(text)
        broad_activity_query = query.strip() in {"work", "worked"}
        prior_memories = None
        if context is not None and context.sources == ("authorized_canonical_result",):
            facts = context.values.get("canonical_facts")
            if isinstance(facts, dict) and isinstance(facts.get("memories"), list):
                prior_memories = tuple(item for item in facts["memories"] if isinstance(item, dict))
        if query.strip() and not broad_activity_query:
            memories = self._search_memories(query, start, end)
        elif prior_memories is not None and query.strip() == "":
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.COMPLETED,
                message="More detail from the authorized prior memory result",
                evidence={"memories": list(prior_memories)},
                correlation_id=intent.correlation_id,
            )
        else:
            memories = tuple(
                memory
                for memory in self.state.memories.values()
                if memory.superseded_by is None
                and (start is None or end is None or start <= memory.occurred_at <= end)
            )[:10]
        evidence = {
            "memories": [
                {
                    "content": memory.content,
                    "occurred_at": memory.occurred_at.isoformat(),
                    "provenance": memory.provenance.value,
                }
                for memory in memories
            ]
        }
        message = (
            f"Found {len(memories)} relevant memor{'y' if len(memories) == 1 else 'ies'}"
            if memories
            else "No matching personal memories found"
        )
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.COMPLETED,
            message=message,
            evidence=evidence,
            correlation_id=intent.correlation_id,
        )

    def _search_memories(
        self, query: str, start: datetime | None, end: datetime | None
    ) -> tuple[MemoryRecord, ...]:
        if self.embedding_provider is None or self.vector_index is None or self.vault_id is None:
            return self.state.search_memories(query, start=start, end=end)
        embedding = self.embedding_provider.embed((query,))[0]
        ids = self.vector_index.search(self.vault_id, embedding, 10, max_distance=0.50)
        memories = tuple(
            self.state.memories[memory_id]
            for memory_id in ids
            if memory_id in self.state.memories
            and self.state.memories[memory_id].superseded_by is None
            and (
                start is None
                or end is None
                or start <= self.state.memories[memory_id].occurred_at <= end
            )
        )
        return memories or self.state.search_memories(query, start=start, end=end)

    def _temporal_window(self, text: str) -> tuple[datetime | None, datetime | None]:
        local_now = self.now.astimezone()
        if "last night" in text:
            today = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
            return today.replace(hour=18) - timedelta(days=1), today.replace(hour=6)
        if "yesterday" in text:
            today = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
            return today - timedelta(days=1), today
        if "last week" in text:
            return local_now - timedelta(days=7), local_now
        return None, None

    def _projects_result(self, intent: IntentFrame) -> Result:
        projects = sorted(self.state.projects.values(), key=lambda project: project.created_at)
        evidence = {
            "projects": [
                {
                    "name": project.name,
                    "created_at": project.created_at.isoformat(),
                }
                for project in projects
            ]
        }
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.COMPLETED,
            message=f"Found {len(projects)} personal project{'s' if len(projects) != 1 else ''}",
            evidence=evidence,
            correlation_id=intent.correlation_id,
        )

    def _goals_result(self, intent: IntentFrame) -> Result:
        goals = sorted(self.state.goals.values(), key=lambda goal: goal.created_at)
        project_names = {
            project.project_id: project.name for project in self.state.projects.values()
        }
        evidence = {
            "goals": [
                {
                    "description": goal.description,
                    "project": (
                        project_names.get(goal.project_id) if goal.project_id is not None else None
                    ),
                    "created_at": goal.created_at.isoformat(),
                }
                for goal in goals
            ]
        }
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.COMPLETED,
            message=f"Found {len(goals)} personal goal{'s' if len(goals) != 1 else ''}",
            evidence=evidence,
            correlation_id=intent.correlation_id,
        )
