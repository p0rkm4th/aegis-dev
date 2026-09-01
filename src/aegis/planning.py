"""Small, read-only cross-domain planning projection over canonical state."""

from __future__ import annotations

import re
from typing import Any, cast
from uuid import uuid4

from .contracts import IntentFrame, ObjectiveState, Result
from .personal import PersonalState
from .tasks import Task


class MultiActionFastPath:
    """Reject compound mutations until durable continuation exists."""

    _ACTION = r"(?:add|complete|create|remove|update)"
    _TARGET = r"(?:a task|tasks|a chore|chores|an event|events|groceries|a grocery)"

    @classmethod
    def matches(cls, utterance: str) -> bool:
        text = utterance.casefold()
        return bool(
            re.search(rf"{cls._ACTION}.*\band\b.*{cls._TARGET}", text)
            or re.search(rf"{cls._TARGET}.*\band\b.*{cls._ACTION}", text)
        )

    @classmethod
    def resolve(cls, intent: IntentFrame) -> Result | None:
        if not cls.matches(intent.utterance):
            return None
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=(
                "This request contains multiple actions. Ask AEGIS to perform one "
                "action at a time so each result can be verified independently."
            ),
            correlation_id=intent.correlation_id,
        )


class DomainClarificationFastPath:
    """Give unsupported or underspecified alpha requests a useful next step."""

    _KNOWN_TERMS = (
        "task",
        "todo",
        "to-do",
        "chore",
        "event",
        "grocery",
        "groceries",
        "food",
        "household",
        "utility",
        "rent",
        "memory",
        "goal",
        "project",
        "personal",
        "finance",
        "afford",
        "homelab",
        "service",
        "network",
    )

    @classmethod
    def matches(cls, utterance: str) -> bool:
        text = utterance.casefold()
        return not any(term in text for term in cls._KNOWN_TERMS)

    @classmethod
    def resolve(cls, intent: IntentFrame) -> Result | None:
        if not cls.matches(intent.utterance):
            return None
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=(
                "I need a little more direction. Ask about tasks, groceries, "
                "household chores or events, personal goals or memories, finance, "
                "Homelab, or Network."
            ),
            correlation_id=intent.correlation_id,
        )


class PersonalTaskComposer:
    """Resolve an explicit personal-goal-to-task request from Vault state."""

    _ACTION_TERMS = ("create", "add", "turn", "make")
    _TARGET = "task"

    @classmethod
    def matches(cls, utterance: str) -> bool:
        text = utterance.casefold()
        return (
            cls._TARGET in text
            and "goal" in text
            and any(term in text for term in cls._ACTION_TERMS)
        )

    @classmethod
    def resolve(cls, utterance: str, personal: PersonalState) -> tuple[str | None, str | None]:
        if not cls.matches(utterance):
            return None, None
        text = utterance.casefold()
        goals = tuple(personal.goals.values())
        if len(goals) == 1:
            return goals[0].description, None
        matches = tuple(
            goal
            for goal in goals
            if any(
                term in goal.description.casefold()
                for term in text.replace("'", "").split()
                if len(term) >= 4 and term not in {"create", cls._TARGET, "goal", "into"}
            )
        )
        if len(matches) == 1:
            return matches[0].description, None
        return (
            None,
            f"Which personal goal should I turn into a {cls._TARGET}? Please name the goal.",
        )


class PersonalChoreComposer(PersonalTaskComposer):
    """Resolve an explicit personal-goal-to-shared-chore request."""

    _TARGET = "chore"


class _PersonalMemoryComposer:
    """Resolve an explicit personal-memory-to-action request from Vault state."""

    _ACTION_TERMS = ("create", "add", "turn", "make")
    _TARGET = "task"
    _STOPWORDS = frozenset({"create", "task", "memory", "into", "from", "my", "a", "the"})

    @classmethod
    def matches(cls, utterance: str) -> bool:
        text = utterance.casefold()
        return (
            cls._TARGET in text
            and "memory" in text
            and any(term in text for term in cls._ACTION_TERMS)
        )

    @classmethod
    def resolve(cls, utterance: str, personal: PersonalState) -> tuple[str | None, str | None]:
        if not cls.matches(utterance):
            return None, None
        text = utterance.casefold()
        terms = tuple(
            term.strip(".,!?;:")
            for term in text.replace("'", "").split()
            if len(term.strip(".,!?;:")) >= 4 and term.strip(".,!?;:") not in cls._STOPWORDS
        )
        candidates = tuple(
            (
                sum(term in memory.content.casefold() for term in terms),
                memory,
            )
            for memory in personal.memories.values()
            if memory.superseded_by is None
        )
        scored = tuple(candidate for candidate in candidates if candidate[0] > 0)
        if scored:
            highest = max(score for score, _memory in scored)
            matches = tuple(memory for score, memory in scored if score == highest)
            if len(matches) == 1:
                return matches[0].content, None
        return (
            None,
            f"Which personal memory should I turn into a {cls._TARGET}? Please name the memory.",
        )


class PersonalMemoryTaskComposer(_PersonalMemoryComposer):
    """Resolve an explicit personal-memory-to-task request from Vault state."""


class PersonalMemoryChoreComposer(_PersonalMemoryComposer):
    """Resolve an explicit personal-memory-to-shared-chore request."""

    _TARGET = "chore"


class CrossDomainPlanningFastPath:
    """Assemble bounded authorized context for explicit planning questions.

    This is deliberately a projection, not a second store or an authority
    layer.  Each source has already enforced its Vault/Space boundary before
    this adapter is called.
    """

    _PLANNING_TERMS = (
        "prioritize",
        "priority",
        "priorities",
        "focus on",
        "plan my",
        "plan this",
        "what should i",
        "what do i need to",
        "what tasks",
    )
    _PERSONAL_TERMS = ("personal", "goal", "goals", "project", "projects", "memory")
    _SHARED_TERMS = (
        "household",
        "obligation",
        "obligations",
        "chore",
        "chores",
        "utility",
        "utilities",
    )
    _TASK_TERMS = ("task", "tasks", "to-do", "todo")
    _FINANCE_TERMS = ("finance", "afford", "affordable", "cost", "budget")
    _MAX_CONTEXT_ITEMS = 5

    def __init__(
        self,
        personal: PersonalState,
        household_snapshot: dict[str, object],
        tasks: tuple[Task, ...],
        finance: dict[str, Any] | None = None,
    ) -> None:
        self.personal = personal
        self.household_snapshot = household_snapshot
        self.tasks = tasks
        self.finance = finance

    @classmethod
    def matches(cls, utterance: str) -> bool:
        text = utterance.casefold()
        domains = sum(
            (
                any(term in text for term in cls._PERSONAL_TERMS),
                any(term in text for term in cls._SHARED_TERMS),
                any(term in text for term in cls._TASK_TERMS),
                any(term in text for term in cls._FINANCE_TERMS),
            )
        )
        return domains >= 2 and any(term in text for term in cls._PLANNING_TERMS)

    def resolve(self, intent: IntentFrame) -> Result | None:
        if not self.matches(intent.utterance):
            return None
        goals = sorted(self.personal.goals.values(), key=lambda goal: goal.created_at)[
            : self._MAX_CONTEXT_ITEMS
        ]
        projects = {project.project_id: project.name for project in self.personal.projects.values()}
        obligations = cast(tuple[Any, ...], self.household_snapshot.get("obligations", ()))
        open_obligations = tuple(item for item in obligations if not item.settled)[
            : self._MAX_CONTEXT_ITEMS
        ]
        open_tasks = tuple(task for task in self.tasks if task.status.value == "open")[
            : self._MAX_CONTEXT_ITEMS
        ]
        query_terms = {
            term.strip(".,!?;:")
            for term in intent.utterance.casefold().split()
            if len(term.strip(".,!?;:")) >= 4
            and term.strip(".,!?;:")
            not in {
                "considering",
                "personal",
                "memory",
                "utilities",
                "open",
                "what",
                "should",
                "prioritize",
            }
        }
        memory_candidates = tuple(
            (
                sum(term in memory.content.casefold() for term in query_terms),
                memory,
            )
            for memory in self.personal.memories.values()
            if memory.superseded_by is None
        )
        scored_memories = tuple(candidate for candidate in memory_candidates if candidate[0] > 0)
        highest_memory_score = max((score for score, _memory in scored_memories), default=0)
        memory_matches = tuple(
            memory for score, memory in scored_memories if score == highest_memory_score
        )[: self._MAX_CONTEXT_ITEMS]
        priorities = [f"household obligation: {item.title}" for item in open_obligations]
        priorities.extend(f"personal goal: {goal.description}" for goal in goals)
        priorities.extend(f"task: {task.title}" for task in open_tasks)
        priorities.extend(f"personal memory: {memory.content}" for memory in memory_matches)
        planning: dict[str, object] = {
            "goals": [
                {
                    "description": goal.description,
                    "project": (
                        projects.get(goal.project_id) if goal.project_id is not None else None
                    ),
                }
                for goal in goals
            ],
            "open_obligations": [
                {"title": item.title, "responsible_id": item.responsible_id}
                for item in open_obligations
            ],
            "open_tasks": [
                {"task_id": str(task.task_id), "title": task.title} for task in open_tasks
            ],
            "memories": [
                {
                    "content": memory.content,
                    "occurred_at": memory.occurred_at.isoformat(),
                    "provenance": memory.provenance.value,
                }
                for memory in memory_matches
            ],
            "priority_candidates": priorities,
            "sources": ("personal_vault", "household_space", "tasks_space"),
        }
        if self.finance is not None:
            planning["affordability"] = {
                key: self.finance[key]
                for key in (
                    "affordable",
                    "purchase_cents",
                    "shared_obligations_cents",
                    "shortfall_cents",
                )
                if key in self.finance
            }
            planning["sources"] = (
                "personal_vault",
                "household_space",
                "tasks_space",
                "finance",
            )
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.COMPLETED,
            message="Cross-domain planning context assembled from canonical state",
            evidence={"planning": planning},
            correlation_id=intent.correlation_id,
        )
