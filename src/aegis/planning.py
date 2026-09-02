"""Small, read-only cross-domain planning projection over canonical state."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from uuid import uuid4

from .contracts import IntentFrame, ObjectiveState, Result
from .personal import PersonalState
from .tasks import Task
from .utterance import is_mutation_request


class MultiActionFastPath:
    """Reject compound mutations until durable continuation exists."""

    _ACTION = r"(?:add|complete|create|remove|update)"
    _ACTION_INFLECTED = (
        r"(?:add|adding|added|complete|completing|completed|create|creating|created|"
        r"remove|removing|removed|update|updating|updated)"
    )
    _TARGET = r"(?:a task|tasks|a chore|chores|an event|events|groceries|a grocery)"
    _READ = ("can i afford", "can we afford", "affordability", "show", "list", "what")
    _UNRESOLVED_ACTION_TERMS = (
        "handle",
        "take care of",
        "deal with",
        "make sure",
        "review",
        "check",
        "inspect",
        "prepare",
    )

    @classmethod
    def matches(cls, utterance: str) -> bool:
        text = utterance.casefold()
        mixed_read_and_mutation = (
            " and " in text
            and re.search(rf"\b{cls._ACTION}\b", text) is not None
            and any(term in text for term in cls._READ)
        )
        unresolved_compound = (
            " and " in text
            and re.search(rf"\b{cls._ACTION}\b", text) is not None
            and any(term in text for term in cls._UNRESOLVED_ACTION_TERMS)
        )
        sequential_compound = (
            bool(
                re.search(
                    r"\bthen\b|,\s*then\b|;|as well as|\bplus\b|\bwhile\b|"
                    r"\balso\b|along with",
                    text,
                )
            )
            and len(re.findall(rf"\b{cls._ACTION_INFLECTED}\b", text)) >= 2
        )
        return bool(
            re.search(rf"{cls._ACTION}.*\band\b.*{cls._TARGET}", text)
            or re.search(rf"{cls._TARGET}.*\band\b.*{cls._ACTION}", text)
            or mixed_read_and_mutation
            or unresolved_compound
            or sequential_compound
        )

    @classmethod
    def task_chore_titles(cls, utterance: str) -> tuple[str, str] | None:
        """Extract the narrow, reversible task/chore plan supported by Core."""
        text = utterance.casefold().strip().rstrip(".!?")
        if not ("task" in text and "chore" in text and " and " in text):
            return None
        if not any(term in text for term in ("add", "complete", "create", "remove", "update")):
            return None
        distinct_match = re.search(
            r"\b(?:a\s+)?task\s+(?:to|for)\s+(.+?)\s+and\s+"
            r"(?:a\s+)?chore\s+(?:to|for)\s+(.+)$",
            text,
        )
        if distinct_match is not None:
            task_title, chore_title = (part.strip() for part in distinct_match.groups())
            if task_title and chore_title:
                return task_title, chore_title
        match = re.search(r"\b(?:to|for)\s+(.+)$", text)
        if match is None:
            return None
        title = match.group(1).strip()
        if not title:
            return None
        return title, title

    @classmethod
    def task_event_details(cls, utterance: str) -> tuple[str, str, str] | None:
        """Extract a bounded task/event plan with the existing tomorrow rule."""
        text = utterance.casefold().strip().rstrip(".!?")
        if not ("task" in text and "event" in text and " and " in text):
            return None
        if not any(term in text for term in ("add", "complete", "create", "remove", "update")):
            return None
        distinct_match = re.search(
            r"\b(?:a\s+)?task\s+(?:to|for)\s+(.+?)\s+and\s+"
            r"(?:an?\s+)?event\s+(?:to|for)\s+(.+)$",
            text,
        )
        if distinct_match is not None:
            task_title, event_title = (part.strip() for part in distinct_match.groups())
            starts_at = datetime.now(timezone.utc)
            if event_title.endswith(" tomorrow"):
                event_title = event_title.removesuffix(" tomorrow").strip()
                starts_at += timedelta(days=1)
            if task_title and event_title:
                return task_title, event_title, starts_at.isoformat()
        match = re.search(r"\b(?:to|for)\s+(.+)$", text)
        if match is None:
            return None
        title = match.group(1).strip()
        starts_at = datetime.now(timezone.utc)
        if title.endswith(" tomorrow"):
            title = title.removesuffix(" tomorrow").strip()
            starts_at += timedelta(days=1)
        if not title:
            return None
        return title, title, starts_at.isoformat()

    @classmethod
    def resolve(cls, intent: IntentFrame) -> Result | None:
        if not cls.matches(intent.utterance):
            return None
        if (
            cls.task_chore_titles(intent.utterance) is not None
            or cls.task_event_details(intent.utterance) is not None
        ):
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


class ContextualMutationGuard:
    """Block unsupported context-to-mutation transformations before dispatch."""

    _REFERENCE = re.compile(
        r"\b(?:those|these|it|that|first|second|third|fourth|last|next|previous)\b"
    )

    @classmethod
    def resolve(cls, intent: IntentFrame) -> Result | None:
        text = intent.utterance.casefold()
        if is_mutation_request(text) and cls._REFERENCE.search(text) is not None:
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message=(
                    "I need the specific item or record to change. I will not guess "
                    "what a reference such as 'those' or 'the first two' means."
                ),
                correlation_id=intent.correlation_id,
            )
        if not (
            is_mutation_request(text)
            and any(term in text for term in ("event", "events"))
            and any(term in text for term in ("memory", "remember", "from my"))
        ):
            return None
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=(
                "I cannot safely create an event from a memory reference yet. "
                "Provide the event date and time directly."
            ),
            correlation_id=intent.correlation_id,
        )


class DomainClarificationFastPath:
    """Give unsupported or underspecified alpha requests a useful next step."""

    _REMINDER_TERMS = ("remind me", "remember to", "make sure i remember")

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
        "purchase",
        "prioritize",
        "priority",
        "utilities",
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
        if any(term in intent.utterance.casefold() for term in cls._REMINDER_TERMS):
            message = (
                "I can help with a reminder as a task. Say, for example, "
                "'Create a task to review the backup.'"
            )
        else:
            message = (
                "I need a little more direction. Ask about tasks, groceries, "
                "household chores or events, personal goals or memories, finance, "
                "Homelab, or Network."
            )
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=message,
            correlation_id=intent.correlation_id,
        )

    @classmethod
    def resolve_underspecified_reminder(cls, intent: IntentFrame) -> Result | None:
        """Keep bare reminders out of model action selection until scoped."""

        text = intent.utterance.casefold()
        if not any(term in text for term in cls._REMINDER_TERMS):
            return None
        if "on my list" in text or any(
            term in text for term in ("task", "todo", "to-do", "grocery", "groceries")
        ):
            return None
        return cls.resolve(intent)


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
    _PERSONAL_TERMS = (
        "personal",
        "goal",
        "goals",
        "project",
        "projects",
        "memory",
        "remember",
        "remembers",
    )
    _SHARED_TERMS = (
        "household",
        "obligation",
        "obligations",
        "chore",
        "chores",
        "inspection",
        "utility",
        "utilities",
    )
    _TASK_TERMS = ("task", "tasks", "to-do", "todo")
    _FINANCE_TERMS = ("finance", "afford", "affordable", "cost", "budget", "purchase")
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
                any(term in text for term in cls._TASK_TERMS)
                or any(term in text for term in ("need to", "take care of", "prepare")),
                any(term in text for term in cls._FINANCE_TERMS),
            )
        )
        return domains >= 2 and (
            any(term in text for term in cls._PLANNING_TERMS)
            or ("which" in text and "should" in text)
            or " and " in text
        )

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
        chores = cast(tuple[Any, ...], self.household_snapshot.get("chores", ()))
        open_chores = tuple(item for item in chores if not item.completed)[
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
        priorities.extend(f"chore: {chore.title}" for chore in open_chores)
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
                {
                    "task_id": str(task.task_id),
                    "title": task.title,
                    "due_at": task.due_at.isoformat() if task.due_at is not None else None,
                }
                for task in open_tasks
            ],
            "open_chores": [
                {"chore_id": str(chore.chore_id), "title": chore.title} for chore in open_chores
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


class ContextualCrossDomainPriorityFastPath:
    """Ground a simple cross-domain starting point in an authorized projection."""

    _START_TERMS = ("first", "priority", "prioritize", "focus", "start", "begin")

    def resolve(self, intent: IntentFrame, context: Any) -> Result | None:
        text = intent.utterance.casefold()
        if is_mutation_request(text) or not any(term in text for term in self._START_TERMS):
            return None
        if getattr(context, "sources", ()) != ("authorized_canonical_result",):
            return None
        facts = context.values.get("canonical_facts")
        planning = facts.get("planning") if isinstance(facts, dict) else None
        if not isinstance(planning, dict):
            return None
        tasks = planning.get("open_tasks", ())
        chores = planning.get("open_chores", ())
        if not isinstance(tasks, list) or not isinstance(chores, list):
            return None
        candidates = [item for item in (*tasks, *chores) if isinstance(item, dict)]
        if not candidates:
            return Result(
                objective_id=uuid4(),
                state=ObjectiveState.BLOCKED,
                message="I could not find an open task or chore to prioritize.",
                correlation_id=intent.correlation_id,
            )
        dated_tasks = [
            item for item in tasks if isinstance(item, dict) and isinstance(item.get("due_at"), str)
        ]
        if dated_tasks:
            selected = min(dated_tasks, key=lambda item: str(item["due_at"]))
            title = selected.get("title")
            if isinstance(title, str) and title:
                return Result(
                    objective_id=uuid4(),
                    state=ObjectiveState.COMPLETED,
                    message=f"Based on the earliest recorded task deadline, start with: {title}",
                    evidence={
                        "collection": "planning",
                        "priority_basis": "authorized_prior_result_earliest_task_deadline",
                        "task": selected,
                    },
                    correlation_id=intent.correlation_id,
                )
        return Result(
            objective_id=uuid4(),
            state=ObjectiveState.BLOCKED,
            message=(
                "I can show the open tasks and chores, but I cannot safely rank them "
                "because no authorized task deadline distinguishes them."
            ),
            correlation_id=intent.correlation_id,
        )
