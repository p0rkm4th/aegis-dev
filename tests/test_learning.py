from datetime import datetime, timezone
from uuid import uuid4

from aegis.contracts import (
    IntentFrame,
    LearningCandidate,
    LearningCandidateKind,
    LearningDisposition,
    ObjectiveState,
    Principal,
)
from aegis.personal import OwnerCorrectionLearning, PersonalState, Provenance


def _intent(utterance: str) -> IntentFrame:
    return IntentFrame(
        principal=Principal(id="alice", vault_id="alice-vault"),
        utterance=utterance,
    )


def test_owner_correction_supersedes_fact_and_replaces_entity_link() -> None:
    state = PersonalState()
    hypnos = state.add_entity("Hypnos")
    erebus = state.add_entity("Erebus")
    original = state.add_memory(
        "Hypnos is the game server.",
        datetime(2026, 9, 8, tzinfo=timezone.utc),
        Provenance.EXPLICIT_USER,
        (hypnos.entity_id,),
    )

    result = OwnerCorrectionLearning(state).resolve(
        _intent("Erebus is the game server, not Hypnos.")
    )

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    corrected_id = result.evidence["learning"]["new_memory_id"]
    corrected = next(
        memory for memory in state.memories.values() if str(memory.memory_id) == corrected_id
    )
    assert corrected.content == "Erebus is the game server."
    assert corrected.provenance is Provenance.CORRECTED
    assert corrected.entity_ids == (erebus.entity_id,)
    assert original.superseded_by == corrected.memory_id
    assert state.search_memories("game server")[0] == corrected


def test_owner_correction_prefix_uses_owner_fact_without_model_prose() -> None:
    state = PersonalState()
    state.add_memory(
        "Hypnos is the game server.",
        datetime(2026, 9, 8, tzinfo=timezone.utc),
        Provenance.EXPLICIT_USER,
    )

    result = OwnerCorrectionLearning(state).resolve(_intent("No, Erebus is the game server."))

    assert result is not None
    assert result.state is ObjectiveState.COMPLETED
    assert result.message == "Got it. Erebus is the game server."
    assert all("Scotty" not in memory.content for memory in state.memories.values())


def test_ambiguous_owner_correction_does_not_mutate_memory() -> None:
    state = PersonalState()
    state.add_memory(
        "Hypnos is the game server for the girls.",
        datetime(2026, 9, 8, tzinfo=timezone.utc),
        Provenance.EXPLICIT_USER,
    )
    state.add_memory(
        "Hypnos is the game server for testing.",
        datetime(2026, 9, 8, tzinfo=timezone.utc),
        Provenance.EXPLICIT_USER,
    )

    result = OwnerCorrectionLearning(state).resolve(_intent("No, Erebus is the game server."))

    assert result is not None
    assert result.state is ObjectiveState.BLOCKED
    assert result.evidence["learning"]["disposition"] == LearningDisposition.CONFIRM.value
    assert len(state.memories) == 2
    assert all(memory.superseded_by is None for memory in state.memories.values())


def test_learning_candidate_is_strict_and_model_cannot_add_fields() -> None:
    candidate = LearningCandidate(
        kind=LearningCandidateKind.OWNER_CORRECTION,
        disposition=LearningDisposition.COMMIT,
        target_memory_id=uuid4(),
        old_text="Hypnos",
        replacement_text="Erebus",
        owner_source_spans=((0, 6),),
        correlation_id=uuid4(),
    )
    assert candidate.disposition is LearningDisposition.COMMIT
