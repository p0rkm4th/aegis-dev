import json

import pytest

from development.luna_sprint import (
    LaneState,
    RepoState,
    SprintState,
    SprintStateError,
    SprintStatus,
    SprintTask,
    TaskStatus,
    load,
    save,
)


def make_state(*tasks: SprintTask) -> SprintState:
    state = SprintState(
        schema_version=1,
        sprint_id="test-sprint",
        objective="prove bounded lane scheduling",
        status=SprintStatus.READY,
        repo=RepoState("aegis-dev", "a" * 40, "a" * 40),
        lanes={
            "ui": LaneState("ui", 80),
            "reliability": LaneState("reliability", 70),
        },
        tasks={task.task_id: task for task in tasks},
    )
    state.validate()
    return state


def task(
    task_id,
    lane="ui",
    *,
    dependencies=(),
    priority=50,
    max_attempts=3,
    max_identical_failure_repeats=2,
):
    return SprintTask(
        task_id=task_id,
        lane_id=lane,
        title=task_id,
        priority=priority,
        dependencies=tuple(dependencies),
        max_attempts=max_attempts,
        max_identical_failure_repeats=max_identical_failure_repeats,
    )


def test_state_round_trip_is_durable_and_corruption_fails(tmp_path):
    path = tmp_path / "LUNA_SPRINT.json"
    state = make_state(task("first", priority=91))
    state.select_next_task()
    save(path, state)

    restored = load(path)
    assert restored.current_task == "first"
    assert restored.tasks["first"].priority == 91

    path.write_text("not json", encoding="utf-8")
    with pytest.raises(SprintStateError):
        load(path)


def test_dependency_cycle_and_orphan_are_rejected():
    with pytest.raises(SprintStateError, match="cycle"):
        make_state(task("a", dependencies=("b",)), task("b", dependencies=("a",)))
    with pytest.raises(SprintStateError, match="unknown dependencies"):
        make_state(task("a", dependencies=("missing",)))


def test_completed_dependency_unblocks_dependent_and_priority_is_deterministic():
    state = make_state(
        task("foundation", lane="reliability", priority=10),
        task("dependent", dependencies=("foundation",), priority=100),
        task("independent", priority=20),
    )
    assert [item.task_id for item in state.ready_tasks()] == ["independent", "foundation"]
    state.select_next_task()
    state.mark_complete("independent", tests=("focused",))
    state.select_next_task()
    state.mark_complete("foundation", tests=("focused",))
    assert [item.task_id for item in state.ready_tasks()] == ["dependent"]


def test_waiting_human_task_switches_to_independent_ready_task():
    state = make_state(task("owner-ui", priority=100), task("safe-work", lane="reliability"))
    assert state.select_next_task().task_id == "owner-ui"
    state.mark_waiting_human("owner-ui", "subjective owner review", "natural dogfood")
    assert state.select_next_task().task_id == "safe-work"
    assert state.tasks["owner-ui"].status is TaskStatus.WAITING_HUMAN
    assert state.status is SprintStatus.ACTIVE


def test_waiting_external_task_switches_to_independent_ready_task():
    state = make_state(task("external-work", priority=100), task("safe-work", lane="reliability"))
    assert state.select_next_task().task_id == "external-work"
    state.mark_waiting_external(
        "external-work", "hosted provider is unavailable", "next hosted run"
    )
    assert state.select_next_task().task_id == "safe-work"
    assert state.tasks["external-work"].status is TaskStatus.WAITING_EXTERNAL
    assert state.status is SprintStatus.ACTIVE


def test_bounded_failure_blocks_dependents_but_not_independent_work():
    state = make_state(
        task("broken", priority=100, max_attempts=1),
        task("dependent", dependencies=("broken",)),
        task("independent"),
    )
    assert state.select_next_task().task_id == "broken"
    state.record_failure("broken", "pytest:abc", "assertion output", hypothesis_changed=True)
    assert state.tasks["broken"].status is TaskStatus.FAILED_BOUNDED
    assert state.tasks["dependent"].status is TaskStatus.WAITING_DEPENDENCY
    assert state.select_next_task().task_id == "independent"


def test_identical_failure_is_not_retried_forever():
    state = make_state(task("loop", max_attempts=3, max_identical_failure_repeats=1))
    state.select_next_task()
    state.record_failure("loop", "same", "first evidence", hypothesis_changed=True)
    state.select_next_task()
    state.record_failure("loop", "same", "second evidence", hypothesis_changed=False)
    assert state.tasks["loop"].status is TaskStatus.FAILED_BOUNDED
    assert state.tasks["loop"].identical_failure_repeats == 1


def test_resume_updates_repo_sha_and_keeps_active_task(tmp_path):
    path = tmp_path / "state.json"
    state = make_state(task("work"))
    state.select_next_task()
    save(path, state)
    resumed = load(path)
    assert resumed.resume("b" * 40).task_id == "work"
    assert resumed.repo.observed_head_sha == "b" * 40


def test_worker_report_alone_cannot_complete_task():
    state = make_state(task("work"))
    with pytest.raises(SprintStateError, match="active"):
        state.mark_complete("work", evidence=("worker says done",))


def test_json_is_human_inspectable(tmp_path):
    path = tmp_path / "state.json"
    save(path, make_state(task("work")))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["tasks"]["work"]["status"] == "READY"
