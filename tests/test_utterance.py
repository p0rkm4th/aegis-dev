from aegis.utterance import has_multiple_question_clauses, is_task_destination_request


def test_compound_question_detects_embedded_and_read_clause() -> None:
    assert has_multiple_question_clauses(
        "We're running low on groceries, can I spend eighty dollars tonight, "
        "and check why Plex is down?"
    )


def test_single_affordability_question_is_not_compound() -> None:
    assert not has_multiple_question_clauses("Can I afford eighty dollars?")


def test_possessive_task_collection_is_an_explicit_destination():
    assert is_task_destination_request("Add checking the lock to my tasks for tomorrow.")
    assert is_task_destination_request("Put this on the task list.")
    assert not is_task_destination_request("Add milk to the grocery list.")
