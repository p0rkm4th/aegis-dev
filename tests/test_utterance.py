from aegis.utterance import is_task_destination_request


def test_possessive_task_collection_is_an_explicit_destination():
    assert is_task_destination_request("Add checking the lock to my tasks for tomorrow.")
    assert is_task_destination_request("Put this on the task list.")
    assert not is_task_destination_request("Add milk to the grocery list.")
