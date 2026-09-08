from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from aegis.cli import _apply_migrations
from aegis.conversation import PostgresConversationStore


@pytest.mark.skipif(
    not os.environ.get("AEGIS_TEST_DATABASE_URL"), reason="requires disposable PostgreSQL"
)
def test_conversation_store_returns_recent_bounded_window_in_order():
    import psycopg

    url = os.environ["AEGIS_TEST_DATABASE_URL"]
    principal = f"conversation-test-{uuid4().hex}"
    other_principal = f"conversation-test-other-{uuid4().hex}"
    conversation_id = uuid4()
    short_conversation_id = uuid4()
    connection = psycopg.connect(url)
    try:
        _apply_migrations(connection)
        connection.execute(
            "INSERT INTO conversations (id, principal_id) VALUES (%s, %s), (%s, %s)",
            (conversation_id, principal, short_conversation_id, principal),
        )
        base = datetime(2026, 9, 8, tzinfo=timezone.utc)
        messages = []
        for index in range(102):
            messages.append(
                (
                    UUID(int=index + 1),
                    conversation_id,
                    principal,
                    "owner" if index % 2 == 0 else "aegis",
                    f"turn-{index:03d}",
                    base + timedelta(seconds=index),
                )
            )
        # Equal timestamps exercise the stable UUID tie-breaker for the newest pair.
        messages[-2] = (*messages[-2][:5], base + timedelta(seconds=101))
        messages[-1] = (*messages[-1][:5], base + timedelta(seconds=101))
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO conversation_messages "
                "(id, conversation_id, principal_id, role, display_text, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                messages,
            )
        connection.execute(
            "INSERT INTO conversation_messages "
            "(id, conversation_id, principal_id, role, display_text, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (uuid4(), short_conversation_id, principal, "owner", "short-turn", base),
        )
        connection.commit()

        store = PostgresConversationStore(lambda: psycopg.connect(url), _apply_migrations)
        recent = store.messages(principal, conversation_id)
        short = store.messages(principal, short_conversation_id)

        assert len(recent) == 100
        assert recent[0]["display_text"] == "turn-002"
        assert recent[-2]["display_text"] == "turn-100"
        assert recent[-1]["display_text"] == "turn-101"
        assert [item["display_text"] for item in recent] == sorted(
            (item["display_text"] for item in recent), key=lambda value: int(value[-3:])
        )
        assert [item["display_text"] for item in short] == ["short-turn"]
        with pytest.raises(PermissionError):
            store.messages(other_principal, conversation_id)
    finally:
        connection.rollback()
        connection.execute(
            "DELETE FROM conversations WHERE id IN (%s, %s)",
            (conversation_id, short_conversation_id),
        )
        connection.commit()
        connection.close()
