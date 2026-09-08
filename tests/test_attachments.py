from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from aegis.attachments import AttachmentError, PostgresAttachmentStore


class _Result:
    def __init__(self, row: Any = None, rows: Any = ()) -> None:
        self.row = row
        self.rows = list(rows)

    def fetchone(self) -> Any:
        return self.row

    def fetchall(self) -> list[Any]:
        return self.rows


class _Connection:
    def __init__(self, conversation_id: UUID, *, owned: bool = True) -> None:
        self.conversation_id = conversation_id
        self.owned = owned
        self.attachment_id: UUID | None = None
        self.closed = False
        self.committed = False
        self.created_at = datetime(2026, 9, 8, tzinfo=timezone.utc)

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> _Result:
        if sql.startswith("SELECT 1 FROM conversations"):
            return _Result((1,) if self.owned and params[0] == self.conversation_id else None)
        if sql.startswith("INSERT INTO conversation_attachments"):
            self.attachment_id = params[0]
            return _Result(
                (
                    self.attachment_id,
                    params[3],
                    params[4],
                    params[5],
                    params[6],
                    "extracted",
                    self.created_at,
                )
            )
        if sql.startswith("SELECT id, original_filename, media_type"):
            if len(params) == 3 and self.attachment_id is None:
                return _Result()
            return _Result(
                rows=(
                    (
                        self.attachment_id,
                        "notes.md",
                        "text/markdown",
                        7,
                        "digest",
                        "extracted",
                        self.created_at,
                    ),
                )
            )
        if sql.startswith("SELECT id, original_filename, extracted_text"):
            return _Result(rows=((self.attachment_id, "notes.md", "# Notes\nOwner data"),))
        raise AssertionError(f"unexpected SQL: {sql}")

    def commit(self) -> None:
        self.committed = True

    def close(self) -> None:
        self.closed = True


def test_attachment_store_persists_bounded_owner_file_and_context(tmp_path: Path) -> None:
    conversation_id = uuid4()
    connection = _Connection(conversation_id)
    store = PostgresAttachmentStore(lambda: connection, lambda _: None, tmp_path)

    metadata = store.create(
        "alice", conversation_id, "../notes.md", "text/markdown", b"# Notes\nOwner data"
    )

    attachment_id = UUID(metadata["attachment_id"])
    assert metadata["original_filename"] == "notes.md"
    assert metadata["extraction_state"] == "extracted"
    assert connection.committed and connection.closed
    stored = list((tmp_path / "attachments" / "alice" / str(conversation_id)).iterdir())
    expected = tmp_path / "attachments" / "alice" / str(conversation_id) / f"{attachment_id}.md"
    assert stored == [expected]
    assert store.list("alice", conversation_id)[0]["attachment_id"] == str(attachment_id)
    assert "Untrusted attachment data" in store.context("alice", conversation_id, [attachment_id])


def test_attachment_store_denies_unowned_conversation_before_writing(tmp_path: Path) -> None:
    conversation_id = uuid4()
    connection = _Connection(conversation_id, owned=False)
    store = PostgresAttachmentStore(lambda: connection, lambda _: None, tmp_path)

    with pytest.raises(PermissionError):
        store.create("bob", conversation_id, "notes.txt", "text/plain", b"private")

    assert not (tmp_path / "attachments").exists()
    assert connection.closed


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("run.sh", b"echo unsafe", "only bounded text and data files are supported"),
        ("notes.txt", b"", "file is empty or exceeds the 200 KB limit"),
        ("notes.txt", b"\xff", "file must be UTF-8 text"),
    ],
)
def test_attachment_store_rejects_unsafe_or_unreadable_files(
    tmp_path: Path, filename: str, content: bytes, message: str
) -> None:
    def no_database() -> None:
        pytest.fail("database must not be touched")

    store = PostgresAttachmentStore(no_database, lambda _: None, tmp_path)

    with pytest.raises(AttachmentError, match=message):
        store.create("alice", uuid4(), filename, "text/plain", content)
