"""Principal-scoped, bounded owner attachment storage for conversation analysis."""

from __future__ import annotations

import hashlib
import re
from builtins import list as builtin_list
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

MAX_ATTACHMENT_BYTES = 200_000
MAX_EXTRACTED_CHARS = 100_000
MAX_CONTEXT_CHARS = 24_000
_ALLOWED_SUFFIXES = {
    ".cfg",
    ".conf",
    ".csv",
    ".css",
    ".html",
    ".ini",
    ".json",
    ".js",
    ".log",
    ".md",
    ".py",
    ".sql",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}
_SAFE_NAME = re.compile(r"^[^\x00-\x1f\x7f]{1,200}$")


class AttachmentError(ValueError):
    """The owner supplied an attachment outside the bounded text contract."""


class PostgresAttachmentStore:
    def __init__(self, connection_factory: Any, migrate: Any, root: Path) -> None:
        self.connection_factory = connection_factory
        self.migrate = migrate
        self.root = root

    def _connection(self) -> Any:
        connection = self.connection_factory()
        self.migrate(connection)
        return connection

    def create(
        self,
        principal_id: str,
        conversation_id: UUID,
        filename: str,
        media_type: str,
        content: bytes,
    ) -> dict[str, Any]:
        safe_name = Path(filename).name
        suffix = Path(safe_name).suffix.casefold()
        if not _SAFE_NAME.fullmatch(safe_name) or safe_name in {".", ".."}:
            raise AttachmentError("filename is invalid")
        if suffix not in _ALLOWED_SUFFIXES:
            raise AttachmentError("only bounded text and data files are supported")
        if not content or len(content) > MAX_ATTACHMENT_BYTES:
            raise AttachmentError("file is empty or exceeds the 200 KB limit")
        try:
            extracted = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AttachmentError("file must be UTF-8 text") from exc
        attachment_id = uuid4()
        digest = hashlib.sha256(content).hexdigest()
        bounded = extracted[:MAX_EXTRACTED_CHARS]
        connection = self._connection()
        try:
            owned = connection.execute(
                "SELECT 1 FROM conversations WHERE id = %s AND principal_id = %s",
                (conversation_id, principal_id),
            ).fetchone()
            if owned is None:
                raise PermissionError("conversation is not owned by principal")
            target_dir = self.root / "attachments" / principal_id / str(conversation_id)
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{attachment_id}{suffix}"
            target.write_bytes(content)
            row = connection.execute(
                "INSERT INTO conversation_attachments "
                "(id, principal_id, conversation_id, original_filename, media_type, byte_size, "
                "sha256, extraction_state, extracted_text) VALUES "
                "(%s, %s, %s, %s, %s, %s, %s, 'extracted', %s) "
                "ON CONFLICT (principal_id, conversation_id, sha256) DO UPDATE SET "
                "original_filename = EXCLUDED.original_filename "
                "RETURNING id, original_filename, media_type, byte_size, sha256, "
                "extraction_state, created_at",
                (
                    attachment_id,
                    principal_id,
                    conversation_id,
                    safe_name,
                    media_type[:120] or "text/plain",
                    len(content),
                    digest,
                    bounded,
                ),
            ).fetchone()
            connection.commit()
            assert row is not None
            return self._row(row)
        finally:
            connection.close()

    def list(self, principal_id: str, conversation_id: UUID) -> builtin_list[dict[str, Any]]:
        connection = self._connection()
        try:
            rows = connection.execute(
                "SELECT id, original_filename, media_type, byte_size, sha256, "
                "extraction_state, created_at FROM conversation_attachments "
                "WHERE principal_id = %s AND conversation_id = %s "
                "ORDER BY created_at ASC",
                (principal_id, conversation_id),
            ).fetchall()
            return [self._row(row) for row in rows]
        finally:
            connection.close()

    def context(self, principal_id: str, conversation_id: UUID, ids: builtin_list[UUID]) -> str:
        if not ids:
            return ""
        connection = self._connection()
        try:
            rows = connection.execute(
                "SELECT id, original_filename, extracted_text FROM conversation_attachments "
                "WHERE principal_id = %s AND conversation_id = %s AND id = ANY(%s)",
                (principal_id, conversation_id, ids[:3]),
            ).fetchall()
            parts: list[str] = []
            remaining = MAX_CONTEXT_CHARS
            for attachment_id, filename, text in rows:
                if remaining <= 0:
                    break
                excerpt = str(text)[:remaining]
                parts.append(
                    f"[Untrusted attachment data: {filename} ({attachment_id})]\n{excerpt}\n"
                    "[End untrusted attachment data]"
                )
                remaining -= len(excerpt)
            return "\n\n".join(parts)
        finally:
            connection.close()

    @staticmethod
    def _row(row: tuple[Any, ...]) -> dict[str, Any]:
        return {
            "attachment_id": str(row[0]),
            "original_filename": row[1],
            "media_type": row[2],
            "byte_size": row[3],
            "sha256": row[4],
            "extraction_state": row[5],
            "created_at": row[6].isoformat(),
        }
