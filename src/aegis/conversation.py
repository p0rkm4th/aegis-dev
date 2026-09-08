"""Small Principal-scoped PostgreSQL conversation projection for the owner shell."""

from __future__ import annotations

from builtins import list as builtin_list
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4


class PostgresConversationStore:
    def __init__(self, connection_factory: Any, migrate: Any) -> None:
        self.connection_factory = connection_factory
        self.migrate = migrate

    def _connection(self) -> Any:
        connection = self.connection_factory()
        self.migrate(connection)
        return connection

    def list(self, principal_id: str, limit: int = 12) -> list[dict[str, Any]]:
        connection = self._connection()
        try:
            rows = connection.execute(
                "SELECT c.id, c.created_at, c.updated_at, "
                "(SELECT LEFT(m.display_text, 96) FROM conversation_messages m "
                "WHERE m.conversation_id = c.id AND m.principal_id = c.principal_id "
                "AND m.role = 'owner' ORDER BY m.created_at ASC LIMIT 1) "
                "FROM conversations c "
                "WHERE c.principal_id = %s ORDER BY c.updated_at DESC LIMIT %s",
                (principal_id, limit),
            ).fetchall()
            return [self._conversation_row(row) for row in rows]
        finally:
            connection.close()

    def create(self, principal_id: str) -> dict[str, Any]:
        conversation_id = uuid4()
        connection = self._connection()
        try:
            row = connection.execute(
                "INSERT INTO conversations (id, principal_id) VALUES (%s, %s) "
                "RETURNING id, created_at, updated_at",
                (conversation_id, principal_id),
            ).fetchone()
            connection.commit()
            assert row is not None
            return self._conversation_row(row)
        finally:
            connection.close()

    def messages(
        self, principal_id: str, conversation_id: UUID, limit: int = 100
    ) -> builtin_list[dict[str, Any]]:
        connection = self._connection()
        try:
            access = connection.execute(
                "SELECT 1 FROM conversations WHERE id = %s AND principal_id = %s",
                (conversation_id, principal_id),
            ).fetchone()
            if access is None:
                raise PermissionError("conversation is not owned by principal")
            rows = connection.execute(
                "SELECT m.id, m.role, m.display_text, m.created_at, m.correlation_id "
                "FROM conversation_messages m JOIN conversations c ON c.id = m.conversation_id "
                "WHERE c.id = %s AND c.principal_id = %s AND m.principal_id = %s "
                "ORDER BY m.created_at DESC, m.id DESC LIMIT %s",
                (conversation_id, principal_id, principal_id, limit),
            ).fetchall()
            rows = reversed(rows)
            return [
                {
                    "message_id": str(row[0]),
                    "role": row[1],
                    "display_text": row[2],
                    "created_at": _timestamp(row[3]),
                    "correlation_id": str(row[4]) if row[4] else None,
                }
                for row in rows
            ]
        finally:
            connection.close()

    def append(
        self,
        principal_id: str,
        conversation_id: UUID,
        role: str,
        text: str,
        correlation_id: UUID | None = None,
    ) -> None:
        if role not in {"owner", "aegis"} or not text.strip():
            raise ValueError("invalid conversation message")
        connection = self._connection()
        try:
            inserted = connection.execute(
                "INSERT INTO conversation_messages "
                "(id, conversation_id, principal_id, role, display_text, correlation_id) "
                "SELECT %s, id, principal_id, %s, %s, %s FROM conversations "
                "WHERE id = %s AND principal_id = %s ON CONFLICT DO NOTHING",
                (uuid4(), role, text[:20_000], correlation_id, conversation_id, principal_id),
            )
            if inserted.rowcount == 0:
                raise PermissionError("conversation is not owned by principal")
            connection.execute(
                "UPDATE conversations SET updated_at = now() WHERE id = %s AND principal_id = %s",
                (conversation_id, principal_id),
            )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _conversation_row(row: tuple[Any, ...]) -> dict[str, Any]:
        return {
            "conversation_id": str(row[0]),
            "created_at": _timestamp(row[1]),
            "updated_at": _timestamp(row[2]),
            "title": row[3] if len(row) > 3 else None,
        }


def _timestamp(value: datetime) -> str:
    return value.isoformat()
