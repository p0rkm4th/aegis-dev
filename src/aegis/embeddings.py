"""Optional local embedding and pgvector retrieval adapters."""

from __future__ import annotations

import json
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import UUID


class EmbeddingProvider(Protocol):
    model: str
    dimensions: int

    def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]: ...


class OllamaEmbeddingProvider:
    """Use Ollama's embedding endpoint without coupling it to cognition."""

    dimensions = 768

    def __init__(self, model: str, base_url: str, timeout: float = 120.0) -> None:
        if not model:
            raise ValueError("embedding model is required")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("Ollama base URL must use HTTP or HTTPS")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        request = Request(
            f"{self.base_url}/api/embed",
            data=json.dumps({"model": self.model, "input": list(texts)}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise EmbeddingResponseError("Ollama embedding request failed") from exc
        raw_embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
        if not isinstance(raw_embeddings, list) or len(raw_embeddings) != len(texts):
            raise EmbeddingResponseError("Ollama returned an invalid embedding count")
        embeddings: list[tuple[float, ...]] = []
        for raw in raw_embeddings:
            if not isinstance(raw, list) or len(raw) != self.dimensions:
                raise EmbeddingResponseError("Ollama returned an invalid embedding dimension")
            try:
                embeddings.append(tuple(float(value) for value in raw))
            except (TypeError, ValueError) as exc:
                raise EmbeddingResponseError("Ollama returned a non-numeric embedding") from exc
        return tuple(embeddings)


class EmbeddingResponseError(ValueError):
    """The embedding provider returned no usable vectors."""


class MemoryVectorIndex(Protocol):
    def upsert(
        self, vault_id: str, memory_id: UUID, embedding: tuple[float, ...], model: str
    ) -> None: ...

    def search(
        self,
        vault_id: str,
        embedding: tuple[float, ...],
        limit: int,
        max_distance: float = 0.50,
    ) -> tuple[UUID, ...]: ...


class PostgresMemoryVectorIndex:
    """Persist/retrieve vectors while keeping memory rows as canonical truth."""

    def __init__(self, connection: Any, dimensions: int = 768) -> None:
        if dimensions != 768:
            raise ValueError("the personal memory vector schema is fixed at 768 dimensions")
        self.connection = connection

    def upsert(
        self, vault_id: str, memory_id: UUID, embedding: tuple[float, ...], model: str
    ) -> None:
        if len(embedding) != 768:
            raise ValueError("personal memory embedding must have 768 dimensions")
        literal = "[" + ",".join(str(value) for value in embedding) + "]"
        self.connection.execute(
            "INSERT INTO personal_memory_vectors "
            "(memory_id, vault_id, embedding, model) VALUES (%s, %s, %s::vector, %s) "
            "ON CONFLICT (memory_id) DO UPDATE SET vault_id = EXCLUDED.vault_id, "
            "embedding = EXCLUDED.embedding, model = EXCLUDED.model, updated_at = now()",
            (str(memory_id), vault_id, literal, model),
        )

    def search(
        self,
        vault_id: str,
        embedding: tuple[float, ...],
        limit: int,
        max_distance: float = 0.50,
    ) -> tuple[UUID, ...]:
        if len(embedding) != 768:
            raise ValueError("personal memory embedding must have 768 dimensions")
        if limit < 1:
            raise ValueError("semantic memory search limit must be positive")
        if not 0.0 <= max_distance <= 2.0:
            raise ValueError("semantic memory distance threshold must be between zero and two")
        literal = "[" + ",".join(str(value) for value in embedding) + "]"
        rows = self.connection.execute(
            "SELECT memory_id FROM personal_memory_vectors "
            "WHERE vault_id = %s AND embedding <=> %s::vector <= %s "
            "ORDER BY embedding <=> %s::vector LIMIT %s",
            (vault_id, literal, max_distance, literal, limit),
        ).fetchall()
        return tuple(UUID(str(row[0])) for row in rows)
