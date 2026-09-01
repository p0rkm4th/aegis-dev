from uuid import uuid4

from aegis.embeddings import OllamaEmbeddingProvider, PostgresMemoryVectorIndex


def test_postgres_vector_index_uses_vault_scoped_vector_queries():
    class Cursor:
        def fetchall(self) -> list[tuple[str]]:
            return [(str(memory_id),)]

    class Connection:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        def execute(self, query: str, params: tuple[object, ...] = ()) -> Cursor:
            self.calls.append((query, params))
            return Cursor()

        def commit(self) -> None:
            pass

    memory_id = uuid4()
    connection = Connection()
    index = PostgresMemoryVectorIndex(connection)
    vector = tuple(0.1 for _ in range(768))
    index.upsert("alice-vault", memory_id, vector, "nomic-embed-text")
    assert index.search("alice-vault", vector, 10) == (memory_id,)
    assert all("vector" in query for query, _ in connection.calls)
    assert "<=> %s::vector <= %s" in connection.calls[-1][0]
    assert all(
        params[0] == "alice-vault" for query, params in connection.calls if "SELECT" in query
    )


def test_ollama_embedding_provider_rejects_wrong_dimensions():
    provider = OllamaEmbeddingProvider("nomic-embed-text", "http://127.0.0.1")
    try:
        class Connection:
            def execute(self, query: str, params: tuple[object, ...] = ()) -> object:
                raise AssertionError("invalid vector should be rejected before SQL")

        PostgresMemoryVectorIndex(Connection()).upsert(
            "alice-vault", uuid4(), (0.1,), provider.model
        )
    except ValueError as exc:
        assert "768" in str(exc)
    else:
        raise AssertionError("invalid embedding dimension was accepted")
