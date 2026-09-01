-- Optional semantic retrieval index; personal_memories remains canonical truth.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS personal_memory_vectors (
    memory_id UUID PRIMARY KEY REFERENCES personal_memories(id) ON DELETE CASCADE,
    vault_id TEXT NOT NULL REFERENCES vaults(id) ON DELETE CASCADE,
    embedding vector(768) NOT NULL,
    model TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS personal_memory_vectors_vault_embedding_idx
    ON personal_memory_vectors USING hnsw (embedding vector_cosine_ops);
