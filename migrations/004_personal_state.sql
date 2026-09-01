-- Canonical personal intelligence state, partitioned by private Vault.
CREATE TABLE IF NOT EXISTS personal_entities (
    id UUID PRIMARY KEY,
    vault_id TEXT NOT NULL REFERENCES vaults(id),
    canonical_name TEXT NOT NULL,
    aliases JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS personal_projects (
    id UUID PRIMARY KEY,
    vault_id TEXT NOT NULL REFERENCES vaults(id),
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS personal_goals (
    id UUID PRIMARY KEY,
    vault_id TEXT NOT NULL REFERENCES vaults(id),
    project_id UUID REFERENCES personal_projects(id),
    description TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS personal_memories (
    id UUID PRIMARY KEY,
    vault_id TEXT NOT NULL REFERENCES vaults(id),
    content TEXT NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    provenance TEXT NOT NULL,
    entity_ids JSONB NOT NULL,
    superseded_by UUID REFERENCES personal_memories(id)
);
