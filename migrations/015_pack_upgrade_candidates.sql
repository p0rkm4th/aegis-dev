-- Candidate Pack software updates remain separate until explicit authority approval.
CREATE TABLE IF NOT EXISTS pack_upgrade_candidates (
    pack_id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    manifest JSONB NOT NULL,
    cards JSONB NOT NULL,
    requested_permissions JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
