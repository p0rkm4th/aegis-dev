-- Durable Pack discovery, permission grants, and lifecycle status.
CREATE TABLE IF NOT EXISTS pack_installations (
    pack_id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    manifest JSONB NOT NULL,
    cards JSONB NOT NULL,
    granted_permissions JSONB NOT NULL,
    status TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
