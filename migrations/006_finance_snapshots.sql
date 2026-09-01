-- Canonical private finance snapshots, partitioned by owning principal.
CREATE TABLE IF NOT EXISTS finance_snapshots (
    owner_id TEXT PRIMARY KEY REFERENCES aegis_principals(id),
    payload JSONB NOT NULL,
    provider_id TEXT NOT NULL,
    captured_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
