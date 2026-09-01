-- Source-grounded OSINT investigations, partitioned by owning principal.
CREATE TABLE IF NOT EXISTS osint_investigations (
    investigation_id UUID PRIMARY KEY,
    owner_id TEXT NOT NULL REFERENCES aegis_principals(id),
    payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS osint_investigations_owner_idx
    ON osint_investigations (owner_id);
