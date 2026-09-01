-- Allowlisted derived household finance output; never store private source data.
CREATE TABLE IF NOT EXISTS household_projections (
    space_id TEXT PRIMARY KEY REFERENCES spaces(id),
    payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
