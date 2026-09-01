-- Canonical shared Space state. Membership and authorization remain owned by
-- the identity/policy tables; this row stores only shared household data.
CREATE TABLE IF NOT EXISTS household_spaces (
    space_id TEXT PRIMARY KEY REFERENCES spaces(id),
    payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
