-- Evidence from explicitly authorized security-lab scopes.
CREATE TABLE IF NOT EXISTS security_lab_findings (
    finding_id UUID PRIMARY KEY,
    owner_id TEXT NOT NULL REFERENCES aegis_principals(id),
    space_id TEXT NOT NULL REFERENCES spaces(id),
    payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS security_lab_findings_owner_space_idx
    ON security_lab_findings (owner_id, space_id);
