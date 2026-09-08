CREATE TABLE IF NOT EXISTS developer_jobs (
    id UUID PRIMARY KEY,
    principal_id TEXT NOT NULL REFERENCES aegis_principals(id),
    project_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('inspect', 'modify')),
    objective TEXT NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('queued', 'running', 'approval_required', 'completed', 'failed', 'interrupted')
    ),
    result JSONB NOT NULL DEFAULT '{}'::jsonb,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS developer_jobs_recent
    ON developer_jobs (principal_id, updated_at DESC);
