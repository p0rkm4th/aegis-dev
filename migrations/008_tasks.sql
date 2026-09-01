-- Canonical shared Tasks Pack state.
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY,
    space_id TEXT NOT NULL REFERENCES spaces(id),
    title TEXT NOT NULL,
    created_by TEXT NOT NULL REFERENCES aegis_principals(id),
    assignee_id TEXT REFERENCES aegis_principals(id),
    due_at TIMESTAMPTZ,
    status TEXT NOT NULL CHECK (status IN ('open', 'completed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
