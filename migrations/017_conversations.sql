CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY,
    principal_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    principal_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('owner', 'aegis')),
    display_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    correlation_id UUID,
    UNIQUE (conversation_id, correlation_id, role)
);
CREATE INDEX IF NOT EXISTS conversation_messages_recent
    ON conversation_messages (conversation_id, created_at DESC);
