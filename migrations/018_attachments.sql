CREATE TABLE IF NOT EXISTS conversation_attachments (
    id UUID PRIMARY KEY,
    principal_id TEXT NOT NULL REFERENCES aegis_principals(id),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    original_filename TEXT NOT NULL,
    media_type TEXT NOT NULL,
    byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
    sha256 TEXT NOT NULL,
    extraction_state TEXT NOT NULL CHECK (extraction_state IN ('extracted', 'rejected')),
    extracted_text TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (principal_id, conversation_id, sha256)
);
CREATE INDEX IF NOT EXISTS conversation_attachments_recent
    ON conversation_attachments (principal_id, conversation_id, created_at DESC);
