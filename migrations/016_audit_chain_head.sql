-- PostgreSQL owns the predecessor chosen by concurrent audit writers.
CREATE TABLE IF NOT EXISTS audit_chain_heads (
    chain_name TEXT PRIMARY KEY,
    head_hash TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO audit_chain_heads (chain_name, head_hash)
VALUES (
    'default',
    COALESCE(
        (SELECT event_hash FROM audit_events
         WHERE event_hash IS NOT NULL ORDER BY created_at DESC, id DESC LIMIT 1),
        'GENESIS'
    )
)
ON CONFLICT (chain_name) DO NOTHING;
