-- Persist the same tamper-evident audit chain used by the local audit adapter.
ALTER TABLE audit_events
    ADD COLUMN IF NOT EXISTS previous_hash TEXT,
    ADD COLUMN IF NOT EXISTS event_hash TEXT;
