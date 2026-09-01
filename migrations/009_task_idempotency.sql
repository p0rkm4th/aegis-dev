-- Make task mutations replay-safe across process recovery.
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS idempotency_key TEXT;
UPDATE tasks SET idempotency_key = 'task:' || id::text WHERE idempotency_key IS NULL;
ALTER TABLE tasks ALTER COLUMN idempotency_key SET NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS tasks_idempotency_key_idx ON tasks (idempotency_key);
