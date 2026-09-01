-- Space-scoped Homelab hosts and services; execution remains separately scoped.
CREATE TABLE IF NOT EXISTS homelab_inventories (
    space_id TEXT PRIMARY KEY REFERENCES spaces(id),
    payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
