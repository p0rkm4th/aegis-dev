-- Authorized Network and discovered device state, partitioned by Space.
CREATE TABLE IF NOT EXISTS network_scopes (
    space_id TEXT NOT NULL REFERENCES spaces(id),
    scope_id TEXT NOT NULL,
    cidrs JSONB NOT NULL,
    purpose TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (space_id, scope_id)
);

CREATE TABLE IF NOT EXISTS network_devices (
    space_id TEXT NOT NULL REFERENCES spaces(id),
    address TEXT NOT NULL,
    hostname TEXT,
    services JSONB NOT NULL DEFAULT '[]'::jsonb,
    discovered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (space_id, address)
);
