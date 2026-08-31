-- PostgreSQL canonical schema. Apply migrations in filename order.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS aegis_principals (
    id TEXT PRIMARY KEY,
    external_subject TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS vaults (
    id TEXT PRIMARY KEY,
    owner_principal_id TEXT NOT NULL REFERENCES aegis_principals(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS spaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS space_memberships (
    principal_id TEXT NOT NULL REFERENCES aegis_principals(id),
    space_id TEXT NOT NULL REFERENCES spaces(id),
    role TEXT NOT NULL CHECK (role IN ('owner', 'member')),
    active BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY (principal_id, space_id)
);

CREATE TABLE IF NOT EXISTS objectives (
    id UUID PRIMARY KEY,
    principal_id TEXT NOT NULL REFERENCES aegis_principals(id),
    vault_id TEXT NOT NULL REFERENCES vaults(id),
    space_id TEXT REFERENCES spaces(id),
    state TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS actions (
    id UUID PRIMARY KEY,
    objective_id UUID NOT NULL REFERENCES objectives(id),
    idempotency_key TEXT NOT NULL UNIQUE,
    capability TEXT NOT NULL,
    state TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS observations (
    id UUID PRIMARY KEY,
    action_id UUID NOT NULL REFERENCES actions(id),
    command_succeeded BOOLEAN NOT NULL,
    evidence JSONB NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS results (
    id UUID PRIMARY KEY,
    objective_id UUID NOT NULL REFERENCES objectives(id),
    idempotency_key TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL,
    evidence JSONB NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    principal_id TEXT NOT NULL REFERENCES aegis_principals(id),
    objective_id UUID REFERENCES objectives(id),
    action_id UUID REFERENCES actions(id),
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
