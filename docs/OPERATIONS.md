# Operations

Run the local validation gate with:

```bash
python -m pip install -e '.[dev]'
bash scripts/validate.sh
```

The clean-install smoke gate also executes the generated `aegis` command and
its private first-run bootstrap from an isolated installation prefix.

`./scripts/aegis` prefers the repository `.venv`, then an active `python3`
environment. Use `AEGIS_PYTHON=/path/to/python` when the runtime is installed
elsewhere; if no usable environment is found, the launcher reports the
installation command instead of failing with an opaque missing-file error.

The canonical production database is PostgreSQL. Apply migrations in filename
order; the current schema starts at `migrations/001_initial.sql`. Do not use a
destructive recreate as an upgrade path.

For PostgreSQL canonical state, use `aegis.backup.backup_postgres` and
`restore_postgres` with `pg_dump`/`pg_restore` from the same major version as
the server. The acceptance runtime uses PostgreSQL 16.15 tooling; mismatched
newer clients can emit restore directives unsupported by the pinned server.
For local SQLite rehearsal state, use `aegis.backup.backup_sqlite` and
`restore_sqlite`, then run the integrity check and the validation suite. Keep
backup destinations protected like the source database because canonical state
may contain private data.

Health reports distinguish overall health from readiness. A non-required
integration may be unhealthy without preventing startup, while a required
component makes readiness false. Live OpenClaw, Ollama, and Home Assistant
acceptance must be recorded from the actual environment; simulator tests do not
substitute for that evidence.

`./scripts/aegis --check` is non-mutating: in addition to testing PostgreSQL
connectivity and the configured Ollama model, it verifies the required
canonical PostgreSQL tables exist. An incomplete schema is reported with
migration remediation instead of being mistaken for a ready runtime. Normal
alpha startup applies the checked-in migrations unless `AEGIS_AUTO_MIGRATE=0`.

The reusable `AegisConfig` accepts the local development alpha with only the
database configured; local identity and authorization defaults are used by the
launcher, and OpenClaw remains optional until an external mutation is needed.
Production configuration still requires explicit Keycloak and OpenFGA URLs so
authority is never silently replaced by local development defaults.
