# Operations

Run the local validation gate with:

```bash
python -m pip install -e '.[dev]'
bash scripts/validate.sh
```

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
