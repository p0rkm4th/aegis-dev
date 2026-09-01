# AEGIS

AEGIS is a local-first intelligence platform. The interactive alpha routes
CLI and browser requests through one Core interaction boundary; PostgreSQL is
canonical state, models propose actions, and policy plus independent
verification remain authoritative.

## Quick start

Use Python 3.11 or newer. Install the development and live dependencies in a
virtual environment, then copy the safe configuration template:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,live]'
cp examples/aegis.env.example .env
```

Replace the database placeholders in `.env`. Ollama must be running at the
address in `AEGIS_OLLAMA_URL` with the configured model installed. The value
must match Ollama's actual listening address; it is commonly
`http://127.0.0.1:11434`, but `./scripts/aegis --check` explains how to fix an
unreachable endpoint.

Check readiness without changing canonical state:

```sh
./scripts/aegis --env-file .env --check
```

Start the loopback browser alpha:

```sh
./scripts/aegis --env-file .env --web
```

Open the printed local URL. For a scriptable request, use
`./scripts/aegis --env-file .env --once "Show my tasks."`; add `--json` for
the canonical machine-readable Result envelope.

Read [docs/ALPHA.md](docs/ALPHA.md) for the browser and workflow contract and
[docs/OPERATIONS.md](docs/OPERATIONS.md) for migrations, backups, and runtime
operations. Architecture and security invariants are documented in
[ARCHITECTURE.md](ARCHITECTURE.md) and [SECURITY_MODEL.md](SECURITY_MODEL.md).

The browser binds to loopback by default. OpenClaw, Home Assistant, mobile,
and voice integrations are optional lanes with separate live-acceptance
requirements; do not expose their services beyond the documented local
boundary.
