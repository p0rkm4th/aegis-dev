# AEGIS

AEGIS is a local-first intelligence platform. The interactive alpha routes
CLI and browser requests through one Core interaction boundary; PostgreSQL is
canonical state, models propose actions, and policy plus independent
verification remain authoritative.

## Quick start

Use Python 3.11 or newer. Install the runtime dependencies in a virtual
environment, then copy the safe configuration template:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[live]'
./scripts/aegis --init
```

Contributor checks additionally need the test and lint tools; install them
with `.venv/bin/python -m pip install -e '.[dev,live]'` when running the
repository validation gate.

The package also installs an `aegis` command; the repository wrapper above is
useful when working directly from a checkout.

`--init` refuses to overwrite an existing file and creates it with private
permissions. Replace its placeholders before running readiness checks.

Replace the database placeholders in `.env`. Ollama must be running at the
address in `AEGIS_OLLAMA_URL` with the configured model installed. The value
must match Ollama's actual listening address; it is commonly
`http://127.0.0.1:11434`, but `./scripts/aegis --check` explains how to fix an
unreachable endpoint.

Check readiness without changing canonical state:

```sh
./scripts/aegis --check
```

Start the loopback browser alpha:

```sh
./scripts/aegis --web
```

Open the printed local URL. For a scriptable request, use
`./scripts/aegis --once "Show my tasks."`; add `--json` for the canonical
machine-readable Result envelope. Use `--env-file PATH` when configuration is
stored outside the repository.

To inspect owner feedback without exposing transcripts, run
`./scripts/aegis --feedback --json`. Add `--harvest` to emit only bounded
defect candidates; every candidate requires fresh reproduction and consequential
actions are never replayed automatically.

If a required service is unavailable, `--web` can still open a loopback
diagnostics shell. It will show the same readiness remediation as `--check`,
while protected state and interaction remain fail-closed until the service is
repaired.

Read [docs/ALPHA.md](docs/ALPHA.md) for the browser and workflow contract and
[docs/OPERATIONS.md](docs/OPERATIONS.md) for migrations, backups, and runtime
operations. Architecture and security invariants are documented in
[ARCHITECTURE.md](ARCHITECTURE.md) and [SECURITY_MODEL.md](SECURITY_MODEL.md).

The browser binds to loopback by default. OpenClaw, Home Assistant, mobile,
and voice integrations are optional lanes with separate live-acceptance
requirements; do not expose their services beyond the documented local
boundary.

## Project documents

- [VISION.md](VISION.md) — ideal self-hostable Jarvis-like product.
- [CORE_CONCEPTS.md](CORE_CONCEPTS.md) — durable cognition, truth, and security invariants.
- [ROADMAP.md](ROADMAP.md) — sequencing and maturity gates.
- [MASTER_ROADMAP.md](MASTER_ROADMAP.md) — detailed checkpoint route and proof requirements.
- [CURRENT_STATE.json](CURRENT_STATE.json) — current implementation and evidence truth.
- [ARCHITECTURE.md](ARCHITECTURE.md) — ownership and system boundaries.
- [MODEL_STRATEGY.md](MODEL_STRATEGY.md) — model roles, scale, and evaluation.
- [SECURITY_MODEL.md](SECURITY_MODEL.md) — privacy, authorization, and verification.
- [PACK_SPEC.md](PACK_SPEC.md) — modular Pack contracts.
- [docs/OWNER_DOGFOOD.md](docs/OWNER_DOGFOOD.md) — installed-runtime dogfood protocol.
