# AEGIS end-user alpha

The alpha is an interactive CLI adapter over the existing Core, PostgreSQL
canonical store, Qwen3:8B provider, Pack lifecycle, policy, and OpenClaw
executor. It does not create a second workflow or state store.

## Launch

From the repository root, configure the local runtime endpoints and launch:

```sh
export AEGIS_DATABASE_URL='postgresql://USER:PASSWORD@127.0.0.1:55432/DB'
export AEGIS_OLLAMA_URL='http://127.0.0.1:11434'
./scripts/aegis
```

The launcher uses the repository `.venv` when available and otherwise falls
back to an active `python3` environment with the checkout on `PYTHONPATH`. Set
`AEGIS_PYTHON` to choose a specific Python executable.

Use `./scripts/aegis --help` to inspect the client options. For automation or
health checks that need one request, use non-interactive mode:

```sh
./scripts/aegis --once "Show my tasks."
```

`--once` prints one canonical human-readable result and exits with status 0;
handled request errors print an actionable message and exit with status 1. It
does not bypass Core policy, authorization, execution, or verification. Add
`--no-banner` when embedding the interactive client in a terminal wrapper.

Before starting a workflow, run `./scripts/aegis --check` for readable
configuration/readiness diagnostics, or `./scripts/aegis --check --json` for a
machine-readable report. PostgreSQL and Ollama are required; OpenClaw is
reported as optional until a workflow needs an external mutation. A failed
readiness check exits with status 1 and does not alter canonical state.

The smallest browser proof is available on loopback with:

```sh
./scripts/aegis --web
```

It serves conversation through the same `handle()` boundary and renders a
small authorized view of canonical Pack/domain state, including available and
installed Pack hubs plus Tasks/Kitchen context. The browser adapter
owns only HTTP and presentation; authorization, persistence, and meaning stay
below it. Use `--port` to select another local port; it listens on loopback by
default.

Browser API requests resolve the configured identity again for each request.
Malformed or unavailable identity is returned as an unauthorized response;
state access is still rechecked against current Space/Vault policy below the
browser adapter.

The browser also exposes `/api/health`, using the same structured readiness
report as `--check`. Health is available for diagnosis without exposing
identity-protected state. Message bodies are bounded and malformed or
oversized requests are rejected before reaching Core.

The browser keeps the current conversation visible for the session and locks
the input while a request is in flight, so a double click cannot intentionally
submit the same visible turn twice. Core correlation/idempotency remains the
authoritative protection for retries and process recovery.

Each browser message carries a UUID correlation ID. If transport fails, the
Retry action resubmits that same ID; Core can therefore reuse its durable
Result instead of treating the retry as a new consequential request.

For grocery mutation, also set `AEGIS_OPENCLAW_GATEWAY_URL`,
`AEGIS_OPENCLAW_TOKEN`, `AEGIS_OPENCLAW_DEVICE_TOKEN`, and
`AEGIS_OPENCLAW_IDENTITY_DB`. The CLI defaults to the explicitly configured
local development Principal `alice` / `alice-vault` / `apartment`; a validated
Keycloak bearer token can instead be supplied with
`AEGIS_KEYCLOAK_ISSUER` and `AEGIS_KEYCLOAK_ACCESS_TOKEN`.

Try `Add rice to groceries.`, `What's on my grocery list.`,
`Create a task to buy cat food.`, and `Show my tasks.`. The CLI applies the
existing migrations by default, persists canonical state in PostgreSQL, and
uses independent readback verification. Restarting the CLI does not clear
state. Routine task-list, household, and affordability reads use deterministic
canonical fast paths; model-backed mutations still require Ollama readiness.
Type `quit` to exit.
