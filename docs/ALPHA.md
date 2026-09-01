# AEGIS end-user alpha

The alpha is an interactive CLI adapter over the existing Core, PostgreSQL
canonical store, Qwen3:8B provider, Pack lifecycle, policy, and OpenClaw
executor. The reusable `aegis.cli.run_interaction()` boundary returns the
canonical `Result`; CLI and browser code only adapt transport and presentation
around it. It does not create a second workflow or state store.

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
handled request errors print an actionable message and exit with status 1.
Canonical non-completed results (for example denied or incomplete work) also
exit with status 1. It does not bypass Core policy, authorization, execution,
or verification. Add
`--no-banner` when embedding the interactive client in a terminal wrapper.
For machine consumers, add `--json` to `--once`; it emits the canonical
Result fields (`objective_id`, `state`, `message`, `evidence`, and
`correlation_id`) rather than presentation text. JSON request failures return
an error object and exit with status 1.

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
below it. Use `--port` to select another local port (1–65535); invalid values
are rejected before startup. It listens on loopback by default, and bootstrap
or bind failures return an actionable error rather than a traceback.

Browser API requests resolve the configured identity again for each request.
Malformed or unavailable identity is returned as an unauthorized response;
state access is still rechecked against current Space/Vault policy below the
browser adapter.

API failures include a stable `code` alongside generic user-safe `error` text
(`identity_unavailable`, `state_access_denied`, `state_unavailable`,
`invalid_request`, `request_denied`, `request_unavailable`, or
`route_not_found`) so clients do not parse prose to choose their behavior.
Successful message responses use a stable envelope containing `message`,
`correlation_id`, and (when available) canonical `state` and `objective_id`;
undocumented callback fields are rejected before reaching the browser.

The browser also exposes `/api/health`, using the same structured readiness
report as `--check`. Health is available for diagnosis without exposing
identity-protected state. Message bodies are bounded and malformed or
oversized requests are rejected before reaching Core.

The browser keeps the current conversation visible for the session and locks
the input while a request is in flight, so a double click cannot intentionally
submit the same visible turn twice. Core correlation/idempotency remains the
authoritative protection for retries and process recovery.

The state refresh control preserves the last authorized view during a transient
service outage and labels the view as needing refresh. Identity failure or
authorization denial clears the displayed nodes and relationships immediately;
stale private state is never retained across a failed authorization refresh.

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
