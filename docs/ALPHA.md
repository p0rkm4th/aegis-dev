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
state. Type `quit` to exit.
