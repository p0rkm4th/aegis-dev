# Live vertical-slice acceptance

This procedure is intentionally not part of the simulator test gate. It must
run in an environment where the required services are reachable.

1. Set the real endpoints and credentials through the environment. Do not put
   credentials in this repository:

   ```bash
   export AEGIS_OLLAMA_URL=http://127.0.0.1:11434
   export AEGIS_DATABASE_URL='postgresql://...'
   export AEGIS_OPENCLAW_GATEWAY_URL='ws://...'
   ```

2. Confirm the baseline model and record the returned digest:

   ```bash
   curl --fail "$AEGIS_OLLAMA_URL/api/tags"
   ```

   The response must contain `qwen3:8b`. Exercise it through
   `aegis.ollama.OllamaProvider`, then pass the response to
   `StrictDecisionDecoder`; malformed or invented decisions must result in no
   execution.

3. Apply migrations to a disposable PostgreSQL database and record the exact
   migration version:

   ```bash
   psql "$AEGIS_DATABASE_URL" -v ON_ERROR_STOP=1 -f migrations/001_initial.sql
   ```

4. Supply a tested OpenClaw Gateway transport implementing the existing
   `GatewayClient` boundary. It must preserve the Core correlation and
   idempotency key, enforce runtime policy/approval, and return external
   observation evidence.

5. Run one real workflow, preferably `Add rice to groceries`, through Core,
   PostgreSQL, the OpenClaw Gateway, and the existing Kitchen Pack. Read the
   resulting external/canonical state independently, verify the postcondition,
   restart the process, and replay the same correlation. The replay must not
   duplicate the mutation.

6. Repeat with one consequential authorized service/device action and an
   independent health/state readback. Capture malformed output, invented
   action, semantic denial, runtime denial, execution failure, verification
   failure, disconnect-after-dispatch, and restart-before-completion evidence.

Do not mark the live vertical slice complete from simulator tests, a successful
API response, or a model statement alone. Record commands, versions, digests,
database migration state, result evidence, and limitations in
`CURRENT_STATE.json`.
