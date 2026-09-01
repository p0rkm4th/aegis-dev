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

4. Use the pinned OpenClaw compatibility tuple. These are separate values:

   ```text
   root CLI/Gateway release:       2026.8.1
   @openclaw/gateway-client:       2026.8.1
   @openclaw/gateway-protocol:     2026.8.1
   Gateway wire protocol:          v4
   ```

   The AEGIS transport uses the documented protocol-4 WebSocket handshake. A
   backend client that requests operator write/admin scopes must use a paired
   device identity; the shared Gateway token alone is insufficient for those
   scopes. Keep the Gateway loopback-only and keep the device token/private key
   outside Git.

   For the disposable acceptance Gateway, pairing was performed with:

   ```bash
   export OPENCLAW_STATE_DIR=/tmp/aegis-openclaw-state
   export OPENCLAW_CONFIG_PATH=/tmp/aegis-openclaw-state/openclaw.json
   openclaw devices list --token "$OPENCLAW_TOKEN" --json
   openclaw devices approve "$REQUEST_ID" --token "$OPENCLAW_TOKEN" --json
   ```

   `OpenClawWebSocketChannel` must keep one persistent authenticated socket
   open across `terminal.open`, `terminal.input`, terminal events, and
   `terminal.close`; closing after input can leave the outcome unknown. The
   `terminal.data` event and an independent external read are the evidence,
   not the RPC acknowledgement.

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

## Current live transport evidence

On 2026-08-31, the disposable loopback Gateway was running OpenClaw 2026.8.1
with a paired device. Through the AEGIS protocol-4 WebSocket channel, `health`
and `terminal.open` succeeded; `terminal.input` returned an acknowledgement;
the same persistent connection then delivered `terminal.data` containing the
command; and an independent read of `/tmp/aegis-openclaw-external-proof`
returned `AEGIS_GATEWAY_EXEC_OK`. This proves the authenticated transport and
external observation boundary. It does not by itself prove the complete
Core-to-Pack workflow; that remains the next acceptance task.
