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

   The repository includes a repeatable acceptance runner for this path:

   ```bash
   AEGIS_DATABASE_URL='postgresql://...' \
   AEGIS_OLLAMA_URL='http://...' \
   AEGIS_OLLAMA_MODEL='qwen3:8b' \
   AEGIS_OPENCLAW_GATEWAY_URL='ws://127.0.0.1:18789' \
   AEGIS_OPENCLAW_TOKEN='...' \
   AEGIS_OPENCLAW_DEVICE_TOKEN='...' \
   AEGIS_OPENCLAW_IDENTITY_DB='/path/to/openclaw.sqlite' \
   AEGIS_LIVE_GROCERY_PATH='/tmp/aegis-live-groceries.tsv' \
   python scripts/live_vertical_slice.py
   ```

   The runner creates a fresh correlation when one is not supplied, uses a
   new Core/store/channel instance for the initial request and replay, and
   reports canonical Results plus the independently read external record
   count. Run it once per fresh external-state path.

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

## Verified grocery vertical slice

On 2026-08-31, the live runner completed the existing Kitchen
`kitchen.groceries.add` ActionCard using Qwen3:8B, PostgreSQL 16.4, and the
paired OpenClaw 2026.8.1 Gateway. Correlation
`4fd15a56-1622-42e9-8f30-58acc467fa76` produced a completed canonical Result;
the Gateway terminal event was observed; an independent read of the external
grocery record found exactly one `rice` record; and a fresh process replay
returned the same completed Result with the same objective and exactly one
external record. PostgreSQL had been restarted before this replay.

This proves the first simple end-to-end live capability. It does not claim
interrupted-before-completion recovery, a consequential Homelab/HA action, or
live hardware acceptance.

The interrupted-run probe was subsequently executed on the same live stack:
the real OpenClaw dispatch completed, Core was intentionally failed before
Result persistence, and a fresh Core/PostgreSQL process retried the same
correlation. Independent readback completed successfully and found exactly one
external record. The probe used correlation
`7f5e7a10-ec5b-4a7e-b653-1212b4e0f4c0`.

The runner’s semantic policy reads active membership from PostgreSQL. In a
live revocation probe, Alice’s active `apartment` membership allowed
`kitchen.write`; setting that membership inactive immediately denied the same
authorization request below the model and Gateway layers. The fixture was
restored to active afterward.

The live runner discovers all three reference Packs from manifest-backed
`PackBundle` values, then installs/enables Kitchen with `kitchen.write` before
retrieving its bounded ActionCard. The live grocery proof therefore crosses
Pack discovery/lifecycle and Core; it does not rely on a Kitchen-specific Core
branch.

The live runner also uses `PostgresAuditLog`. After migration
`002_audit_hash_chain.sql`, a fresh process loaded the persisted objective
creation, action observation, and result events; the audit chain contained four
events and `verify()` returned true.

OpenFGA acceptance is separate from authentication. The disposable OpenFGA
server was exposed only at `127.0.0.1:58080`, using image digest
`sha256:78d1fa601d42340ecb131305d80d3767d0f254f9b1bc3646f9a557e11b24c63a`.
Through `OpenFGAHttpClient` and the existing `OpenFGAAuthorization` port, the
real `/stores/{store}/check` API allowed Alice’s `can_read` tuple and denied
Bob’s missing tuple. Keycloak authentication remains a separate integration
checkpoint.

The disposable Keycloak identity runtime was then started at
`http://127.0.0.1:58081` from `quay.io/keycloak/keycloak:26.7.2`, resolved to
image digest
`sha256:9d1f1b2b7261ff53c66cb1092dfcdc34a5fb77e81f9e6a6e75b8b6a795de8067`.
The management `/health/ready` endpoint returned `UP` on port 9000 inside the
container. OIDC discovery, token issuance, and `/realms/aegis/.../userinfo`
returned the disposable Alice subject. `KeycloakIdentityProvider` then maps
that validated subject to canonical Vault/Space context; it does not validate
tokens or place identity rules in prompts.
