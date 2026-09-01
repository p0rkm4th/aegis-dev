# Ecosystem harvest audit

Audit date: 2026-09-01. Pins below are upstream default-branch heads observed
on this date unless a release pin is stated. This is a reuse decision record,
not permission to import code with unclear provenance.

## Decisions

| Candidate / exact upstream path | License / pin | Classification | Capability, cost, and ownership decision |
|---|---|---|---|
| [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent): `scripts/install.sh`, `hermes_cli/__init__.py`, `hermes_cli/setup.py`, `hermes_cli/doctor.py`, `hermes_cli/gateway.py`, `apps/desktop/src/app/session/hooks/use-message-stream/gateway-event/` | MIT; `18a76be124d7c16ed98b629a358b23fef76a7f46` (`v0.21.0` in `hermes_cli/__init__.py`) | DESIGN-HARVEST, TEST-HARVEST | Strong install/bootstrap, setup/status/doctor, service, channel, voice/mobile, and reconnect UX patterns. Porting would add a second agent/gateway model and substantial dependencies. Do not import agent semantics, memory, allowlists, or policy. No AEGIS deletion. |
| [openclaw/openclaw](https://github.com/openclaw/openclaw): `packages/gateway-client/`, `packages/gateway-protocol/`, `packages/gateway-client/src/{client.ts,protocol-client.ts,device-auth.ts,reconnect-policy.ts}`, `apps/desktop/src/store/gateway-reconnect.ts`, `apps/android/app/src/main/java/ai/openclaw/app/gateway/` | Repository API reports no SPDX identifier; upstream README and both published package manifests state MIT. Packages `2026.8.1`; source `bd15d91e0814326f68ac2ca027c228abf5a57e80` | ADAPT interface; TEST-HARVEST; no direct Node runtime | Official packages provide v4 frame validation, challenge/device auth, correlation, timeouts, reconnect, events, browser helpers, and lifecycle tests. A Node bridge adds packaging, IPC, secret, and failure boundaries. Keep AEGIS Core and Python transport for now; use release-time schema/conformance generation. No current deletion. |
| [cytoscape/cytoscape.js](https://github.com/cytoscape/cytoscape.js): `src/`, `dist/`, `playwright-tests/`, `test/` | MIT; release `v3.33.4`; audit head `fd3595bbf0eaac76ef2a6984a29e85703c239703` | REJECT-FOR-NOW; DESIGN-HARVEST | Strong graph model/layout/interaction and browser tests, but adds npm supply chain to a dependency-free proof. If later adopted, it consumes only authorized Core projection. No deletion. |
| [xyflow/xyflow](https://github.com/xyflow/xyflow): `packages/react/`, `packages/system/`, `packages/react/src/additional-components/`, `packages/react/src/components/A11yDescriptions/` | MIT; `@xyflow/react` `12.11.5`; source `0a1f9575b25679f2880175de8d3eae21aedde921` | REJECT-FOR-NOW; DESIGN-HARVEST | Best future fit for functional node UI, controls, custom nodes, accessibility, and React tests, but requires React/npm and presentation state. Revisit once the API projection is stable. No deletion. |
| [Textualize/textual](https://github.com/Textualize/textual): `src/textual/app.py`, `src/textual/pilot.py`, `docs/guide/testing.md`, `tests/test_pilot.py` | MIT; source `06dbeef4bb70fb718236aa418ed658ef4667a126` | DESIGN-HARVEST; REJECT-FOR-NOW | Useful TUI composition and pilot tests, but current scriptable CLI needs no new runtime dependency. Future TUI may adapt it as a client only. No deletion. |
| [tiangolo/typer](https://github.com/tiangolo/typer): CLI parser/help surface | MIT; source `99eb220df7c69a0f14a0b69214042677e0760b9d` | REJECT-FOR-NOW; DESIGN-HARVEST | Friendly typed CLI patterns, but argparse already provides the needed help, `--once`, `--check`, and `--web` with no dependency. No deletion. |
| [fastapi/fastapi](https://github.com/fastapi/fastapi): application/dependency/testing patterns | MIT; source `b93bf331621c7332dfba54d706fa3bafc1be1650` | REJECT-FOR-NOW; DESIGN-HARVEST | Strong schema/OpenAPI/ASGI/TestClient patterns, but adds Starlette/Uvicorn/httpx before AEGIS needs them. Revisit for authenticated multi-client streaming. No deletion. |
| [zellij-org/zellij](https://github.com/zellij-org/zellij): server/daemon, web client, plugin model | MIT; `v0.44.3` release / `0.45.0` workspace | REJECT | Terminal workspace, not an interaction or service-lifecycle library. Operational dependency is disproportionate. No reuse. |

## OpenClaw protocol special review

`@openclaw/gateway-client` is a Node/browser reference client, not a Python
library. Its published README assigns host callbacks responsibility for device
identity persistence, signing, token storage, transport, and product-specific
close/reconnect behavior, while the package owns challenge ordering, frame
correlation, timeout cleanup, sequence gaps, and reconnect scheduling. The
protocol package owns typed schemas and runtime validators.

The smallest durable AEGIS path is:

1. Pin OpenClaw `2026.8.1` and exact source/package hashes as a compatibility
   release input.
2. At development/release time, consume
   `packages/gateway-protocol/src/public-schema.ts`, `src/schema/`, and frame
   guards to generate or refresh checked-in Python frame DTOs and conformance
   fixtures. This is a build-time tool, not a Node sidecar.
3. Test challenge ordering, exact v4 connect envelope, device signature inputs,
   response correlation, event buffering, timeout, disconnect/unknown outcome,
   and reconnect safety against those fixtures.
4. Keep the minimal Python transport and AEGIS-specific secret/lifecycle
   callbacks until generated coverage proves replacing it reduces risk.

This removes manually maintained protocol drift without moving semantic
authorization, sessions, or completion into OpenClaw. A Node bridge is rejected
for now: it would add a Python↔Node RPC contract, a second secret-bearing
process, another lifecycle/error boundary, and Node as a new install
requirement for the Python alpha.

## UI, operator, and channel conclusions

The strongest harvest is test/design material, not a framework swap. Use
Hermes/OpenClaw setup, doctor, gateway reconnect, approval, channel, voice,
and browser tests as TEST-HARVEST/DESIGN-HARVEST. When AEGIS’s interaction and
authorized state projection are stable enough for a real frontend, evaluate
`@xyflow/react` first for graph navigation plus conventional detail panels,
with Cytoscape.js as the fallback if graph analysis/layout becomes dominant.
Keep the inline browser until migration has a measured capability gain.

OpenClaw channel and Android/voice implementations are reference integrations,
not candidates for direct AEGIS adoption: they assume OpenClaw session,
pairing, and gateway authority. Preserve typed AEGIS seams and mark physical
mobile/voice acceptance separately.

Current AEGIS reuse status: PostgreSQL, Pydantic, pytest, Ruff, mypy, and the
OpenClaw runtime interface are already present and recorded in
`THIRD_PARTY.md`/`provenance/SBOM.json`. No upstream implementation is copied
by this audit.
