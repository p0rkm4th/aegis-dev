# AEGIS security model

This document describes the security boundary for the interactive alpha. It
is an engineering contract, not a claim that a self-hosted deployment is safe
against a compromised host or database.

## Protected assets

- Vault-private memories, projects, goals, finance data, credentials, and
  conversation-derived context.
- Explicitly shared Space projections and membership/role state.
- Canonical PostgreSQL state, audit records, objective lifecycle, and
  idempotency records.
- Runtime credentials, identity tokens, Pack permissions, and external
  integration endpoints.
- Integrity of action proposals, observations, verification evidence, and
  completion claims.

## Trust boundaries

1. **Client to Core.** CLI, browser, and future clients are untrusted
   transports. They may submit bounded text/JSON, but do not define identity,
   policy, canonical state, or completion.
2. **Identity to Core.** A bearer token is authenticated and mapped to a
   canonical Principal below model reasoning. Missing, malformed, expired, or
   revoked identity fails closed.
3. **Core to model/provider.** Model endpoints are replaceable and potentially
   mistaken, compromised, or prompt-injection-sensitive. Their output is an
   untrusted proposal decoded against a bounded schema and candidate set.
4. **Pack to Core.** Pack manifests, metadata, schemas, executors, and
   verifiers are extension inputs. Declared permissions are necessary but do
   not grant access to other Vaults, Spaces, or capabilities.
5. **Core to external runtime.** OpenClaw, Home Assistant, network targets,
   and other integrations are separate execution boundaries. Transport
   reachability is not authorization, and an accepted command is not proof of
   its objective.
6. **Core to storage/retrieval.** PostgreSQL is canonical structured truth.
   Embeddings, caches, projections, and retrieved context are derived and
   non-authoritative.

## Attacker capabilities and abuse paths

The threat model includes a hostile user sharing a client, a stolen or stale
identity token, a malicious Pack or Pack upgrade, malicious retrieved text,
hostile model output or model endpoint, forged OpenClaw payloads, replayed
executions, browser/loopback requests, SSRF-style endpoint input, and a user
whose Space membership is revoked during an interaction. A database or host
compromise is treated as an infrastructure compromise; application controls
cannot restore confidentiality after the attacker has unrestricted database or
process access.

Important abuse paths include:

- using a model-generated action, transcript instruction, or retrieved memory
  to bypass authorization or invent a canonical fact;
- using an authorized projection, graph edge, error, or stale browser cache to
  disclose another Vault or Space;
- changing a read into a mutation, completing the wrong ambiguous target, or
  replaying a consequential action after a timeout/crash;
- submitting a Pack with overly broad metadata, mismatched permission/schema,
  an incompatible verifier, or an executor that reaches undeclared targets;
- treating HTTP success, transport reachability, or an executor response as
  independently verified objective completion;
- injecting private or hostile retrieved text into a model prompt so it is
  treated as instructions or authority;
- directing a server-side integration to an attacker-chosen host or network
  scope through unvalidated arguments;
- racing revocation, retry, disable, removal, or Pack upgrade against an
  in-flight objective.

## Required mitigations

- Resolve Principal and current Vault/Space policy before constructing an
  authorized working set. Recheck policy at the execution boundary.
- Keep the lifecycle explicit:

  `intent -> objective -> proposal -> authorization -> execution ->
  observation -> verification -> canonical result -> completion`.

- Decode model output strictly. Candidate ActionCards, argument schemas,
  declared permissions, and canonical grounding are deterministic Core
  constraints. Models cannot invent tools, permissions, facts, or success.
- Treat transcript and retrieval output as labeled context only. Never promote
  them to canonical state without an authorized, verified action.
- Require independent postconditions for consequential actions. Unknown or
  partial outcomes remain UNKNOWN/INCOMPLETE; retries preserve correlation and
  idempotency and never blindly replay a consequential mutation.
- Keep Vault private by default. Project only explicitly authorized Space
  data, filter graph/detail projections through the same policy boundary, and
  clear protected client state after revocation or identity loss.
- Bind external actions to declared Pack runtime/verifier contracts and
  allowlisted network/target policy. Validate URLs, addresses, paths, payload
  sizes, and integration identities at the boundary; do not let user/model
  text select unrestricted destinations.
- Validate Pack manifests, permission coverage, schemas, provenance, and
  lifecycle transitions. Treat upgrades/removal as compatibility and
  authorization events, not metadata-only changes.
- Bind browser services to safe local interfaces by default, reject malformed
  or oversized requests before Core, avoid traceback/secret disclosure, and
  make source/install/runtime mismatches diagnosable without exposing secrets.
- Use tamper-evident correlated audit records for identity, policy,
  execution, observation, verification, and feedback events.

## Testable security invariants

The deterministic suite and controlled owner dogfood must continue to prove:

- wrong Vault, wrong Space, missing identity, malformed identity, revoked
  membership, and stale permissions cannot read or mutate protected state;
- model prompt injection, malicious Pack metadata, unsupported actions, and
  malformed proposals fail closed without granting authority;
- ambiguous targets clarify, read-only requests do not mutate, and model or
  provider failure cannot widen permissions or claim completion;
- duplicate requests and crash/restart recovery preserve correlation and do
  not duplicate consequential mutations;
- timeout/unknown observation is never converted into assumed failure or
  success;
- graph nodes, edges, conventional detail views, error messages, and feedback
  responses contain no unauthorized private data;
- executor output is insufficient without independent canonical verification;
- external target/network scope is explicit and SSRF-like destination changes
  are rejected;
- Pack disable/removal/upgrade and revocation take effect at the runtime
  authorization boundary.

## Residual risks

A self-hosted alpha cannot protect secrets from a compromised host, kernel,
database administrator, or model endpoint with process-level access. External
integration acceptance may also remain lane-limited when hardware, accounts, or
credentials are unavailable. Those limitations must be recorded as missing
evidence, never replaced with simulated success. The project must preserve
the fail-closed behavior and verification seams so later deployments can add
stronger isolation, approval UX, secrets management, and external attestations
without moving authority into the model or client.
