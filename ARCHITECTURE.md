# Architecture

Core is the sole semantic owner. PostgreSQL owns canonical domain state;
pgvector is retrieval only. Keycloak owns identity, OpenFGA owns relationship
authorization, OpenClaw owns channels/runtime execution, Ollama initially owns
local model serving, and Home Assistant owns device truth. Model output is
untrusted input and never grants authority.

The initial implementation uses in-process ports and fakes so Core can be
tested independently. The Phase 2 boundary will use the OpenClaw Gateway
protocol rather than a fork.

## Capability spaces

AEGIS separates cognitive possibility from executable authority. ActionCards
are an executable vocabulary, not the complete universe of outcomes AEGIS may
reason about.

The architecture recognizes three capability classes:

1. **Stable typed domain capabilities** — durable Pack contracts with explicit
   schemas, permissions, runtime bindings, observation, and verification.
2. **General-purpose scoped capabilities** — replaceable runtime providers
   that can investigate or operate within an explicit authority envelope,
   bounded inputs, budgets, and stop conditions. They remain subordinate to
   Core semantics and do not become authority merely because they are
   reachable.
3. **Acquired or generated reusable capabilities** — discovered, adapted, or
   built behavior that must pass sandboxing, testing, permission analysis,
   installation/enablement, and the ordinary approval and verification
   lifecycle before it can execute.

Core may preserve an unresolved capability need and investigate or escalate
it; it must not silently drop that requirement or declare the objective
complete. Models and workers may propose a capability, procedure, or
acquisition path, but may not install, enable, authorize, or verify it.

## Recovery-first objective runtime

An internal blockage is recovery input, not an automatic owner dead end.
Objective JSON carries a small orthogonal recovery state with a bounded step
budget, failure fingerprint, and capability/provider counters. Core-owned
coordination may inspect already-authorized canonical or public evidence and
re-enter the ordinary validation path. A persisted legacy `BLOCKED` objective
without recovery metadata remains owner-visible and is never silently
reawakened. Owner blockage is emitted only when bounded recovery cannot safely
continue without owner input; recovery never changes Kernel authority,
approval, verification, or external-unknown semantics.

## Compounding owner corrections

Interaction output may produce a bounded `OWNER_CORRECTION` learning
candidate, but only Core-validated, provenance-bearing, Vault-scoped owner
evidence enters canonical PersonalState. The correction path grounds
replacement content in the current owner utterance, supersedes the old
MemoryRecord, and reuses normal retrieval; transcripts, assistant prose, and
retrieved evidence are not canonical memory. Learned context can inform
identity, but never creates permission or authority.

## Development sprint control plane

Luna's repo-associated development sprint state is a separate engineering
control plane, not an AEGIS Objective and not canonical owner state. It may
persist bounded task dependencies, evidence, failures, leases, and human
blockers so work can resume without reconstructing a campaign from chat.

A task that needs Scotty is persisted as `WAITING_HUMAN`; independent approved
ready tasks remain runnable. Worker output is evidence only: deterministic
tests, hosted validation, and installed checks establish engineering success.
The sprint scheduler never grants product authority, changes Kernel semantics,
or turns a model/worker report into completion.
