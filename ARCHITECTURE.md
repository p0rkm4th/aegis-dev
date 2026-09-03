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
