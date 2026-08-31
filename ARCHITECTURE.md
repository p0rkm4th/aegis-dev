# Architecture

Core is the sole semantic owner. PostgreSQL owns canonical domain state;
pgvector is retrieval only. Keycloak owns identity, OpenFGA owns relationship
authorization, OpenClaw owns channels/runtime execution, Ollama initially owns
local model serving, and Home Assistant owns device truth. Model output is
untrusted input and never grants authority.

The initial implementation uses in-process ports and fakes so Core can be
tested independently. The Phase 2 boundary will use the OpenClaw Gateway
protocol rather than a fork.
