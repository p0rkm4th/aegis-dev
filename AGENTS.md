# AEGIS contribution entrypoint

Before substantial architecture, implementation, roadmap, or dogfood work,
consult these repository control surfaces in order:

1. [VISION.md](VISION.md) — ideal end product and North Star.
2. [CORE_CONCEPTS.md](CORE_CONCEPTS.md) — durable invariants and design philosophy.
3. [ROADMAP.md](ROADMAP.md) — product sequencing and maturity gates.
4. [CURRENT_STATE.json](CURRENT_STATE.json) — authoritative implementation and evidence state.
5. [ARCHITECTURE.md](ARCHITECTURE.md) — technical ownership boundaries.
6. [MODEL_STRATEGY.md](MODEL_STRATEGY.md) — cognition and model constraints.
7. [SECURITY_MODEL.md](SECURITY_MODEL.md) — authorization, privacy, and completion guarantees.
8. [PACK_SPEC.md](PACK_SPEC.md) — Pack contracts and extensibility.
9. [docs/OWNER_DOGFOOD.md](docs/OWNER_DOGFOOD.md) — installed-runtime behavior evidence.

CURRENT_STATE chooses immediate executable work, but implementation must not
silently redefine the Vision or Core Concepts. If implementation pressure
conflicts with a core invariant, stop and document the conflict before coding.

GitHub `origin/aegis-dev` is repository truth. Inspect remote HEAD, local HEAD,
worktree, tests, recent commits, and live evidence before acting. Do not assume
source, installed wheels, running services, and last-green releases match.

During one uninterrupted session, do not repeatedly reread large unchanged
documents; use git state, hashes, and mtimes to detect meaningful changes.
