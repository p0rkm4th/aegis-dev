# Third-party ledger

| Component | Version/pin | License | Use | Status |
|---|---|---|---|---|
| Pydantic | `>=2.7,<3` | MIT | typed contracts | direct dependency; verify lock at release |
| Psycopg | `>=3.2,<4` (`binary` extra) | LGPL-3.0-only | PostgreSQL canonical state and migrations | direct runtime dependency; verify lock at release |
| Pytest | `>=8,<9` | MIT | tests | development-only |
| Ruff | `>=0.6,<1` | MIT | lint/format | development-only |
| Mypy | `>=1.11,<2` | MIT | type checking | development-only |
| OpenClaw | `2026.8.1` tested release; source audit `bd15d91e0814326f68ac2ca027c228abf5a57e80` | upstream README and `@openclaw/gateway-{client,protocol}` manifests state MIT; repository API has no SPDX result | external runtime/interface; official Node packages are not runtime dependencies | adapter boundary; release-time protocol conformance harvest recorded in `ECOSYSTEM_HARVEST.md` |
| Hades/Odysseus | pushed SHA recorded in `reference/` | AGPL-3.0 | clean-room behavioral reference | no source reuse |
| spaCy / `en_core_web_sm` | `3.8.16` / `3.8.0` benchmark model | MIT engine / CC BY 4.0 model | optional structural-parser adapter for untrusted coverage evidence only | not a Core dependency or vendored model; full benchmark and distribution boundary are recorded in `docs/OBJECTIVE_FIDELITY_STRUCTURAL_BENCHMARK.md` |

The tested direct-dependency inventory is recorded in
`provenance/SBOM.json`. It is intentionally marked incomplete for transitive
locking until a release lockfile is established; the inventory must not be
presented as a complete production SBOM before then.
