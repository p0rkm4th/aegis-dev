# Third-party ledger

| Component | Version/pin | License | Use | Status |
|---|---|---|---|---|
| Pydantic | `>=2.7,<3` | MIT | typed contracts | direct dependency; verify lock at release |
| Pytest | `>=8,<9` | MIT | tests | development-only |
| Ruff | `>=0.6,<1` | MIT | lint/format | development-only |
| Mypy | `>=1.11,<2` | MIT | type checking | development-only |
| OpenClaw | `2026.8.1` tested release | review exact pinned distribution | external runtime/interface | adapter boundary |
| Hades/Odysseus | pushed SHA recorded in `reference/` | AGPL-3.0 | clean-room behavioral reference | no source reuse |

The tested direct-dependency inventory is recorded in
`provenance/SBOM.json`. It is intentionally marked incomplete for transitive
locking until a release lockfile is established; the inventory must not be
presented as a complete production SBOM before then.
