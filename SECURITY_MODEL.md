# Security model

- Vault data is private by default; Space projections are explicit.
- Model proposals are validated, then checked by deterministic semantic policy.
- Runtime policy and approval are separate gates.
- Reachability never implies authorization.
- A command result is not an outcome; postconditions require canonical evidence.
- Stable correlation and idempotency keys prevent blind replay.
- Packs receive declared least-privilege permissions.
