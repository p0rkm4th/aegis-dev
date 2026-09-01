# Owner dogfood lane

The local owner dogfood instance is a loopback-only browser service. It is
intended to let an owner use AEGIS at `http://127.0.0.1:18080/` without
running a terminal for each interaction.

The provisioned instance uses:

- the installed package release under
  `/home/scootz/.local/share/aegis-owner/releases/<green-sha>`;
- PostgreSQL canonical state from the persistent acceptance data volume;
- Keycloak OIDC userinfo and the canonical `aegis_principals` subject mapping;
- Ollama `qwen3:8b` at the configured local runtime endpoint;
- the existing Core authorization, verification, correlation, and idempotency
  paths.

The user service is `/home/scootz/.config/systemd/user/aegis-owner.service`.
Luna can inspect or operate it with:

```bash
systemctl --user status aegis-owner.service
systemctl --user restart aegis-owner.service
curl http://127.0.0.1:18080/api/health
curl -i http://127.0.0.1:18080/api/ready
```

The service binds only to loopback. Its bearer and database secrets are held
in the user service manager environment, not in this repository. Consequential
actions continue to use the normal Core policy and independent verification;
browser refresh or service restart never automatically replays a mutation.

Upgrade procedure:

1. Run the full validation gate on a new green commit.
2. Install that commit into a new versioned release directory.
3. Change the service's release path and reload/restart the user service.
4. Check `/api/health`, `/api/ready`, and the authorized Constellation before
   inviting owner use.
5. Preserve the PostgreSQL volume and service logs; never recreate canonical
   state as an upgrade shortcut.

Owner activity is evidence, not an implicit authority grant. Record the
running release SHA, model/provider endpoint, correlation ID, objective ID,
Result state, and any failure classification in `CURRENT_STATE.json` without
recording credentials or private raw payloads.

## Evaluation terminology

**Exact canary green** means one particular tested utterance behaved correctly.
It does not prove the general language capability.

**Capability green** means a varied family of previously unseen natural-language
variants behaved correctly through the installed runtime. Families should
include paraphrases, typos, word-order changes, shorthand, politeness, and
informal language. The evaluation family must not be copied into production as
a synonym list or phrase parser.

**Product failure** means AEGIS may have returned a technically valid Result
but failed the reasonable human objective. **Safety success** means a blocked
or clarification Result was correct because guessing would have been unsafe.
A completed canonical Result is not automatically a successful dogfood result,
and a blocked Result is not automatically a failure.

Retire solved exact canaries to cheap deterministic regressions and move
routine model dogfood to novel language, combinations, and failure boundaries.
Capture enough correlation and bounded evidence for Luna to inspect manual
feedback without treating owner feedback as canonical domain truth.
