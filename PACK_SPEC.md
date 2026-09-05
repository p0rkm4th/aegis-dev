# Pack specification

A Pack declares identity, version, permissions, schemas, capabilities, action
contracts, integrations, verification behavior, and migrations. Core retrieves
bounded ActionCards from registered Packs and does not gain domain-specific
branches when a Pack is installed.

Pack installers may register one runtime factory per ActionCard through the
typed `PackRuntimeRegistry`. Registration is atomic and must cover every
declared action; each resolved binding supplies a separate executor, verifier,
and policy map. This seam does not grant authority: Core still validates the
proposal, rechecks authorization, observes execution, verifies canonical state,
and decides completion.

Packs may optionally declare non-authoritative `ui` metadata in their manifest:
`label`, `category`, and `detail_view`. The browser may use these hints for
labels and grouping, but they grant no permission, do not define canonical
state, and cannot bypass Core policy or verification. A missing `ui` block is
valid and falls back to a safe generated label.

Consequential ActionCards may also declare `argument_grounding` rules. A Pack
grounder may propose explicit utterance spans, authorized canonical references,
approved deterministic derivation IDs, or a predeclared approved default for
each argument. These remain proposals: Core independently verifies the value,
current utterance, authorized context, declared rule, resolver ID, or default
contract before authorization. Optional unstated arguments remain omitted.
The same contract applies to ACTIONs and every PLAN step. Grounding never
grants or caches authorization, execution, observation, verification, or
completion.
