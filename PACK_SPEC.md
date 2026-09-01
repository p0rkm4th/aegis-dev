# Pack specification

A Pack declares identity, version, permissions, schemas, capabilities, action
contracts, integrations, verification behavior, and migrations. Core retrieves
bounded ActionCards from registered Packs and does not gain domain-specific
branches when a Pack is installed.

Packs may optionally declare non-authoritative `ui` metadata in their manifest:
`label`, `category`, and `detail_view`. The browser may use these hints for
labels and grouping, but they grant no permission, do not define canonical
state, and cannot bypass Core policy or verification. A missing `ui` block is
valid and falls back to a safe generated label.
