# AEGIS Roadmap

This roadmap records product sequencing. It does not override the active
implementation priorities in `CURRENT_STATE.json`.

## End-user interaction boundary

Stabilize the proven CLI/TUI alpha interaction boundary first. All client
surfaces—CLI/TUI, web, graph, Telegram, voice, and future clients—route into
the same AEGIS interaction boundary and then Core:

```text
CLI/TUI --------\
Web/Graph -------\
Telegram ---------> AEGIS interaction boundary -> Core
Voice -----------/
Other clients ---/
```

## AEGIS CONSTELLATION UI

Placement: after the end-user CLI/TUI alpha interaction boundary is proven and
before later ambient/Jarvis maturity work.

AEGIS is the central intelligence hub. Major Packs and domains appear as hub
nodes around AEGIS: Home, Finance, Tasks, Projects, Memory, Calendar,
Communications, Infrastructure, Travel, and future domains. Domain hubs expand
through semantic zoom into semantic areas and contextual objects/resources.
Cross-domain relationships are first-class graph edges. The graph visualizes
the user's world and the context AEGIS is using; it must not render every
database row as a permanent node.

### Node hierarchy

1. Domain hubs.
2. Semantic areas, such as Finance/Accounts, Home/Kitchen, and
   Infrastructure/Services.
3. Contextual objects, such as accounts, tasks, goals, devices, projects,
   trips, bills, and services.

### Interaction

The graph supports navigation, relationships, context, state, and activity.
Conversation/chat remains a primary interaction surface. Selecting a node may
open conventional panels, tables, cards, charts, timelines, or forms. Requests
may illuminate relevant domains/resources. Active work may show:

```text
proposal -> authorization -> execution -> observation -> verification
         -> completion/approval
```

Conventional and accessibility-friendly navigation remains an alternative to
spatial interaction.

### Security and privacy

Vault and Space boundaries must be visually understandable. Hidden resources
must never be revealed through the graph. Graph reachability never implies
authorization. The UI consumes canonical authorization results from Core and
OpenFGA.

### Pack extensibility

The UI must not create a second hard-coded frontend domain ontology. Packs may
optionally expose stable UI metadata for grouping, labels, categories, icons,
relationships, and preferred detail views. Installing a Pack can add a new
constellation without redesigning Core or the frontend. UI metadata grants no
authority.

### Delivery order

1. Do not delay the current end-user alpha for the graph UI.
2. Stabilize CLI/TUI and the interaction boundary first.
3. Build a minimal browser-based Constellation prototype against that same
   boundary.
4. Add domain detail panels incrementally.
5. Prioritize real workflows over decorative graph complexity.
6. Preserve conventional/accessibility-friendly navigation as an alternative.

The spatial concept may draw inspiration from the network/constellation
presentation discussed with the owner for the Hackers game node map, while
remaining a functional AEGIS UI rather than a game interface.
