# AEGIS MASTER ROADMAP

> **Purpose:** This is the durable execution map from a greenfield AEGIS Core to a mature, self-hostable, personalizable Jarvis-style intelligence platform.
>
> This file is deliberately more detailed than `ROADMAP.md`. `ROADMAP.md` should remain the readable product map. This file is the implementation campaign that Luna/Codex can follow checkpoint by checkpoint.
>
> **Do not treat this document as current state.** `CURRENT_STATE.json` records what is actually implemented, proven, running, blocked, and green. This document records the intended road.
>
> **Do not silently redefine this roadmap from implementation pressure.** If a checkpoint reveals a real architectural conflict, document the conflict and update the roadmap deliberately.

---

## 0. DOCUMENT AUTHORITY

AEGIS repository control surfaces should have distinct jobs:

1. `VISION.md`
   - The snap-your-fingers end state.
   - What AEGIS should be if the product were mature tomorrow.
   - Stable North Star.

2. `CORE_CONCEPTS.md`
   - Durable design philosophy and invariants.
   - Stable unless project direction intentionally changes.

3. `ROADMAP.md`
   - Human-readable product sequence and major milestones.
   - Short enough to read regularly.

4. `MASTER_ROADMAP.md`
   - This file.
   - Detailed execution checkpoints, dependencies, proof, dogfood, exit criteria, and non-goals.
   - Luna should use this to choose the next unproven checkpoint.

5. `CURRENT_STATE.json`
   - Authoritative present-tense implementation and evidence state.
   - Current branch/head, last green, running release, tests, live evidence, blocked lanes, next checkpoint.
   - Never infer current truth from this roadmap.

6. `ARCHITECTURE.md`, `MODEL_STRATEGY.md`, `SECURITY_MODEL.md`, `PACK_SPEC.md`
   - Detailed contracts for their respective areas.

7. `docs/OWNER_DOGFOOD.md`
   - Evaluation doctrine for real installed-runtime behavior.

If documents conflict:
- current runtime/code/evidence truth comes from GitHub + `CURRENT_STATE.json`;
- architectural invariants come from `CORE_CONCEPTS.md`, `ARCHITECTURE.md`, `SECURITY_MODEL.md`, and `MODEL_STRATEGY.md`;
- intended sequence comes from this roadmap;
- the mature end-state comes from `VISION.md`.

---

# 1. NORTH STAR

AEGIS should mature into a **self-hostable, personalizable Jarvis-style intelligence platform**.

A person, household, family, or explicitly shared group installs Core, chooses models appropriate to their hardware and budget, and adds domains through modular Packs.

The mature user should experience **one coherent intelligence**, not a menu of apps or a pile of commands.

AEGIS should naturally understand and coordinate, where explicitly enabled:

- memory and knowledge;
- goals, projects, tasks, reminders, and calendars;
- communications;
- household responsibilities and shared Spaces;
- food, groceries, pantry, recipes, and meal planning;
- finance, budgets, bills, liabilities, investments, and forecasting;
- smart home and Home Assistant;
- devices, infrastructure, networks, homelab, and vehicles;
- documents and research;
- relationships / CRM;
- travel;
- business operations;
- wellness data;
- future domains added without redesigning Core.

The mature experience should support:

- natural conversation;
- typo, shorthand, paraphrase, and informal-language tolerance;
- real multi-turn context;
- grounded cross-domain reasoning;
- safe action;
- verification before success is claimed;
- proactive but bounded assistance;
- browser, desktop, terminal, mobile/chat, and voice/ambient interaction;
- local-first operation with replaceable models;
- selective stronger-model use only when worth the cost;
- usefulness on approximately 8B models as the baseline;
- useful bounded operation on 2B–4B models where harness structure can compensate.

"Jarvis-like" describes **continuity, context, initiative, semantic understanding, and coherent action**, not literal sentience.

---

# 2. NON-NEGOTIABLE CORE PRINCIPLES

Every milestone below inherits these rules.

## 2.1 Models are cognition, never authority

Models may:
- interpret natural language;
- resolve likely intent;
- choose among bounded candidate capabilities;
- reason over authorized context;
- synthesize grounded answers;
- propose actions;
- help decompose objectives;
- write non-authoritative creative/general content.

Models may not:
- grant permissions;
- invent authority;
- create new executable capabilities merely by naming them;
- decide what canonical state is true;
- own persistence;
- bypass Vault/Space boundaries;
- claim execution succeeded without observation and verification;
- declare consequential objective completion without Core evidence.

## 2.1.1 Capability boundary is not the cognitive ceiling

The capability boundary constrains execution authority, not cognitive
expressiveness. AEGIS may understand and investigate objectives beyond its
installed typed capabilities. A missing capability may remain an unsatisfied
objective requirement and lead to bounded investigation, composition through
general-purpose scoped tools, capability discovery, adapter or Pack
generation, or owner escalation.

Models and workers may propose capabilities, procedures, implementations, and
acquisition paths, but proposals grant no authority. Discovery is not
permission; generation is not installation; installation is not enablement;
enablement is not approval; reachability is not authority; and tool success is
not objective completion. Core continues to own objective meaning, canonical
truth, permission, lifecycle, verification, completion, and parent-objective
resumption.

ActionCards remain the strict vocabulary of executable candidates, not the
complete universe of outcomes AEGIS may reason about. General-purpose tools
are replaceable providers inside explicit authority envelopes. Reusable
acquired behavior should converge toward typed, testable capability contracts.

## 2.2 Determinism protects truth and authority, not English vocabulary

The division of labor is:

> **Determinism decides what is true and permitted.**  
> **Semantic retrieval decides what is probably relevant.**  
> **Models decide what the human probably means and what should be proposed.**  
> **Core decides what may actually happen.**  
> **Verification decides whether it did.**

Deterministic fast paths are valuable for:
- high-confidence canonical reads;
- exact protocol/schema validation;
- policy;
- authorization;
- state machines;
- idempotency/correlation;
- verification;
- safety boundaries;
- known cheap operations.

They must not evolve into a hand-built English parser.

Do not fix natural-language failures by indefinitely adding:
- synonyms;
- regexes;
- phrase templates;
- verb lists;
- special-case branching.

If deterministic interpretation is not high-confidence, use bounded semantic cognition.

## 2.3 Canonical truth remains outside the model

- PostgreSQL owns canonical structured state.
- pgvector or equivalent is retrieval only.
- Model inference may be stored as a proposal, note, hypothesis, or derived artifact only when its provenance and epistemic status are explicit.
- Financial truth is never inferred when a canonical provider/store should supply it.
- External execution truth comes from independent observation/readback where feasible.

## 2.4 Objective completion is stronger than tool success

The canonical path is:

```text
intent
-> objective
-> proposal
-> authorization
-> execution
-> observation
-> verification
-> canonical result
-> completion
```

A successful API/tool call is not sufficient proof that the human objective succeeded.

## 2.5 Modularity is an invariant

New domains should mostly require:
- Pack schemas;
- capabilities;
- permissions;
- events;
- actions;
- verification contracts;
- retrieval hooks;
- model-facing descriptions;
- optional UI metadata;
- adapters.

New domains should not require repeated Core redesign.

Cross-domain composition belongs in Core.

## 2.6 Small-model friendliness is architectural

Before escalating model size:
- reduce context;
- shortlist candidates;
- improve schemas;
- decompose work;
- clarify state;
- improve retrieval;
- strengthen deterministic policy and verification;
- separate planning from execution;
- make errors recoverable.

Approximately 8B local models are the primary baseline.
2B–4B usefulness is a design target.
Stronger models are optional escalation, not a dependency for basic use.

## 2.7 Open source is the supply chain

Before building commodity functionality:
1. search broadly;
2. inspect licenses and maintenance;
3. classify candidates:
   - DIRECTLY USE
   - PORT
   - ADAPT
   - FORK
   - VENDOR
   - TEST-HARVEST
   - DESIGN-HARVEST
   - REJECT
   - ALREADY PRESENT
4. record provenance;
5. prefer reuse where it saves meaningful time without compromising architecture.

## 2.8 Dogfood proves capability, not phrases

## User-competency dogfood gate

AEGIS must be dogfooded as if operated by users with widely different levels
of technical knowledge.

Do not assume users know:
- Pack names;
- capability names;
- action vocabulary;
- system architecture;
- databases;
- permissions terminology;
- networking concepts;
- model terminology;
- how to phrase requests precisely.

Routine dogfood should deliberately rotate among user styles.

### Nontechnical / casual user

Examples:
- "make that thing from earlier done"
- "do I need to buy anything?"
- "the wifi thing is being weird again"
- "can you remind me about that tomorrow?"
- "why am I broke this month?"
- "make the house ready before everyone gets here"

Expect incomplete terminology, vague references, typos, and no knowledge of
AEGIS internals.

AEGIS should infer what it safely can and clarify only what actually matters.

### Typical competent consumer

Examples:
- "add milk to the grocery list"
- "what tasks do I have due this week?"
- "can I afford to spend $80 tonight?"
- "schedule dinner with Sarah Friday"
- "is anything wrong with my home network?"

User understands apps and ordinary technology but not infrastructure details.

### Enthusiast / power user

Examples:
- "show me tasks due before Friday and prioritize anything blocking the backup"
- "check whether the NAS is reachable and whether its backup service is healthy"
- "compare this month's discretionary spending against the previous three months"

User gives better context and may expect deeper control, but still should not
need internal AEGIS syntax.

### Technical professional / administrator

Examples:
- "probe the authorized 192.168.10.0/24 scope and summarize hosts with unusual open services"
- "restart the backup service if health verification fails, but don't reboot the host"
- "show me provenance for the finance numbers used in that recommendation"

AEGIS may expose more technical detail when useful, but authorization and
verification rules remain identical.

### Domain expert who is not technically sophisticated

Examples include users highly knowledgeable about:
- finance;
- cooking;
- medicine/wellness;
- business operations;
- travel;
- vehicles;

but unfamiliar with computing terminology.

Do not equate domain expertise with technical expertise.

### Impatient / sloppy user

Test:
- fragments;
- misspellings;
- voice-like phrasing;
- corrections;
- contradictory follow-ups;
- missing nouns;
- excessive pronouns;
- shorthand;
- accidental duplicate submissions.

Examples:
- "nah friday"
- "do the other one"
- "groceres for sat"
- "mark that done"
- "wait don't"
- "same thing as last time but cheaper"

### Overconfident user

Test requests that assume AEGIS can do things it cannot or is not authorized
to do.

AEGIS should remain helpful without fabricating capability or success.

### Evaluation rule

The same semantic capability should be tested across multiple competency levels.

Do not call a capability green merely because a technically precise user can
operate it successfully.

Capability-level green means reasonable users can express the objective in
their own language without needing to understand AEGIS internals.

Technical competency may change:
- how much explanation AEGIS provides;
- terminology used in the response;
- how much detail is shown.

It must NOT change:
- truth;
- permissions;
- authorization;
- verification;
- privacy;
- completion semantics.
Use these evaluation levels:

- **EXACT CANARY GREEN**: the exact tested utterance works.
- **CAPABILITY GREEN**: unseen paraphrases, typos, word-order changes, shorthand, and nearby ambiguous language work across a meaningful test family.
- **SAFETY SUCCESS**: refusal/clarification is correct when ambiguity or permission requires it.
- **PRODUCT FAILURE**: the system can be technically valid yet fail the reasonable human objective.

Once an exact canary is green, retire it to cheap deterministic regression.
Routine model dogfood moves to unseen language, combinations, failures, and boundaries.

---

# 3. LUNA EXECUTION PROTOCOL

Luna should not treat this roadmap as one enormous coding prompt.

For every checkpoint:

```text
inspect authoritative current state
-> find the earliest/highest-value unmet checkpoint whose dependencies are satisfied
-> inspect existing implementation
-> search open source for reusable parts where meaningful
-> define the smallest durable slice
-> implement
-> run deterministic tests
-> run novel objective-level dogfood where appropriate
-> distinguish exact-canary proof from capability proof
-> commit
-> push
-> update CURRENT_STATE.json
-> promote owner dogfood only from green evidence
-> continue
```

## 3.1 Checkpoint status vocabulary

Every checkpoint should be one of:

- `not_started`
- `in_progress`
- `deterministic_green`
- `live_green`
- `capability_green`
- `lane_blocked`
- `retired_regression`

`CURRENT_STATE.json` should eventually record:

```json
{
  "roadmap_version": 1,
  "roadmap_checkpoint": "M08.04",
  "roadmap_checkpoint_status": "in_progress"
}
```

Do not renumber historical checkpoints casually.

## 3.2 Blocker policy

### Local blocker
Luna can fix it.
- diagnose;
- fix;
- prove;
- continue.

### Lane blocker
Requires unavailable external hardware, credentials, provider access, owner presence, mobile device, Home Assistant installation, etc.
- record exact missing evidence;
- preserve the seam and tests;
- mark only that lane blocked;
- continue elsewhere.

### Campaign blocker
Only valid when **no meaningful safe in-scope work remains** without owner intervention or a consequential owner decision is genuinely required.

A lane blocker is not an apocalypse.

---

# 4. ROADMAP OVERVIEW

```text
M00  Repository Doctrine and Campaign Control
  ↓
M01  Semantic Core Contracts and Completion Model
  ↓
M02  Replaceable Model / Runtime Boundary
  ↓
M03  Durable Canonical PostgreSQL State
  ↓
M04  Identity, Vaults, Spaces, Authorization
  ↓
M05  Pack Contract and Domain Extensibility
  ↓
M06  Proving Domain Vertical Slices
  ↓
M07  Memory, Knowledge, Retrieval
  ↓
M08  Natural Semantic Front Door
  ↓
M09  Multi-Turn Conversation and Working Context
  ↓
M10  Cross-Domain Composition and Planning
  ↓
M11  Owner Dogfood and Productization Loop
  ↓
M12  Constellation and Multi-Client Experience
  ↓
M13  Proactive Intelligence and Durable Background Work
  ↓
M14  External World: Home, Communications, Devices
  ↓
M15  Voice and Ambient Interaction
  ↓
M16  Distribution, Installation, Pack Ecosystem
  ↓
M17  Small-Model Optimization and Adaptive Cognition
  ↓
M18  Release, Recovery, Security, and Trust Hardening
  ↓
M19  Jarvis Maturity
```

Some later lanes may begin early when dependencies allow, but no later feature may bypass the invariants established earlier.

---

# M00 — REPOSITORY DOCTRINE AND CAMPAIGN CONTROL

## Goal

Make the repository sufficient to recover:
- what AEGIS is;
- why it exists;
- what must remain true;
- where it is going;
- where it currently stands;
- how Luna should evaluate progress.

## Required artifacts

- `AGENTS.md`
- `VISION.md`
- `CORE_CONCEPTS.md`
- `ROADMAP.md`
- `MASTER_ROADMAP.md`
- `CURRENT_STATE.json`
- `ARCHITECTURE.md`
- `MODEL_STRATEGY.md`
- `SECURITY_MODEL.md`
- `PACK_SPEC.md`
- `docs/OWNER_DOGFOOD.md`
- `ECOSYSTEM_HARVEST.md`

## Checkpoints

### M00.01 — Document authority
Define which document answers which question.

### M00.02 — Vision
Document the snap-your-fingers mature product.

### M00.03 — Core concepts
Document the semantic-cognition/authority split, modularity, model replaceability, truth, dogfood, and OSS principles.

### M00.04 — Executable roadmap
Land this file and reference it from `AGENTS.md` and `ROADMAP.md`.

### M00.05 — Current-state pointer
Make `CURRENT_STATE.json` identify current roadmap checkpoint without becoming the roadmap itself.

### M00.06 — Luna operating instructions
Ensure future Codex sessions know to inspect repository truth before acting and do not repeatedly reread huge unchanged instructions.

## Proof

A fresh agent with no chat history should be able to inspect only the repo and accurately explain:
- the mature goal;
- current state;
- architecture;
- current milestone;
- next executable checkpoint;
- security invariants;
- evaluation doctrine.

## Exit criteria

M00 is complete when repository documentation is a sufficient project memory.

---

# M01 — SEMANTIC CORE CONTRACTS AND COMPLETION MODEL

## Goal

Build the stable semantic kernel that every domain and client can share.

## Required concepts

Typed contracts for:
- principal;
- intent;
- objective;
- proposal/decision;
- action;
- observation;
- verification contract/result;
- canonical result;
- completion;
- correlation;
- idempotency;
- provenance;
- Pack capability;
- events.

## Checkpoints

### M01.01 — IntentFrame
Represent who is asking, what was said/requested, correlation identity, and request context.

### M01.02 — ObjectiveSpec
Represent desired end state separately from individual actions.

### M01.03 — Bounded proposal
Models propose only against Core-provided capability/action candidates.

### M01.04 — Policy-before-execution
A model-selected action is not executable until policy/authorization permits it.

### M01.05 — Observation contract
Execution produces observed evidence, not assumed success.

### M01.06 — Verification contract
Consequential operations define how success is independently checked.

### M01.07 — Completion contract
Completion is based on objective evidence, not tool-return status.

### M01.08 — Retry/replay semantics
Correlation and idempotency prevent duplicate consequences.

### M01.09 — Failure taxonomy
Distinguish:
- blocked;
- clarify;
- retryable failure;
- terminal failure;
- unknown outcome;
- partial completion;
- success.

## Dogfood

Attack:
- malformed model decisions;
- invented action names;
- missing arguments;
- stale retries;
- duplicate submissions;
- verifier exceptions;
- executor exceptions;
- model failure before decision;
- partial observation.

## Exit criteria

Core remains correct even if the model is confused, malicious, malformed, or offline.

---

# M02 — REPLACEABLE MODEL / RUNTIME BOUNDARY

## Goal

Make cognition and execution providers replaceable.

## Baseline components

- Ollama for local inference.
- Qwen-class ~8B baseline.
- OpenClaw for runtime/channel execution where applicable.
- provider interfaces that do not leak provider-specific semantics into Core.

## Checkpoints

### M02.01 — Model provider interface
Structured input/output boundary.

### M02.02 — Strict decision decoding
Reject invalid structures rather than repairing dangerous ambiguity.

### M02.03 — Model capability contract
Models see bounded candidate actions/capabilities, not arbitrary executable namespaces.

### M02.04 — Provider failure handling
Timeouts/unavailability become truthful retryable results.

### M02.05 — Runtime execution boundary
Execution adapters receive validated requests only.

### M02.06 — OpenClaw integration
Use upstream protocol boundaries instead of owning a fork unless evidence justifies it.

### M02.07 — Provider swap proof
At least one substitute/mock provider can run without semantic-core redesign.

## Exit criteria

AEGIS does not require a particular model vendor or runtime vendor to preserve meaning.

---

# M03 — DURABLE CANONICAL POSTGRESQL STATE

## Goal

Move all important structured truth and objective lifecycle state into durable canonical storage.

## Checkpoints

### M03.01 — Schema/migration discipline
Contiguous migrations, clean-install validation, rollback/backup guidance.

### M03.02 — Objective/result persistence
Restart-safe objectives, results, correlations, and lifecycle records.

### M03.03 — Action/observation persistence
Consequential work survives restarts with stable identity.

### M03.04 — Audit/provenance
Tamper-evident or otherwise integrity-aware audit trail.

### M03.05 — Domain state primitives
Reusable repository patterns for Pack state.

### M03.06 — Restart/replay acceptance
Kill/restart between proposal, execution, observation, verification, and completion.

### M03.07 — Backup/restore
Prove compatible backup/restore on the supported PostgreSQL version.

### M03.08 — Health/readiness
Readiness verifies canonical schema without mutating it.

## Exit criteria

Restarting AEGIS does not erase what happened, fabricate what happened, or blindly repeat what may have happened.

---

# M04 — IDENTITY, VAULTS, SPACES, AUTHORIZATION

## Goal

Make privacy and sharing first-class and independent of model reasoning.

## Components

- Keycloak identity.
- OpenFGA relationship authorization.
- canonical principal mappings.
- Vault private state.
- explicit shared Spaces.

## Checkpoints

### M04.01 — Principal identity
Validated external identity maps to a canonical principal.

### M04.02 — Vault isolation
Private state is owner-partitioned and inaccessible across owners.

### M04.03 — Space membership
Shared state requires active membership.

### M04.04 — Role/capability policy
Capabilities depend on explicit role/policy, not model judgment.

### M04.05 — Revocation
Membership/session revocation immediately fails closed.

### M04.06 — No authority from reachability
The graph, network, tool listing, or discovered device does not imply permission.

### M04.07 — Client identity propagation
CLI/browser/mobile/voice resolve the same canonical identity semantics.

### M04.08 — Multi-user adversarial proof
Test cross-owner, cross-Space, stale-token, revoked-member, and replay scenarios.

## Exit criteria

Security survives model failure and client confusion.

---

# M05 — PACK CONTRACT AND DOMAIN EXTENSIBILITY

## Goal

Make new domains addable without redesigning Core.

## Pack manifest should declare

- Pack identity/version;
- schemas/state;
- capabilities;
- model-facing capability descriptions;
- permissions;
- actions;
- events;
- verification;
- retrieval hooks;
- optional UI metadata;
- dependencies;
- provenance/license;
- migration needs.

## Checkpoints

### M05.01 — Pack manifest contract

### M05.02 — Install/enable/disable lifecycle

### M05.03 — Permission declaration and grants

### M05.04 — Capability registration

### M05.05 — Model-facing semantic descriptions

### M05.06 — Verification adapters

### M05.07 — Retrieval hooks

### M05.08 — Optional Constellation metadata

### M05.09 — Forge/generation path
Generated Packs must pass validation before installation.

### M05.10 — Third-party Pack proof
Add a small new domain mostly through Pack/adapters, with little or no Core modification.

## Exit criteria

A novel domain can be added without inventing a second architecture.

---

# M06 — PROVING DOMAIN VERTICAL SLICES

## Goal

Prove Core against real domains rather than abstractions.

## Priority proving domains

### Tasks
- create;
- read;
- complete;
- deadlines;
- assignment;
- shared/private semantics;
- verification.

### Household
- chores;
- events;
- obligations;
- shared state;
- membership.

### Kitchen/Food
- groceries;
- pantry/inventory;
- recommendations;
- meal planning;
- external mutation where applicable.

### Finance
Finance is the hardest truth discipline:
- accounts;
- transactions;
- balances;
- cash flow;
- spending;
- budgets;
- bills;
- liabilities;
- investments;
- shared/private projections;
- goals;
- forecasting;
- anomaly detection.

Financial facts must remain canonical/provenance-aware.
Models analyze and recommend but never invent account truth.

### Homelab/Infrastructure
- hosts;
- services;
- health;
- restart workflows;
- independent readback.

### Network
- authorized scopes;
- discovered devices;
- safe probes;
- explicit scope boundaries.

## Checkpoints

### M06.01 — At least one safe read vertical slice

### M06.02 — At least one verified write vertical slice

### M06.03 — Shared-state vertical slice

### M06.04 — Private-state vertical slice

### M06.05 — External execution vertical slice

### M06.06 — Finance truth/projection slice

### M06.07 — Cross-restart replay proof

## Exit criteria

Core semantics work on real domains without special-casing one domain as "the architecture."

---

# M07 — MEMORY, KNOWLEDGE, RETRIEVAL

## Goal

Give AEGIS durable memory without turning retrieval into truth.

## Checkpoints

### M07.01 — Canonical memories
Store memory records with provenance, timestamps, ownership, and supersession.

### M07.02 — Projects/goals/entities
Stable entity identity and relationships.

### M07.03 — Deterministic retrieval
Exact/entity/temporal retrieval for cheap high-confidence cases.

### M07.04 — Semantic retrieval
pgvector/equivalent indexes authorized content for retrieval only.

### M07.05 — Incremental indexing
Do not unnecessarily re-embed entire memory sets per interaction.

### M07.06 — Supersession
Old/corrected memory is not silently returned as current truth.

### M07.07 — Provenance in answers
Grounded answers retain where relevant state came from.

### M07.08 — Context compaction
Retrieve only what the current objective needs.

### M07.09 — Memory privacy
Semantic indexes cannot leak Vault/Space boundaries.

## Exit criteria

AEGIS can remember and retrieve useful personal context while canonical ownership and provenance remain explicit.

---

# M08 — NATURAL SEMANTIC FRONT DOOR

## Goal

Make AEGIS understand humans rather than a command grammar.

This is a core product milestone.

## Desired behavior

These should not require hard-coded phrase support:

- "set task status get gud scrub complete"
- "knock get gud scrub off my list"
- "i'm done with get gud scrub"
- "what grocerieus should i get?"
- "anything around here need attention?"
- "whats in the fridge i should use soon?"
- harmless general requests such as "write me a story about fish"

## Architecture

```text
utterance
   |
   +--> high-confidence deterministic intent?
   |         |
   |         `--> fast canonical path
   |
   `--> semantic capability retrieval / candidate reduction
             |
             `--> bounded model intent proposal
                        |
                        `--> Core validates
```

## Checkpoints

### M08.01 — Confidence-aware fast paths
Fast paths should run only when interpretation is genuinely high-confidence.

### M08.02 — General benign ANSWER path
Qwen may answer harmless non-authoritative requests that need no Pack mutation.

### M08.03 — Semantic Pack/capability retrieval
Embed compact capability descriptions and shortlist relevant Packs/actions.

### M08.04 — Bounded intent schema
Model may choose:
- ANSWER;
- ACTION;
- CLARIFY;
- NEED_CONTEXT;
- BLOCKED;
from Core-defined contracts.

### M08.05 — Candidate action selection
The model selects IDs/indexes from a provided set; Core expands to canonical action specs.

### M08.06 — Argument extraction
Model can map informal language into structured bounded action arguments.

### M08.07 — Ambiguity handling
Unfamiliar wording should not mean BLOCKED.
Real ambiguity should CLARIFY.

### M08.08 — Typos and paraphrases
Capability-level dogfood across unseen variants.

### M08.09 — Near-miss safety
Read vs write language must not silently flip consequences.

### M08.10 — Remove phrase overfit
Retire production synonym/regex growth that only compensates for model routing.

## Capability-green dogfood

For each semantic capability:
- unseen paraphrases;
- misspellings;
- shorthand;
- changed word order;
- politeness;
- slang;
- misleading nearby vocabulary;
- adversarial wording;
- model malformed output;
- ambiguous entity names.

Do not use the evaluation family as a production synonym table.

## Exit criteria

A user does not need to know Pack names, action IDs, or magic verbs.

---

# M09 — MULTI-TURN CONVERSATION AND WORKING CONTEXT

## Goal

Make conversation semantically continuous without treating transcript text as truth.

## Required examples

```text
User: What's on my grocery list?
AEGIS: ...
User: Which of those should I buy first?
```

```text
User: Show the tasks due next week.
AEGIS: ...
User: Complete the first one.
```

```text
User: Plan dinner Friday.
AEGIS: ...
User: Keep it under $40.
AEGIS: ...
User: Add what we need.
```

## Checkpoints

### M09.01 — Conversation/session identity
Stable session identity distinct from action correlation IDs.

### M09.02 — Recent-turn context
Bounded relevant turns, not unlimited transcript dumping.

### M09.03 — Resolved referents
Store non-authoritative contextual bindings for:
- it;
- that;
- those;
- first/second;
- previous result;
- selected node/entity.

### M09.04 — Objective continuation
A follow-up can continue an objective rather than starting from nothing.

### M09.05 — Context grounding
Prior model text is not automatically treated as canonical fact.

### M09.06 — Context-to-mutation guard
"Add those" can resolve candidates but still crosses authorization and verification.

### M09.07 — Clarification on stale/ambiguous references

### M09.08 — Restart continuity
Important conversational/objective context survives process restart where appropriate.

### M09.09 — Context compaction for small models

## Exit criteria

Ordinary pronouns and follow-up commands work naturally without weakening canonical truth.

---

# M10 — CROSS-DOMAIN COMPOSITION AND PLANNING

## Goal

Make AEGIS valuable because domains compose.

## Example mature objectives

- "I have $75 left for groceries and dinner this week. What should I buy?"
- "Given my goals and everything due this week, what should I focus on tonight?"
- "We're leaving Friday. What tasks, bills, house things, and car stuff need attention first?"
- "I want to host six people Saturday. Plan food, groceries, calendar, chores, and budget."
- "My server backup failed. Add what needs doing, check whether this affects anything scheduled, and remind me if it is still unresolved tomorrow."

## Checkpoints

### M10.01 — Authorized cross-domain working set
Each domain filters before composition.

### M10.02 — Planner contract
Decompose one objective into bounded steps.

### M10.03 — Per-step authorization
Authorization is not inherited blindly from the overall objective.

### M10.04 — Durable plan persistence

### M10.05 — Sequential continuation

### M10.06 — Partial failure
Preserve what succeeded and explain what did not.

### M10.07 — Compensation/cancellation where supported

### M10.08 — Cross-domain verification

### M10.09 — Model planning with small candidate sets

### M10.10 — Cross-domain capability dogfood
Vary combinations rather than repeatedly proving one chain.

### M10.11 — Capability coverage and unresolved capability needs
Plans preserve unsatisfied requirements instead of silently dropping them or treating the objective as complete. Missing capability is durable product state that may require bounded investigation, composition, discovery, acquisition, owner escalation, or a later reusable capability.

## Exit criteria

AEGIS can coordinate multiple domains as one objective while each domain retains its truth/permission boundaries.

---

# M11 — OWNER DOGFOOD AND PRODUCTIZATION LOOP

## Goal

Turn real human use into a continuous evaluation frontier.

## Checkpoints

### M11.01 — Persistent owner instance
Installed release, canonical DB, identity, real model.

### M11.02 — Last-green promotion
Owner dogfood follows last-green release, not raw branch HEAD.

Desired flow:

```text
commit
-> deterministic green
-> required live acceptance
-> mark last-green
-> install versioned owner release
-> restart
-> health/readiness check
-> rollback if unhealthy
```

### M11.03 — Release identity visible
UI can display running SHA/version unobtrusively.

### M11.04 — Dogfood event capture
Capture:
- correlation;
- objective;
- running release;
- model;
- result state;
- relevant safe evidence.

### M11.05 — Owner feedback
PASS / FAIL / WEIRD / UNSURE plus optional note.
Feedback is evaluation metadata, never canonical domain truth.

### M11.06 — Exact canary vs capability green
Formalize evaluation levels.

### M11.07 — Automatic defect harvesting
Luna can inspect recent failures and convert them into:
- defect;
- deterministic regression;
- fix;
- unseen live re-evaluation.

### M11.08 — Retired-regression policy
Do not spend model inference endlessly reproving solved behavior.

### M11.09 — Product-utility gate
Ask:
"Did the human objective succeed naturally?"
not merely:
"Did a valid Result exist?"

## Exit criteria

Real use continuously improves the product without contaminating truth or overfitting to owner phrasing.

---

# M12 — CONSTELLATION AND MULTI-CLIENT EXPERIENCE

## Goal

Expose the same AEGIS intelligence through useful clients.

All clients share the same interaction boundary.

```text
CLI/TUI --------\
Web/Graph -------\
Telegram ---------> Interaction Boundary -> Core
Voice -----------/
Other clients ---/
```

## Constellation vision

AEGIS is the central hub.
Domains/Packs are major nodes.
Semantic zoom reveals:
1. domain hubs;
2. semantic areas;
3. contextual objects.

Graph reachability never implies authorization.

## Checkpoints

### M12.01 — Stable client-neutral interaction API

### M12.02 — CLI/TUI usability

### M12.03 — Browser conversation surface

### M12.04 — Authorized state/detail views

### M12.05 — Dynamic Pack-driven UI metadata

### M12.06 — Graph relationships

### M12.07 — Semantic zoom

### M12.08 — Active-objective visualization
Proposal -> authorization -> execution -> observation -> verification -> completion.

### M12.09 — Conventional accessible navigation
Graph is not mandatory.

### M12.10 — Contextual UI
Selecting nodes can bias/restrict context without granting authority.

### M12.11 — Mobile/chat client
Use the same boundary.

## Exit criteria

The UI feels like a window into one intelligence, not a second implementation of AEGIS.

---

# M13 — PROACTIVE INTELLIGENCE AND DURABLE BACKGROUND WORK

## Goal

Move from reactive assistant to bounded proactive intelligence.

## Checkpoints

### M13.01 — Event model
Canonical domain events can trigger evaluation.

### M13.02 — Background objective contract

### M13.03 — Scheduling

### M13.04 — Cancellation

### M13.05 — Deduplication/idempotency

### M13.06 — Proactive suggestion
Suggest without acting when authority is absent.

### M13.07 — Notification routing

### M13.08 — Approval-required background actions

### M13.09 — Quiet-hours / preference policy

### M13.10 — Stale-state reconciliation

### M13.11 — Bounded autonomy
Background work has explicit limits, timeouts, budgets, and stop conditions.

### M13.12 — Bounded investigation loop
An unresolved objective may inspect already-authorized state, integrations, public information, and available capabilities under explicit budgets and stop conditions before escalating to the owner. Investigation proposes no authority and cannot silently complete an unsatisfied requirement.

## Example mature behavior

- "Your electric bill is unusually high."
- "You usually order groceries by Thursday; the pantry looks short for this weekend."
- "The backup is still failing and tomorrow's maintenance window is approaching."
- "You have overlapping obligations Friday and travel time makes both impossible."

## Exit criteria

AEGIS can notice and help without becoming an uncontrolled daemon.

---

# M14 — EXTERNAL WORLD: HOME, COMMUNICATIONS, DEVICES

## Goal

Connect AEGIS to real-world systems behind typed replaceable adapters.

## Home Assistant

### M14.01 — Device discovery as data, not permission

### M14.02 — Explicit authorized device scopes

### M14.03 — Command observation/readback

### M14.04 — Device-state reconciliation

## Communications

### M14.05 — Email/message read adapters

### M14.06 — Draft before send where appropriate

### M14.07 — Explicit send authority and verification

## Calendar

### M14.08 — Canonical event read

### M14.09 — conflict-aware scheduling

### M14.10 — verified create/update/cancel

## Other physical/digital integrations

### M14.11 — Vehicles

### M14.12 — Business systems

### M14.13 — Documents/cloud storage

### M14.14 — Governed general browser/API execution
General external digital interaction is available only behind explicit observation, preparation, mutation, and submission authority boundaries. Replaceable runtime providers remain subordinate to Core authorization, privacy, verification, and completion semantics.

## Exit criteria

Integrations remain replaceable, scoped, and subordinate to Core semantics.

---

# M15 — VOICE AND AMBIENT INTERACTION

## Goal

Make AEGIS available without requiring a keyboard while preserving the same authority model.

## Checkpoints

### M15.01 — Speech-to-text adapter

### M15.02 — Voice session identity

### M15.03 — Wake/activation semantics

### M15.04 — Interruption and correction

### M15.05 — Voice-specific clarification

### M15.06 — Text-to-speech

### M15.07 — Ambient context boundaries

### M15.08 — Sensitive-output policy

### M15.09 — Consequential action confirmation

### M15.10 — Multi-device continuity

## Exit criteria

Voice is another client, not a privileged bypass around Core.

---

# M16 — DISTRIBUTION, INSTALLATION, PACK ECOSYSTEM

## Goal

Make AEGIS something ordinary technical users can actually install, configure, update, and extend.

## Checkpoints

### M16.01 — Reproducible install

### M16.02 — First-run setup

### M16.03 — Readiness diagnostics

### M16.04 — Safe secrets/configuration

### M16.05 — Upgrade/migration path

### M16.06 — Backup/restore UX

### M16.07 — Versioned releases

### M16.08 — Pack installation workflow

### M16.09 — Pack provenance/license display

### M16.10 — Compatibility contracts

### M16.11 — Self-hosting documentation

### M16.12 — Hardware/model sizing guidance

### M16.13 — Capability acquisition lifecycle
Discover → reuse/adapt/build → sandbox → test → permission analysis → install/enable → verify. Each transition is explicit; discovery, generation, installation, and enablement do not grant approval or execution authority.

### M16.14 — Pack Forge Phase B
Pack Forge becomes the deterministic compiler/scaffolder below model- or worker-assisted capability generation, without granting model or worker authority. Generated behavior must converge toward typed, testable capability contracts.

## Exit criteria

A new user can install Core, choose a supported model, add Packs, and recover from ordinary setup mistakes without reading source code.

---

# M17 — SMALL-MODEL OPTIMIZATION AND ADAPTIVE COGNITION

## Goal

Make the harness carry as much intelligence burden as possible so users are not forced into frontier-model economics.

## Checkpoints

### M17.01 — Benchmark suite
Representative objective families across:
- 2B–4B;
- ~8B;
- larger local;
- optional frontier.

### M17.02 — Candidate reduction
Measure semantic shortlist size and routing accuracy.

### M17.03 — Context compaction

### M17.04 — Schema simplification

### M17.05 — Decomposition strategies

### M17.06 — Deterministic pre/post-processing

### M17.07 — Optional micro-router experiment
Only if measurements justify another inference hop.

Potential architecture:

```text
embedding retrieval
-> optional 2B-4B router
-> ~8B reasoning model
-> deterministic Core
```

### M17.08 — Selective stronger-model escalation
Escalation policy based on:
- complexity;
- confidence;
- risk;
- expected gain;
- user preference/cost.

### M17.09 — Provider fallback

### M17.10 — Efficiency telemetry
Latency, tokens, context size, failure rate, model escalations.

## Exit criteria

Basic AEGIS usefulness does not require a frontier model, and stronger models are used intentionally rather than reflexively.

---

# M18 — RELEASE, RECOVERY, SECURITY, AND TRUST HARDENING

## Goal

Turn a capable system into a trustworthy one.

## Checkpoints

### M18.01 — Threat model refresh

### M18.02 — Prompt-injection/adversarial content handling

### M18.03 — Privilege/revocation races

### M18.04 — Crash/restart at every consequential lifecycle boundary

### M18.05 — Unknown outcome recovery

### M18.06 — Correlation/idempotency abuse testing

### M18.07 — Audit integrity

### M18.08 — Data export/deletion

### M18.09 — Pack supply-chain controls

### M18.10 — Dependency/license/security scan

### M18.11 — Migration rollback/recovery

### M18.12 — Owner/admin emergency controls

### M18.13 — Performance/load

### M18.14 — Multi-user privacy stress

### M18.15 — Release acceptance matrix

## Exit criteria

A user can trust AEGIS not only when everything works, but when models, networks, providers, adapters, and humans behave badly.

---

# M19 — JARVIS MATURITY

## Goal

AEGIS should now behave like a coherent, persistent, trustworthy personal intelligence.

This is not one feature. It is the integration quality of everything above.

## Maturity dimensions

### M19.01 — Naturalness
Users speak normally.
No Pack vocabulary ceremony.

### M19.02 — Continuity
AEGIS understands ongoing objectives and relevant prior context.

### M19.03 — Knowledge
AEGIS knows authorized canonical facts and relevant durable memories.

### M19.04 — Composition
AEGIS naturally combines domains where the objective requires them.

### M19.05 — Initiative
AEGIS notices relevant changes and offers useful help.

### M19.06 — Restraint
AEGIS knows when it lacks authority, evidence, or confidence.

### M19.07 — Trust
Claims about the world trace back to canonical/provenance-aware state.

### M19.08 — Action
Consequential work is policy-gated, observed, verified, and replay-safe.

### M19.09 — Personalization
Different users/Spaces can have different:
- models;
- Packs;
- preferences;
- privacy;
- proactivity;
- clients;
- resource budgets.

### M19.10 — Model independence
The product remains coherent when models/providers change.

### M19.11 — Extensibility
A new domain can be added without redesigning Core.

### M19.12 — Ambient availability
Users can reach AEGIS through the interface appropriate to the moment.

### M19.13 — Novel objective generalization
AEGIS can pursue useful objectives for which no bespoke high-level Pack action was pre-authored by combining bounded general capabilities or acquiring the smallest appropriate reusable capability. Novel cognition and capability proposals remain non-authoritative until Core validates, authorizes, executes, and verifies them.

## Final product acceptance

A mature AEGIS should pass tests like:

> "I'm hosting friends Saturday. Keep dinner under $80, remember Maya doesn't eat pork, check whether we already have anything useful, add what we need, and make sure the apartment is ready before they arrive."

AEGIS should:
1. understand the objective naturally;
2. identify relevant Packs;
3. retrieve only authorized context;
4. use canonical household/food/finance/calendar/task state;
5. recognize missing or ambiguous information;
6. propose a plan;
7. ask only necessary clarification;
8. authorize each consequential step;
9. execute permitted actions;
10. observe and verify outcomes;
11. retain durable objective state;
12. respond conversationally with what is true, what changed, and what still needs attention.

Another mature test:

> "Something seems off with my finances this month. Figure out why, but don't move any money."

AEGIS should:
- respect the no-action boundary;
- inspect canonical finance data;
- correlate recurring bills, spending, household obligations, and relevant events;
- distinguish facts from hypotheses;
- surface provenance;
- avoid exposing private data to unauthorized Spaces;
- give a useful explanation without needing the user to name a Finance Pack.

Another:

> "The server is acting weird again. Handle what you safely can and tell me before you do anything risky."

AEGIS should:
- identify Infrastructure/Network/Tasks relevance;
- inspect authorized state;
- perform low-risk approved diagnostics;
- preserve evidence;
- stop for approval before consequential actions;
- verify any action it does take;
- create/continue durable work if needed.

At this point the user should experience:

> **"AEGIS understands my world, knows what it may do, remembers what matters, coordinates the right systems, and tells me the truth."**

That is the target.

---

# 5. CROSS-CUTTING GATES

These gates apply throughout the roadmap rather than at one milestone.

## Security gate

No new capability ships if it:
- lets model output grant authority;
- leaks Vault/Space data;
- treats reachability as permission;
- treats tool success as objective success;
- blindly replays consequential work.

## Domain-extensibility gate

If a new domain requires repeated Core special-casing, stop and inspect whether the Pack contract is missing a reusable primitive.

## Small-model gate

If weak-model performance is poor:
1. inspect candidate count;
2. inspect context quality;
3. inspect schema;
4. inspect ambiguity;
5. inspect decomposition;
6. inspect verification;
7. only then consider a larger model.

## Product gate

Do not count infrastructure correctness as sufficient product success.

Ask:
- did the human objective succeed?
- was the interaction natural?
- did AEGIS expose unnecessary plumbing?
- did it ask only necessary clarification?
- did it use model cognition where language understanding was needed?

## OSS gate

Before building commodity functionality, record whether suitable upstream code exists.

## Dogfood gate

Routine dogfood must target:
- new capability;
- weak boundaries;
- new combinations;
- stale/conflicting state;
- failures;
- recovery;
- small-model stress;
- adversarial input;
- multi-user boundaries;
- proactivity;
- temporal reasoning.

Do not repeatedly spend inference on solved exact canaries.

---

# 6. HOW CURRENT_STATE SHOULD DRIVE THIS ROADMAP

`CURRENT_STATE.json` should not duplicate every paragraph above.

It should point into it.

Recommended fields:

```json
{
  "roadmap_version": 1,
  "active_milestone": "M08",
  "active_checkpoint": "M08.04",
  "checkpoint_status": "in_progress",
  "checkpoint_started_at": "...",
  "checkpoint_evidence": {
    "deterministic": [],
    "live": [],
    "capability": []
  },
  "blocked_lanes": [],
  "next_candidates": [
    "M08.05",
    "M09.01"
  ]
}
```

When early milestones are already green, record them as complete from existing evidence.
Do **not** rerun solved campaigns merely to satisfy the numbering.

If a past "green" only proved one phrase, downgrade the interpretation to exact-canary evidence rather than deleting the historical evidence.

---

# 7. HOW LUNA CHOOSES WHAT TO DO NEXT

When there are multiple available checkpoints, rank them by:

1. closes a real user-facing capability gap;
2. closes a security/truth/completion gap;
3. unblocks multiple future milestones;
4. improves generality/extensibility;
5. improves 8B/2B-4B viability;
6. improves recovery/reliability;
7. replaces brittle special cases with a reusable primitive;
8. reduces owner setup burden;
9. enables better dogfood;
10. cosmetic polish.

Avoid:
- large refactors without capability gain;
- decorative UI ahead of interaction quality;
- repetitive dogfood;
- new domains just to increase domain count;
- model-size escalation before harness improvements;
- microservices without a real boundary need;
- prompt-driven business logic;
- duplicated semantic ownership.

---

# 8. ROADMAP CHANGE CONTROL

This roadmap is allowed to evolve.

Change it when:
- evidence invalidates an assumption;
- a better reusable architecture is discovered;
- open-source reuse materially changes the sequence;
- a dependency becomes obsolete;
- a new safety requirement appears;
- a milestone should split because proof is too broad.

Do not change it merely because:
- the current implementation took a shortcut;
- a single dogfood phrase failed;
- a library was inconvenient;
- a lane is temporarily blocked.

Material roadmap changes should state:
- what changed;
- why;
- which invariants remain;
- migration impact;
- whether CURRENT_STATE checkpoints must be remapped.

---

# 9. DEFINITION OF SUCCESS

AEGIS is successful when ordinary users can:

1. install it;
2. run models suited to their hardware/budget;
3. add modular Packs;
4. interact naturally without learning an internal command grammar;
5. trust canonical facts and provenance;
6. keep private data private;
7. share intentionally through Spaces;
8. have AEGIS reason across domains;
9. let it safely execute authorized actions;
10. see verified outcomes rather than hallucinated success;
11. recover from crashes/provider failures without blind replay;
12. use it through terminal, browser, mobile/chat, and eventually voice/ambient interfaces;
13. retain usefulness on approximately 8B local models and meaningful bounded usefulness on smaller models;
14. extend the system into future domains without redesigning Core.

The product-level end state is:

> **A trustworthy, private, modular, self-hostable Jarvis-style intelligence that understands the human naturally, understands their authorized world, coordinates useful work, and never confuses model confidence with reality.**
