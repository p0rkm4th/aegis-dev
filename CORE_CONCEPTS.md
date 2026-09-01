# AEGIS core concepts

These are durable product and architecture invariants. They are not a
checklist and should change only through a deliberate project-direction
decision.

## Semantic cognition is the front door

AEGIS has language models for semantic understanding, reasoning, context
resolution, and proposal generation. Deterministic fast paths are valuable
when intent is already high-confidence, and deterministic code enforces hard
constraints. They must not become an ever-growing English parser of regexes,
verb lists, synonyms, and phrase-specific branches. Do not repair each
dogfood failure by adding another phrase.

The preferred shape is:

```text
user + bounded conversational context
        |
        +-> high-confidence deterministic fast path
        |
        `-> semantic Pack/capability retrieval
                -> bounded model reasoning
                -> structured proposal
                -> Core validation
                -> authorization
                -> execution
                -> observation
                -> verification
                -> canonical Result
```

Determinism decides what is true and permitted. Semantic retrieval decides what
is probably relevant. Models decide what the human probably means and what to
propose. Core decides what may happen. Verification decides whether it did.

## Models are cognition, not authority

Models may understand language, resolve likely intent, select bounded
capabilities, reason over authorized context, synthesize grounded answers, and
propose or help plan actions. Models may not grant permissions, invent tools or
facts, own persistence, decide authorization, claim execution success, own
verification, or declare objective completion without Core evidence.

Generated answers are explicitly non-authoritative unless separately grounded
by canonical state. Transcript text is context, never a capability grant.

## Semantic Pack retrieval and model scale

Packs expose compact model-facing capability descriptions. Retrieval should
shortlist likely relevant authorized Packs and ActionCards; every capability
must not be stuffed into every prompt. This supports the Qwen-class ~8B
baseline and keeps 2B–4B models viable through context reduction, schemas,
candidate reduction, deterministic fast paths, and state machines.

Multiple models are allowed only when measurement justifies the cost. A future
optional embedding or small routing model may precede an ~8B reasoning model,
but a micro-router is not required merely because it sounds elegant. Providers
and models remain replaceable.

## Conversation is first-class

AEGIS should understand references such as “which of those?”, “do the first
two”, “make it Friday”, and “the one I mentioned yesterday”. Use compact,
structured, authorization-checked context rather than unlimited transcript
dumping. Earlier model text never becomes canonical fact merely by appearing in
conversation. Ambiguous references clarify rather than guess.

## Core owns meaning and completion

The authoritative lifecycle remains:

```text
intent -> objective -> proposal -> authorization -> execution -> observation
       -> verification -> canonical Result -> completion
```

Tool success is not objective success. PostgreSQL owns structured canonical
truth; pgvector and other retrieval indexes only help find context. Models do
not silently promote inference into facts.

## Extensibility, security, and provenance

New domains should primarily be Packs and adapters, not Core surgery. Finance
is a proving domain, while cross-domain composition belongs in Core. Vaults are
private by default and Spaces are explicit. Least privilege, semantic/runtime/
human approval distinctions, independent verification, provenance, and
correlation/idempotency must survive model failure, revocation, retries, and
crashes. Reachability is never authority.

Before implementing commodity functionality, search the open-source ecosystem
for code, tests, schemas, protocols, and designs to directly use, port, adapt,
fork, vendor, test-harvest, design-harvest, or reject. Track provenance and
licenses.

The product test is whether a choice makes AEGIS more like a trustworthy,
coherent personal intelligence and less like a command parser, hard-coded demo,
disconnected tool collection, database UI, or prompt-driven business system.
