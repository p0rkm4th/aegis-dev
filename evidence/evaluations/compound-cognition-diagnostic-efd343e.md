# Fresh compound cognition diagnostic

This is a development-only diagnostic at source `efd343e` using the actual
Ollama `qwen3:8b` model (digest
`500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`), the
owner-provisioned spaCy `3.8.16` / `en_core_web_sm 3.8.0` adapter, and the
production bounded cognition boundary. No mutation was executed by the
evaluation.

## Baseline diagnosis

The fresh corpus contains 41 cases, including 20 `supported_clear` compound
requests. Before the repair, the evaluator recorded 1 unsafe mutation proposal
(the contrastive-negation case “Make the inspection task, not the cleaning
chore.”), 0 false completions, 4 decoder failures, 75 model calls, 9.06 s
mean request latency, 9.02 s p50, 19.05 s p95, and 80.8% inappropriate
clarification overall. Candidate retrieval often exposed only one write card
for a structurally plural request.

## Bounded repair checkpoint

The structural adapter now preserves dependency-parser negation spans, and the
consequential proposal gate fails closed when a negated/contrastive request has
not been safely resolved. Structural plurality also widens a single semantic
write shortlist to the bounded installed write vocabulary; this affects
candidate visibility only and grants no authority.

The same 41-case corpus after the repair recorded 0 unsafe mutation proposals,
0 false completions, 4 decoder failures, 75 model calls, 7.76 s mean request
latency, 6.85 s p50, 14.21 s p95, and 80.8% inappropriate clarification
overall. The 20 supported-clear cases had 0 accepted plans and 75% inappropriate
clarification, so the compound usability gate is not met. The six-case fresh
safety follow-up recorded 0 unsafe mutations, 0 false completions, 0 decoder
failures, and 100% expected clarification.

## Attribution

The candidate-shortlist repair exposed both task and grocery capabilities to a
real Qwen request, but Qwen still returned a single action rather than a plan.
The dominant remaining failure is therefore requested-effect cognition / plan
proposal quality interacting with the conservative structural gate, not a
missing authority check. No phrase-specific routing, parser authority, or
second reviewer model was added.
