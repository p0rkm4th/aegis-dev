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

## Follow-up parser experiment

A bounded follow-up widened structural anchors for argument-bearing `dep` and
`advcl` predicates so previously under-segmented clauses could expose more
write candidates. On the same 41-case Qwen3:8B corpus, supported-clear route
accuracy was 3/20, inappropriate clarification was 80%, mean latency was
11.24 s, and model calls increased to 91. The broader signal increased
decision noise without improving usable compound planning, so the experiment
was reverted and is not part of the production checkpoint.

## Action-agnostic decomposition experiment

At source `c195f6f`, the same 41-case corpus (SHA
`641a694d0d9b850f968e0c99108ac4ee8a5ca81614431bbaec0db00eccdc4d33`) was run
through a temporary action-agnostic discovery contract followed by bounded
per-effect capability mapping. Discovery received no ActionCards and returned
only effect text, grounded span, and polarity; Core performed grounding and
structural checks before mapping.

The first decomposed run recorded 0/20 supported-clear plans, 95%
inappropriate clarification, 1 decoder failure, 85 model calls, 10.10 s mean
latency, and 20.85 s p95. A bounded mapping refinement supplied the original
utterance plus a Core-derived focus effect and an explicit mapping-only prompt.
It removed decoder failures, but still recorded 0/20 supported-clear plans,
95% inappropriate clarification, 84 model calls, 8.89 s mean latency, and
16.53 s p95. Safety remained clean in both runs: zero false completions,
unsafe mutations, and Core false acceptances.

The decomposition hypothesis is rejected for this baseline: isolating effect
discovery did not improve Qwen3:8B compound usability and materially increased
clarification. The temporary contracts and runtime path were removed. This is
an evaluated model/cognition boundary, not permission to weaken structural
safety or to add another equivalent reviewer.
