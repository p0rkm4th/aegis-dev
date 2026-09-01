# Model strategy

Models are replaceable cognition. Qwen-class ~8B local inference is the first
baseline, with 2B–4B usefulness as an explicit design target. Models never own
authorization, truth, persistence, verification, or completion.

“Deterministic paths run first” means deterministic answers/actions are used
when intent is already high-confidence and hard safety, policy, and
verification stay deterministic. It does not mean every request must pass a
lexical parser first. Ambiguous, informal, typo-ridden, or novel language
should reach bounded semantic cognition rather than being rejected for lacking
a known keyword. Do not turn dogfood repairs into a growing synonym table.

The preferred pipeline is semantic Pack/capability retrieval, a compact
authorized working set, bounded schema-constrained model reasoning, and Core
validation before authorization and execution. Candidate reduction is important
for small models; every Pack and database row must not be placed in context.
Structured context may resolve references across turns, but transcript text is
never canonical truth.

An optional 2B–4B routing model or embedding retriever may be added only when
measurement shows useful accuracy, latency, context, or resource gains. A
Qwen-class reasoning model remains replaceable by provider and model. Selective
stronger models are justified only for measured tasks where their cost and
complexity are worthwhile. Add inference hops only after measuring the current
boundary.
