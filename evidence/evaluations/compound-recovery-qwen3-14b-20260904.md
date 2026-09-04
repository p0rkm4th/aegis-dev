# Qwen3:14B validation-guided recovery comparison

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases, unchanged)
- Source: `46d11fd7eccbe80417064a0316d784f2bdcada6e`
- Provider: Ollama `qwen3:14b`, digest `bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`
- Structural evidence: spaCy `3.8.16`, `en_core_web_sm 3.8.0` (CC BY 4.0)
- Execution: disabled; evaluation boundary only

Under the identical recovery contract, candidates, prompts, validators, and corpus, Qwen3:14B made 129 model calls across 17 repair requests and 22 repair attempts. Maximum per-case repair attempts was 2. Request modes were `effect_segmentation=18`, `effect_repair=10`, `objective_mapping=8`, and `ordinary_decision=93`.

Overall valid-route rate was 23.08%; supported-clear valid-route was 5/20 (25%), and inappropriate clarification was 13/20 (65%). Relative to the Qwen3:8B current-HEAD replay (7.69% overall, 1/20 supported-clear, 85% inappropriate clarification), this is a material capability improvement. It remains below the 80% recovery target.

Latency was materially higher: average request latency 25.81s, p50 24.35s, p95 44.56s. Decoder failures were 1. Safety remained green: zero false completions, false mutations, unsafe mutations, and security hard failures. The evidence supports Qwen3:14B as a measured hard-cognition/reference candidate, not an automatic daily resident replacement.
