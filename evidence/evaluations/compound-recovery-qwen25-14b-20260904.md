# Qwen2.5:14B validation-guided recovery comparison

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases, unchanged)
- Source: `69858b50f0f8e327c7f32c7e4c6316fb32e62f92`
- Provider: Ollama `qwen2.5:14b`, digest `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6`
- Structural evidence: spaCy `3.8.16`, `en_core_web_sm 3.8.0` (CC BY 4.0)
- Execution: disabled; evaluation boundary only

Under the same contract and corpus, Qwen2.5:14B made 94 model calls across 10 repair requests and 12 repair attempts. Overall and supported-clear valid-route rates were both 0%; supported-clear inappropriate clarification was 12/20 (60%), with five decoder failures. Request modes were `effect_segmentation=2`, `effect_repair=2`, and `ordinary_decision=91`.

Safety remained green: zero false completions, false mutations, unsafe mutations, and security hard failures. Average/p50/p95 latency was 18.50s/13.09s/42.57s. The model is rejected for this role because it failed to produce trustworthy final behavior and did not exercise the objective-mapping seam meaningfully.
