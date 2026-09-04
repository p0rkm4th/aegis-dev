# Strict segmentation-only effect replay

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases, unchanged)
- Source: `0dfd9a0`
- Provider: Ollama `qwen3:8b`, digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`
- Structural evidence: spaCy `3.8.16`, `en_core_web_sm 3.8.0` (CC BY 4.0)
- Execution: disabled; evaluation boundary only

The strict effect schema replay made 107 model calls and 19 bounded repairs. Request-mode telemetry proved the new seam was exercised: `effect_segmentation=7`, `effect_repair=5`, `objective_mapping=2`, and `ordinary_decision=93`. One supported-clear case reached a PLAN through the separate mapping pass, but aggregate supported-clear final completion remained 0/20, inappropriate clarification 85%, and route accuracy 41.5%.

Safety remained green: zero false completions, false mutations, unsafe mutations, and security hard failures. The strict contract is therefore structurally correct and exercised, but not yet product-useful; the next work should diagnose mapping/coverage validation failures rather than add more schema permissiveness.
