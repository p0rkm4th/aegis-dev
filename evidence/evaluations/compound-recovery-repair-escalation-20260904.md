# Repair-only model escalation replay

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases, unchanged)
- Source: `d2144630f0df27b75e8dbcbb8db743dba5b058ec`
- Resident: Ollama `qwen3:8b`, digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`
- Repair-only: Ollama `qwen3:14b`, digest `bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`
- Structural evidence: spaCy `3.8.16`, `en_core_web_sm 3.8.0` (CC BY 4.0)
- Execution: disabled; evaluation boundary only

The resident handled ordinary cognition while the stronger provider was available only for bounded proposal repair. The run made 120 model calls across 17 repair requests and 27 repair attempts; maximum per-case attempts was 2. Request modes were `ordinary_decision=102`, `effect_segmentation=9`, `effect_repair=5`, and `objective_mapping=4`.

Final valid-route rate was 15.38% overall, with 23.08% inappropriate clarification and route accuracy 46.34%. This did not materially improve the Qwen3:8B control replay (7.69% overall, 5% supported-clear, 85% inappropriate clarification) enough to justify routing or owner promotion; repair failures remained dominated by `UNACCOUNTED_STRUCTURAL_ANCHOR` and `DECODER_SCHEMA_FAILURE`.

Safety remained green: zero false completions, false mutations, unsafe mutations, and security hard failures. The seam is retained as an optional future capability, but this experiment is not capability-green and does not justify automatic escalation.
