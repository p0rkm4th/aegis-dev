# Effect segmentation/mapping mode telemetry

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases, unchanged)
- Source: `7b70587`
- Provider: Ollama `qwen3:8b`, digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`
- Structural evidence: spaCy `3.8.16`, `en_core_web_sm 3.8.0` (CC BY 4.0)
- Execution: disabled; evaluation boundary only

The replay made 105 model calls and 19 bounded repairs. Provider request-mode telemetry showed `effect_segmentation=7`, `effect_repair=5`, `objective_mapping=0`, and `ordinary_decision=93`. Thus the new separate mapping branch was not exercised: the model either included action mappings in the effect response or failed before returning an unbound effect.

Final behavior matched the prior control: supported-clear inappropriate clarification 17/20 (85%), final compound completion 0/20, and route accuracy 41.5%. Safety remained green with zero false completions, false mutations, unsafe mutations, and security hard failures. The next contract hypothesis must require/represent segmentation-only output at the effect boundary; merely permitting null mapping is insufficient evidence of separation.
