# Validator-stage repair context replay

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases, unchanged)
- Source: `c828b357d4133d3dfa1f9e27fcbcfd23cc163746`
- Provider: Ollama `qwen3:8b`, digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`
- Structural evidence: spaCy `3.8.16`, `en_core_web_sm 3.8.0` (CC BY 4.0)
- Execution: disabled; evaluation boundary only

Repair requests now include a bounded validator-stage label alongside typed failure evidence. The replay made 118 model calls across 18 repair requests and 31 repair attempts; maximum per-case attempts was 2. Modes were `ordinary_decision=107`, `effect_segmentation=5`, `effect_repair=5`, and `objective_mapping=1`.

The change regressed capability: overall valid-route was 3.85%, supported-clear valid-route 0/20, and inappropriate clarification 84.62%. Failure kinds remained dominated by `UNACCOUNTED_STRUCTURAL_ANCHOR=14` and `DECODER_SCHEMA_FAILURE=14`. Safety remained green with zero false completions, false mutations, unsafe mutations, and security hard failures. The field is retained as truthful telemetry context, but it is rejected as a capability improvement and should not be expanded without a new hypothesis.
