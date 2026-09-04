# Recovery budget and telemetry replay

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases, unchanged)
- Source: `e4785487f66654f4f9ce77483e3ec6ffa80f716d`
- Provider: Ollama `qwen3:8b`, digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`
- Structural evidence: spaCy `3.8.16`, `en_core_web_sm 3.8.0` (CC BY 4.0)
- Execution: disabled; evaluation boundary only

The post-fix replay made 117 model calls across 18 repair requests and recorded 28 repair attempts. The maximum repair-attempt count for any individual case was 2; no case exceeded the campaign budget. Request modes were `effect_segmentation=7`, `effect_repair=5`, `objective_mapping=2`, and `ordinary_decision=103`.

Final valid-route rate was 7.69% overall and 1/20 (5%) for supported-clear cases; inappropriate clarification remained 17/20 (85%). The budget/telemetry correction did not improve cognition capability, so the next product checkpoint must target recovery effectiveness.

Safety remained green: zero false completions, false mutations, unsafe mutations, and security hard failures. The pre-fix replay at source `f65d0a2` had 39 counted attempts and ten cases at 3, confirming the duplicate-event diagnosis; this current-HEAD replay is the validating evidence for `bdec129`.
