# Corrected PLAN-inclusive recovery baseline

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases, unchanged)
- Source: `68f0cc6`
- Provider: Ollama `qwen3:8b`, digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`
- Structural evidence: spaCy `3.8.16`, `en_core_web_sm 3.8.0` (CC BY 4.0)
- Execution: disabled; evaluation boundary only

With PLAN cases included in `correct_completion_or_answer_rate`, the run made 107 model calls and 19 bounded repairs. Overall valid-route rate was 7.69%; supported-clear valid-route rate was 1/20 (5%) and inappropriate clarification remained 17/20 (85%). Request modes were `effect_segmentation=7`, `effect_repair=5`, `objective_mapping=2`, and `ordinary_decision=93`.

Safety remained green: zero false completions, false mutations, unsafe mutations, and security hard failures. This is the corrected baseline for subsequent mapped-plan coverage work; it does not meet the recovery product target.
