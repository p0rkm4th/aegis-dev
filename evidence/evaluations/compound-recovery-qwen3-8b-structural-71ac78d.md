# Qwen3:8B structural-anchor replay

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases; unchanged)
- Source revision: `71ac78d8ebc59d547d2be5bd40b4f0351e244a6f`
- Provider: Ollama Qwen3:8B, digest `500a1f067a9f...b2b8b41`
- Independent structural model: `en_core_web_sm`
- Report: `evaluation/reports/qwen3-8b-compound-41-structural-71ac78d.json`

The bounded structural-anchor refinement passed the full deterministic gate and
hosted validation, and it recognized previously missed independent predicates.
The frozen replay made 120 model calls, including 17 repair requests and 30
bounded attempts. Supported-clear final correctness was 2/20 (10%), with 90%
inappropriate clarification; overall route accuracy was 43.90% and overall
correct completion/answer rate was 11.54%. The result is not capability-green
and does not justify owner promotion or further parser expansion.

Safety remained green: Core false completions 0, false mutations 0, unsafe
mutations 0, and security hard failure false. The structural change is retained
as a small evidence-quality improvement, while the next hypothesis moves to
the recovery/decision boundary rather than weakening validation.
