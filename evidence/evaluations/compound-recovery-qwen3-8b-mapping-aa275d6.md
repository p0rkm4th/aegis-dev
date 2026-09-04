# Qwen3:8B capability-mapping recovery replay

- Corpus: `evaluation/compound_dev_owner_fresh.json` (41 cases; unchanged)
- Source revision: `aa275d6f8041e8656f1551d9980c7df0128689b9`
- Provider: Ollama Qwen3:8B, unchanged digest `500a1f067a9f...b2b8b41`
- Independent structural model: `en_core_web_sm`
- Report: `evaluation/reports/qwen3-8b-compound-41-mapping-recovery-aa275d6.json`

The production mapping seam now has bounded typed repair for malformed,
incomplete, unavailable-card, and undeclared-argument mappings, using an
ObjectiveSpec repair schema and the same candidate-bound validator. In this
replay the mapping pass ran four times but requested no mapping repair, so the
overall metrics were unchanged: supported-clear final correctness 2/20 (10%),
90% inappropriate clarification, overall route accuracy 46.34%, and overall
correct completion/answer rate 15.38%.

Safety remained green: false completions 0, false mutations 0, unsafe mutations
0, and security hard failure false. Deterministic validation passed 619 tests
with 5 skips. This is a production seam completion and safety-preserving
evidence, not a capability-green result; owner promotion is not warranted.
